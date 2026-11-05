import fcntl
import hashlib
import json
import os
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cdcn_msu_mfsd.config import Config, load_config
from cdcn_msu_mfsd.training import output_directory


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_config: Path
    width_experiment: str
    theta_experiment: str
    model: str = "cdcn"
    supervision: str = "pseudo_depth"
    widths: list[float] = Field(default_factory=lambda: [1.0, 0.5, 0.25])
    primary_width: float = 0.5
    theta_values: list[float] = Field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 1.0])
    seeds: list[int] = Field(default_factory=lambda: [42, 2026, 2027])
    folds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    test_videos: int = 160
    test_genuine: int = 40
    test_attacks: int = 120


@dataclass(frozen=True)
class Condition:
    width: float
    theta: float

    @property
    def key(self) -> str:
        return f"{self.model_name}-a{self.width:.2f}-t{self.theta:.2f}-pseudo_depth"

    @property
    def model_name(self) -> str:
        return "cdcn"


class Campaign:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        raw = yaml.safe_load(self.path.read_text())
        self.settings = CampaignConfig.model_validate(raw)
        self.settings = self.settings.model_copy(
            update={"base_config": (self.path.parent / self.settings.base_config).resolve()}
        )
        self.base = load_config(self.settings.base_config)
        self.root = self.base.paths.output_root / "pipelines" / self.settings.name
        self.state_path = self.root / "state.json"
        self.status_path = self.root / "status.json"

    @property
    def conditions(self) -> list[Condition]:
        return sorted(
            {
                Condition(width, theta)
                for width, theta in [(width, 0.7) for width in self.settings.widths]
                + [(self.settings.primary_width, theta) for theta in self.settings.theta_values]
            },
            key=lambda item: (item.width, item.theta),
        )

    @property
    def plan_path(self) -> Path:
        return self.root / "frozen" / "final_plan.json"

    @property
    def plan_digest_path(self) -> Path:
        return self.root / "frozen" / "final_plan.sha256"

    def config(
        self,
        condition: Condition,
        fold: int = 0,
        seed: int = 2026,
        phase: str = "cv",
        epochs: int | None = None,
        experiment_name: str | None = None,
    ) -> Config:
        if experiment_name is None:
            if phase == "refit":
                experiment_name = f"thesis-completion-{condition.key}"
            elif condition.theta == 0.7 and condition.width in self.settings.widths:
                experiment_name = self.settings.width_experiment
            else:
                experiment_name = self.settings.theta_experiment
        overrides = [
            f"model.name={self.settings.model}",
            f"model.width_multiplier={condition.width}",
            f"model.theta={condition.theta}",
            f"supervision.kind={self.settings.supervision}",
            f"experiment.name={experiment_name}",
            f"experiment.fold={fold}",
            f"experiment.seed={seed}",
            f"experiment.phase={phase}",
        ]
        if epochs is not None:
            overrides.append(f"training.epochs={epochs}")
        return load_config(self.settings.base_config, overrides)

    def cv_directory(self, condition: Condition, fold: int, seed: int) -> Path:
        return output_directory(self.config(condition, fold, seed))

    def oof_directory(self, condition: Condition, seed: int) -> Path:
        return self.root / "oof" / condition.key / f"seed-{seed}"

    def refit_directory(self, condition: Condition, seed: int) -> Path:
        return self.root / "refits" / condition.key / f"seed-{seed}"

    def test_directory(self, condition: Condition, seed: int) -> Path:
        return self.root / "tests" / condition.key / f"seed-{seed}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(campaign: Campaign) -> dict[str, str]:
    files = {
        "campaign": campaign.path,
        "base_config": campaign.settings.base_config,
        "frames": campaign.base.paths.cache_root / "frames.jsonl",
        "folds": campaign.base.paths.cache_root / "folds.json",
        "depth_provenance": campaign.base.paths.depth_root / "provenance.json",
    }
    result = {name: sha256_file(path) for name, path in files.items() if path.is_file()}
    source_digest = hashlib.sha256()
    source_root = Path("src/cdcn_msu_mfsd")
    for path in sorted(source_root.rglob("*.py")):
        source_digest.update(str(path.relative_to(source_root)).encode())
        source_digest.update(bytes.fromhex(sha256_file(path)))
    result["source_sha256"] = source_digest.hexdigest()
    return result


@contextmanager
def campaign_lock(campaign: Campaign) -> Iterator[None]:
    if os.getenv("CDCN_PIPELINE_LOCK_HELD") == "1":
        yield
        return
    lock_path = campaign.root / "campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process holds the campaign lock: {lock_path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def update_campaign_state(
    campaign: Campaign,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    state = read_json(campaign.state_path) if campaign.state_path.exists() else {}
    state.setdefault("campaign", campaign.settings.name)
    state.setdefault("stages", {})
    record = state["stages"].setdefault(stage, {})
    record.update(
        {
            "status": status,
            "updated_at_unix": time.time(),
            "input_fingerprint": input_fingerprint(campaign),
        }
    )
    if status == "running":
        record["started_at_unix"] = time.time()
    if details:
        record.update(details)
    write_json(campaign.state_path, state)


class Progress:
    def __init__(self, campaign: Campaign, stage: str, total: int) -> None:
        self.campaign = campaign
        self.stage = stage
        self.total = total
        self.started = time.monotonic()
        self.durations: list[float] = []
        self.completed = 0
        self.update(0, "starting")

    def update(self, completed: int, current: str, elapsed_job: float | None = None) -> None:
        now = time.monotonic()
        if elapsed_job is not None:
            self.durations.append(elapsed_job)
        elapsed = now - self.started
        remaining = max(0, self.total - completed)
        eta = statistics.fmean(self.durations) * remaining if self.durations else None
        report = {
            "stage": self.stage,
            "completed": completed,
            "total": self.total,
            "percent": 100.0 * completed / self.total if self.total else 100.0,
            "current": current,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "updated_at_unix": time.time(),
        }
        write_json(self.campaign.status_path, report)
        eta_text = f"{eta / 60:.1f}m" if eta is not None else "unknown"
        print(
            f"[{self.stage}] {completed}/{self.total} ({report['percent']:.1f}%) "
            f"{current}; elapsed={elapsed / 60:.1f}m; eta={eta_text}",
            flush=True,
        )
        self.completed = completed


def conditions_for_widths(campaign: Campaign) -> list[Condition]:
    return [Condition(width, 0.7) for width in campaign.settings.widths]


def conditions_for_theta(campaign: Campaign) -> list[Condition]:
    return [
        Condition(campaign.settings.primary_width, theta)
        for theta in campaign.settings.theta_values
    ]
