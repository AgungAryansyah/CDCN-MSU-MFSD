import json
import os
import time
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import wandb
from dotenv import load_dotenv

from cdcn_msu_mfsd.config import Config, resolved_config


class Tracker(Protocol):
    def log(self, metrics: Mapping[str, int | float], step: int | None = None) -> None: ...

    def summary(self, metrics: Mapping[str, int | float | str]) -> None: ...

    def finish(self) -> None: ...


@dataclass(frozen=True)
class TrackingContext:
    name: str
    group: str
    job_type: str
    tags: tuple[str, ...] = ()
    config: Mapping[str, Any] | None = None
    required_online: bool = False
    custom_step_metric: bool = False


class MetricsTracker:
    def __init__(
        self,
        local_path: Path,
        run: Any | None = None,
        required_online: bool = False,
        custom_step_metric: bool = False,
        metadata_path: Path | None = None,
    ) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path = local_path
        self.run = run
        self.required_online = required_online
        self.custom_step_metric = custom_step_metric
        self.metadata_path = metadata_path
        self.errors: list[str] = []
        self.last_custom_step: int = 0

    def _attempt(self, action: Any, label: str) -> None:
        error: Exception | None = None
        for attempt in range(3):
            try:
                action()
                return
            except Exception as caught:
                error = caught
                if attempt < 2:
                    time.sleep(2**attempt)
        message = f"W&B {label} failed after three attempts: {error}"
        self.errors.append(message)
        if self.required_online:
            raise RuntimeError(message) from error
        warnings.warn(message, stacklevel=2)

    def log(self, metrics: Mapping[str, int | float], step: int | None = None) -> None:
        record: dict[str, int | float] = dict(metrics)
        if step is not None:
            record["step"] = step
            if self.custom_step_metric:
                record["pipeline/global_step"] = step
                self.last_custom_step = step
        elif self.custom_step_metric:
            record["pipeline/global_step"] = self.last_custom_step
        with self.local_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if self.run is not None:
            payload: dict[str, int | float] = dict(metrics)
            if self.custom_step_metric:
                payload["pipeline/global_step"] = (
                    step if step is not None else self.last_custom_step
                )
                self._attempt(lambda: self.run.log(payload), "logging")
            else:
                self._attempt(lambda: self.run.log(payload, step=step), "logging")

    def summary(self, metrics: Mapping[str, int | float | str]) -> None:
        path = self.local_path.with_name("summary.json")
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing.update(metrics)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        if self.run is not None:
            self._attempt(lambda: self.run.summary.update(dict(metrics)), "summary update")

    def finish(self) -> None:
        if self.run is not None:
            self._attempt(self.run.finish, "finish")
        if self.metadata_path is not None:
            state = (
                json.loads(self.metadata_path.read_text()) if self.metadata_path.exists() else {}
            )
            state["status"] = "finished" if not self.errors else "failed"
            state["errors"] = self.errors
            _write_json(self.metadata_path, state)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluation_metrics(split: str, report: Mapping[str, Any]) -> dict[str, int | float]:
    false_positive = int(report["false_accepts"])
    false_negative = int(report["false_rejects"])
    genuine = int(report["genuine"])
    attacks = int(report["attacks"])
    return {
        f"{split}/acer": float(report["acer"]),
        f"{split}/apcer": float(report["apcer"]),
        f"{split}/bpcer": float(report["bpcer"]),
        f"{split}/eer": float(report["eer"]),
        f"{split}/roc_auc": float(report["auc"]),
        f"{split}/f1_attack": float(report["f1"]),
        f"{split}/accuracy": float(report["accuracy"]),
        f"{split}/threshold": float(report["threshold"]),
        f"{split}/confusion/true_positive": genuine - false_negative,
        f"{split}/confusion/true_negative": attacks - false_positive,
        f"{split}/confusion/false_positive": false_positive,
        f"{split}/confusion/false_negative": false_negative,
        f"{split}/count/genuine": genuine,
        f"{split}/count/attack": attacks,
    }


def create_tracker(
    config: Config,
    output_dir: Path,
    provenance: Mapping[str, Any],
    context: TrackingContext | None = None,
) -> MetricsTracker:
    load_dotenv()
    local_path = output_dir / "metrics.jsonl"
    mode = os.getenv("WANDB_MODE", "offline")
    enabled = config.tracking.enabled and mode != "disabled"
    if context is not None and context.required_online and mode != "online":
        raise ValueError("This experiment stage requires WANDB_MODE=online")
    if not enabled:
        if context is not None and context.required_online:
            raise ValueError("Online W&B tracking is required for this experiment stage")
        return MetricsTracker(local_path)
    if mode not in {"online", "offline"}:
        raise ValueError("WANDB_MODE must be online, offline, or disabled")
    wandb_mode = cast(Literal["online", "offline"], mode)
    project = os.getenv("WANDB_PROJECT")
    if not project:
        raise ValueError("WANDB_PROJECT must be configured in .env")
    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY must be configured in .env for online tracking")
    run_config = resolved_config(config)
    run_config["provenance"] = dict(provenance)
    metadata_path: Path | None = None
    run_id: str | None = None
    run_name = f"{config.experiment.name}-f{config.experiment.fold}-s{config.experiment.seed}"
    group = config.experiment.name
    job_type: str = config.experiment.phase
    tags: list[str] = [config.model.name, config.supervision.kind]
    resume: str | None = None
    if context is not None:
        run_config["pipeline"] = dict(context.config or {})
        run_name = context.name
        group = context.group
        job_type = context.job_type
        tags.extend(context.tags)
        if context.custom_step_metric:
            run_config["pipeline_step_metric"] = "pipeline/global_step"
        metadata_path = output_dir / "wandb_run.json"
        previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        run_id = context_run_id = previous.get("run_id")
        if not isinstance(context_run_id, str):
            run_id = uuid.uuid4().hex[:8]
            resume = "never"
        else:
            resume = "allow" if previous.get("status") == "starting" else "must"
        _write_json(
            metadata_path,
            {
                "run_id": run_id,
                "name": run_name,
                "group": group,
                "job_type": job_type,
                "status": "starting",
            },
        )
    init_args: dict[str, Any] = {
        "entity": os.getenv("WANDB_ENTITY") or None,
        "project": project,
        "dir": output_dir,
        "name": run_name,
        "group": group,
        "job_type": job_type,
        "tags": tags,
        "config": run_config,
        "mode": wandb_mode,
        "save_code": False,
        "settings": wandb.Settings(
            allow_offline_artifacts=False,
            disable_code=True,
            disable_git=True,
            x_save_requirements=False,
            ignore_globs=("*.pt", "*.pth", "*.ckpt", "*.jpg", "*.png", "*.npy"),
        ),
    }
    if run_id is not None:
        init_args.update({"id": run_id, "resume": resume})
    run = wandb.init(**init_args)
    if context is not None and context.custom_step_metric:
        run.define_metric("pipeline/global_step")
        run.define_metric("*", step_metric="pipeline/global_step")
    if metadata_path is not None:
        _write_json(
            metadata_path,
            {
                "run_id": run.id,
                "name": run.name,
                "url": run.url,
                "group": group,
                "job_type": job_type,
                "status": "running",
            },
        )
    return MetricsTracker(
        local_path,
        run,
        required_online=context.required_online if context is not None else False,
        custom_step_metric=context.custom_step_metric if context is not None else False,
        metadata_path=metadata_path,
    )
