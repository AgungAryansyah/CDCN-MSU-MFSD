import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Sampler

from cdcn_msu_mfsd.config import Config, config_hash, save_resolved_config
from cdcn_msu_mfsd.data.dataset import MSUMFSDDataset, Sample, TargetProvider
from cdcn_msu_mfsd.data.preprocessing import load_frame_records
from cdcn_msu_mfsd.data.protocol import load_folds
from cdcn_msu_mfsd.data.supervision import (
    BinaryTarget,
    PseudoDepthTarget,
    PseudoDepthValidation,
    validate_pseudo_depth_targets,
)
from cdcn_msu_mfsd.losses import DepthLoss
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.tracking import MetricsTracker, create_tracker, evaluation_metrics


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class EpochRandomSampler(Sampler[int]):
    def __init__(self, dataset: MSUMFSDDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed

    def __iter__(self) -> Any:
        generator = torch.Generator().manual_seed(self.seed + self.dataset.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def _loader(
    dataset: MSUMFSDDataset,
    config: Config,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[Sample]:
    generator = torch.Generator().manual_seed(config.experiment.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=EpochRandomSampler(dataset, config.experiment.seed) if shuffle else None,
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def make_dataloaders(
    config: Config, device: torch.device
) -> tuple[DataLoader[Sample], DataLoader[Sample]]:
    frame_records = load_frame_records(config.paths.cache_root / "frames.jsonl")
    fold = load_folds(config.paths.cache_root / "folds.json")[config.experiment.fold]
    target_provider: TargetProvider
    if config.supervision.kind == "binary":
        target_provider = BinaryTarget()
    else:
        target_provider = PseudoDepthTarget(config.paths.depth_root)
    train_dataset = MSUMFSDDataset(
        frame_records,
        config.paths.cache_root,
        target_provider,
        set(fold.train_subjects),
        True,
        config.data.frames_per_video,
        config.experiment.seed,
        config.data.crop_scale,
        (config.data.crop_scale_min, config.data.crop_scale_max),
    )
    validation_dataset = MSUMFSDDataset(
        frame_records,
        config.paths.cache_root,
        target_provider,
        set(fold.validation_subjects),
        False,
        config.data.frames_per_video,
        config.experiment.seed,
        config.data.crop_scale,
    )
    return _loader(train_dataset, config, True, device), _loader(
        validation_dataset, config, False, device
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(
    config: Config, depth_validation: PseudoDepthValidation | None = None
) -> dict[str, str]:
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"
    files = {
        "frames_sha256": config.paths.cache_root / "frames.jsonl",
        "folds_sha256": config.paths.cache_root / "folds.json",
    }
    result = {"git_sha": git_sha, "config_sha256": config_hash(config)}
    result.update({key: _file_hash(path) for key, path in files.items() if path.exists()})
    depth_provenance = config.paths.depth_root / "provenance.json"
    if depth_validation is not None:
        result["pseudo_depth_targets"] = str(depth_validation.targets)
        result["pseudo_depth_sha256"] = depth_validation.sha256
    if depth_provenance.exists():
        result["pseudo_depth_provenance_sha256"] = _file_hash(depth_provenance)
        metadata = json.loads(depth_provenance.read_text())
        for key in ("contract_sha256", "generation_sha256", "checkpoint_sha256"):
            value = metadata.get(key)
            if isinstance(value, str):
                result[f"pseudo_depth_{key}"] = value
    return result


def output_directory(config: Config) -> Path:
    run_name = f"{config.model.name}-{config.supervision.kind}"
    return (
        config.paths.output_root
        / config.experiment.name
        / run_name
        / f"fold-{config.experiment.fold}"
        / f"seed-{config.experiment.seed}"
    )


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    epoch: int,
    best_epoch: int,
    best_loss: float,
    global_step: int,
    epochs_without_improvement: int,
    model: nn.Module,
    optimizer: Adam,
    scheduler: StepLR,
    scaler: Any,
    config: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "global_step": global_step,
            "epochs_without_improvement": epochs_without_improvement,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config_hash": config_hash(config),
            "rng": _rng_state(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    scheduler: StepLR,
    scaler: Any,
    config: Config,
) -> tuple[int, int, float, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["config_hash"] != config_hash(config):
        raise ValueError("Checkpoint configuration does not match the requested run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    _restore_rng_state(checkpoint["rng"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["best_epoch"]),
        float(checkpoint["best_loss"]),
        int(checkpoint["global_step"]),
        int(checkpoint["epochs_without_improvement"]),
    )


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader[Sample],
        validation_loader: DataLoader[Sample],
        tracker: MetricsTracker,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.training_model = (
            cast(nn.Module, torch.compile(self.model)) if config.runtime.compile else self.model
        )
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.tracker = tracker
        self.device = device
        self.loss = DepthLoss(config.loss.absolute_weight, config.loss.contrastive_weight).to(
            device
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scheduler = StepLR(
            self.optimizer, step_size=config.scheduler.step_size, gamma=config.scheduler.gamma
        )
        amp_enabled = config.runtime.amp and device.type == "cuda"
        self.scaler: Any = torch.amp.GradScaler("cuda", enabled=amp_enabled)  # type: ignore[attr-defined]
        self.amp_enabled = amp_enabled

    def _batch(self, batch: Sample) -> tuple[Tensor, Tensor]:
        images = batch["image"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        return images, targets

    def train_epoch(self, epoch: int, global_step: int) -> tuple[dict[str, float], int]:
        self.training_model.train()
        dataset = self.train_loader.dataset
        if isinstance(dataset, MSUMFSDDataset):
            dataset.set_epoch(epoch)
        totals = {"total": 0.0, "absolute": 0.0, "contrastive": 0.0}
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(self.train_loader):
            images, targets = self._batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                predictions = cast(Tensor, self.training_model(images))
                components = self.loss.components(predictions, targets)
                scaled_loss = components["total"] / self.config.training.gradient_accumulation
            cast(Tensor, self.scaler.scale(scaled_loss)).backward()  # type: ignore[no-untyped-call]
            should_step = (
                batch_index + 1
            ) % self.config.training.gradient_accumulation == 0 or batch_index + 1 == len(
                self.train_loader
            )
            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            for name in totals:
                totals[name] += float(components[name].detach())
            global_step += 1
            if global_step % self.config.tracking.log_every_steps == 0:
                self.tracker.log(
                    {f"train/{name}": float(value.detach()) for name, value in components.items()},
                    global_step,
                )
        return {name: value / len(self.train_loader) for name, value in totals.items()}, global_step

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        self.training_model.eval()
        totals = {"total": 0.0, "absolute": 0.0, "contrastive": 0.0}
        for batch in self.validation_loader:
            images, targets = self._batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                predictions = cast(Tensor, self.training_model(images))
                components = self.loss.components(predictions, targets)
            for name in totals:
                totals[name] += float(components[name])
        return {name: value / len(self.validation_loader) for name, value in totals.items()}

    def fit(self, output_dir: Path, resume: Path | None = None) -> TrainingResult:
        best_loss = float("inf")
        best_epoch = -1
        start_epoch = 0
        global_step = 0
        epochs_without_improvement = 0
        if resume is not None:
            (
                start_epoch,
                best_epoch,
                best_loss,
                global_step,
                epochs_without_improvement,
            ) = load_checkpoint(
                resume,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.config,
            )
        epochs_completed = start_epoch
        for epoch in range(start_epoch, self.config.training.epochs):
            train_metrics, global_step = self.train_epoch(epoch, global_step)
            validation_metrics = self.validate()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                **{f"train_epoch/{key}": value for key, value in train_metrics.items()},
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
            }
            if self.device.type == "cuda":
                epoch_metrics["gpu/max_memory_gib"] = torch.cuda.max_memory_reserved() / (1024**3)
            self.tracker.log(epoch_metrics, global_step)
            current_loss = validation_metrics["total"]
            improved = current_loss < best_loss
            if improved:
                best_loss = current_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            self.scheduler.step()
            if improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    epoch,
                    best_epoch,
                    best_loss,
                    global_step,
                    epochs_without_improvement,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.config,
                )
            save_checkpoint(
                output_dir / "last.pt",
                epoch,
                best_epoch,
                best_loss,
                global_step,
                epochs_without_improvement,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.config,
            )
            epochs_completed = epoch + 1
            if epochs_without_improvement >= self.config.training.patience:
                break
        self.tracker.summary(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "epochs_completed": epochs_completed,
            }
        )
        return TrainingResult(output_dir, best_epoch, best_loss, epochs_completed)


def run_training(config: Config, resume: Path | None = None) -> TrainingResult:
    seed_everything(config.experiment.seed, config.runtime.deterministic)
    device = select_device(config.runtime.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    depth_validation = None
    if config.supervision.kind == "pseudo_depth":
        frame_records = load_frame_records(config.paths.cache_root / "frames.jsonl")
        depth_validation = validate_pseudo_depth_targets(frame_records, config.paths.depth_root)
    output_dir = output_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "config.yaml")
    run_provenance = provenance(config, depth_validation)
    (output_dir / "provenance.json").write_text(
        json.dumps(run_provenance, indent=2, sort_keys=True) + "\n"
    )
    train_loader, validation_loader = make_dataloaders(config, device)
    tracker = create_tracker(config, output_dir, run_provenance)
    try:
        trainer = Trainer(
            config,
            build_model(config.model.name, config.model.theta),
            train_loader,
            validation_loader,
            tracker,
            device,
        )
        result = trainer.fit(output_dir, resume)
        from cdcn_msu_mfsd.evaluation import evaluate_checkpoint

        report = evaluate_checkpoint(config, output_dir / "best.pt", "validation", output_dir)
        pad_metrics = evaluation_metrics("validation", report)
        tracker.log(pad_metrics)
        tracker.summary(pad_metrics)
        return result
    finally:
        tracker.finish()
