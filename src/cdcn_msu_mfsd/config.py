import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(StrictModel):
    dataset_root: Path
    train_subjects: Path
    test_subjects: Path
    cache_root: Path
    output_root: Path
    depth_root: Path


class DataConfig(StrictModel):
    candidates_per_video: int = Field(default=32, gt=0)
    frames_per_video: int = Field(default=8, gt=0)
    crop_scale: float = Field(default=1.3, gt=0)
    crop_scale_min: float = Field(default=1.2, gt=0)
    crop_scale_max: float = Field(default=1.5, gt=0)
    workers: int = Field(default=8, ge=0)

    @model_validator(mode="after")
    def validate_crop_range(self) -> "DataConfig":
        if self.crop_scale_min > self.crop_scale_max:
            raise ValueError("crop_scale_min must not exceed crop_scale_max")
        return self


class ModelConfig(StrictModel):
    name: Literal["cdcn", "cdcnpp"] = "cdcn"
    theta: float = Field(default=0.7, ge=0, le=1)


class SupervisionConfig(StrictModel):
    kind: Literal["binary", "pseudo_depth"] = "binary"


class LossConfig(StrictModel):
    absolute_weight: float = Field(default=1.0, ge=0)
    contrastive_weight: float = Field(default=1.0, ge=0)


class OptimizerConfig(StrictModel):
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=5e-5, ge=0)


class SchedulerConfig(StrictModel):
    step_size: int = Field(default=60, gt=0)
    gamma: float = Field(default=0.5, gt=0, le=1)


class TrainingConfig(StrictModel):
    epochs: int = Field(default=200, gt=0)
    batch_size: int = Field(default=8, gt=0)
    gradient_accumulation: int = Field(default=2, gt=0)
    patience: int = Field(default=30, gt=0)


class RuntimeConfig(StrictModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    amp: bool = True
    compile: bool = False
    deterministic: bool = True
    allow_tf32: bool = True


class TrackingConfig(StrictModel):
    enabled: bool = True
    log_every_steps: int = Field(default=20, gt=0)


class ExperimentConfig(StrictModel):
    name: str = "baseline"
    fold: int = Field(default=0, ge=0, le=4)
    seed: int = 2026
    phase: Literal["cv", "refit", "test", "smoke"] = "cv"


class Config(StrictModel):
    paths: PathsConfig
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    supervision: SupervisionConfig = SupervisionConfig()
    loss: LossConfig = LossConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    training: TrainingConfig = TrainingConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    tracking: TrackingConfig = TrackingConfig()
    experiment: ExperimentConfig = ExperimentConfig()


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_mapping(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        raise ValueError(f"Cyclic YAML inheritance detected at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ValueError(f"extends must be a string or list: {path}")
    merged: dict[str, Any] = {}
    for parent in parents:
        if not isinstance(parent, str):
            raise ValueError(f"extends entries must be strings: {path}")
        merged = _merge(merged, _load_mapping(path.parent / parent, (*stack, resolved)))
    return _merge(merged, raw)


def _set_override(mapping: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    target = mapping
    for key in keys[:-1]:
        value = target.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"Cannot override nested key below {key}")
        target = value
    target[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: Path, overrides: list[str] | None = None) -> Config:
    mapping = _load_mapping(path)
    for expression in overrides or []:
        _set_override(mapping, expression)
    return Config.model_validate(mapping)


def resolved_config(config: Config) -> dict[str, Any]:
    return config.model_dump(mode="json")


def config_hash(config: Config) -> str:
    payload = json.dumps(resolved_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def save_resolved_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved_config(config), sort_keys=True))
