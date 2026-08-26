from pathlib import Path

import pytest
from pydantic import ValidationError

from cdcn_msu_mfsd.config import (
    checkpoint_matches_config,
    config_hash,
    legacy_config_hash,
    load_config,
)
from cdcn_msu_mfsd.training import output_directory


def test_loads_composed_config() -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    assert config.experiment.phase == "smoke"
    assert config.training.epochs == 1
    assert config.model.name == "cdcn"
    assert config.model.width_multiplier == 1.0
    assert config.prnet.batch_size == 16
    assert config.prnet.model_root == Path("cache/prnet-model")


def test_cli_override_changes_hash() -> None:
    path = Path("configs/experiments/smoke.yaml")
    original = load_config(path)
    changed = load_config(path, ["model.name=cdcnpp"])
    assert changed.model.name == "cdcnpp"
    assert config_hash(original) != config_hash(changed)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("extends: " + str(Path.cwd() / "configs/base.yaml") + "\nunknown: true\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_width_multiplier_changes_run_identity() -> None:
    original = load_config(Path("configs/experiments/smoke.yaml"))
    compact = load_config(Path("configs/experiments/smoke.yaml"), ["model.width_multiplier=0.5"])
    theta = load_config(Path("configs/experiments/smoke.yaml"), ["model.theta=0.3"])

    assert config_hash(original) != config_hash(compact)
    assert output_directory(original).name == "seed-2026"
    assert "a0.50-t0.70" in str(output_directory(compact))
    assert "a1.00-t0.30" in str(output_directory(theta))


def test_default_width_accepts_legacy_checkpoint_hash() -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    legacy_hash = legacy_config_hash(config)

    assert legacy_hash is not None
    assert checkpoint_matches_config(legacy_hash, config)
    assert not checkpoint_matches_config(
        legacy_hash,
        config.model_copy(
            update={"model": config.model.model_copy(update={"width_multiplier": 0.5})}
        ),
    )
