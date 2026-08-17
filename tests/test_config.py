from pathlib import Path

import pytest
from pydantic import ValidationError

from cdcn_msu_mfsd.config import config_hash, load_config


def test_loads_composed_config() -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    assert config.experiment.phase == "smoke"
    assert config.training.epochs == 1
    assert config.model.name == "cdcn"


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
