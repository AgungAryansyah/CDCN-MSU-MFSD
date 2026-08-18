from pathlib import Path

import pytest
from typer.testing import CliRunner

from cdcn_msu_mfsd import cli
from cdcn_msu_mfsd.cli import app
from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.data.manifest import VideoRecord
from cdcn_msu_mfsd.data.supervision import PRNetRecord
from cdcn_msu_mfsd.prnet_generation import PRNetGenerationReport

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_doctor_reports_torch_runtime() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert '"cuda_available"' in result.stdout
    assert '"ffmpeg_available"' in result.stdout
    assert '"ffprobe_available"' in result.stdout
    assert '"prnet"' in result.stdout
    assert '"torch": "2.11.0' in result.stdout


def test_inspect_data_skips_metadata_probe_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_manifest(*args: object, **kwargs: object) -> list[VideoRecord]:
        assert args[-1] is False
        return []

    monkeypatch.setattr(cli, "build_manifest", build_manifest)
    result = runner.invoke(app, ["inspect-data"])

    assert result.exit_code == 0


def test_matrix_dry_run_does_not_train() -> None:
    result = runner.invoke(
        app,
        ["run-matrix", "configs/ablations/contrastive.yaml", "--dry-run"],
    )
    assert result.exit_code == 0
    assert result.stdout.count('"status": "planned"') == 15


def test_prepare_prnet_runs_integrated_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(Path("configs/base.yaml"))
    config = config.model_copy(
        update={"prnet": config.prnet.model_copy(update={"workers": 0, "device": "cpu"})}
    )
    record = PRNetRecord("video", 0, "inputs/video.png", "video__000000")
    report = PRNetGenerationReport(1, 1, 0, "outputs", "cpu", "contract", "model", 0.1)
    monkeypatch.setattr(cli, "load_config", lambda *_: config)
    monkeypatch.setattr(cli, "fetch_prnet_artifacts", lambda *_: [])
    monkeypatch.setattr(cli, "load_frame_records", lambda *_: [])
    monkeypatch.setattr(cli, "export_prnet_inputs", lambda *_: [record])
    monkeypatch.setattr(cli, "generate_prnet_outputs", lambda *_: report)
    monkeypatch.setattr(cli, "import_prnet_outputs", lambda *_: 1)

    result = runner.invoke(app, ["prepare-prnet"])

    assert result.exit_code == 0
    assert '"exported": 1' in result.stdout
    assert '"generated": 1' in result.stdout
    assert '"imported": 1' in result.stdout
