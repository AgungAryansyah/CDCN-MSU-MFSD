import pytest
from typer.testing import CliRunner

from cdcn_msu_mfsd import cli
from cdcn_msu_mfsd.cli import app
from cdcn_msu_mfsd.data.manifest import VideoRecord

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
