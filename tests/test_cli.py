from typer.testing import CliRunner

from cdcn_msu_mfsd.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_doctor_reports_torch_runtime() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert '"cuda_available"' in result.stdout
    assert '"torch": "2.11.0' in result.stdout


def test_matrix_dry_run_does_not_train() -> None:
    result = runner.invoke(
        app,
        ["run-matrix", "configs/ablations/contrastive.yaml", "--dry-run"],
    )
    assert result.exit_code == 0
    assert result.stdout.count('"status": "planned"') == 15
