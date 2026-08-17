from typer.testing import CliRunner

from cdcn_msu_mfsd.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_matrix_dry_run_does_not_train() -> None:
    result = runner.invoke(
        app,
        ["run-matrix", "configs/ablations/contrastive.yaml", "--dry-run"],
    )
    assert result.exit_code == 0
    assert result.stdout.count('"status": "planned"') == 15
