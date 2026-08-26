from pathlib import Path

from cdcn_msu_mfsd.experiments import expand_matrix, median_best_epoch


def test_main_matrix_has_sixty_unique_runs() -> None:
    configurations = expand_matrix(Path("configs/matrices/main.yaml"))
    assert len(configurations) == 60


def test_ablation_matrix_sizes() -> None:
    assert len(expand_matrix(Path("configs/ablations/theta.yaml"))) == 25
    assert len(expand_matrix(Path("configs/ablations/contrastive.yaml"))) == 15


def test_width_matrix_has_three_widths_across_folds_and_seeds() -> None:
    configs = expand_matrix(Path("configs/matrices/small_width.yaml"))

    assert len(configs) == 45
    assert {config.model.width_multiplier for config in configs} == {1.0, 0.5, 0.25}
    assert {config.model.theta for config in configs} == {0.7}
    assert {config.experiment.seed for config in configs} == {42, 2026, 2027}


def test_matrix_overrides_select_a_theta_sweep_width() -> None:
    configs = expand_matrix(
        Path("configs/ablations/theta.yaml"),
        ["model.width_multiplier=0.25", "experiment.name=small-theta-a025"],
    )

    assert len(configs) == 25
    assert {config.model.width_multiplier for config in configs} == {0.25}
    assert {config.experiment.name for config in configs} == {"small-theta-a025"}


def test_median_best_epoch_uses_one_based_epochs(tmp_path: Path) -> None:
    paths = []
    for index, epoch in enumerate((3, 7, 5, 4, 8)):
        path = tmp_path / f"summary-{index}.json"
        path.write_text(f'{{"best_epoch": {epoch}}}\n')
        paths.append(path)
    assert median_best_epoch(paths) == 6
