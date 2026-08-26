import json
from pathlib import Path

from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.experiments import expand_matrix
from cdcn_msu_mfsd.reporting import collect_results
from cdcn_msu_mfsd.training import output_directory


def test_collect_results_writes_oof_comparison(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                f"base: {Path('configs/base.yaml').resolve()}",
                "name: report-test",
                "fixed:",
                "  model.name: cdcn",
                "  model.width_multiplier: 0.25",
                "  model.theta: 0.7",
                "  supervision.kind: pseudo_depth",
                "axes:",
                "  experiment.fold: [0, 1, 2, 3, 4]",
                "  experiment.seed: [42]",
                "",
            ]
        )
    )
    output_root = tmp_path / "outputs"
    configs = expand_matrix(matrix_path, [f"paths.output_root={json.dumps(str(output_root))}"])
    for config in configs:
        output_dir = output_directory(config)
        output_dir.mkdir(parents=True)
        videos = []
        for index in range(6):
            videos.append(
                {
                    "video_id": f"real-{config.experiment.fold}-{index}",
                    "label": 1,
                    "score": 0.9,
                    "attack_type": "genuine",
                    "device": "android",
                }
            )
        for index in range(18):
            videos.append(
                {
                    "video_id": f"attack-{config.experiment.fold}-{index}",
                    "label": 0,
                    "score": 0.1,
                    "attack_type": "printed_photo",
                    "device": "android",
                }
            )
        (output_dir / "validation_videos.jsonl").write_text(
            "\n".join(json.dumps(video) for video in videos) + "\n"
        )
        (output_dir / "split.json").write_text(
            json.dumps(
                {
                    "train_subjects": ["001"],
                    "validation_subjects": [f"00{config.experiment.fold + 2}"],
                    "test_subjects": ["020"],
                }
            )
        )
        (output_dir / "provenance.json").write_text(json.dumps({"git_sha": "test-sha"}))
        (output_dir / "efficiency.json").write_text(
            json.dumps(
                {
                    "device": "cpu",
                    "input_shape": [1, 3, 256, 256],
                    "trainable_parameters": 10,
                    "total_parameters": 10,
                    "macs": 20,
                    "model_state_size_mb": 0.1,
                    "latency_ms": 1.5,
                }
            )
        )

    output_csv = tmp_path / "results" / "width_ablation.csv"
    rows = collect_results(configs, output_csv)

    assert len(rows) == 1
    assert rows[0]["metric_split"] == "validation_oof"
    assert rows[0]["f1"] == 1.0
    assert output_csv.exists()
    assert output_csv.with_name("width_ablation_summary.csv").exists()
    assert output_csv.with_suffix(".json").exists()


def test_collect_results_refuses_to_replace_existing_artifacts(tmp_path: Path) -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    output_csv = tmp_path / "results.csv"
    output_csv.write_text("existing\n")

    try:
        collect_results([config], output_csv)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing result artifact to be rejected")
