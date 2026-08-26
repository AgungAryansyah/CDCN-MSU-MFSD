import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from cdcn_msu_mfsd.config import Config
from cdcn_msu_mfsd.efficiency import benchmark_config, save_efficiency_report
from cdcn_msu_mfsd.experiments import aggregate_oof
from cdcn_msu_mfsd.training import output_directory

_RESULT_FIELDS = (
    "experiment_name",
    "architecture",
    "width_multiplier",
    "theta",
    "seed",
    "metric_split",
    "input_resolution",
    "folds",
    "train_subjects",
    "validation_subjects",
    "test_subjects",
    "apcer",
    "bpcer",
    "acer",
    "f1",
    "auc",
    "eer",
    "accuracy",
    "threshold",
    "trainable_parameters",
    "total_parameters",
    "macs",
    "model_state_size_mb",
    "latency_ms_cpu",
    "latency_ms_gpu",
    "checkpoint_paths",
    "git_commit",
)

_SUMMARY_FIELDS = (
    "experiment_name",
    "architecture",
    "width_multiplier",
    "theta",
    "seeds",
    "apcer_mean",
    "apcer_std",
    "bpcer_mean",
    "bpcer_std",
    "acer_mean",
    "acer_std",
    "f1_mean",
    "f1_std",
    "auc_mean",
    "auc_std",
    "trainable_parameters",
    "total_parameters",
    "macs",
    "model_state_size_mb",
    "latency_ms_cpu",
    "latency_ms_gpu",
)


def _architecture_key(config: Config) -> tuple[str, float, float, str]:
    return (
        config.model.name,
        config.model.width_multiplier,
        config.model.theta,
        config.supervision.kind,
    )


def _result_key(config: Config) -> tuple[str, str, float, float, str, int]:
    return (
        config.experiment.name,
        config.model.name,
        config.model.width_multiplier,
        config.model.theta,
        config.supervision.kind,
        config.experiment.seed,
    )


def _architecture_label(config: Config) -> str:
    return f"{config.model.name.upper()}-a{config.model.width_multiplier:.2f}"


def benchmark_configurations(
    configurations: list[Config],
    device: str = "cpu",
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> list[dict[str, str | float | int]]:
    grouped: dict[tuple[str, float, float, str], list[Config]] = defaultdict(list)
    for config in configurations:
        grouped[_architecture_key(config)].append(config)
    results: list[dict[str, str | float | int]] = []
    for configs in grouped.values():
        report = benchmark_config(configs[0], device, warmup_runs, timed_runs)
        for config in configs:
            save_efficiency_report(config, report)
        results.append(
            {
                "architecture": _architecture_label(configs[0]),
                "width_multiplier": configs[0].model.width_multiplier,
                "theta": configs[0].model.theta,
                "device": report.device,
                "latency_ms": report.latency_ms,
            }
        )
    return results


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _efficiency(configurations: list[Config]) -> dict[str, Any]:
    for config in configurations:
        path = output_directory(config) / "efficiency.json"
        if path.exists():
            return _read_json(path)
    raise FileNotFoundError(
        "Missing efficiency report. Run benchmark-matrix before collect-results."
    )


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment_name"]),
            str(row["architecture"]),
            float(row["width_multiplier"]),
            float(row["theta"]),
        )
        grouped[key].append(row)
    summaries = []
    for (_, _, width_multiplier, theta), group in grouped.items():
        first = group[0]
        summary: dict[str, Any] = {
            "experiment_name": first["experiment_name"],
            "architecture": first["architecture"],
            "width_multiplier": width_multiplier,
            "theta": theta,
            "seeds": json.dumps(sorted(int(row["seed"]) for row in group)),
        }
        for metric in ("apcer", "bpcer", "acer", "f1", "auc"):
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std"] = statistics.pstdev(values)
        for field in _SUMMARY_FIELDS[15:]:
            summary[field] = first[field]
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (float(row["width_multiplier"]), float(row["theta"])))


def collect_results(configurations: list[Config], output_csv: Path) -> list[dict[str, Any]]:
    output_json = output_csv.with_suffix(".json")
    summary_csv = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    if any(path.exists() for path in (output_csv, output_json, summary_csv)):
        raise FileExistsError(f"Result artifact already exists for {output_csv}")
    grouped: dict[tuple[str, str, float, float, str, int], list[Config]] = defaultdict(list)
    for config in configurations:
        grouped[_result_key(config)].append(config)
    rows = []
    for configs in grouped.values():
        ordered = sorted(configs, key=lambda config: config.experiment.fold)
        prediction_paths = [
            output_directory(config) / "validation_videos.jsonl" for config in ordered
        ]
        missing = [path for path in prediction_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing validation predictions: {missing[0]}")
        seed = ordered[0].experiment.seed
        oof_dir = output_directory(ordered[0]).parents[1] / "oof" / f"seed-{seed}"
        metrics = aggregate_oof(prediction_paths, oof_dir)
        splits = [_read_json(output_directory(config) / "split.json") for config in ordered]
        efficiency = _efficiency(ordered)
        input_shape = efficiency["input_shape"]
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError("Efficiency report has an invalid input shape")
        device = str(efficiency["device"])
        row: dict[str, Any] = {
            "experiment_name": ordered[0].experiment.name,
            "architecture": _architecture_label(ordered[0]),
            "width_multiplier": ordered[0].model.width_multiplier,
            "theta": ordered[0].model.theta,
            "seed": seed,
            "metric_split": "validation_oof",
            "input_resolution": f"{input_shape[3]}x{input_shape[2]}",
            "folds": json.dumps([config.experiment.fold for config in ordered]),
            "train_subjects": json.dumps([split["train_subjects"] for split in splits]),
            "validation_subjects": json.dumps([split["validation_subjects"] for split in splits]),
            "test_subjects": json.dumps(splits[0]["test_subjects"]),
            "apcer": metrics["apcer"],
            "bpcer": metrics["bpcer"],
            "acer": metrics["acer"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "eer": metrics["eer"],
            "accuracy": metrics["accuracy"],
            "threshold": metrics["threshold"],
            "trainable_parameters": efficiency["trainable_parameters"],
            "total_parameters": efficiency["total_parameters"],
            "macs": efficiency["macs"],
            "model_state_size_mb": efficiency["model_state_size_mb"],
            "latency_ms_cpu": efficiency["latency_ms"] if device == "cpu" else None,
            "latency_ms_gpu": efficiency["latency_ms"] if device == "cuda" else None,
            "checkpoint_paths": json.dumps(
                [str(output_directory(config) / "best.pt") for config in ordered]
            ),
            "git_commit": _read_json(output_directory(ordered[0]) / "provenance.json")["git_sha"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (float(row["width_multiplier"]), int(row["seed"])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, _RESULT_FIELDS, rows)
    _write_csv(summary_csv, _SUMMARY_FIELDS, _summary_rows(rows))
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows
