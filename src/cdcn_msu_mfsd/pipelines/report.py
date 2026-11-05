import csv
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from cdcn_msu_mfsd.pipelines.common import Campaign, read_json, write_json

_METRICS = ("acer", "apcer", "bpcer", "auc", "f1", "eer", "accuracy")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty report")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(campaign: Campaign) -> dict[str, Any]:
    result_path = campaign.root / "test_results.json"
    if not result_path.is_file():
        raise FileNotFoundError("Official test results are missing")
    results = read_json(result_path)["results"]
    expected = len(campaign.conditions) * len(campaign.settings.seeds)
    if len(results) != expected or expected != 21:
        raise ValueError(f"Expected 21 official test reports, found {len(results)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in results:
        metrics = record["metrics"]
        if int(metrics["genuine"]) != campaign.settings.test_genuine:
            raise ValueError(f"Wrong genuine count in {record['condition']} seed {record['seed']}")
        if int(metrics["attacks"]) != campaign.settings.test_attacks:
            raise ValueError(f"Wrong attack count in {record['condition']} seed {record['seed']}")
        grouped[str(record["condition"])].append(record)

    summaries = []
    detailed = []
    for condition in campaign.conditions:
        key = condition.key
        seed_records = sorted(grouped[key], key=lambda item: int(item["seed"]))
        if len(seed_records) != len(campaign.settings.seeds):
            raise ValueError(f"Expected three test seeds for {key}")
        summary: dict[str, Any] = {
            "condition": key,
            "width_multiplier": condition.width,
            "theta": condition.theta,
            "seeds": ",".join(str(item["seed"]) for item in seed_records),
            "test_videos_per_seed": campaign.settings.test_videos,
        }
        for metric in _METRICS:
            values = [float(item["metrics"][metric]) for item in seed_records]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std_population"] = statistics.pstdev(values)
        summary["false_accepts_by_seed"] = ",".join(
            str(int(item["metrics"]["false_accepts"])) for item in seed_records
        )
        summary["false_rejects_by_seed"] = ",".join(
            str(int(item["metrics"]["false_rejects"])) for item in seed_records
        )
        summaries.append(summary)
        for item in seed_records:
            detailed.append(
                {
                    "condition": key,
                    "width_multiplier": condition.width,
                    "theta": condition.theta,
                    "seed": int(item["seed"]),
                    **{metric: item["metrics"][metric] for metric in _METRICS},
                    "false_accepts": item["metrics"]["false_accepts"],
                    "false_rejects": item["metrics"]["false_rejects"],
                    "threshold": item["threshold"],
                    "genuine_videos": item["metrics"]["genuine"],
                    "attack_videos": item["metrics"]["attacks"],
                    "f1_positive_class": item["metrics"]["positive_class"],
                }
            )

    subgroup_summaries: dict[str, dict[str, Any]] = {"attack_type": {}, "device": {}}
    for subgroup_key in subgroup_summaries:
        names = sorted(
            {name for record in results for name in record["metrics"].get(f"by_{subgroup_key}", {})}
        )
        for name in names:
            subgroup_summaries[subgroup_key][name] = {}
            for metric in ("apcer", "bpcer", "acer", "false_accepts", "false_rejects"):
                values = [
                    float(record["metrics"][f"by_{subgroup_key}"][name][metric])
                    for record in results
                    if name in record["metrics"].get(f"by_{subgroup_key}", {})
                ]
                subgroup_summaries[subgroup_key][name][metric] = {
                    "mean": statistics.fmean(values),
                    "std_population": statistics.pstdev(values),
                    "n_condition_seed_pairs": len(values),
                }

    result = {
        "campaign": campaign.settings.name,
        "official_test_conditions": len(summaries),
        "seeds_per_condition": len(campaign.settings.seeds),
        "test_protocol": {
            "videos_per_seed": campaign.settings.test_videos,
            "genuine_per_seed": campaign.settings.test_genuine,
            "attacks_per_seed": campaign.settings.test_attacks,
            "threshold_selection": "five-fold pooled OOF training videos",
            "positive_class_for_f1": "attack",
        },
        "summaries_mean_and_population_sd": summaries,
        "per_seed_results": detailed,
        "subgroup_results_across_condition_seed_pairs": subgroup_summaries,
        "efficiency": (
            read_json(campaign.root / "width_efficiency.json")["records"]
            + read_json(campaign.root / "theta_efficiency.json")["records"]
        ),
        "model_statistics": read_json(campaign.root / "model_stats.json")["conditions"],
        "width_selection": read_json(campaign.root / "width_selection.json"),
        "frozen_plan_sha256": hashlib.sha256(campaign.plan_path.read_bytes()).hexdigest(),
    }
    report_dir = campaign.root / "reports"
    write_json(report_dir / "final_results.json", result)
    _write_csv(report_dir / "final_results_per_seed.csv", detailed)
    _write_csv(report_dir / "final_results_mean_sd.csv", summaries)
    lines = [
        f"# Final experiment results: {campaign.settings.name}",
        "",
        "Official test results use one frozen OOF threshold for each condition and seed.",
        "F1 uses attack as its positive class. Each row summarizes three training seeds.",
        "",
        "| Width | Theta | ACER mean ± SD | APCER mean ± SD | "
        "BPCER mean ± SD | AUC mean ± SD | F1 mean ± SD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {width_multiplier:.2f} | {theta:.2f} | {acer_mean:.4f} ± {acer_std_population:.4f} "
            "| {apcer_mean:.4f} ± {apcer_std_population:.4f} "
            "| {bpcer_mean:.4f} ± {bpcer_std_population:.4f} "
            "| {auc_mean:.4f} ± {auc_std_population:.4f} "
            "| {f1_mean:.4f} ± {f1_std_population:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The seed values use the same 15 official training subjects. They are repeated "
            "training runs, not independent datasets.",
            "",
            "False accepts and false rejects are listed by seed in the CSV file.",
            "",
        ]
    )
    (report_dir / "final_results.md").parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "final_results.md").write_text("\n".join(lines))
    return result
