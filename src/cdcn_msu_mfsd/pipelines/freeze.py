from typing import Any, cast

from cdcn_msu_mfsd.config import config_hash
from cdcn_msu_mfsd.experiments import median_best_epoch
from cdcn_msu_mfsd.pipelines.common import Campaign, read_json, sha256_file, write_json


def build_plan(campaign: Campaign) -> dict[str, Any]:
    entries = []
    for condition in campaign.conditions:
        for seed in campaign.settings.seeds:
            oof_dir = campaign.oof_directory(condition, seed)
            threshold_path = oof_dir / "threshold.json"
            metrics_path = oof_dir / "oof_metrics.json"
            predictions_path = oof_dir / "oof_videos.jsonl"
            if not threshold_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(
                    f"Missing OOF threshold report for {condition.key}, seed {seed}"
                )
            if not predictions_path.is_file():
                raise FileNotFoundError(f"Missing OOF video predictions: {predictions_path}")
            summaries = [
                campaign.cv_directory(condition, fold, seed) / "summary.json"
                for fold in campaign.settings.folds
            ]
            missing = [path for path in summaries if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing CV summary: {missing[0]}")
            epochs = median_best_epoch(summaries)
            refit_config = campaign.config(
                condition,
                seed=seed,
                phase="refit",
                epochs=epochs,
            )
            threshold = float(read_json(threshold_path)["threshold"])
            entries.append(
                {
                    "condition": condition.key,
                    "width_multiplier": condition.width,
                    "theta": condition.theta,
                    "seed": seed,
                    "threshold": threshold,
                    "threshold_sha256": sha256_file(threshold_path),
                    "oof_predictions_sha256": sha256_file(predictions_path),
                    "oof_metrics": read_json(metrics_path),
                    "refit_epochs": epochs,
                    "refit_config_hash": config_hash(refit_config),
                    "refit_config": refit_config.model_dump(mode="json"),
                    "cv_summary_sha256": {
                        str(campaign.settings.folds[index]): sha256_file(path)
                        for index, path in enumerate(summaries)
                    },
                }
            )
    expected = len(campaign.conditions) * len(campaign.settings.seeds)
    if len(entries) != expected or expected != 21:
        raise ValueError(f"Expected 21 frozen refit/test conditions, found {len(entries)}")
    return {
        "campaign": campaign.settings.name,
        "selection": read_json(campaign.root / "width_selection.json"),
        "test_protocol": {
            "videos": campaign.settings.test_videos,
            "genuine": campaign.settings.test_genuine,
            "attacks": campaign.settings.test_attacks,
            "evaluation_count": "one per refit checkpoint",
            "threshold_source": "five-fold out-of-fold training predictions",
        },
        "conditions": entries,
    }


def run(campaign: Campaign) -> dict[str, Any]:
    result = build_plan(campaign)
    if campaign.plan_path.exists():
        existing = read_json(campaign.plan_path)
        if existing != result:
            raise FileExistsError(
                "A frozen final plan already exists and differs from current inputs. "
                "The official test plan is immutable."
            )
        if (
            not campaign.plan_digest_path.is_file()
            or campaign.plan_digest_path.read_text().strip() != sha256_file(campaign.plan_path)
        ):
            raise ValueError("Frozen plan digest does not match the plan contents")
        return cast(dict[str, Any], existing)
    write_json(campaign.plan_path, result)
    campaign.plan_digest_path.write_text(sha256_file(campaign.plan_path) + "\n")
    return result
