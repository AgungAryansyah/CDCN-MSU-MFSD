from typing import Any, cast

from cdcn_msu_mfsd.experiments import aggregate_oof
from cdcn_msu_mfsd.pipelines.common import (
    Campaign,
    Condition,
    Progress,
    conditions_for_theta,
    conditions_for_widths,
    write_json,
)


def _aggregate(campaign: Campaign, stage: str, conditions: list[Condition]) -> dict[str, Any]:
    jobs = [(condition, seed) for condition in conditions for seed in campaign.settings.seeds]
    progress = Progress(campaign, stage, len(jobs))
    records = []
    for index, (condition, seed) in enumerate(jobs, start=1):
        paths = [
            campaign.cv_directory(condition, fold, seed) / "validation_videos.jsonl"
            for fold in campaign.settings.folds
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing fold predictions: {missing[0]}")
        report = aggregate_oof(paths, campaign.oof_directory(condition, seed))
        records.append(
            {
                "condition": condition.key,
                "seed": seed,
                "threshold": report["threshold"],
                "acer": report["acer"],
                "apcer": report["apcer"],
                "bpcer": report["bpcer"],
                "auc": report["auc"],
                "f1": report["f1"],
                "accuracy": report["accuracy"],
                "videos": report["genuine"] + report["attacks"],
                "genuine": report["genuine"],
                "attacks": report["attacks"],
            }
        )
        progress.update(index, f"{condition.key}, seed {seed}")
    result = {"stage": stage, "records": records}
    write_json(campaign.root / f"{stage}.json", result)
    return cast(dict[str, Any], result)


def run_width(campaign: Campaign) -> dict[str, Any]:
    return _aggregate(campaign, "width_oof", conditions_for_widths(campaign))


def run_theta(campaign: Campaign) -> dict[str, Any]:
    return _aggregate(campaign, "theta_oof", conditions_for_theta(campaign))
