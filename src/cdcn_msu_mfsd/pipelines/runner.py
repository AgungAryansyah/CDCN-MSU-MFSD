import traceback
from collections.abc import Callable
from typing import Any, cast

from cdcn_msu_mfsd.pipelines.common import (
    Campaign,
    campaign_lock,
    input_fingerprint,
    read_json,
    update_campaign_state,
)


def plan(campaign: Campaign) -> dict[str, Any]:
    return {
        "campaign": campaign.settings.name,
        "reused_width_cv_records": 45,
        "new_theta_cv_records": 60,
        "primary_width": campaign.settings.primary_width,
        "official_conditions": [condition.key for condition in campaign.conditions],
        "refit_jobs": len(campaign.conditions) * len(campaign.settings.seeds),
        "test_jobs": len(campaign.conditions) * len(campaign.settings.seeds),
        "pipeline_order": [
            "preflight",
            "wandb-smoke",
            "verify",
            "width",
            "width-oof",
            "model-stats",
            "width-efficiency",
            "selection",
            "theta",
            "theta-oof",
            "theta-efficiency",
            "freeze",
            "refit",
            "test",
            "report",
        ],
        "manualisasi": "excluded",
    }


def run_component(component: str, campaign: Campaign) -> Any:
    if component == "plan":
        return plan(campaign)
    from cdcn_msu_mfsd.pipelines import (
        cross_validation,
        efficiency,
        freeze,
        model_stats,
        oof,
        preflight,
        refit,
        report,
        selection,
        smoke,
        test_evaluation,
        verification,
    )

    stages: dict[str, Callable[[], Any]] = {
        "preflight": lambda: preflight.run(campaign),
        "wandb-smoke": lambda: smoke.run(campaign),
        "verify": lambda: verification.run(campaign),
        "width": lambda: cross_validation.run_width(campaign),
        "width-oof": lambda: oof.run_width(campaign),
        "model-stats": lambda: model_stats.run(campaign),
        "width-efficiency": lambda: efficiency.run_scope(campaign, "width"),
        "selection": lambda: selection.run(campaign),
        "theta": lambda: cross_validation.run_theta(campaign),
        "theta-oof": lambda: oof.run_theta(campaign),
        "theta-efficiency": lambda: efficiency.run_scope(campaign, "theta"),
        "freeze": lambda: freeze.run(campaign),
        "refit": lambda: refit.run(campaign),
        "test": lambda: test_evaluation.run(campaign),
        "report": lambda: report.run(campaign),
    }
    if component not in stages:
        raise ValueError(f"Unknown pipeline component: {component}")
    with campaign_lock(campaign):
        state = read_json(campaign.state_path) if campaign.state_path.exists() else {}
        preflight_state = state.get("stages", {}).get("preflight", {})
        current_fingerprint = input_fingerprint(campaign)
        if component != "preflight":
            if preflight_state.get("status") != "completed":
                raise RuntimeError("Run preflight before any other campaign stage")
            if preflight_state.get("input_fingerprint") != current_fingerprint:
                raise RuntimeError(
                    "Campaign inputs or source code changed after preflight. "
                    "Start a new campaign to protect saved results."
                )
        elif (
            preflight_state.get("status") == "completed"
            and preflight_state.get("input_fingerprint") != current_fingerprint
        ):
            raise RuntimeError(
                "Campaign inputs or source code changed after preflight. "
                "Start a new campaign to protect saved results."
            )
        update_campaign_state(campaign, component, "running")
        try:
            result = stages[component]()
            update_campaign_state(campaign, component, "completed")
            return result
        except BaseException as error:
            update_campaign_state(
                campaign,
                component,
                "failed",
                {"error": str(error), "traceback": traceback.format_exc()},
            )
            raise


def read_status(campaign: Campaign) -> dict[str, Any]:
    if not campaign.status_path.exists():
        return {"campaign": campaign.settings.name, "status": "not_started"}
    from cdcn_msu_mfsd.pipelines.common import read_json

    return cast(dict[str, Any], read_json(campaign.status_path))
