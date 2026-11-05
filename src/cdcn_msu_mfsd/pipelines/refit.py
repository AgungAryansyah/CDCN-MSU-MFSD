import time
from typing import Any

import torch

from cdcn_msu_mfsd.config import (
    Config,
    checkpoint_matches_config,
    config_hash,
    save_resolved_config,
)
from cdcn_msu_mfsd.data.supervision import PseudoDepthValidation
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.pipelines.common import (
    Campaign,
    Progress,
    read_json,
    sha256_file,
    write_json,
)
from cdcn_msu_mfsd.tracking import TrackingContext, create_tracker
from cdcn_msu_mfsd.training import (
    Trainer,
    make_refit_loader,
    provenance,
    seed_everything,
    select_device,
)


def _refit_entry(campaign: Campaign) -> list[dict[str, Any]]:
    if not campaign.plan_path.is_file():
        raise FileNotFoundError("The frozen final plan is missing")
    if (
        not campaign.plan_digest_path.is_file()
        or campaign.plan_digest_path.read_text().strip() != sha256_file(campaign.plan_path)
    ):
        raise ValueError("Frozen plan digest does not match the plan contents")
    return list(read_json(campaign.plan_path)["conditions"])


def run(campaign: Campaign) -> dict[str, object]:
    entries = _refit_entry(campaign)
    progress = Progress(campaign, "refit", len(entries))
    results = []
    preflight = read_json(campaign.root / "preflight.json")
    depth_validation = PseudoDepthValidation(
        int(preflight["pseudo_depth_targets"]), str(preflight["pseudo_depth_sha256"])
    )
    plan_hash = sha256_file(campaign.plan_path)
    for index, entry in enumerate(entries, start=1):
        config = Config.model_validate(entry["refit_config"])
        if config_hash(config) != entry["refit_config_hash"]:
            raise ValueError(
                f"Frozen config hash mismatch for {entry['condition']} seed {entry['seed']}"
            )
        condition_key = str(entry["condition"])
        seed = int(entry["seed"])
        condition = next(item for item in campaign.conditions if item.key == condition_key)
        output_dir = campaign.refit_directory(condition, seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        complete_path = output_dir / "refit_complete.json"
        final_path = output_dir / "final.pt"
        if complete_path.is_file() and final_path.is_file():
            complete = read_json(complete_path)
            wandb_path = output_dir / "wandb_run.json"
            if (
                complete.get("frozen_plan_sha256") == plan_hash
                and complete.get("config_hash") == entry["refit_config_hash"]
                and complete.get("checkpoint_sha256") == sha256_file(final_path)
                and wandb_path.is_file()
                and read_json(wandb_path).get("status") == "finished"
            ):
                results.append({**complete, "status": "reused_complete"})
                progress.update(index, f"reused {condition_key}, seed {seed}")
                continue
            raise ValueError(
                f"Completed refit artifacts do not match the frozen plan: {output_dir}"
            )

        started = time.monotonic()
        save_resolved_config(config, output_dir / "config.yaml")
        seed_everything(config.experiment.seed, config.runtime.deterministic)
        device = select_device(config.runtime.device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
        loader = make_refit_loader(config, device)
        run_provenance = provenance(config, depth_validation)
        run_provenance["frozen_plan_sha256"] = plan_hash
        run_provenance["oof_threshold_sha256"] = str(entry["threshold_sha256"])
        context = TrackingContext(
            name=f"refit-{condition_key}-s{seed}",
            group=campaign.settings.name,
            job_type="refit",
            tags=("final-fit", condition_key, f"seed-{seed}"),
            config={
                "campaign": campaign.settings.name,
                "condition": condition_key,
                "seed": seed,
                "epochs": config.training.epochs,
                "threshold": entry["threshold"],
                "frozen_plan_sha256": plan_hash,
            },
            required_online=True,
            custom_step_metric=True,
        )
        tracker = create_tracker(config, output_dir, run_provenance, context)
        try:
            trainer = Trainer(
                config,
                build_model(config.model.name, config.model.theta, config.model.width_multiplier),
                loader,
                loader,
                tracker,
                device,
            )
            last_path = output_dir / "last.pt"
            trainer.fit_fixed_epochs(output_dir, last_path if last_path.is_file() else None)
            checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
            if not checkpoint_matches_config(checkpoint.get("config_hash"), config):
                raise ValueError("Refit checkpoint config hash does not match the frozen config")
            complete = {
                "condition": condition_key,
                "seed": seed,
                "status": "completed",
                "refit_epochs": config.training.epochs,
                "threshold": float(entry["threshold"]),
                "frozen_plan_sha256": plan_hash,
                "config_hash": entry["refit_config_hash"],
                "checkpoint": str(final_path),
                "checkpoint_sha256": sha256_file(final_path),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            tracker.summary(
                {
                    "refit/epochs": config.training.epochs,
                    "refit/checkpoint_bytes": final_path.stat().st_size,
                    "refit/frozen_threshold": float(entry["threshold"]),
                }
            )
        finally:
            tracker.finish()
        write_json(complete_path, complete)
        results.append(complete)
        write_json(output_dir / "refit_result.json", complete)
        progress.update(index, f"{condition_key}, seed {seed}", complete["elapsed_seconds"])
    report = {"jobs": len(results), "results": results}
    write_json(campaign.root / "refit_results.json", report)
    return report
