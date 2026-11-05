from dataclasses import asdict

import torch

from cdcn_msu_mfsd.efficiency import benchmark_model
from cdcn_msu_mfsd.models import build_model
from cdcn_msu_mfsd.pipelines.common import Campaign, Progress, write_json


def run(campaign: Campaign) -> dict[str, object]:
    progress = Progress(campaign, "model-stats", len(campaign.conditions))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 6))
    records = []
    try:
        for index, condition in enumerate(campaign.conditions, start=1):
            config = campaign.config(condition)
            model = build_model(
                config.model.name, config.model.theta, config.model.width_multiplier
            )
            measured = benchmark_model(
                model,
                torch.device("cpu"),
                input_shape=(1, 3, 256, 256),
                warmup_runs=0,
                timed_runs=1,
            )
            record = asdict(measured)
            record.update(
                {
                    "condition": condition.key,
                    "width_multiplier": condition.width,
                    "theta": condition.theta,
                    "meaning": "architecture counts and serialized state_dict size",
                }
            )
            records.append(record)
            progress.update(index, condition.key)
    finally:
        torch.set_num_threads(previous_threads)
    result = {"input_shape": [1, 3, 256, 256], "conditions": records}
    write_json(campaign.root / "model_stats.json", result)
    return result
