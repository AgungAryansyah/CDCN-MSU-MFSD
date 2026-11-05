import statistics
from typing import Any, cast

from cdcn_msu_mfsd.pipelines.common import (
    Campaign,
    Condition,
    read_json,
    write_json,
)


def run(campaign: Campaign) -> dict[str, Any]:
    source = campaign.root / "width_oof.json"
    efficiency_source = campaign.root / "width_efficiency.json"
    if not source.is_file() or not efficiency_source.is_file():
        raise FileNotFoundError("Width OOF and width efficiency reports are required")
    records = cast(list[dict[str, Any]], read_json(source)["records"])
    efficiency = cast(list[dict[str, Any]], read_json(efficiency_source)["records"])
    summaries: list[dict[str, Any]] = []
    for width in campaign.settings.widths:
        condition = Condition(width, 0.7)
        seed_records = [item for item in records if item["condition"] == condition.key]
        if len(seed_records) != len(campaign.settings.seeds):
            raise ValueError(f"Expected one OOF report per seed for {condition.key}")
        cpu = next(
            item
            for item in efficiency
            if item["condition"] == condition.key and item["device"].startswith("cpu")
        )
        summaries.append(
            {
                "condition": condition.key,
                "width_multiplier": width,
                "theta": 0.7,
                "acer_mean": statistics.fmean(float(item["acer"]) for item in seed_records),
                "acer_std_population": statistics.pstdev(
                    float(item["acer"]) for item in seed_records
                ),
                "cpu_latency_ms": float(cpu["latency_ms"]),
                "macs": int(cpu["macs"]),
                "total_parameters": int(cpu["total_parameters"]),
            }
        )

    compact = [item for item in summaries if float(item["width_multiplier"]) < 1.0]
    frontier = []
    for candidate in compact:
        dominated = any(
            other is not candidate
            and float(other["acer_mean"]) <= float(candidate["acer_mean"])
            and float(other["cpu_latency_ms"]) <= float(candidate["cpu_latency_ms"])
            and int(other["macs"]) <= int(candidate["macs"])
            and (
                float(other["acer_mean"]) < float(candidate["acer_mean"])
                or float(other["cpu_latency_ms"]) < float(candidate["cpu_latency_ms"])
                or int(other["macs"]) < int(candidate["macs"])
            )
            for other in compact
        )
        if not dominated:
            frontier.append(candidate)
    selected = min(
        frontier,
        key=lambda item: (
            round(float(item["acer_mean"]), 3),
            float(item["cpu_latency_ms"]),
            int(item["macs"]),
        ),
    )
    if float(selected["width_multiplier"]) != campaign.settings.primary_width:
        raise ValueError(
            "The frozen thesis plan expects the Chapter 3 selection to choose width "
            f"{campaign.settings.primary_width}, but the measured rule chose "
            f"{selected['width_multiplier']}. Review the width analysis before theta runs."
        )
    result = {
        "rule": "Among compact Pareto candidates, choose lower mean OOF ACER. "
        "Break ties rounded to three decimals with CPU latency, then MACs.",
        "compact_candidates": compact,
        "pareto_frontier": frontier,
        "selected": selected,
        "full_width_reference": next(
            item for item in summaries if float(item["width_multiplier"]) == 1.0
        ),
    }
    write_json(campaign.root / "width_selection.json", result)
    return result
