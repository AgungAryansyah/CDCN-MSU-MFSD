import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

COLORS = ["#0072B2", "#E69F00", "#009E73"]


def export(figure, directory, name):
    for suffix in ("pdf", "png"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Exported {name}: PDF and PNG", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/labkc-comm3-analysis"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/results-analysis-2026-11-05")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = json.loads((args.data / "summary.json").read_text())
    per_seed = json.loads((args.data / "per_seed.json").read_text())
    efficiency = json.loads((args.data / "static_efficiency.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )
    print("Phase: rendering three figures from the saved analysis tables", flush=True)
    figure, axis = plt.subplots(figsize=(5.8, 3.6), layout="constrained")
    widths = [1.0, 0.5, 0.25]
    parameters = []
    for alpha, color in zip(widths, COLORS, strict=True):
        row = next(r for r in summaries if r["campaign"] == "small-width" and r["alpha"] == alpha)
        x = (
            next(r["parameters"] for r in efficiency if r["alpha"] == alpha and r["theta"] == 0.7)
            / 1e6
        )
        parameters.append(x)
        seeds = [r for r in per_seed if r["campaign"] == "small-width" and r["alpha"] == alpha]
        axis.scatter([x] * 3, [100 * r["acer"] for r in seeds], color=color, s=28, alpha=0.6)
        axis.errorbar(
            x,
            100 * row["acer_mean"],
            yerr=100 * row["acer_std"],
            fmt="D",
            color="black",
            capsize=4,
            markersize=5,
        )
        axis.annotate(
            rf"$\alpha={alpha:g}$",
            (x, max(100 * r["acer"] for r in seeds)),
            xytext=(10, 12) if alpha == 0.25 else (0, 12),
            textcoords="offset points",
            ha="left" if alpha == 0.25 else "center",
        )
    axis.set(
        xscale="log",
        xlabel="Parameters (millions; logarithmic scale)",
        ylabel="OOF ACER (%)",
        title=r"Width comparison at $\theta=0.7$",
        ylim=(0, 20),
    )
    axis.set_xticks(parameters, labels=[f"{x:.3f}" for x in parameters])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.scatter([], [], s=28, color="gray", label="Individual seed")
    axis.errorbar(
        [], [], yerr=[], fmt="D", color="black", capsize=4, label="Mean ± population SD (3 seeds)"
    )
    axis.legend(loc="lower left", fontsize=8)
    export(figure, args.output, "width_vs_capacity")
    figure, axis = plt.subplots(figsize=(5.8, 3.6), layout="constrained")
    theta_rows = sorted(
        [r for r in summaries if r["campaign"] == "small-theta-a02"], key=lambda r: r["theta"]
    )
    for seed, color in zip((42, 2026, 2027), COLORS, strict=True):
        rows = sorted(
            [r for r in per_seed if r["campaign"] == "small-theta-a02" and r["seed"] == seed],
            key=lambda r: r["theta"],
        )
        axis.plot(
            [r["theta"] for r in rows],
            [100 * r["acer"] for r in rows],
            linestyle=":",
            marker="o",
            color=color,
            alpha=0.7,
            label=f"Seed {seed}",
        )
    axis.errorbar(
        [r["theta"] for r in theta_rows],
        [100 * r["acer_mean"] for r in theta_rows],
        yerr=[100 * r["acer_std"] for r in theta_rows],
        fmt="D-",
        color="black",
        capsize=4,
        label="Mean ± population SD",
        markersize=5,
    )
    axis.set(
        xlabel=r"Central difference weight $\theta$",
        ylabel="OOF ACER (%)",
        title=r"Theta ablation at $\alpha=0.25$",
        ylim=(-1, 50),
        xticks=[0, 0.3, 0.5, 0.7, 1],
    )
    axis.legend(loc="upper left", fontsize=8)
    export(figure, args.output, "theta_oof")
    with (args.data / "subgroups.csv").open() as stream:
        subgroups = list(csv.DictReader(stream))
    configs = [
        ("small-width", 1.0, 0.7),
        ("small-width", 0.5, 0.7),
        ("small-width", 0.25, 0.7),
        ("small-theta-a02", 0.25, 0),
        ("small-theta-a02", 0.25, 0.3),
    ]
    labels = [rf"$\alpha={a:g}$" + "\n" + rf"$\theta={t:g}$" for _, a, t in configs]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), layout="constrained")
    settings = [
        (
            "attack_type",
            ["printed_photo", "iphone_video", "ipad_video"],
            "apcer",
            ["Printed photo", "iPhone replay", "iPad replay"],
            "Attack false acceptance",
        ),
        (
            "device",
            ["android", "laptop"],
            "bpcer",
            ["Android", "Laptop"],
            "Bona fide false rejection",
        ),
    ]
    x = np.arange(len(configs))
    for axis, (field, groups, metric, names, title) in zip(axes, settings, strict=True):
        width = 0.76 / len(groups)
        for index, (group, name, color) in enumerate(zip(groups, names, COLORS, strict=False)):
            means = []
            for campaign, alpha, theta in configs:
                values = [
                    float(r[metric]) * 100
                    for r in subgroups
                    if r["campaign"] == campaign
                    and float(r["alpha"]) == alpha
                    and float(r["theta"]) == theta
                    and r["grouping"] == field
                    and r["group"] == group
                ]
                means.append(statistics.fmean(values))
            axis.bar(
                x + (index - (len(groups) - 1) / 2) * width,
                means,
                width * 0.92,
                color=color,
                label=name,
            )
        axis.set(
            xticks=x, xticklabels=labels, ylabel=metric.upper() + " (%)", title=title, ylim=(0, 40)
        )
        axis.legend(fontsize=8, loc="upper right")
    export(figure, args.output, "subgroup_errors")


if __name__ == "__main__":
    main()
