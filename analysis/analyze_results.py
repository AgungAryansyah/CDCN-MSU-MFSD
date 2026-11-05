import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rates(rows, threshold):
    attacks = [r for r in rows if r["label"] == 0]
    genuine = [r for r in rows if r["label"] == 1]
    fa = sum(r["score"] >= threshold for r in attacks)
    fr = sum(r["score"] < threshold for r in genuine)
    apcer, bpcer = fa / len(attacks), fr / len(genuine)
    tp = len(attacks) - fa
    return dict(
        apcer=apcer,
        bpcer=bpcer,
        acer=(apcer + bpcer) / 2,
        accuracy=1 - (fa + fr) / len(rows),
        f1=2 * tp / (2 * tp + fa + fr),
        false_accepts=fa,
        false_rejects=fr,
        attacks=len(attacks),
        genuine=len(genuine),
    )


def metrics(rows):
    scores = sorted({r["score"] for r in rows})
    epsilon = max(1e-8, (scores[-1] - scores[0]) * 1e-8)
    thresholds = [
        scores[0] - epsilon,
        *((a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)),
        scores[-1] + epsilon,
    ]
    candidates = [(t, rates(rows, t)) for t in thresholds]
    threshold, result = min(candidates, key=lambda x: (x[1]["acer"], -x[1]["accuracy"], x[0]))
    attacks = [r["score"] for r in rows if r["label"] == 0]
    genuine = [r["score"] for r in rows if r["label"] == 1]
    result["auc"] = sum((a < g) + 0.5 * (a == g) for a in attacks for g in genuine) / (
        len(attacks) * len(genuine)
    )
    result["eer"] = min((abs(m["apcer"] - m["bpcer"]), m["acer"]) for _, m in candidates)[1]
    result["threshold"] = threshold
    return result


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def static_efficiency(alpha, theta):
    stem, block, expanded, fusion, head = [round(c * alpha) for c in (64, 128, 196, 128, 64)]
    layers = [(3, stem, 256, True)]
    for incoming, size in [(stem, 256), (block, 128), (block, 64)]:
        layers.extend(
            [
                (incoming, block, size, True),
                (block, expanded, size, True),
                (expanded, block, size, True),
            ]
        )
    layers.extend([(3 * block, fusion, 32, True), (fusion, head, 32, True), (head, 1, 32, False)])
    parameters = sum(9 * i * o + (2 * o if bn else 0) for i, o, _, bn in layers)
    macs = sum(h * h * o * i * (9 + int(theta != 0)) for i, o, h, _ in layers)
    tensor_bytes = parameters * 4 + sum(8 * o + 8 for _, o, _, bn in layers if bn)
    return dict(
        alpha=alpha,
        theta=theta,
        parameters=parameters,
        convolution_macs=macs,
        state_tensor_payload_bytes=tensor_bytes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/labkc-comm3"))
    parser.add_argument("--output", type=Path, default=Path("results/labkc-comm3-analysis"))
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.source.rglob("validation_videos.jsonl"))
    groups, runs = defaultdict(list), []
    curves, checks = [], Counter()
    provenance, controls, split_by_fold = defaultdict(set), set(), defaultdict(set)
    print(f"Phase: auditing {len(paths)} saved validation runs; ETA unavailable", flush=True)
    for index, path in enumerate(paths, 1):
        directory = path.parent
        campaign = path.relative_to(args.source).parts[0]
        config = yaml.safe_load((directory / "config.yaml").read_text())
        model, experiment = config["model"], config["experiment"]
        alpha, theta = model.get("width_multiplier", 1.0), model["theta"]
        seed, fold = experiment["seed"], experiment["fold"]
        identity = dict(
            campaign=campaign,
            model=model["name"],
            alpha=alpha,
            theta=theta,
            supervision=config["supervision"]["kind"],
            seed=seed,
        )
        key = tuple(identity.values())
        rows = read_jsonl(path)
        ids = [r["video_id"] for r in rows]
        assert len(ids) == len(set(ids)) == 24, path
        assert Counter(r["label"] for r in rows) == {0: 18, 1: 6}, path
        assert all(math.isfinite(r["score"]) for r in rows), path
        calculated = metrics(rows)
        saved = read_json(directory / "validation_metrics.json")
        for name in calculated:
            if name not in saved:
                assert campaign == "main" and name == "f1", (path, name)
                checks["legacy_runs_without_saved_f1"] += 1
                continue
            assert math.isclose(calculated[name], saved[name], rel_tol=0, abs_tol=1e-12), (
                path,
                name,
            )
        frames = read_jsonl(directory / "validation_frames.jsonl")
        frame_groups = defaultdict(list)
        for row in frames:
            frame_groups[row["video_id"]].append(row)
        assert set(frame_groups) == set(ids), path
        for row in rows:
            video_frames = frame_groups[row["video_id"]]
            assert len(video_frames) == len({f["frame_index"] for f in video_frames}) == 8, path
            assert all(
                all(f[k] == row[k] for k in ("label", "device", "attack_type"))
                for f in video_frames
            ), path
            assert math.isclose(
                row["score"],
                statistics.fmean(f["score"] for f in video_frames),
                rel_tol=0,
                abs_tol=1e-12,
            ), path
        summary = read_json(directory / "summary.json")
        history = [r for r in read_jsonl(directory / "metrics.jsonl") if "epoch" in r]
        assert len(history) == summary["epochs_completed"], path
        assert math.isclose(
            min(r["validation/total"] for r in history),
            summary["best_validation_loss"],
            rel_tol=0,
            abs_tol=1e-12,
        ), path
        record = dict(
            **identity,
            fold=fold,
            **calculated,
            best_epoch_one_based=summary["best_epoch"] + 1,
            epochs_completed=summary["epochs_completed"],
            best_validation_loss=summary["best_validation_loss"],
            best_train_loss=history[summary["best_epoch"]]["train_epoch/total"],
            logged_peak_gpu_reserved_gib=max(r.get("gpu/max_memory_gib", 0) for r in history),
            source=str(path),
        )
        runs.append(record)
        groups[key].append((record, rows))
        checks[f"{campaign}_runs"] += 1
        if campaign in {"small-width", "small-theta-a02"}:
            for name in ("split.json", "best.pt", "last.pt", "provenance.json", "threshold.json"):
                assert (directory / name).exists(), (directory, name)
            split = read_json(directory / "split.json")
            train, val, test = (
                set(split[f"{x}_subjects"]) for x in ("train", "validation", "test")
            )
            assert (len(train), len(val), len(test)) == (12, 3, 20), directory
            assert not (train & val or train & test or val & test), directory
            assert len(split["train_videos"]) == 96 and len(split["test_videos"]) == 160
            for prefix, subjects in [("train", train), ("validation", val), ("test", test)]:
                assert {
                    re.search(r"client(\d+)_", x)[1] for x in split[f"{prefix}_videos"]
                } == subjects, directory
            assert set(ids) == set(split["validation_videos"]), directory
            assert {re.search(r"client(\d+)_", x)[1] for x in ids} == val, directory
            split_by_fold[fold].add(json.dumps(split, sort_keys=True))
            for name, value in read_json(directory / "provenance.json").items():
                if name != "config_sha256":
                    provenance[name].add(value)
            control = json.loads(json.dumps(config))
            for name in ("name", "fold", "seed"):
                control["experiment"].pop(name)
            for name in ("theta", "width_multiplier"):
                control["model"].pop(name)
            controls.add(json.dumps(control, sort_keys=True))
            for row in history:
                curves.append(
                    dict(
                        **identity,
                        fold=fold,
                        epoch=row["epoch"] + 1,
                        train_loss=row["train_epoch/total"],
                        validation_loss=row["validation/total"],
                    )
                )
            assert summary["epochs_completed"] == summary["best_epoch"] + 31, directory
            checks["current_runs_audited"] += 1
        if index % 30 == 0 or index == len(paths):
            elapsed = time.monotonic() - started
            print(
                f"Auditing: {index}/{len(paths)} ({100 * index / len(paths):.0f}%), "
                f"elapsed {elapsed:.1f}s, ETA {elapsed / index * (len(paths) - index):.1f}s",
                flush=True,
            )
    assert checks["small-width_runs"] == 45 and checks["small-theta-a02_runs"] == 75
    assert len(controls) == 1 and all(len(v) == 1 for v in split_by_fold.values())
    assert all(len(v) == 1 for v in provenance.values()), dict(provenance)
    print("Phase: combining five validation folds per configuration and seed", flush=True)
    per_seed, errors, subgroup, predictions = [], [], [], []
    oof = {}
    for key, items in sorted(groups.items()):
        assert sorted(r["fold"] for r, _ in items) == list(range(5)), key
        rows = sorted([v for _, vs in items for v in vs], key=lambda r: r["video_id"])
        assert len(rows) == len({r["video_id"] for r in rows}) == 120, key
        identity = dict(
            zip(("campaign", "model", "alpha", "theta", "supervision", "seed"), key, strict=True)
        )
        result = metrics(rows)
        row = dict(
            **identity,
            **result,
            mean_fold_acer=statistics.fmean(r["acer"] for r, _ in items),
            refit_epochs=int(statistics.median(r["best_epoch_one_based"] for r, _ in items)),
        )
        per_seed.append(row)
        oof[key] = (row, rows)
        for item in rows:
            item = dict(
                **identity,
                **item,
                threshold=result["threshold"],
                predicted_label=int(item["score"] >= result["threshold"]),
            )
            predictions.append(item)
            if item["predicted_label"] != item["label"]:
                errors.append(item)
        for field in ("attack_type", "device"):
            for value in sorted({r[field] for r in rows} - {"genuine"}):
                subset = [
                    r
                    for r in rows
                    if r[field] == value or (field == "attack_type" and r["label"] == 1)
                ]
                subgroup.append(
                    dict(
                        **identity,
                        grouping=field,
                        group=value,
                        **rates(subset, result["threshold"]),
                    )
                )
    summaries = []
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[
            tuple(row[k] for k in ("campaign", "model", "alpha", "theta", "supervision"))
        ].append(row)
    metric_names = (
        "apcer",
        "bpcer",
        "acer",
        "f1",
        "auc",
        "eer",
        "accuracy",
        "false_accepts",
        "false_rejects",
        "mean_fold_acer",
    )
    for key, rows in sorted(grouped.items()):
        summary = dict(
            zip(("campaign", "model", "alpha", "theta", "supervision"), key, strict=True)
        )
        summary["seeds"] = sorted(r["seed"] for r in rows)
        for name in metric_names:
            summary[name + "_mean"] = statistics.fmean(r[name] for r in rows)
            summary[name + "_std"] = statistics.pstdev(r[name] for r in rows)
        summaries.append(summary)
        print(
            key,
            "OOF ACER %",
            round(100 * summary["acer_mean"], 6),
            "+/-",
            round(100 * summary["acer_std"], 6),
            flush=True,
        )
    repeated = []
    for seed in (42, 2026, 2027):
        width = oof[("small-width", "cdcn", 0.25, 0.7, "pseudo_depth", seed)]
        theta = oof[("small-theta-a02", "cdcn", 0.25, 0.7, "pseudo_depth", seed)]
        repeated.append(
            dict(
                seed=seed,
                predictions_identical=width[1] == theta[1],
                metrics_identical={k: v for k, v in width[0].items() if k not in {"campaign"}}
                == {k: v for k, v in theta[0].items() if k not in {"campaign"}},
            )
        )
    existing = read_json(args.source / "small-width/cdcn-pseudo_depth/oof/seed-42/oof_metrics.json")
    calculated = oof[("small-width", "cdcn", 1.0, 0.7, "pseudo_depth", 42)][0]
    assert all(
        math.isclose(existing[k], calculated[k], rel_tol=0, abs_tol=1e-12)
        for k in (*metric_names[:9], "threshold")
    )
    source_hashes = {}
    for path in sorted(args.source.rglob("*")):
        if (
            not path.is_symlink()
            and path.is_file()
            and path.suffix in {".json", ".jsonl", ".yaml"}
            and "wandb" not in path.parts
        ):
            source_hashes[str(path.relative_to(args.source))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    audit = dict(
        source=str(args.source),
        checks=dict(checks),
        control_configurations=len(controls),
        split_variants_per_fold={k: len(v) for k, v in split_by_fold.items()},
        provenance={k: sorted(v) for k, v in provenance.items()},
        repeated_theta07=repeated,
        existing_oof_matches=True,
        test_prediction_files=len(list(args.source.rglob("test_videos.jsonl"))),
        efficiency_report_files=len(list(args.source.rglob("efficiency*.json"))),
        source_sha256=source_hashes,
    )
    efficiency = [
        static_efficiency(a, t)
        for a, t in [
            (1.0, 0.7),
            (0.5, 0.7),
            (0.25, 0.0),
            (0.25, 0.3),
            (0.25, 0.5),
            (0.25, 0.7),
            (0.25, 1.0),
        ]
    ]
    print("Phase: saving analysis tables and source audit", flush=True)
    for name, rows in [
        ("per_seed", per_seed),
        ("summary", summaries),
        ("fold_metrics", runs),
        ("subgroups", subgroup),
        ("errors", errors),
        ("oof_predictions", predictions),
        ("learning_curves", curves),
        ("static_efficiency", efficiency),
    ]:
        save_csv(args.output / f"{name}.csv", rows)
        if name in {"summary", "per_seed", "static_efficiency"}:
            (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"Complete: {len(runs)} runs, {len(per_seed)} OOF results; "
        f"elapsed {time.monotonic() - started:.1f}s; output {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
