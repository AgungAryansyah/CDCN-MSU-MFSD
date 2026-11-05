import os
from collections import Counter

import torch
from dotenv import load_dotenv

from cdcn_msu_mfsd.data.preprocessing import load_frame_records
from cdcn_msu_mfsd.data.protocol import load_folds
from cdcn_msu_mfsd.data.supervision import validate_pseudo_depth_targets
from cdcn_msu_mfsd.pipelines.common import Campaign, Progress, write_json


def run(campaign: Campaign) -> dict[str, object]:
    progress = Progress(campaign, "preflight", 5)
    settings = campaign.settings
    if settings.model != "cdcn" or settings.supervision != "pseudo_depth":
        raise ValueError("This campaign requires CDCN with pseudo-depth supervision")
    if set(settings.widths) != {1.0, 0.5, 0.25}:
        raise ValueError("The width phase requires widths 1.0, 0.5, and 0.25")
    if settings.primary_width not in settings.widths:
        raise ValueError("The primary width must appear in the width comparison")
    if settings.primary_width != 0.5 or set(settings.theta_values) != {0.0, 0.3, 0.5, 0.7, 1.0}:
        raise ValueError("The theta phase requires width 0.5 and theta values 0, 0.3, 0.5, 0.7, 1")
    if settings.seeds != [42, 2026, 2027] or settings.folds != [0, 1, 2, 3, 4]:
        raise ValueError("The thesis campaign requires seeds 42, 2026, 2027 and five folds")
    progress.update(1, "campaign scope accepted")

    frames_path = campaign.base.paths.cache_root / "frames.jsonl"
    folds_path = campaign.base.paths.cache_root / "folds.json"
    if not frames_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError("Prepared frame records or subject folds are missing")
    records = load_frame_records(frames_path)
    folds = load_folds(folds_path)
    if len(folds) != len(settings.folds):
        raise ValueError(f"Expected five folds, found {len(folds)}")
    protocols = Counter(record.protocol for record in records)
    train_subjects = sorted({record.subject for record in records if record.protocol == "train"})
    test_subjects = sorted({record.subject for record in records if record.protocol == "test"})
    if len(train_subjects) != 15 or len(test_subjects) != 20:
        raise ValueError(
            f"Expected 15 train and 20 test subjects, got {len(train_subjects)} and "
            f"{len(test_subjects)}"
        )
    progress.update(2, f"loaded {len(records)} frame records and five folds")

    depth = validate_pseudo_depth_targets(records, campaign.base.paths.depth_root)
    progress.update(3, f"validated {depth.targets} pseudo-depth targets")

    load_dotenv()
    required = ("WANDB_PROJECT", "WANDB_API_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing required W&B settings: {', '.join(missing)}")
    if os.getenv("WANDB_MODE", "offline") != "online":
        raise ValueError("Set WANDB_MODE=online before starting the thesis campaign")
    progress.update(4, "W&B credentials found and online mode is active")

    if not torch.cuda.is_available():
        raise RuntimeError("The thesis campaign requires an available CUDA device")
    properties = torch.cuda.get_device_properties(0)
    report: dict[str, object] = {
        "campaign": settings.name,
        "records": len(records),
        "protocol_frame_counts": dict(protocols),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "folds": len(folds),
        "pseudo_depth_targets": depth.targets,
        "pseudo_depth_sha256": depth.sha256,
        "device": properties.name,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "vram_gib": properties.total_memory / (1024**3),
        "wandb_entity": os.getenv("WANDB_ENTITY"),
        "wandb_project": os.getenv("WANDB_PROJECT"),
        "wandb_mode": os.getenv("WANDB_MODE"),
        "width_conditions": len(settings.widths),
        "theta_conditions": len(settings.theta_values),
        "seeds": settings.seeds,
        "folds_expected": settings.folds,
        "manualisasi": "excluded by user request",
    }
    write_json(campaign.root / "preflight.json", report)
    progress.update(5, "preflight passed")
    return report
