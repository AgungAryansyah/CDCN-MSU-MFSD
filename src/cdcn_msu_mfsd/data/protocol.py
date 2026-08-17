import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from cdcn_msu_mfsd.data.manifest import VideoRecord


@dataclass(frozen=True)
class Fold:
    index: int
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]


def make_folds(records: list[VideoRecord], count: int = 5, seed: int = 2026) -> list[Fold]:
    subjects = sorted({record.subject for record in records if record.protocol == "train"})
    if len(subjects) < count:
        raise ValueError(f"Cannot create {count} folds from {len(subjects)} subjects")
    random.Random(seed).shuffle(subjects)
    validation_groups = [tuple(subjects[index::count]) for index in range(count)]
    folds = []
    for index, validation in enumerate(validation_groups):
        train = tuple(sorted(set(subjects) - set(validation)))
        folds.append(Fold(index, train, tuple(sorted(validation))))
    validate_folds(folds, set(subjects))
    return folds


def validate_folds(folds: list[Fold], expected_subjects: set[str]) -> None:
    seen: list[str] = []
    for fold in folds:
        if set(fold.train_subjects) & set(fold.validation_subjects):
            raise ValueError(f"Fold {fold.index} has subject leakage")
        if set(fold.train_subjects) | set(fold.validation_subjects) != expected_subjects:
            raise ValueError(f"Fold {fold.index} does not cover the training protocol")
        seen.extend(fold.validation_subjects)
    if sorted(seen) != sorted(expected_subjects):
        raise ValueError("Each subject must appear in validation exactly once")


def save_folds(folds: list[Fold], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([asdict(fold) for fold in folds], indent=2, sort_keys=True) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_folds(path: Path) -> list[Fold]:
    return [
        Fold(
            index=item["index"],
            train_subjects=tuple(item["train_subjects"]),
            validation_subjects=tuple(item["validation_subjects"]),
        )
        for item in json.loads(path.read_text())
    ]

