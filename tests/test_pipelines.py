from pathlib import Path

import pytest
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cdcn_msu_mfsd.config import config_hash
from cdcn_msu_mfsd.pipelines import test_evaluation
from cdcn_msu_mfsd.pipelines.common import Campaign, Condition, input_fingerprint, write_json
from cdcn_msu_mfsd.pipelines.freeze import run as freeze_plan
from cdcn_msu_mfsd.pipelines.runner import plan, run_component
from cdcn_msu_mfsd.pipelines.selection import run as select_width


def _campaign(tmp_path: Path) -> Campaign:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "dataset_root": str(tmp_path / "data"),
                    "train_subjects": str(tmp_path / "train.txt"),
                    "test_subjects": str(tmp_path / "test.txt"),
                    "cache_root": str(tmp_path / "cache"),
                    "output_root": str(tmp_path / "outputs"),
                    "depth_root": str(tmp_path / "depth"),
                }
            }
        )
    )
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "name": "test-campaign",
                "base_config": "base.yaml",
                "width_experiment": "small-width",
                "theta_experiment": "small-theta-a050",
            }
        )
    )
    return Campaign(campaign_path)


def test_campaign_plan_separates_all_stages_and_counts(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    result = plan(campaign)

    assert result["reused_width_cv_records"] == 45
    assert result["new_theta_cv_records"] == 60
    assert result["refit_jobs"] == 21
    assert result["test_jobs"] == 21
    assert len(result["official_conditions"]) == 7
    assert "manualisasi" in result and result["manualisasi"] == "excluded"


def test_pipeline_requires_unchanged_inputs_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path)
    write_json(
        campaign.state_path,
        {
            "stages": {
                "preflight": {
                    "status": "completed",
                    "input_fingerprint": input_fingerprint(campaign),
                }
            }
        },
    )
    monkeypatch.setattr(
        "cdcn_msu_mfsd.pipelines.verification.run", lambda _campaign: {"verified": True}
    )

    assert run_component("verify", campaign) == {"verified": True}
    campaign.path.write_text(campaign.path.read_text() + "# changed\n")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        run_component("verify", campaign)


def test_campaign_paths_do_not_mix_refit_conditions(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    first = campaign.refit_directory(Condition(0.5, 0.3), 42)
    second = campaign.refit_directory(Condition(0.5, 0.5), 42)

    assert first != second
    assert campaign.cv_directory(Condition(0.5, 0.7), 0, 2026).parts[-4] == "small-width"
    assert campaign.cv_directory(Condition(0.5, 0.3), 0, 2026).parts[-4] == "small-theta-a050"


def test_selection_applies_compact_pareto_then_acer(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    records = []
    efficiency = []
    for width, acer, latency, macs in (
        (1.0, 0.05, 4.0, 50),
        (0.5, 0.10, 2.0, 13),
        (0.25, 0.12, 1.0, 3),
    ):
        condition = Condition(width, 0.7)
        for seed in campaign.settings.seeds:
            records.append({"condition": condition.key, "seed": seed, "acer": acer})
        efficiency.append(
            {
                "condition": condition.key,
                "device": "cpu",
                "latency_ms": latency,
                "macs": macs,
                "total_parameters": macs * 100,
            }
        )
    write_json(campaign.root / "width_oof.json", {"records": records})
    write_json(campaign.root / "width_efficiency.json", {"records": efficiency})

    result = select_width(campaign)

    assert result["selected"]["width_multiplier"] == 0.5
    assert {item["width_multiplier"] for item in result["pareto_frontier"]} == {0.5, 0.25}


def test_freeze_plan_rejects_changed_threshold_source(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    write_json(campaign.root / "width_selection.json", {"selected": {"width_multiplier": 0.5}})
    for condition in campaign.conditions:
        for seed in campaign.settings.seeds:
            for fold in campaign.settings.folds:
                write_json(
                    campaign.cv_directory(condition, fold, seed) / "summary.json",
                    {"best_epoch": fold + 2},
                )
            oof_dir = campaign.oof_directory(condition, seed)
            write_json(oof_dir / "threshold.json", {"threshold": 0.7})
            write_json(oof_dir / "oof_metrics.json", {"acer": 0.1, "auc": 0.9})
            (oof_dir / "oof_videos.jsonl").write_text("{}\n")

    frozen = freeze_plan(campaign)
    assert len(frozen["conditions"]) == 21
    assert all(item["refit_epochs"] == 5 for item in frozen["conditions"])

    threshold_path = campaign.oof_directory(campaign.conditions[0], 42) / "threshold.json"
    write_json(threshold_path, {"threshold": 0.71})
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_plan(campaign)


class _TestDataset(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": torch.full((3, 8, 8), float(index)),
            "target": torch.zeros(32, 32),
            "label": torch.tensor(1 if index == 0 else 0),
            "video_id": f"video-{index}",
            "frame_index": index,
            "attack_type": "genuine" if index == 0 else "printed_photo",
            "device": "android",
        }


class _CountModel(nn.Module):
    forward_calls = 0

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        type(self).forward_calls += len(inputs)
        return F.interpolate(self.conv(inputs), size=(32, 32)).squeeze(1)


def test_test_predictions_resume_from_video_frame_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path)
    campaign.settings = campaign.settings.model_copy(
        update={"test_videos": 2, "test_genuine": 1, "test_attacks": 1}
    )
    write_json(campaign.plan_path, {"frozen": True})
    config = campaign.config(Condition(0.5, 0.7), seed=42, phase="refit", epochs=1)
    model = _CountModel()
    checkpoint_path = tmp_path / "final.pt"
    torch.save({"model": model.state_dict(), "config_hash": config_hash(config)}, checkpoint_path)
    monkeypatch.setattr(test_evaluation, "build_model", lambda *_args: _CountModel())
    monkeypatch.setattr(
        test_evaluation,
        "_evaluation_loader",
        lambda *_args: DataLoader(_TestDataset(), batch_size=1),
    )
    monkeypatch.setattr(test_evaluation, "select_device", lambda _requested: torch.device("cpu"))
    _CountModel.forward_calls = 0
    output_dir = tmp_path / "test-output"

    report = test_evaluation._predict_resumable(config, checkpoint_path, output_dir, 0.5, campaign)
    assert report["genuine"] == 1
    assert report["attacks"] == 1
    assert _CountModel.forward_calls == 2

    test_evaluation._predict_resumable(config, checkpoint_path, output_dir, 0.5, campaign)
    assert _CountModel.forward_calls == 2
