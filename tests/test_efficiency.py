from pathlib import Path

import torch

from cdcn_msu_mfsd.config import load_config
from cdcn_msu_mfsd.efficiency import EfficiencyReport, benchmark_model, save_efficiency_report
from cdcn_msu_mfsd.models import CDCN


def test_benchmark_reports_architecture_costs() -> None:
    report = benchmark_model(
        CDCN(width_multiplier=0.25),
        torch.device("cpu"),
        input_shape=(1, 3, 32, 32),
        warmup_runs=0,
        timed_runs=1,
    )

    assert report.trainable_parameters > 0
    assert report.total_parameters == report.trainable_parameters
    assert report.macs > 0
    assert report.model_state_size_mb > 0
    assert report.latency_ms >= 0
    assert report.environment["device"] == "cpu"


def test_cdc_macs_include_difference_path() -> None:
    ordinary = benchmark_model(
        CDCN(theta=0.0, width_multiplier=0.25), torch.device("cpu"), (1, 3, 32, 32), 0, 1
    )
    cdc = benchmark_model(
        CDCN(theta=0.7, width_multiplier=0.25), torch.device("cpu"), (1, 3, 32, 32), 0, 1
    )

    assert cdc.macs > ordinary.macs


def test_efficiency_reports_are_separated_by_device(tmp_path: Path) -> None:
    config = load_config(Path("configs/experiments/smoke.yaml"))
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"output_root": tmp_path / "outputs"})}
    )
    cpu = EfficiencyReport("cdcn", 1.0, 0.7, "cpu", (1, 3, 256, 256), 1, 1, 1, 1.0, 1.0, 0, 1, {})
    cuda = EfficiencyReport("cdcn", 1.0, 0.7, "cuda", (1, 3, 256, 256), 1, 1, 1, 1.0, 1.0, 0, 1, {})

    cpu_path = save_efficiency_report(config, cpu)
    cuda_path = save_efficiency_report(config, cuda)

    assert cpu_path.name == "efficiency_cpu.json"
    assert cuda_path.name == "efficiency_cuda.json"
    assert cpu_path.exists()
    assert cuda_path.exists()
