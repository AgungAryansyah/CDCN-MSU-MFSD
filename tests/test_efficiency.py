import torch

from cdcn_msu_mfsd.efficiency import benchmark_model
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
