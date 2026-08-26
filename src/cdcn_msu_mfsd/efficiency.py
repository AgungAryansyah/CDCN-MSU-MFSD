import io
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from cdcn_msu_mfsd.config import Config
from cdcn_msu_mfsd.models import CentralDifferenceConv2d, build_model
from cdcn_msu_mfsd.training import output_directory, select_device

_DEFAULT_INPUT_SHAPE = (1, 3, 256, 256)


@dataclass(frozen=True)
class EfficiencyReport:
    model: str
    width_multiplier: float
    theta: float
    device: str
    input_shape: tuple[int, int, int, int]
    trainable_parameters: int
    total_parameters: int
    macs: int
    model_state_size_mb: float
    latency_ms: float
    warmup_runs: int
    timed_runs: int
    environment: dict[str, str | int | None]


@dataclass(frozen=True)
class BenchmarkMeasurements:
    input_shape: tuple[int, int, int, int]
    trainable_parameters: int
    total_parameters: int
    macs: int
    model_state_size_mb: float
    latency_ms: float
    environment: dict[str, str | int | None]


def _cpu_name() -> str:
    if processor := platform.processor():
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.machine()


def _environment(device: torch.device) -> dict[str, str | int | None]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "cpu": _cpu_name(),
        "gpu": gpu_name,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }


def estimate_macs(model: nn.Module, inputs: Tensor) -> int:
    total = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _: tuple[Tensor], outputs: Tensor) -> None:
        nonlocal total
        kernel_height, kernel_width = module.kernel_size
        per_output = (module.in_channels // module.groups) * kernel_height * kernel_width
        total += outputs.numel() * per_output

    def cdc_hook(module: CentralDifferenceConv2d, _: tuple[Tensor], outputs: Tensor) -> None:
        nonlocal total
        if module.theta != 0:
            total += outputs.numel() * (module.conv.in_channels // module.conv.groups)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        if isinstance(module, CentralDifferenceConv2d):
            handles.append(module.register_forward_hook(cdc_hook))
    try:
        with torch.inference_mode():
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return total


def _model_state_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getbuffer()) / (1024**2)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, int, int, int] = _DEFAULT_INPUT_SHAPE,
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> BenchmarkMeasurements:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be positive")
    model = model.to(device).eval()
    inputs = torch.randn(input_shape, device=device)
    macs = estimate_macs(model, inputs)
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(timed_runs):
            model(inputs)
        _synchronize(device)
    return BenchmarkMeasurements(
        input_shape=input_shape,
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        macs=macs,
        model_state_size_mb=_model_state_size_mb(model),
        latency_ms=(time.perf_counter() - started) * 1000 / timed_runs,
        environment=_environment(device),
    )


def benchmark_config(
    config: Config,
    requested_device: str = "cpu",
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> EfficiencyReport:
    device = select_device(requested_device)
    values = benchmark_model(
        build_model(config.model.name, config.model.theta, config.model.width_multiplier),
        device,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )
    return EfficiencyReport(
        model=config.model.name,
        width_multiplier=config.model.width_multiplier,
        theta=config.model.theta,
        device=str(device),
        input_shape=values.input_shape,
        trainable_parameters=values.trainable_parameters,
        total_parameters=values.total_parameters,
        macs=values.macs,
        model_state_size_mb=values.model_state_size_mb,
        latency_ms=values.latency_ms,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        environment=values.environment,
    )


def save_efficiency_report(config: Config, report: EfficiencyReport) -> Path:
    path = output_directory(config) / "efficiency.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    return path
