import torch
from torch.nn import functional as F

from cdcn_msu_mfsd.models import CentralDifferenceConv2d, build_model
from cdcn_msu_mfsd.pipelines.common import Campaign, Progress, write_json


def run(campaign: Campaign) -> dict[str, object]:
    conditions = campaign.conditions
    progress = Progress(campaign, "verify", len(conditions))
    records = []
    operator_errors = {}
    for theta in (0.0, 0.7, 1.0):
        layer = CentralDifferenceConv2d(3, 4, theta=theta).eval()
        inputs = torch.randn(1, 3, 9, 9)
        with torch.inference_mode():
            actual = layer(inputs)
            reference = layer.conv(inputs)
            if theta != 0.0:
                difference_kernel = layer.conv.weight.sum(dim=(2, 3), keepdim=True)
                difference = F.conv2d(
                    inputs,
                    difference_kernel,
                    bias=layer.conv.bias,
                    stride=layer.conv.stride,
                    groups=layer.conv.groups,
                )
                reference = reference - theta * difference
        error = float((actual - reference).abs().max())
        operator_errors[str(theta)] = error
        if not torch.allclose(actual, reference, atol=1e-6, rtol=1e-6):
            raise ValueError(f"Central-difference operator check failed at theta={theta}")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 6))
    try:
        for index, condition in enumerate(conditions, start=1):
            config = campaign.config(condition)
            model = build_model(
                config.model.name, config.model.theta, config.model.width_multiplier
            ).eval()
            shapes: dict[str, list[int]] = {}
            handles = []
            for name in ("stem", "block1", "block2", "block3", "head"):
                module = getattr(model, name)
                handles.append(
                    module.register_forward_hook(
                        lambda _, __, output, module_name=name, shape_map=shapes: (
                            shape_map.__setitem__(module_name, list(output.shape))
                        )
                    )
                )
            try:
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, 256, 256))
            finally:
                for handle in handles:
                    handle.remove()
            if list(output.shape) != [1, 32, 32]:
                raise ValueError(f"Unexpected output shape for {condition.key}: {output.shape}")
            if not {"stem", "block1", "block2", "block3", "head"}.issubset(shapes):
                raise ValueError(f"Missing intermediate shape evidence for {condition.key}")
            parameters = sum(parameter.numel() for parameter in model.parameters())
            records.append(
                {
                    "condition": condition.key,
                    "width_multiplier": condition.width,
                    "theta": condition.theta,
                    "intermediate_shapes": shapes,
                    "output_shape": list(output.shape),
                    "total_parameters": parameters,
                    "operator_check": "forward completed with finite output",
                    "finite_output": bool(torch.isfinite(output).all()),
                }
            )
            if not records[-1]["finite_output"]:
                raise ValueError(f"Model output contains non-finite values: {condition.key}")
            progress.update(index, condition.key)
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "input_shape": [1, 3, 256, 256],
        "expected_output_shape": [1, 32, 32],
        "central_difference_max_abs_error": operator_errors,
        "central_difference_tolerance": 1e-6,
        "conditions": records,
    }
    write_json(campaign.root / "verification" / "model_shapes.json", report)
    return report
