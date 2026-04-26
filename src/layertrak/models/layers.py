from typing import TypedDict

import torch.nn as nn


class LayerConfig(TypedDict):
    """A named TRAK layer configuration."""

    name: str
    prefixes: list[str] | None


def _feature_prefixes(start: int, end: int) -> list[str]:
    """Generate MobileNetV2 feature block prefixes for range [start, end)."""
    return [f"features.{i}." for i in range(start, end)]


RESNET_LAYER_CONFIGS: list[LayerConfig] = [
    {"name": "head_only", "prefixes": ["fc."]},
    {"name": "late", "prefixes": ["layer4.", "fc."]},
    {"name": "mid_late", "prefixes": ["layer3."]},
    {"name": "early", "prefixes": ["conv1.", "layer1."]},
    {"name": "full_model", "prefixes": None},
]

MOBILENETV2_LAYER_CONFIGS: list[LayerConfig] = [
    {"name": "head_only", "prefixes": ["classifier."]},
    {"name": "late", "prefixes": [*_feature_prefixes(14, 19), "classifier."]},
    {"name": "mid_late", "prefixes": _feature_prefixes(7, 14)},
    {"name": "early", "prefixes": _feature_prefixes(0, 7)},
    {"name": "full_model", "prefixes": None},
]

_CONFIGS_BY_ARCH: dict[str, list[LayerConfig]] = {
    "resnet18": RESNET_LAYER_CONFIGS,
    "resnet34": RESNET_LAYER_CONFIGS,
    "mobilenetv2": MOBILENETV2_LAYER_CONFIGS,
}


def get_layer_configs(arch: str) -> list[LayerConfig]:
    """Return all layer configurations for a given architecture.

    Args:
        arch: Architecture name ("resnet18", "resnet34", "mobilenetv2").

    Returns:
        List of LayerConfig dicts for the architecture.

    Raises:
        KeyError: If architecture is not recognized.
    """
    return _CONFIGS_BY_ARCH[arch]


def get_grad_wrt(model: nn.Module, prefixes: list[str] | None) -> list[str] | None:
    """Return parameter names matching any of the given prefixes.

    Args:
        model: PyTorch model.
        prefixes: List of parameter name prefixes to match.
            If None, gradients are computed for the full model.

    Returns:
        List of matching parameter names for use with TRAKer's grad_wrt.
        Returns None for the full-model baseline.
    """
    if prefixes is None:
        return None

    return [name for name, _ in model.named_parameters() if any(name.startswith(p) for p in prefixes)]
