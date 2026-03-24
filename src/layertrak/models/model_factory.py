from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from layertrak.models.mobilenet import make_mobilenetv2_cifar
from layertrak.models.resnet import make_resnet_cifar

ModelName = Literal["resnet18", "resnet34", "mobilenetv2"]


def create_model(
    name: ModelName,
    num_classes: int = 10,
    *,
    device: str = "cpu",
    freeze_early: bool = True,
    pretrained: bool = True,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    """Create a model by name, optionally loading a checkpoint.

    Args:
        name: Model architecture name.
        num_classes: Number of output classes.
        device: Device to move the model to.
        freeze_early: Whether to freeze early layers.
        pretrained: Whether to load pretrained ImageNet weights.
        checkpoint_path: Optional path to a saved state_dict.

    Returns:
        Model moved to the specified device.
    """
    if name in ("resnet18", "resnet34"):
        model = make_resnet_cifar(
            variant=name,
            num_classes=num_classes,
            freeze_early=freeze_early,
            pretrained=pretrained,
        )
    elif name == "mobilenetv2":
        model = make_mobilenetv2_cifar(
            num_classes=num_classes,
            freeze_early=freeze_early,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    return model.to(device)
