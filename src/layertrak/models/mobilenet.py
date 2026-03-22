import torch.nn as nn
from torchvision import models


def make_mobilenetv2_cifar(
    num_classes: int = 10,
    *,
    freeze_early: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Create a MobileNetV2 adapted for CIFAR-10 (32x32 images).

    Modifications vs. ImageNet version:
    - First conv stride reduced from 2 to 1 (better for 32x32)
    - Classifier head replaced for `num_classes`

    Args:
        num_classes: Number of output classes.
        freeze_early: If True, freeze all feature layers except features[14:].
        pretrained: If True, load ImageNet pretrained weights.

    Returns:
        A MobileNetV2 model adapted for CIFAR-10.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_early:
        for name, param in model.named_parameters():
            if not (name.startswith("classifier") or _is_late_feature(name)):
                param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _is_late_feature(name: str) -> bool:
    """Check if parameter belongs to features[14:] (late blocks)."""
    if not name.startswith("features."):
        return False
    parts = name.split(".")
    if len(parts) < 2:
        return False
    try:
        idx = int(parts[1])
    except ValueError:
        return False
    return idx >= 14
