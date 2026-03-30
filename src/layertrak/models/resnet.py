from typing import Literal

import torch.nn as nn
from torchvision import models

ResNetVariant = Literal["resnet18", "resnet34"]

_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
}


def make_resnet_cifar(
    variant: ResNetVariant = "resnet18",
    num_classes: int = 10,
    *,
    freeze_early: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Create a ResNet adapted for CIFAR-10 (32x32 images).

    Modifications vs. ImageNet version:
    - conv1: 3x3 kernel, stride 1, padding 1 (instead of 7x7/stride 2)
    - maxpool replaced with Identity
    - fc head replaced for `num_classes`

    Args:
        variant: Which ResNet variant to build.
        num_classes: Number of output classes.
        freeze_early: If True, freeze all layers except layer4 and fc.
        pretrained: If True, load ImageNet pretrained weights.

    Returns:
        A ResNet model adapted for CIFAR-10.
    """
    builder, weights = _BUILDERS[variant]
    model = builder(weights=weights if pretrained else None)

    if freeze_early:
        for name, param in model.named_parameters():
            if not name.startswith(("layer4", "fc")):
                param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
