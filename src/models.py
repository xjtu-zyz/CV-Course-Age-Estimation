"""Model factory for DenseNet121 and ViT-B/16 (torchvision only).

Final classifier is replaced with Linear(in_features -> 1). Forward returns
(B, 1); train/eval code squeezes to (B,) before computing the loss.
"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def _build_densenet(pretrained: bool) -> nn.Module:
    weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    return model


def _build_vit_b_16(pretrained: bool) -> nn.Module:
    weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vit_b_16(weights=weights)
    # torchvision vit_b_16 head is model.heads (Sequential) ending with `head` Linear.
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 1)
    return model


_BUILDERS = {
    "densenet": _build_densenet,
    "densenet121": _build_densenet,
    "vit": _build_vit_b_16,
    "vit_b_16": _build_vit_b_16,
    # `vit_baseline` is the same backbone but reads configs/vit_baseline.yaml
    # (different fine-tune recipe) and writes results to results/vit_baseline/.
    "vit_baseline": _build_vit_b_16,
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Build a regression model by short name.

    name in {densenet, densenet121, vit, vit_b_16}.
    """
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[key](pretrained)
