from __future__ import annotations

import torch
from torch import nn

_SPATIAL_MAX_NORM = 1.0
_CLASSIFIER_MAX_NORM = 0.25
_FIRST_POOL = 4
_SECOND_POOL = 8
_SEPARABLE_KERNEL = 16


def _batch_norm(n_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(n_features, momentum=0.01, eps=1e-3)


def _same_padding(kernel: int) -> nn.ZeroPad2d:
    total = kernel - 1
    return nn.ZeroPad2d((total // 2, total - total // 2, 0, 0))


def _limit_norm(weight: torch.Tensor, limit: float) -> None:
    norms = weight.flatten(1).norm(dim=1).clamp(min=limit)
    weight.mul_((limit / norms).view(-1, *[1] * (weight.dim() - 1)))


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 256,
        n_classes: int = 4,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.temporal_pad = _same_padding(kernel_length)
        self.temporal = nn.Conv2d(1, f1, (1, kernel_length), bias=False)
        self.temporal_norm = _batch_norm(f1)
        self.spatial = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False)
        self.spatial_norm = _batch_norm(f1 * d)
        self.first_pool = nn.AvgPool2d((1, _FIRST_POOL))
        self.first_dropout = nn.Dropout(dropout)
        self.separable_pad = _same_padding(_SEPARABLE_KERNEL)
        self.separable_depthwise = nn.Conv2d(f1 * d, f1 * d, (1, _SEPARABLE_KERNEL), groups=f1 * d, bias=False)
        self.separable_pointwise = nn.Conv2d(f1 * d, f2, 1, bias=False)
        self.separable_norm = _batch_norm(f2)
        self.second_pool = nn.AvgPool2d((1, _SECOND_POOL))
        self.second_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(f2 * (n_samples // (_FIRST_POOL * _SECOND_POOL)), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_norm(self.temporal(self.temporal_pad(x.unsqueeze(1))))
        x = self.activation(self.spatial_norm(self.spatial(x)))
        x = self.first_dropout(self.first_pool(x))
        x = self.separable_pointwise(self.separable_depthwise(self.separable_pad(x)))
        x = self.activation(self.separable_norm(x))
        x = self.second_dropout(self.second_pool(x))
        return self.classifier(x.flatten(1))

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        _limit_norm(self.spatial.weight, _SPATIAL_MAX_NORM)
        _limit_norm(self.classifier.weight, _CLASSIFIER_MAX_NORM)
