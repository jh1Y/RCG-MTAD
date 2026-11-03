"""MobileNetV4 image branch migrated from Road-Surface-Recognition-IMU.

The module deliberately preserves the source module hierarchy so that legacy
state dictionaries keep the same parameter names.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    """Simple parameter-free attention module."""

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x) or x.ndim != 4:
            return x

        spatial_count = max(x.shape[-2] * x.shape[-1] - 1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        squared_delta = (x - mean).pow(2)
        energy = squared_delta / (
            4.0
            * (
                squared_delta.sum(dim=(2, 3), keepdim=True) / spatial_count
                + self.e_lambda
            )
        ) + 0.5
        return x * self.activation(energy)


class BlockWithSimAM(nn.Module):
    """Wrap a MobileNet block as ``block -> SimAM``."""

    def __init__(self, block: nn.Module, simam_lambda: float = 1e-4) -> None:
        super().__init__()
        self.block = block
        self.simam = SimAM(e_lambda=simam_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if torch.is_tensor(x) and x.ndim == 4:
            x = self.simam(x)
        return x


def insert_simam_after_each_block(
    model: nn.Module,
    insert_stages: Tuple[int, ...] = (2, 3, 4),
    simam_lambda: float = 1e-4,
    verbose: bool = False,
) -> nn.Module:
    if hasattr(model, "blocks"):
        container = model.blocks
        container_name = "blocks"
    elif hasattr(model, "stages"):
        container = model.stages
        container_name = "stages"
    else:
        raise AttributeError("MobileNetV4 backbone has neither blocks nor stages")

    inserted = 0
    for stage_idx in insert_stages:
        if stage_idx < 0 or stage_idx >= len(container):
            if verbose:
                print(f"[road-encoder] skip {container_name}[{stage_idx}]")
            continue

        stage = container[stage_idx]
        if isinstance(stage, (nn.Sequential, nn.ModuleList)):
            for block_idx in range(len(stage)):
                if isinstance(stage[block_idx], BlockWithSimAM):
                    continue
                stage[block_idx] = BlockWithSimAM(stage[block_idx], simam_lambda)
                inserted += 1
        elif not isinstance(stage, BlockWithSimAM):
            container[stage_idx] = BlockWithSimAM(stage, simam_lambda)
            inserted += 1

    if inserted == 0:
        raise RuntimeError(
            f"No SimAM block inserted; check insert_stages={insert_stages}"
        )
    if verbose:
        print(f"[road-encoder] inserted {inserted} SimAM blocks")
    return model


@torch.no_grad()
def infer_backbone_output_dim(backbone: nn.Module, img_size: int = 224) -> int:
    was_training = backbone.training
    backbone.eval()
    try:
        device = next(backbone.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    feature = backbone(torch.zeros(2, 3, img_size, img_size, device=device))
    if isinstance(feature, (tuple, list)):
        feature = feature[0]
    if not torch.is_tensor(feature):
        raise TypeError(f"Backbone output must be a tensor, got {type(feature)!r}")
    if feature.ndim == 4:
        feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
    elif feature.ndim > 2:
        feature = feature.flatten(1)
    if feature.ndim != 2:
        raise RuntimeError(f"Cannot infer backbone output from {feature.shape}")

    if was_training:
        backbone.train()
    return int(feature.shape[1])


class MobileNetV4ImageBranch(nn.Module):
    """MobileNetV4-Conv image encoder with optional per-block SimAM."""

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_medium.e500_r224_in1k",
        pretrained: bool = False,
        use_simam: bool = True,
        output_dim: Optional[int] = None,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        insert_stages: Tuple[int, ...] = (2, 3, 4),
        simam_lambda: float = 1e-4,
        img_size: int = 224,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise ImportError("Install timm>=1.0.8,<1.1 to build RTI-MFNet") from exc

        self.model_name = model_name
        self.use_simam = bool(use_simam)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if self.use_simam:
            self.backbone = insert_simam_after_each_block(
                self.backbone,
                insert_stages=insert_stages,
                simam_lambda=simam_lambda,
                verbose=verbose,
            )

        self.backbone_out_dim = infer_backbone_output_dim(self.backbone, img_size)
        if output_dim is None:
            self.proj = nn.Identity()
            self.out_dim = self.backbone_out_dim
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.backbone_out_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.out_dim = int(output_dim)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(image)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        if feature.ndim == 4:
            feature = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        elif feature.ndim > 2:
            feature = feature.flatten(1)
        return self.proj(feature)


def build_mobilenetv4_image_branch(
    exp_name: str = "E2",
    **kwargs,
) -> MobileNetV4ImageBranch:
    exp_name = exp_name.upper()
    if exp_name not in {"E1", "E2"}:
        raise ValueError("exp_name must be E1 (plain) or E2 (SimAM)")
    return MobileNetV4ImageBranch(use_simam=exp_name == "E2", **kwargs)

