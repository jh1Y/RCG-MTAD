from __future__ import annotations

import torch
import torch.nn as nn

from tire_anomaly_system.models.road_encoder import RTIMFNetConfig, RTIMFNetRoadEncoder


class TinyImageBranch(nn.Module):
    out_dim = 12

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(3, self.out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(image).flatten(1))


def build_tiny_model(**overrides) -> RTIMFNetRoadEncoder:
    kwargs = dict(
        image_size=16,
        image_fusion_dim=16,
        imu_channels=9,
        imu_channel_indices=(3, 4, 5),
        freeze_backbone=False,
        train_last_stage=False,
        dropout=0.0,
        pretrained=False,
    )
    kwargs.update(overrides)
    return RTIMFNetRoadEncoder(
        RTIMFNetConfig(**kwargs),
        image_backbone=TinyImageBranch(),
    )


def test_flexible_multimodal_forward() -> None:
    model = build_tiny_model().eval()
    output = model.encode_road(
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 37),
        torch.randn(2, 3),
        torch.randn(2, 19, 3),
    )
    assert output.zroad.shape == (2, 16)
    assert output.logits.shape == (2, 4)
    assert output.probabilities.shape == (2, 4)
    assert torch.allclose(output.probabilities.sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.gates.sum(1), torch.ones(2), atol=1e-5)


def test_dynamic_skip_can_avoid_all_images() -> None:
    model = build_tiny_model(
        enable_dynamic_image_skip=True,
        image_skip_threshold=1.1,
    ).eval()
    output = model.encode_road(
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 41),
        torch.randn(2, 3),
        torch.randn(2, 3, 11),
    )
    assert not output.image_used_mask.any()
    assert torch.allclose(output.gates[:, 0], torch.zeros(2))
    assert torch.allclose(output.gates.sum(1), torch.ones(2), atol=1e-5)


def test_legacy_forward_returns_logits() -> None:
    model = build_tiny_model().eval()
    logits = model(
        torch.randn(2, 3, 16, 16),
        torch.randn(2, 25),
        torch.randn(2, 3),
        torch.randn(2, 13, 3),
    )
    assert logits.shape == (2, 4)

