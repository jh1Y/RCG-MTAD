"""Differentiable image/sensor-to-road-and-tire model.

Images are RGB tensors resized and ImageNet-normalized by the dataset. Sensor
inputs are in physical units: acceleration, IMU, and [kPa, degrees C, km/h].
All sensor normalization and the healthy P-T calibration live in state_dict.
The road branch sees the first 256 acceleration samples; the tire branch sees
the first 1024, matching the source feature-extraction windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
from torch import nn

from .road_encoder import RTIMFNetConfig, RTIMFNetRoadEncoder, RoadEncoderOutput, SOURCE_ROAD_CLASSES
from .tire_classifier import (
    RoadConditionedTireClassifier, TireClassifierConfig, TireClassifierOutput,
)


@dataclass(frozen=True)
class InputConfig:
    acc_length: int = 1024
    road_acc_length: int = 256
    imu_channels: int = 9
    imu_length: int = 1
    gyro_indices: tuple[int, ...] = (3, 4, 5)
    state_dim: int = 3

    def __post_init__(self):
        if min(self.acc_length, self.road_acc_length, self.imu_channels,
               self.imu_length, self.state_dim) <= 0:
            raise ValueError("Input dimensions must be positive")
        if self.road_acc_length > self.acc_length:
            raise ValueError("road_acc_length cannot exceed acc_length")
        if not self.gyro_indices or len(set(self.gyro_indices)) != len(self.gyro_indices):
            raise ValueError("gyro_indices must be nonempty and unique")
        if min(self.gyro_indices) < 0 or max(self.gyro_indices) >= self.imu_channels:
            raise ValueError("gyro_indices must lie within the IMU channels")


@dataclass(frozen=True)
class RCGMTADConfig:
    road: RTIMFNetConfig = field(default_factory=lambda: RTIMFNetConfig(
        freeze_backbone=False, train_last_stage=False, gate_temperature=1.0,
    ))
    tire: TireClassifierConfig = field(default_factory=lambda: TireClassifierConfig(
        acc_length=1024,
    ))
    inputs: InputConfig = field(default_factory=InputConfig)
    freeze_road_encoder: bool = False

    def __post_init__(self):
        if tuple(self.road.class_names) != SOURCE_ROAD_CLASSES or self.tire.num_classes != 4:
            raise ValueError("RCG-MTAD requires the fixed four-road and four-tire class order")
        if self.road.image_fusion_dim != self.tire.road_embedding_dim:
            raise ValueError("Road embedding and tire conditioning dimensions differ")
        if self.road.num_classes != self.tire.road_classes:
            raise ValueError("Road class counts differ between the two branches")
        if self.inputs.acc_length != self.tire.acc_length:
            raise ValueError("Input and tire acceleration lengths differ")
        if (self.inputs.imu_channels != self.road.imu_channels
                or self.inputs.state_dim != self.road.scalar_dim):
            raise ValueError("Road encoder and raw input dimensions differ")
        if (len(self.inputs.gyro_indices) != self.tire.gyro_channels
                or self.inputs.imu_length != self.tire.gyro_length):
            raise ValueError("Tire gyro configuration does not match raw IMU inputs")
        if max(self.tire.pressure_state_index, self.tire.temperature_state_index) >= self.inputs.state_dim:
            raise ValueError("Pressure/temperature indices exceed vehicle_state width")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> RCGMTADConfig:
        road = dict(values.get("road", {}))
        for key in ("class_names", "insert_stages", "imu_channel_indices"):
            if road.get(key) is not None and key in road:
                road[key] = tuple(road[key])
        inputs = dict(values.get("inputs", {}))
        if "gyro_indices" in inputs:
            inputs["gyro_indices"] = tuple(inputs["gyro_indices"])
        defaults = cls()
        road_config = RTIMFNetConfig(**{**asdict(defaults.road), **road})
        input_config = InputConfig(**{
            "imu_channels": road_config.imu_channels, "state_dim": road_config.scalar_dim, **inputs,
        })
        tire_config = TireClassifierConfig(**{
            **asdict(defaults.tire), "acc_length": input_config.acc_length,
            "road_embedding_dim": road_config.image_fusion_dim,
            "road_classes": road_config.num_classes,
            "gyro_channels": len(input_config.gyro_indices), "gyro_length": input_config.imu_length,
            **values.get("tire", {}),
        })
        return cls(
            road=road_config, tire=tire_config, inputs=input_config,
            freeze_road_encoder=bool(values.get("freeze_road_encoder", False)),
        )


class RCGMTADOutput(NamedTuple):
    road: RoadEncoderOutput
    tire: TireClassifierOutput


class SensorPreprocessor(nn.Module):
    """Train-fitted road normalization and deterministic raw-window preparation."""

    def __init__(self, config: InputConfig):
        super().__init__()
        self.config = config
        for name, shape in (
            ("acc", (config.road_acc_length,)),
            ("imu", (config.imu_channels, config.imu_length)),
            ("state", (config.state_dim,)),
        ):
            self.register_buffer(f"{name}_mean", torch.zeros(shape))
            self.register_buffer(f"{name}_std", torch.ones(shape))
        self.register_buffer("fitted", torch.tensor(False))

    def prepare(self, tire_acc, imu, vehicle_state):
        cfg = self.config
        if tire_acc.ndim == 3 and tire_acc.shape[1] == 1:
            tire_acc = tire_acc[:, 0]
        if tire_acc.ndim != 2 or tire_acc.shape[1] < cfg.acc_length:
            raise ValueError(f"tire_acc must be [B,L] with L >= {cfg.acc_length}")
        tire_acc = tire_acc[:, :cfg.acc_length]
        # Explicit layouts avoid guessing channels from short time windows.
        if imu.ndim == 2 and cfg.imu_length == 1 and imu.shape[1] == cfg.imu_channels:
            imu = imu.unsqueeze(-1)
        elif imu.ndim == 3 and tuple(imu.shape[1:]) == (cfg.imu_channels, cfg.imu_length):
            pass
        elif imu.ndim == 3 and tuple(imu.shape[1:]) == (cfg.imu_length, cfg.imu_channels):
            imu = imu.transpose(1, 2)
        else:
            raise ValueError(f"imu must have {cfg.imu_channels} channels and {cfg.imu_length} time steps")
        if vehicle_state.ndim != 2 or vehicle_state.shape[1] != cfg.state_dim:
            raise ValueError(f"vehicle_state must be [B,{cfg.state_dim}]")
        if not (len(tire_acc) == len(imu) == len(vehicle_state)):
            raise ValueError("Sensor batch sizes differ")
        if not all(torch.isfinite(value).all() for value in (tire_acc, imu, vehicle_state)):
            raise ValueError("Sensor inputs contain NaN or infinite values")
        return tire_acc, imu, vehicle_state

    @torch.no_grad()
    def fit(self, tire_acc, imu, vehicle_state):
        tire_acc, imu, vehicle_state = self.prepare(tire_acc, imu, vehicle_state)
        if len(tire_acc) < 2:
            raise ValueError("At least two training rows are required for normalization")
        for name, value in (("acc", tire_acc[:, :self.config.road_acc_length]),
                            ("imu", imu), ("state", vehicle_state)):
            getattr(self, f"{name}_mean").copy_(value.mean(0))
            std = value.std(0, unbiased=False)
            # StandardScaler convention for constant (including padded) features.
            getattr(self, f"{name}_std").copy_(torch.where(std > 1e-6, std, 1.0))
        self.fitted.fill_(True)

    def road_inputs(self, tire_acc, imu, vehicle_state):
        if not bool(self.fitted):
            raise RuntimeError("Road input normalization has not been fitted")
        return (
            (tire_acc[:, :self.config.road_acc_length] - self.acc_mean) / self.acc_std,
            (imu - self.imu_mean) / self.imu_std,
            (vehicle_state - self.state_mean) / self.state_std,
        )


class RCGMTAD(nn.Module):
    """One forward pass produces road and four-state tire probabilities.

    ``forward(image, tire_acc, imu, vehicle_state)`` requires no cached road
    features. Tire loss can backpropagate through Proad and zroad into the road
    encoder. Optional freezing keeps the road encoder in eval mode, including BN.
    """

    def __init__(self, config: RCGMTADConfig | None = None, *, image_backbone=None):
        super().__init__()
        self.config = config or RCGMTADConfig()
        self.preprocessor = SensorPreprocessor(self.config.inputs)
        self.road_encoder = RTIMFNetRoadEncoder(self.config.road, image_backbone=image_backbone)
        self.tire_classifier = RoadConditionedTireClassifier(self.config.tire)
        if self.config.freeze_road_encoder:
            self.road_encoder.requires_grad_(False)
            self.road_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_road_encoder:
            self.road_encoder.eval()
        elif mode and self.config.road.freeze_backbone and not self.config.road.train_last_stage:
            self.road_encoder.image_backbone.eval()
        return self

    def encode_road(self, image, tire_acc, imu, vehicle_state):
        acc, imu, state = self.preprocessor.prepare(tire_acc, imu, vehicle_state)
        road_acc, road_imu, road_state = self.preprocessor.road_inputs(acc, imu, state)
        return self.road_encoder.encode_road(image, road_acc, road_state, road_imu)

    def forward(self, image, tire_acc, imu, vehicle_state) -> RCGMTADOutput:
        acc, imu, state = self.preprocessor.prepare(tire_acc, imu, vehicle_state)
        road = self.encode_road(image, acc, imu, state)
        pt = state[:, [self.config.tire.pressure_state_index, self.config.tire.temperature_state_index]]
        tire = self.tire_classifier(
            acc, imu[:, self.config.inputs.gyro_indices, :], pt,
            road.probabilities, road.zroad,
        )
        return RCGMTADOutput(road=road, tire=tire)
