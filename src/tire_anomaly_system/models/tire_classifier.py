"""Road-conditioned healthy/leak/overheat/imbalance classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TireClassifierConfig:
    acc_length: int = 256
    gyro_channels: int = 3
    gyro_length: int = 1
    response_dim: int = 256
    road_classes: int = 4
    road_feature_dim: int = 16
    use_road_probability: bool = True
    use_dynamic_response: bool = True
    use_gyro: bool = True
    use_pressure_temperature: bool = True
    use_zroad_film: bool = True
    use_zroad_concat: bool = False
    acceleration_encoder_variant: str = "dual_rms"
    road_embedding_dim: int = 256
    film_hidden_dim: int = 128
    pressure_temperature_feature_dim: int = 4
    pressure_temperature_hidden_dim: int = 16
    pressure_temperature_mode: str = "coupled"
    pressure_state_index: int = 0
    temperature_state_index: int = 1
    atmospheric_pressure_kpa: float = 101.325
    reference_temperature_c: float = 20.0
    minimum_residual_scale_kpa: float = 0.5
    hidden_dim: int = 128
    num_classes: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        dimensions = (
            self.acc_length,
            self.gyro_channels,
            self.gyro_length,
            self.response_dim,
            self.road_classes,
            self.road_feature_dim,
            self.road_embedding_dim,
            self.film_hidden_dim,
            self.pressure_temperature_feature_dim,
            self.pressure_temperature_hidden_dim,
            self.hidden_dim,
            self.num_classes,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("All classifier dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        valid_pressure_temperature_modes = {
            "coupled",
            "raw",
            "pressure_only",
            "temperature_only",
            "corrected_pressure",
            "residual",
        }
        if self.pressure_temperature_mode not in valid_pressure_temperature_modes:
            raise ValueError(
                "pressure_temperature_mode must be one of "
                f"{sorted(valid_pressure_temperature_modes)}"
            )
        expected_pt_dim = {
            "coupled": 4,
            "raw": 2,
            "pressure_only": 1,
            "temperature_only": 1,
            "corrected_pressure": 1,
            "residual": 2,
        }[self.pressure_temperature_mode]
        if self.pressure_temperature_feature_dim != expected_pt_dim:
            raise ValueError(
                "pressure_temperature_feature_dim must be "
                f"{expected_pt_dim} for mode {self.pressure_temperature_mode!r}"
            )
        if self.response_dim != 256:
            raise ValueError("response_dim must remain 256")
        if not self.use_dynamic_response and self.use_gyro:
            raise ValueError(
                "use_gyro must be false when use_dynamic_response is false"
            )
        if not any((
            self.use_dynamic_response,
            self.use_pressure_temperature,
            self.use_road_probability,
            self.use_zroad_concat,
        )):
            raise ValueError("At least one classifier input branch must be enabled")
        if self.use_zroad_concat and self.use_zroad_film:
            raise ValueError("Direct zroad concatenation requires use_zroad_film=False")
        if self.pressure_state_index == self.temperature_state_index:
            raise ValueError("Pressure and temperature state indices must differ")
        if min(self.pressure_state_index, self.temperature_state_index) < 0:
            raise ValueError("Pressure and temperature state indices must be non-negative")
        if self.atmospheric_pressure_kpa <= 0.0:
            raise ValueError("atmospheric_pressure_kpa must be positive")
        if self.reference_temperature_c <= -273.15:
            raise ValueError("reference_temperature_c must exceed absolute zero")
        if self.minimum_residual_scale_kpa <= 0.0:
            raise ValueError("minimum_residual_scale_kpa must be positive")
        valid_acceleration_variants = {
            "local_only",
            "long_only",
            "dual_avg",
            "dual_rms",
            "dual_no_dilation",
        }
        if self.acceleration_encoder_variant not in valid_acceleration_variants:
            raise ValueError(
                "acceleration_encoder_variant must be one of "
                f"{sorted(valid_acceleration_variants)}"
            )


class TireClassifierOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    response_feature: torch.Tensor
    road_feature: torch.Tensor
    pressure_temperature_features: torch.Tensor
    pressure_temperature_feature: torch.Tensor
    modulated_feature: torch.Tensor


class TireStateOutput(NamedTuple):
    """Sensor-only features before downstream road-conditioned fusion."""

    acceleration_feature: torch.Tensor
    imu_feature: torch.Tensor
    response_feature: torch.Tensor
    pressure_temperature_features: torch.Tensor
    pressure_temperature_feature: torch.Tensor
    tire_feature: torch.Tensor


class GlobalRMSPool1d(nn.Module):
    """Parameter-free global RMS pooling over a temporal feature map."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 3:
            raise ValueError(
                "GlobalRMSPool1d expects [batch, channels, time], "
                f"got {tuple(feature.shape)}"
            )
        return torch.sqrt(feature.square().mean(dim=-1) + self.eps)


class DualScaleAccelerationEncoder(nn.Module):
    """Configurable acceleration encoder used by the architecture ablations."""

    branch_dim = 64
    output_dim = 64

    def __init__(self, dropout: float, variant: str = "dual_rms") -> None:
        super().__init__()
        if variant not in {
            "local_only",
            "long_only",
            "dual_avg",
            "dual_rms",
            "dual_no_dilation",
        }:
            raise ValueError(f"Unknown acceleration encoder variant: {variant}")
        self.variant = variant
        self.local_branch = (
            nn.Sequential(
                nn.Conv1d(1, 32, 7, stride=2, padding=3),
                nn.GELU(),
                nn.Conv1d(32, 48, 5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv1d(48, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            if variant != "long_only" else None
        )
        no_dilation = variant == "dual_no_dilation"
        long_pool: nn.Module = (
            nn.AdaptiveAvgPool1d(1)
            if variant == "dual_avg" else GlobalRMSPool1d()
        )
        self.long_period_branch = (
            nn.Sequential(
                nn.Conv1d(1, 16, 31, stride=2, padding=15),
                nn.GELU(),
                nn.Conv1d(
                    16,
                    32,
                    15,
                    stride=2,
                    padding=(7 if no_dilation else 14),
                    dilation=(1 if no_dilation else 2),
                ),
                nn.GELU(),
                nn.Conv1d(
                    32,
                    64,
                    7,
                    stride=2,
                    padding=(3 if no_dilation else 12),
                    dilation=(1 if no_dilation else 4),
                ),
                nn.GELU(),
                long_pool,
            )
            if variant != "local_only" else None
        )
        output_input_dim = (
            self.branch_dim
            if variant in {"local_only", "long_only"}
            else self.branch_dim * 2
        )
        self.output_norm = nn.Sequential(
            nn.Linear(output_input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, acceleration: torch.Tensor) -> torch.Tensor:
        if acceleration.ndim != 3 or acceleration.shape[1] != 1:
            raise ValueError(
                "DualScaleAccelerationEncoder expects [batch, 1, time], "
                f"got {tuple(acceleration.shape)}"
            )
        if self.variant == "local_only":
            assert self.local_branch is not None
            feature = self.local_branch(acceleration).flatten(1)
        elif self.variant == "long_only":
            assert self.long_period_branch is not None
            feature = self.long_period_branch(acceleration).flatten(1)
        else:
            assert self.local_branch is not None
            assert self.long_period_branch is not None
            local_feature = self.local_branch(acceleration).flatten(1)
            long_period_feature = self.long_period_branch(acceleration).flatten(1)
            feature = torch.cat((local_feature, long_period_feature), dim=1)
        return self.output_norm(feature)


class RoadConditionedTireClassifier(nn.Module):
    """Classify tire state from road probabilities and tire PT physics."""

    def __init__(self, config: TireClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("acc_mean", torch.zeros(config.acc_length))
        self.register_buffer("acc_std", torch.ones(config.acc_length))
        self.register_buffer(
            "gyro_mean", torch.zeros(config.gyro_channels, config.gyro_length)
        )
        self.register_buffer(
            "gyro_std", torch.ones(config.gyro_channels, config.gyro_length)
        )
        self.register_buffer(
            "zroad_mean", torch.zeros(config.road_embedding_dim)
        )
        self.register_buffer(
            "zroad_std", torch.ones(config.road_embedding_dim)
        )
        self.register_buffer("normalizer_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("pt_kappa_kpa_per_kelvin", torch.tensor(1.0))
        self.register_buffer("pt_residual_center_kpa", torch.tensor(0.0))
        self.register_buffer("pt_residual_scale_kpa", torch.tensor(1.0))
        self.register_buffer(
            "pt_feature_mean", torch.zeros(config.pressure_temperature_feature_dim)
        )
        self.register_buffer(
            "pt_feature_std", torch.ones(config.pressure_temperature_feature_dim)
        )
        self.register_buffer(
            "pt_calibration_fitted", torch.tensor(False, dtype=torch.bool)
        )

        self.acc_encoder = (
            DualScaleAccelerationEncoder(
                config.dropout, config.acceleration_encoder_variant
            )
            if config.use_dynamic_response else None
        )
        self.gyro_encoder = (
            nn.Sequential(
                nn.Conv1d(config.gyro_channels, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            if config.use_dynamic_response and config.use_gyro else None
        )
        self.fusion = (
            nn.Sequential(
                nn.Linear(
                    DualScaleAccelerationEncoder.output_dim
                    + (64 if config.use_gyro else 0),
                    256,
                ),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.response_dim),
                nn.LayerNorm(config.response_dim),
            )
            if config.use_dynamic_response else None
        )
        self.road_encoder = (
            nn.Sequential(
                nn.Linear(config.road_classes, config.road_feature_dim),
                nn.LayerNorm(config.road_feature_dim),
                nn.GELU(),
            )
            if config.use_road_probability
            else None
        )
        self.pressure_temperature_encoder = (
            nn.Sequential(
                nn.Linear(
                    config.pressure_temperature_feature_dim,
                    config.pressure_temperature_hidden_dim,
                ),
                nn.LayerNorm(config.pressure_temperature_hidden_dim),
                nn.GELU(),
            )
            if config.use_pressure_temperature else None
        )
        fusion_input = (
            (config.response_dim if config.use_dynamic_response else 0)
            + (
                config.pressure_temperature_hidden_dim
                if config.use_pressure_temperature else 0
            )
            + (
                config.road_feature_dim
                if config.use_road_probability
                else 0
            )
        )
        if config.use_zroad_concat:
            fusion_input += config.road_embedding_dim
        self.film_input_norm = nn.LayerNorm(fusion_input)
        self.film_generator = (
            nn.Sequential(
                nn.Linear(config.road_embedding_dim, config.film_hidden_dim),
                nn.LayerNorm(config.film_hidden_dim),
                nn.GELU(),
                nn.Linear(config.film_hidden_dim, fusion_input * 2),
            )
            if config.use_zroad_film else None
        )
        if self.film_generator is not None:
            nn.init.zeros_(self.film_generator[-1].weight)
            nn.init.zeros_(self.film_generator[-1].bias)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def _prepare_gyro(self, gyro: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if gyro.ndim == 2 and gyro.shape[1] == cfg.gyro_channels:
            gyro = gyro.unsqueeze(-1)
        elif gyro.ndim == 3 and gyro.shape[1] == cfg.gyro_channels:
            pass
        elif gyro.ndim == 3 and gyro.shape[2] == cfg.gyro_channels:
            gyro = gyro.transpose(1, 2)
        else:
            raise ValueError(
                f"gyro must contain {cfg.gyro_channels} channels, got {tuple(gyro.shape)}"
            )
        if gyro.shape[2] != cfg.gyro_length:
            raise ValueError(
                f"gyro length must be {cfg.gyro_length}, got {gyro.shape[2]}"
            )
        return gyro

    def _prepare_acceleration(self, tire_acc: torch.Tensor) -> torch.Tensor:
        """Deterministically center-crop longer archives for length ablations."""

        if tire_acc.ndim != 2:
            raise ValueError("tire_acc must be [batch,time]")
        target_length = self.config.acc_length
        source_length = tire_acc.shape[1]
        if source_length < target_length:
            raise ValueError(
                f"tire_acc length must be at least {target_length}, got {source_length}"
            )
        if source_length == target_length:
            return tire_acc
        start = (source_length - target_length) // 2
        return tire_acc[:, start:start + target_length]

    @torch.no_grad()
    def fit_normalizer(
        self,
        tire_acc: torch.Tensor,
        gyro: torch.Tensor,
        pressure_temperature_state: torch.Tensor,
        zroad: torch.Tensor,
        anomaly_class: torch.Tensor,
    ) -> None:
        tire_acc = self._prepare_acceleration(tire_acc)
        gyro = self._prepare_gyro(gyro)
        cfg = self.config
        if tire_acc.shape[1] != cfg.acc_length:
            raise AssertionError("Acceleration preparation returned the wrong length")
        if (
            pressure_temperature_state.ndim != 2
            or pressure_temperature_state.shape[1] != 2
        ):
            raise ValueError("pressure_temperature_state must be [N,2]")
        if zroad.ndim != 2 or zroad.shape[1] != cfg.road_embedding_dim:
            raise ValueError(f"zroad must be [N,{cfg.road_embedding_dim}]")
        if not (
            tire_acc.shape[0]
            == gyro.shape[0]
            == pressure_temperature_state.shape[0]
            == zroad.shape[0]
            and tire_acc.shape[0] >= 2
        ):
            raise ValueError("Normalizer requires matched training samples")
        if anomaly_class.ndim != 1 or anomaly_class.shape[0] != tire_acc.shape[0]:
            raise ValueError("anomaly_class must be [N] and aligned with training inputs")
        if not (
            torch.isfinite(tire_acc).all()
            and torch.isfinite(gyro).all()
            and torch.isfinite(pressure_temperature_state).all()
            and torch.isfinite(zroad).all()
        ):
            raise ValueError("Training inputs contain NaN or infinite values")
        self.acc_mean.copy_(tire_acc.mean(0))
        self.acc_std.copy_(tire_acc.std(0, unbiased=False).clamp_min(1e-5))
        self.gyro_mean.copy_(gyro.mean(0))
        self.gyro_std.copy_(gyro.std(0, unbiased=False).clamp_min(1e-5))
        self.zroad_mean.copy_(zroad.mean(0))
        self.zroad_std.copy_(zroad.std(0, unbiased=False).clamp_min(1e-5))

        healthy_state = pressure_temperature_state[anomaly_class == 0].to(
            dtype=torch.float64
        )
        if healthy_state.shape[0] < 2:
            raise ValueError(
                "Pressure-temperature calibration requires at least two healthy "
                "training rows"
            )
        if cfg.pressure_temperature_mode in {
            "coupled",
            "corrected_pressure",
            "residual",
        }:
            pressure_absolute, temperature_kelvin = self._pressure_temperature_values(
                healthy_state
            )
            kappa = torch.quantile(pressure_absolute / temperature_kelvin, 0.5)
            residual = pressure_absolute - kappa * temperature_kelvin
            residual_center = torch.quantile(residual, 0.5)
            mad = torch.quantile(torch.abs(residual - residual_center), 0.5)
            residual_scale = 1.4826 * mad
            minimum_scale = float(self.config.minimum_residual_scale_kpa)
            if float(residual_scale) < minimum_scale:
                residual_scale = residual.std(unbiased=False)
            residual_scale = residual_scale.clamp_min(minimum_scale)
            self.pt_kappa_kpa_per_kelvin.copy_(kappa.to(dtype=torch.float32))
            self.pt_residual_center_kpa.copy_(residual_center.to(dtype=torch.float32))
            self.pt_residual_scale_kpa.copy_(residual_scale.to(dtype=torch.float32))

        healthy_raw_features = self._pressure_temperature_raw_features(
            healthy_state.to(dtype=pressure_temperature_state.dtype)
        )
        self.pt_feature_mean.copy_(healthy_raw_features.mean(0))
        self.pt_feature_std.copy_(
            healthy_raw_features.std(0, unbiased=False).clamp_min(1e-5)
        )
        self.normalizer_fitted.fill_(True)
        self.pt_calibration_fitted.fill_(True)

    def _pressure_temperature_values(
        self, pressure_temperature_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            pressure_temperature_state.ndim != 2
            or pressure_temperature_state.shape[1] != 2
        ):
            raise ValueError("pressure_temperature_state must be [N,2]")
        cfg = self.config
        pressure_absolute = (
            pressure_temperature_state[:, 0] + cfg.atmospheric_pressure_kpa
        )
        temperature_kelvin = pressure_temperature_state[:, 1] + 273.15
        if not (
            torch.isfinite(pressure_absolute).all()
            and torch.isfinite(temperature_kelvin).all()
        ):
            raise ValueError("Pressure-temperature inputs contain non-finite values")
        if bool((pressure_absolute <= 0).any()) or bool(
            (temperature_kelvin <= 0).any()
        ):
            raise ValueError(
                "Absolute pressure and Kelvin temperature must be positive"
            )
        return pressure_absolute, temperature_kelvin

    def _pressure_temperature_raw_features(
        self, pressure_temperature_state: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        if cfg.pressure_temperature_mode == "raw":
            return pressure_temperature_state
        if cfg.pressure_temperature_mode == "pressure_only":
            return pressure_temperature_state[:, 0:1]
        if cfg.pressure_temperature_mode == "temperature_only":
            return pressure_temperature_state[:, 1:2]
        pressure_absolute, temperature_kelvin = self._pressure_temperature_values(
            pressure_temperature_state
        )
        expected_absolute = self.pt_kappa_kpa_per_kelvin * temperature_kelvin
        residual = pressure_absolute - expected_absolute
        residual_z = (
            residual - self.pt_residual_center_kpa
        ) / self.pt_residual_scale_kpa
        reference_temperature_kelvin = cfg.reference_temperature_c + 273.15
        corrected_pressure = (
            pressure_absolute
            * reference_temperature_kelvin
            / temperature_kelvin
            - cfg.atmospheric_pressure_kpa
        )
        relative_residual = pressure_absolute / expected_absolute - 1.0
        if cfg.pressure_temperature_mode == "corrected_pressure":
            return corrected_pressure.unsqueeze(1)
        if cfg.pressure_temperature_mode == "residual":
            return torch.stack((residual_z, relative_residual), dim=1)
        return torch.stack(
            (
                corrected_pressure,
                expected_absolute - cfg.atmospheric_pressure_kpa,
                residual_z,
                relative_residual,
            ),
            dim=1,
        )

    def pressure_temperature_calibration(self) -> dict:
        """Return the healthy-training-only isochoric calibration."""

        mode = self.config.pressure_temperature_mode
        uses_physics_relation = mode in {
            "coupled",
            "corrected_pressure",
            "residual",
        }
        feature_names = {
            "coupled": [
                "temperature_corrected_pressure_kpa",
                "expected_pressure_kpa",
                "pressure_temperature_residual_z",
                "relative_pressure_temperature_residual",
            ],
            "raw": ["pressure_kpa", "temperature_c"],
            "pressure_only": ["pressure_kpa"],
            "temperature_only": ["temperature_c"],
            "corrected_pressure": ["temperature_corrected_pressure_kpa"],
            "residual": [
                "pressure_temperature_residual_z",
                "relative_pressure_temperature_residual",
            ],
        }[mode]
        relation = (
            "P_abs = kappa * T_K"
            if uses_physics_relation
            else "healthy-training standardization without P/T coupling"
        )

        return {
            "enabled": bool(self.pt_calibration_fitted.item()),
            "used_by_classifier": self.config.use_pressure_temperature,
            "mode": mode,
            "relation": relation,
            "fit_source": "healthy training partition only",
            "pressure_state_index": self.config.pressure_state_index,
            "temperature_state_index": self.config.temperature_state_index,
            "atmospheric_pressure_kpa": self.config.atmospheric_pressure_kpa,
            "reference_temperature_c": self.config.reference_temperature_c,
            "minimum_residual_scale_kpa": self.config.minimum_residual_scale_kpa,
            "kappa_kpa_per_kelvin": (
                float(self.pt_kappa_kpa_per_kelvin.item())
                if uses_physics_relation else None
            ),
            "residual_center_kpa": (
                float(self.pt_residual_center_kpa.item())
                if uses_physics_relation else None
            ),
            "residual_scale_kpa": (
                float(self.pt_residual_scale_kpa.item())
                if uses_physics_relation else None
            ),
            "feature_names": feature_names,
            "feature_mean": self.pt_feature_mean.detach().cpu().tolist(),
            "feature_std": self.pt_feature_std.detach().cpu().tolist(),
        }

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict:
        """Load matching legacy sensor features while reinitializing the new road head."""

        own = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own and own[key].shape == value.shape
        }
        result = self.load_state_dict(compatible, strict=False)
        return {
            "loaded": sorted(compatible),
            "missing": sorted(result.missing_keys),
            "ignored": sorted(set(state_dict) - set(compatible)),
        }

    def extract_tire_features(
        self,
        tire_acc: torch.Tensor,
        gyro: torch.Tensor,
        pressure_temperature_state: torch.Tensor,
    ) -> TireStateOutput:
        """Extract tire state from acceleration, IMU and pressure/temperature.

        This boundary has no road inputs. ``tire_feature`` concatenates the
        temporal response and P-T embedding in that order (256 + 16 by default).
        The classifier's road-conditioned fusion is a separate downstream step.
        """
        if not bool(self.normalizer_fitted.item()):
            raise RuntimeError("Input normalizer has not been fitted")
        if not bool(self.pt_calibration_fitted.item()):
            raise RuntimeError("Pressure-temperature calibration has not been fitted")
        tire_acc = self._prepare_acceleration(tire_acc)
        gyro = self._prepare_gyro(gyro)
        if (
            pressure_temperature_state.ndim != 2
            or pressure_temperature_state.shape[1] != 2
        ):
            raise ValueError("pressure_temperature_state must be [B,2]")
        if not (
            tire_acc.shape[0]
            == gyro.shape[0]
            == pressure_temperature_state.shape[0]
        ):
            raise ValueError("All tire extractor inputs must have the same batch size")

        acc = (tire_acc - self.acc_mean) / self.acc_std
        gyro = (gyro - self.gyro_mean) / self.gyro_std
        empty = acc.new_empty((acc.shape[0], 0))
        acc_feature, imu_feature = empty, empty
        if self.acc_encoder is not None:
            acc_feature = self.acc_encoder(acc.unsqueeze(1))
            response_parts = [acc_feature]
            if self.gyro_encoder is not None:
                imu_feature = self.gyro_encoder(gyro).flatten(1)
                response_parts.append(imu_feature)
            assert self.fusion is not None
            response = self.fusion(torch.cat(response_parts, dim=1))
        else:
            response = empty
        if self.pressure_temperature_encoder is not None:
            pressure_temperature_features = self._pressure_temperature_raw_features(
                pressure_temperature_state
            )
            normalized_pressure_temperature_features = (
                pressure_temperature_features - self.pt_feature_mean
            ) / self.pt_feature_std
            pressure_temperature_feature = self.pressure_temperature_encoder(
                normalized_pressure_temperature_features
            )
        else:
            pressure_temperature_features = pressure_temperature_state.new_empty(
                (pressure_temperature_state.shape[0], 0)
            )
            pressure_temperature_feature = pressure_temperature_features
        return TireStateOutput(
            acceleration_feature=acc_feature,
            imu_feature=imu_feature,
            response_feature=response,
            pressure_temperature_features=pressure_temperature_features,
            pressure_temperature_feature=pressure_temperature_feature,
            tire_feature=torch.cat((response, pressure_temperature_feature), dim=1),
        )

    def forward(
        self,
        tire_acc: torch.Tensor,
        gyro: torch.Tensor,
        pressure_temperature_state: torch.Tensor,
        road_probabilities: torch.Tensor,
        zroad: torch.Tensor,
    ) -> TireClassifierOutput:
        cfg = self.config
        state = self.extract_tire_features(tire_acc, gyro, pressure_temperature_state)
        if road_probabilities.ndim != 2 or road_probabilities.shape[1] != cfg.road_classes:
            raise ValueError(f"road_probabilities must be [B,{cfg.road_classes}]")
        if zroad.ndim != 2 or zroad.shape[1] != cfg.road_embedding_dim:
            raise ValueError(f"zroad must be [B,{cfg.road_embedding_dim}]")
        if not (tire_acc.shape[0] == road_probabilities.shape[0] == zroad.shape[0]):
            raise ValueError("All classifier inputs must have the same batch size")
        zroad = (zroad - self.zroad_mean) / self.zroad_std
        road = road_probabilities.clamp_min(0.0)
        road = road / road.sum(dim=1, keepdim=True).clamp_min(1e-8)
        road_feature = (
            self.road_encoder(road)
            if self.road_encoder is not None
            else road.new_empty((road.shape[0], 0))
        )
        response = state.response_feature
        pressure_temperature_features = state.pressure_temperature_features
        pressure_temperature_feature = state.pressure_temperature_feature
        concatenated_parts = []
        if self.config.use_dynamic_response:
            concatenated_parts.append(response)
        if self.config.use_road_probability:
            concatenated_parts.append(road_feature)
        if self.config.use_pressure_temperature:
            concatenated_parts.append(pressure_temperature_feature)
        if self.config.use_zroad_concat:
            concatenated_parts.append(zroad)
        concatenated_feature = torch.cat(concatenated_parts, dim=1)
        normalized_feature = self.film_input_norm(concatenated_feature)
        if self.film_generator is not None:
            gamma_delta, beta = self.film_generator(zroad).chunk(2, dim=1)
            modulated_feature = normalized_feature * (1.0 + gamma_delta) + beta
        else:
            modulated_feature = normalized_feature
        logits = self.classifier(modulated_feature)
        return TireClassifierOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=1),
            response_feature=response,
            road_feature=road_feature,
            pressure_temperature_features=pressure_temperature_features,
            pressure_temperature_feature=pressure_temperature_feature,
            modulated_feature=modulated_feature,
        )
