"""RTI-MFNet road encoder migrated from ``mobilenetv4_simam_gate``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from .mobilenetv4_simam import build_mobilenetv4_image_branch


# Keep the source training label order. A checkpoint cannot encode class names.
SOURCE_ROAD_CLASSES: Tuple[str, ...] = (
    "积水路面",
    "破损路面",
    "水泥路面",
    "沥青路面",
)


@dataclass(frozen=True)
class RTIMFNetConfig:
    num_classes: int = 4
    class_names: Tuple[str, ...] = SOURCE_ROAD_CLASSES
    image_model_name: str = "mobilenetv4_conv_medium.e500_r224_in1k"
    image_exp_name: str = "E2"
    image_size: int = 224
    image_output_dim: Optional[int] = None
    image_fusion_dim: int = 256
    insert_stages: Tuple[int, ...] = (2, 3, 4)
    simam_lambda: float = 1e-4
    pretrained: bool = False
    freeze_backbone: bool = True
    train_last_stage: bool = True

    scalar_dim: int = 3
    imu_channels: int = 9
    imu_hidden: int = 32
    # If a 3-channel gyro is supplied to a legacy 9-channel model, scatter
    # gx/gy/gz into these source-channel positions. Set None for first-C fill.
    imu_channel_indices: Optional[Tuple[int, ...]] = (3, 4, 5)

    dropout: float = 0.3
    gate_hidden: int = 32
    gate_mode: str = "softmax"
    gate_temperature: float = 2.0

    enable_dynamic_image_skip: bool = False
    image_skip_threshold: float = 0.10
    renorm_after_skip: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.num_classes != len(self.class_names):
            raise ValueError("num_classes must match class_names length")
        if self.gate_mode not in {"softmax", "sigmoid"}:
            raise ValueError("gate_mode must be softmax or sigmoid")
        if self.imu_channels <= 0 or self.scalar_dim <= 0:
            raise ValueError("imu_channels and scalar_dim must be positive")
        if self.image_fusion_dim <= 0:
            raise ValueError("image_fusion_dim must be positive")


class RoadEncoderOutput(NamedTuple):
    zroad: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    gates: torch.Tensor
    image_used_mask: torch.Tensor


class RTIMFNetRoadEncoder(nn.Module):
    """Multimodal road encoder with source-checkpoint-compatible names.

    ``forward`` returns logits for backward compatibility. New code should use
    :meth:`encode_road` to obtain ``zroad``, ``Proad`` and gate diagnostics.
    """

    def __init__(
        self,
        config: Optional[RTIMFNetConfig] = None,
        *,
        image_backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config or RTIMFNetConfig()
        cfg = self.config

        self.img_size = cfg.image_size
        self.freeze_backbone_flag = cfg.freeze_backbone
        self.train_last_stage = cfg.train_last_stage
        self.gate_mode = cfg.gate_mode
        self.gate_temperature = float(cfg.gate_temperature)
        self.enable_dynamic_image_skip = cfg.enable_dynamic_image_skip
        self.image_skip_threshold = float(cfg.image_skip_threshold)
        self.renorm_after_skip = cfg.renorm_after_skip

        if image_backbone is None:
            image_backbone = build_mobilenetv4_image_branch(
                exp_name=cfg.image_exp_name,
                model_name=cfg.image_model_name,
                pretrained=cfg.pretrained,
                output_dim=cfg.image_output_dim,
                dropout=cfg.dropout,
                freeze_backbone=False,
                insert_stages=cfg.insert_stages,
                simam_lambda=cfg.simam_lambda,
                img_size=cfg.image_size,
                verbose=cfg.verbose,
            )
        self.image_backbone = image_backbone
        self.image_raw_dim = self._infer_image_dim(cfg.image_size)
        self.image_adapter = nn.Sequential(
            nn.Linear(self.image_raw_dim, cfg.image_fusion_dim),
            nn.LayerNorm(cfg.image_fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.fusion_dim = cfg.image_fusion_dim
        self.image_dim = cfg.image_fusion_dim
        self._configure_image_trainability()

        # Source-compatible tire radial acceleration branch.
        self.seq_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.seq_adapter = nn.Sequential(
            nn.Linear(32, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Scalars control gates and are not directly added as a fourth modality.
        self.scalar_extractor = nn.Sequential(
            nn.Linear(cfg.scalar_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )
        self.scalar_gate_generator = nn.Sequential(
            nn.Linear(16, cfg.gate_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gate_hidden, 3),
        )

        self.imu_sequence_branch = nn.Sequential(
            nn.Conv1d(cfg.imu_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, cfg.imu_hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cfg.imu_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
        )
        self.imu_vector_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, cfg.imu_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.imu_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
        )
        self.imu_adapter = nn.Sequential(
            nn.Linear(cfg.imu_hidden, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.num_classes),
        )

        self.last_gates: Optional[torch.Tensor] = None
        self.last_gates_used: Optional[torch.Tensor] = None
        self.last_image_use_mask: Optional[torch.Tensor] = None

    def _infer_image_dim(self, img_size: int) -> int:
        out_dim = getattr(self.image_backbone, "out_dim", None)
        if out_dim is not None:
            return int(out_dim)

        was_training = self.image_backbone.training
        self.image_backbone.eval()
        try:
            device = next(self.image_backbone.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            feature = self.image_backbone(
                torch.zeros(2, 3, img_size, img_size, device=device)
            )
        if was_training:
            self.image_backbone.train()
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        return int(feature.flatten(1).shape[1])

    def _configure_image_trainability(self) -> None:
        if not self.config.freeze_backbone:
            self.image_no_grad = False
            return

        for parameter in self.image_backbone.parameters():
            parameter.requires_grad = False
        if self.config.train_last_stage:
            trainable_tokens = (
                "stages.4",
                "stages.5",
                "blocks.4",
                "blocks.5",
                "norm",
                "head",
                "classifier",
                "fc",
            )
            for name, parameter in self.image_backbone.named_parameters():
                if any(token in name.lower() for token in trainable_tokens):
                    parameter.requires_grad = True
        self.image_no_grad = not self.config.train_last_stage

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        if self.image_no_grad:
            with torch.no_grad():
                feature = self.image_backbone(image)
        else:
            feature = self.image_backbone(image)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        if feature.ndim > 2:
            feature = feature.flatten(1)
        return self.image_adapter(feature)

    def encode_acc_seq(self, tire_acc: torch.Tensor) -> torch.Tensor:
        if tire_acc.ndim == 2:
            tire_acc = tire_acc.unsqueeze(1)
        elif tire_acc.ndim != 3 or tire_acc.shape[1] != 1:
            raise ValueError(
                f"tire_acc must be [B,L] or [B,1,L], got {tuple(tire_acc.shape)}"
            )
        feature = self.seq_extractor(tire_acc).flatten(1)
        return self.seq_adapter(feature)

    def encode_scalar(self, vehicle_state: torch.Tensor) -> torch.Tensor:
        if vehicle_state.ndim != 2 or vehicle_state.shape[1] != self.config.scalar_dim:
            raise ValueError(
                "vehicle_state must be [B,scalar_dim], got "
                f"{tuple(vehicle_state.shape)} for scalar_dim={self.config.scalar_dim}"
            )
        return self.scalar_extractor(vehicle_state)

    def _sequence_to_channels_first(self, gyro: torch.Tensor) -> torch.Tensor:
        if gyro.ndim != 3:
            raise ValueError(f"gyro sequence must be 3D, got {tuple(gyro.shape)}")

        channel_indices = self.config.imu_channel_indices
        mapped_channels = len(channel_indices) if channel_indices is not None else None

        # Resolve exact model/mapped channel dimensions before using a size
        # heuristic. This keeps both [B,9,1] and [B,1,9] unambiguous.
        if gyro.shape[1] == self.config.imu_channels:
            channels_first = gyro
        elif gyro.shape[-1] == self.config.imu_channels:
            channels_first = gyro.permute(0, 2, 1)
        elif mapped_channels is not None and gyro.shape[1] == mapped_channels:
            channels_first = gyro
        elif mapped_channels is not None and gyro.shape[-1] == mapped_channels:
            channels_first = gyro.permute(0, 2, 1)
        elif gyro.shape[1] < gyro.shape[-1] and gyro.shape[1] <= self.config.imu_channels:
            channels_first = gyro
        elif gyro.shape[-1] <= self.config.imu_channels:
            channels_first = gyro.permute(0, 2, 1)
        else:
            raise ValueError(
                "Cannot infer gyro layout; use [B,L,C] or [B,C,L] with "
                f"C<={self.config.imu_channels}, got {tuple(gyro.shape)}"
            )

        input_channels = int(channels_first.shape[1])
        if input_channels == self.config.imu_channels:
            return channels_first
        if input_channels > self.config.imu_channels:
            raise ValueError(
                f"gyro has {input_channels} channels, model expects {self.config.imu_channels}"
            )

        padded = channels_first.new_zeros(
            channels_first.shape[0],
            self.config.imu_channels,
            channels_first.shape[2],
        )
        indices = self.config.imu_channel_indices
        if indices is None:
            padded[:, :input_channels] = channels_first
        else:
            if len(indices) != input_channels:
                raise ValueError(
                    "imu_channel_indices length must equal supplied gyro channels: "
                    f"{len(indices)} != {input_channels}"
                )
            if min(indices) < 0 or max(indices) >= self.config.imu_channels:
                raise ValueError("imu_channel_indices contains an out-of-range index")
            padded[:, list(indices)] = channels_first
        return padded

    def encode_imu(self, gyro: torch.Tensor) -> torch.Tensor:
        if gyro.ndim == 2:
            feature = self.imu_vector_branch(gyro.unsqueeze(1))
        elif gyro.ndim == 3:
            feature = self.imu_sequence_branch(
                self._sequence_to_channels_first(gyro)
            )
        else:
            raise ValueError(
                f"gyro must be [B,D], [B,L,C], or [B,C,L], got {tuple(gyro.shape)}"
            )
        return self.imu_adapter(feature)

    def compute_scalar_gates(self, scalar_feature: torch.Tensor) -> torch.Tensor:
        logits = self.scalar_gate_generator(scalar_feature)
        if self.gate_mode == "softmax":
            temperature = max(self.gate_temperature, 1e-6)
            return torch.softmax(logits / temperature, dim=1)
        return torch.sigmoid(logits)

    def fuse_by_gates(
        self,
        image_feature: torch.Tensor,
        tire_feature: torch.Tensor,
        gyro_feature: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        fused = (
            gates[:, 0:1] * image_feature
            + gates[:, 1:2] * tire_feature
            + gates[:, 2:3] * gyro_feature
        )
        fused = self.fusion_norm(fused)
        return fused + self.fusion_ffn(fused)

    def _encode_full(
        self,
        image: torch.Tensor,
        tire_acc: torch.Tensor,
        vehicle_state: torch.Tensor,
        gyro: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scalar_feature = self.encode_scalar(vehicle_state)
        gates = self.compute_scalar_gates(scalar_feature)
        zroad = self.fuse_by_gates(
            self.encode_image(image),
            self.encode_acc_seq(tire_acc),
            self.encode_imu(gyro),
            gates,
        )
        image_mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
        self.last_gates = gates.detach()
        self.last_gates_used = gates.detach()
        self.last_image_use_mask = image_mask.detach()
        return zroad, gates, image_mask

    def _encode_dynamic(
        self,
        image: torch.Tensor,
        tire_acc: torch.Tensor,
        vehicle_state: torch.Tensor,
        gyro: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gates = self.compute_scalar_gates(self.encode_scalar(vehicle_state))
        tire_feature = self.encode_acc_seq(tire_acc)
        gyro_feature = self.encode_imu(gyro)
        image_mask = gates[:, 0] >= self.image_skip_threshold
        image_feature = tire_feature.new_zeros(image.shape[0], self.fusion_dim)
        if image_mask.any():
            image_feature[image_mask] = self.encode_image(image[image_mask])

        used_gates = gates.clone()
        used_gates[~image_mask, 0] = 0.0
        if self.renorm_after_skip:
            used_gates = used_gates / used_gates.sum(dim=1, keepdim=True).clamp_min(1e-6)

        zroad = self.fuse_by_gates(
            image_feature,
            tire_feature,
            gyro_feature,
            used_gates,
        )
        self.last_gates = gates.detach()
        self.last_gates_used = used_gates.detach()
        self.last_image_use_mask = image_mask.detach()
        return zroad, used_gates, image_mask

    def encode_road(
        self,
        image: torch.Tensor,
        tire_acc: torch.Tensor,
        vehicle_state: torch.Tensor,
        gyro: torch.Tensor,
        *,
        dynamic_image_skip: Optional[bool] = None,
    ) -> RoadEncoderOutput:
        if not (
            image.shape[0]
            == tire_acc.shape[0]
            == vehicle_state.shape[0]
            == gyro.shape[0]
        ):
            raise ValueError("All modalities must have the same batch size")

        use_dynamic = (
            (not self.training and self.enable_dynamic_image_skip)
            if dynamic_image_skip is None
            else bool(dynamic_image_skip)
        )
        if use_dynamic:
            zroad, gates, image_mask = self._encode_dynamic(
                image, tire_acc, vehicle_state, gyro
            )
        else:
            zroad, gates, image_mask = self._encode_full(
                image, tire_acc, vehicle_state, gyro
            )
        logits = self.classifier(zroad)
        return RoadEncoderOutput(
            zroad=zroad,
            logits=logits,
            probabilities=torch.softmax(logits, dim=1),
            gates=gates,
            image_used_mask=image_mask,
        )

    def forward(
        self,
        image: torch.Tensor,
        acc_seq: torch.Tensor,
        scalars: torch.Tensor,
        imu_seq: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_road(image, acc_seq, scalars, imu_seq).logits

    def get_last_gates(self) -> Optional[dict[str, float]]:
        if self.last_gates is None:
            return None
        gates = self.last_gates.detach().cpu()
        return {
            "image_gate": gates[:, 0].mean().item(),
            "tire_acc_gate": gates[:, 1].mean().item(),
            "gyro_gate": gates[:, 2].mean().item(),
        }


class RawDataAttentionMobileNetV4IMU_gate(RTIMFNetRoadEncoder):
    """Source-constructor-compatible wrapper around :class:`RTIMFNetRoadEncoder`."""

    def __init__(
        self,
        num_classes: int = 4,
        seq_len: int = 256,
        imu_channels: int = 9,
        imu_hidden: int = 32,
        dropout_rate: float = 0.3,
        scalar_dim: int = 3,
        pretrained: bool = False,
        freeze_backbone: bool = True,
        train_last_stage: bool = False,
        use_residual_attention: bool = True,
        init_alpha: float = -4.0,
        img_size: int = 224,
        image_output_dim: Optional[int] = 256,
        image_fusion_dim: int = 512,
        image_exp_name: str = "E2",
        image_model_name: str = "mobilenetv4_conv_medium.e500_r224_in1k",
        insert_stages: Tuple[int, ...] = (2, 3, 4),
        simam_lambda: float = 1e-4,
        verbose: bool = False,
        gate_hidden: int = 32,
        gate_mode: str = "softmax",
        gate_temperature: float = 1.0,
        enable_dynamic_image_skip: bool = False,
        image_skip_threshold: float = 0.10,
        renorm_after_skip: bool = True,
        imu_channel_indices: Optional[Tuple[int, ...]] = (3, 4, 5),
    ) -> None:
        # seq_len, use_residual_attention and init_alpha are retained only for
        # source-call compatibility. Adaptive pooling makes seq_len unnecessary.
        del seq_len, use_residual_attention, init_alpha
        class_names = (
            SOURCE_ROAD_CLASSES
            if num_classes == len(SOURCE_ROAD_CLASSES)
            else tuple(str(index) for index in range(num_classes))
        )
        super().__init__(
            RTIMFNetConfig(
                num_classes=num_classes,
                class_names=class_names,
                image_model_name=image_model_name,
                image_exp_name=image_exp_name,
                image_size=img_size,
                image_output_dim=image_output_dim,
                image_fusion_dim=image_fusion_dim,
                insert_stages=insert_stages,
                simam_lambda=simam_lambda,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                train_last_stage=train_last_stage,
                scalar_dim=scalar_dim,
                imu_channels=imu_channels,
                imu_hidden=imu_hidden,
                imu_channel_indices=imu_channel_indices,
                dropout=dropout_rate,
                gate_hidden=gate_hidden,
                gate_mode=gate_mode,
                gate_temperature=gate_temperature,
                enable_dynamic_image_skip=enable_dynamic_image_skip,
                image_skip_threshold=image_skip_threshold,
                renorm_after_skip=renorm_after_skip,
                verbose=verbose,
            )
        )
