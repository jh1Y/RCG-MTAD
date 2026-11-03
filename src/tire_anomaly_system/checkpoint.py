"""Checkpoint inspection and legacy RTI-MFNet weight loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: Path
    loaded_keys: int
    model_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def loaded_fraction(self) -> float:
        return self.loaded_keys / max(self.model_keys, 1)


def _torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise TypeError("Checkpoint does not contain a recognizable tensor state_dict")


def _strip_common_prefixes(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    prefixes = ("module.", "_orig_mod.", "model.")
    changed = True
    while changed and cleaned:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in cleaned):
                cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
                changed = True
    return cleaned


def inspect_legacy_architecture(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Infer shape-bearing constructor settings before building the model."""

    path = Path(checkpoint_path)
    state = _strip_common_prefixes(
        extract_state_dict(_torch_load(path, map_location))
    )

    inferred: dict[str, Any] = {}
    probes = {
        "num_classes": ("classifier.4.weight", 0),
        "image_fusion_dim": ("fusion_norm.weight", 0),
        "scalar_dim": ("scalar_extractor.0.weight", 1),
        "gate_hidden": ("scalar_gate_generator.0.weight", 0),
        "imu_channels": ("imu_sequence_branch.0.weight", 1),
        "imu_hidden": ("imu_sequence_branch.3.weight", 0),
    }
    for name, (key, axis) in probes.items():
        tensor = state.get(key)
        if tensor is not None:
            inferred[name] = int(tensor.shape[axis])

    projection = state.get("image_backbone.proj.0.weight")
    inferred["image_output_dim"] = (
        int(projection.shape[0]) if projection is not None else None
    )
    inferred["state_dict_keys"] = len(state)
    return inferred


def load_legacy_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadReport:
    path = Path(checkpoint_path)
    state = _strip_common_prefixes(
        extract_state_dict(_torch_load(path, map_location))
    )
    model_state = model.state_dict()

    unexpected = tuple(sorted(key for key in state if key not in model_state))
    shape_mismatches = tuple(
        sorted(
            f"{key}: checkpoint{tuple(value.shape)} != model{tuple(model_state[key].shape)}"
            for key, value in state.items()
            if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
        )
    )
    loadable = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing = tuple(sorted(key for key in model_state if key not in loadable))

    if strict and (missing or unexpected or shape_mismatches):
        details = [
            f"missing={len(missing)}",
            f"unexpected={len(unexpected)}",
            f"shape_mismatches={len(shape_mismatches)}",
        ]
        mismatch_preview = "; ".join(shape_mismatches[:5])
        if mismatch_preview:
            details.append(mismatch_preview)
        raise RuntimeError("Legacy checkpoint is not architecture-compatible: " + ", ".join(details))

    model.load_state_dict(loadable, strict=False)
    return CheckpointLoadReport(
        path=path,
        loaded_keys=len(loadable),
        model_keys=len(model_state),
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=shape_mismatches,
    )

