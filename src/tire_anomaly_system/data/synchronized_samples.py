"""Load synchronized road samples from an archive or a CSV manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a mapping")
    return payload


def _load_archive(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    if suffix in {".pt", ".pth"}:
        return dict(_safe_torch_load(path))
    raise ValueError("Synchronized archive must be .npz, .pt, or .pth")


def _load_array_file(path: Path, key: Optional[str]) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if key:
                if key not in payload.files:
                    raise KeyError(f"{path} does not contain key {key!r}")
                return payload[key]
            if len(payload.files) != 1:
                raise KeyError(
                    f"{path} contains multiple arrays; specify the matching key"
                )
            return payload[payload.files[0]]
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping):
            if key:
                if key not in payload:
                    raise KeyError(f"{path} does not contain key {key!r}")
                return payload[key]
            tensors = [value for value in payload.values() if torch.is_tensor(value)]
            if len(tensors) != 1:
                raise KeyError(
                    f"{path} contains multiple values; specify the matching key"
                )
            return tensors[0]
        return payload
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        return np.loadtxt(path, delimiter=delimiter)
    raise ValueError(f"Unsupported sensor array file: {path}")


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _image_tensor(value: Any, root: Path, image_size: int) -> torch.Tensor:
    if isinstance(value, (str, np.str_)):
        path = _resolve_path(str(value), root)
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    else:
        tensor = torch.as_tensor(np.asarray(value)).detach().to(torch.float32)
        if tensor.ndim != 3:
            raise ValueError(f"image sample must be 3D, got {tuple(tensor.shape)}")
        if tensor.shape[0] == 3:
            pass
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"image sample must have three channels, got {tuple(tensor.shape)}")
        if tensor.max().item() > 1.5:
            tensor = tensor / 255.0
        if tuple(tensor.shape[-2:]) != (image_size, image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    return (tensor - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def _sensor_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32).contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError("sensor sample contains NaN or infinite values")
    return tensor


def _class_label(value: Any, name: str) -> int:
    if value is None or str(value).strip() == "":
        return -1
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer label") from error
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer label")
    return int(number)


class SynchronizedRoadDataset(Dataset):
    """Synchronized image, tire acceleration, gyro and vehicle-state samples.

    Archive input requires ``image`` or ``image_path``, ``tire_acc``, ``gyro``
    and ``vehicle_state``. CSV input uses one row per sample and defaults to
    ``image_path``, ``tire_acc_path``, ``gyro_path``, ``tire_pressure``,
    ``tire_temperature`` and ``speed`` columns.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        data_root: str | Path | None = None,
        image_size: int = 224,
        image_column: str = "image_path",
        tire_acc_column: str = "tire_acc_path",
        gyro_column: str = "gyro_path",
        state_columns: Sequence[str] = (
            "tire_pressure",
            "tire_temperature",
            "speed",
        ),
        tire_acc_key: Optional[str] = None,
        gyro_key: Optional[str] = None,
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        self.root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else self.source.parent
        )
        self.image_size = int(image_size)
        self.image_column = image_column
        self.tire_acc_column = tire_acc_column
        self.gyro_column = gyro_column
        self.state_columns = tuple(state_columns)
        self.tire_acc_key = tire_acc_key
        self.gyro_key = gyro_key

        if len(self.state_columns) == 0:
            raise ValueError("state_columns cannot be empty")
        if self.source.suffix.lower() == ".csv":
            with self.source.open("r", encoding="utf-8-sig", newline="") as file:
                self.rows = list(csv.DictReader(file))
            if not self.rows:
                raise ValueError(f"CSV manifest is empty: {self.source}")
            self.archive: Optional[dict[str, Any]] = None
            required = {
                self.image_column,
                self.tire_acc_column,
                self.gyro_column,
                *self.state_columns,
            }
            missing = sorted(required - set(self.rows[0]))
            if missing:
                raise KeyError(f"CSV manifest missing columns: {missing}")
            self.length = len(self.rows)
        else:
            self.archive = _load_archive(self.source)
            self.rows = []
            image_key = "image" if "image" in self.archive else "image_path"
            required = {image_key, "tire_acc", "gyro", "vehicle_state"}
            missing = sorted(required - set(self.archive))
            if missing:
                raise KeyError(f"Synchronized archive missing keys: {missing}")
            self.image_key = image_key
            self.length = len(self.archive[image_key])
            for key in ("tire_acc", "gyro", "vehicle_state"):
                if len(self.archive[key]) != self.length:
                    raise ValueError(f"{key} length does not match image count")

    def __len__(self) -> int:
        return self.length

    def _metadata(self, values: Mapping[str, Any], index: int) -> dict[str, Any]:
        metadata = {
            "road_label": _class_label(values.get("road_label", -1), "road_label"),
            "anomaly_class": _class_label(values.get("anomaly_class", -1), "anomaly_class"),
            "split": str(values.get("split", "")),
            "group_id": str(values.get("group_id", "")),
            "sample_id": str(values.get("sample_id", index)),
        }
        for key in (
            "source_row_id",
            "source_sample_id",
            "timestamp",
            "window_start_timestamp",
            "window_end_timestamp",
            "event_id",
        ):
            if key in values:
                metadata[key] = str(values[key])
        return metadata

    def _archive_item(self, index: int) -> dict[str, Any]:
        assert self.archive is not None
        values = self.archive
        item = {
            "image": _image_tensor(values[self.image_key][index], self.root, self.image_size),
            "tire_acc": _sensor_tensor(values["tire_acc"][index]).squeeze(),
            "gyro": _sensor_tensor(values["gyro"][index]),
            "vehicle_state": _sensor_tensor(values["vehicle_state"][index]).flatten(),
        }
        metadata = {
            key: values[key][index]
            for key in (
                "road_label",
                "anomaly_class",
                "source_row_id",
                "source_sample_id",
                "split",
                "group_id",
                "sample_id",
                "timestamp",
                "window_start_timestamp",
                "window_end_timestamp",
                "event_id",
            )
            if key in values
        }
        item.update(self._metadata(metadata, index))
        return item

    def _manifest_item(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tire_path = _resolve_path(row[self.tire_acc_column], self.root)
        gyro_path = _resolve_path(row[self.gyro_column], self.root)
        item = {
            "image": _image_tensor(row[self.image_column], self.root, self.image_size),
            "tire_acc": _sensor_tensor(
                _load_array_file(tire_path, self.tire_acc_key)
            ).squeeze(),
            "gyro": _sensor_tensor(_load_array_file(gyro_path, self.gyro_key)),
            "vehicle_state": torch.tensor(
                [float(row[name]) for name in self.state_columns], dtype=torch.float32
            ),
        }
        item.update(self._metadata(row, index))
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = (
            self._manifest_item(index)
            if self.archive is None
            else self._archive_item(index)
        )
        if item["tire_acc"].ndim != 1:
            raise ValueError(
                f"tire_acc sample must reduce to [L], got {tuple(item['tire_acc'].shape)}"
            )
        if item["gyro"].ndim not in {1, 2}:
            raise ValueError(
                f"gyro sample must be [D], [L,C], or [C,L], got {tuple(item['gyro'].shape)}"
            )
        if item["vehicle_state"].numel() != len(self.state_columns):
            raise ValueError("vehicle_state dimension does not match state_columns")
        return item


