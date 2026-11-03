"""Raw multimodal datasets shared by road pretraining and joint training.

Generic NPZ/PT archives use image (or image_path), tire_acc, gyro, vehicle_state,
road_label, anomaly_class, split, sample_id and optional source_row_id/group_id.
CSV manifests use the path columns documented by SynchronizedRoadDataset, with
the same label/split columns. Labels are zero-based; split is train/val/test.
TTSPD reads existing four-state aligned CSVs and their explicit partitions.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .synchronized_samples import (
    SynchronizedRoadDataset, _image_tensor, _load_array_file, _resolve_path, _sensor_tensor,
)
from .ttspd_multimodal_data import load_ttspd_samples, normalize_split


class RawMultimodalDataset(SynchronizedRoadDataset):
    """Each sample is padded/truncated once, independently of batch companions."""

    def __init__(self, source, *, acc_length=1024, **kwargs):
        super().__init__(source, **kwargs)
        self.acc_length = acc_length
        if self.archive is not None:
            for key, value in self.archive.items():
                if np.ndim(value) > 0 and len(value) != self.length:
                    raise ValueError(f"Archive key {key} does not match sample count")

    def metadata(self, index):
        values = self.rows[index] if self.archive is None else {
            key: self.archive[key][index] for key in (
                "road_label", "anomaly_class", "split", "sample_id", "group_id",
                "source_row_id", "source_sample_id", "timestamp",
            ) if key in self.archive
        }
        return self._metadata(values, index)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        acc = item["tire_acc"][:self.acc_length]
        item["tire_acc"] = torch.nn.functional.pad(acc, (0, self.acc_length - len(acc)))
        return item

    def sensor_item(self, index):
        if self.archive is not None:
            values = {key: _sensor_tensor(self.archive[key][index])
                      for key in ("tire_acc", "gyro", "vehicle_state")}
        else:
            row = self.rows[index]
            values = {
                "tire_acc": _sensor_tensor(_load_array_file(_resolve_path(row[self.tire_acc_column], self.root), self.tire_acc_key)),
                "gyro": _sensor_tensor(_load_array_file(_resolve_path(row[self.gyro_column], self.root), self.gyro_key)),
                "vehicle_state": torch.tensor([float(row[key]) for key in self.state_columns]),
            }
        acc = values["tire_acc"].flatten()[:self.acc_length]
        values["tire_acc"] = torch.nn.functional.pad(acc, (0, self.acc_length - len(acc)))
        return {**self.metadata(index), **values}


class TTSPDEndToEndDataset(Dataset):
    """Load existing four-state TTSPD rows in physical units without resplitting."""

    def __init__(self, data_dir, *, sensor_subdir="sensor",
                 image_size=224, acc_length=1024, imu_length=1, imu_channels=9,
                 recursive_images=False, for_inference=False):
        self.root = Path(data_dir).expanduser().resolve()
        self.image_size = image_size
        self.rows = load_ttspd_samples(
            self.root, sensor_subdir=sensor_subdir, acc_length=acc_length,
            imu_length=imu_length, imu_channels=imu_channels,
            recursive_images=recursive_images, for_inference=for_inference,
        )

    def __len__(self):
        return len(self.rows)

    def metadata(self, index):
        return {key: value for key, value in self.rows[index].items()
                if key not in {"image_path", "tire_acc", "gyro", "vehicle_state"}}

    def __getitem__(self, index):
        row = self.rows[index]
        return {**self.sensor_item(index),
                "image": _image_tensor(row["image_path"], self.root, self.image_size)}

    def sensor_item(self, index):
        row = self.rows[index]
        return {**self.metadata(index), **{key: row[key] for key in ("tire_acc", "gyro", "vehicle_state")}}


def collate_raw_samples(samples):
    result = {key: torch.stack([row[key] for row in samples])
              for key in ("image", "tire_acc", "gyro", "vehicle_state")}
    for key in ("road_label", "anomaly_class"):
        result[key] = torch.tensor([row.get(key, -1) for row in samples], dtype=torch.long)
    for key in ("sample_id", "group_id", "timestamp", "split"):
        result[key] = [str(row.get(key, "")) for row in samples]
    return result


def partition_dataset(dataset):
    """Audit explicit splits without opening images or ever resplitting samples."""
    partitions = {key: [] for key in ("train", "val", "test")}
    identities = {key: set() for key in partitions}
    sample_ids = set()
    tire_counts = {key: [0] * 4 for key in partitions}
    for index in range(len(dataset)):
        row = dataset.metadata(index)
        split = normalize_split(row.get("split", ""))
        if split not in partitions:
            raise ValueError(f"Sample {index} needs an explicit train/val/test split")
        road, tire = int(row.get("road_label", -1)), int(row.get("anomaly_class", -1))
        if road not in range(4) or tire not in range(4):
            raise ValueError(f"Sample {index} has missing or invalid labels")
        sample_id = str(row.get("sample_id", index))
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        source = str(row.get("source_sample_id") or row.get("source_row_id") or sample_id)
        identities[split].add(source)
        partitions[split].append(index)
        if tire in range(4):
            tire_counts[split][tire] += 1
    for left, right in combinations(partitions, 2):
        if identities[left] & identities[right]:
            raise ValueError(f"Source sample leakage between {left} and {right}")
    if len(partitions["train"]) < 2 or not partitions["val"]:
        raise ValueError("At least two training samples and a validation partition are required")
    if any(count == 0 for count in tire_counts["train"]):
        raise ValueError("The training partition must contain all four tire classes")
    return {key: np.asarray(value, dtype=np.int64) for key, value in partitions.items()}
