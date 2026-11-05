"""Load prepared splits, including the source project's images/sensor layout."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .labels import validate_tire_label

SPLITS = ("train", "valid", "test")
ROAD_TYPES = ("积水路面", "破损路面", "水泥路面", "沥青路面")
STATE_COLUMNS = ("pressure_kPa", "temperature_C", "speed_kmh")
IMU_COLUMNS = ("imu_ax_mean", "imu_ay_mean", "imu_az_mean",
               "imu_gx_mean", "imu_gy_mean", "imu_gz_mean")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_sort_key(value):
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", str(value))]


def _source_timestamp(value):
    text = str(value).removesuffix(".0")
    return pd.to_datetime(text, format="%Y%m%d%H%M%S" if text.isdigit() and len(text) == 14 else None,
                          errors="raise")


def _pair_source_images(rows, image_dir):
    """Source loader's labelled-row rule: one image per ordered source group.

    This is proportional road-wise association, not timestamp synchronization.
    Every sensor row is retained, including rows sharing a source_row_id.
    """
    for road_label, road_name in enumerate(ROAD_TYPES):
        road_rows = [row for row in rows if row["road_label"] == road_label]
        pending = [row for row in road_rows if not row["image_path"]]
        if not pending:
            continue
        if any(not row.get("source_row_id") or not row.get("timestamp") for row in road_rows):
            raise ValueError(
                f"Missing image_path for {road_name}; source-order pairing requires "
                "source_row_id and a timestamp on every row in that road class"
            )
        ordering = pd.DataFrame({
            "position": range(len(road_rows)),
            "trial_id": [str(row.get("trial_id", "")) for row in road_rows],
            "tms_datetime": [_source_timestamp(row["timestamp"]) for row in road_rows],
        }).sort_values(["trial_id", "tms_datetime"])
        ordered = [road_rows[index] for index in ordering["position"]]
        source_ids = list(dict.fromkeys(row["source_row_id"] for row in ordered))
        positions = {identity: index for index, identity in enumerate(source_ids)}
        explicit = {}
        for row in ordered:
            if row["image_path"]:
                explicit.setdefault(row["source_row_id"], set()).add(row["image_path"])
        folder = image_dir / road_name
        images = sorted((path.resolve() for path in folder.glob("*") if path.is_file()
                         and path.suffix.lower() in IMAGE_EXTENSIONS), key=_natural_sort_key)
        for row in pending:
            existing = explicit.get(row["source_row_id"], set())
            if len(existing) > 1:
                raise ValueError(f"Conflicting explicit images for source_row_id {row['source_row_id']!r}")
            if existing:
                row["image_path"] = next(iter(existing))
            else:
                if not images:
                    raise FileNotFoundError(f"No source-pairing images found in {folder}")
                # Keep the source implementation's floating-point expression.
                index = int(positions[row["source_row_id"]] / len(source_ids) * len(images))
                row["image_path"] = str(images[min(index, len(images) - 1)])
            row["image_pairing"] = "source_order"
    if any(not row["image_path"] for row in rows):
        raise ValueError("Missing image_path; source-order pairing requires a known road label")


def _read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype={
                key: "string" for key in ("image_path", "sample_id", "source_row_id",
                                         "group_id", "timestamp", "tms_datetime",
                                         "tms_timestamp_raw", "trial_id")
            })
            frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode sensor CSV: {path}")


def _text(row, key):
    value = row.get(key, "")
    return "" if pd.isna(value) else str(value).strip()


def _label(row, key, *, required):
    value = _text(row, key)
    if not value:
        if required:
            raise ValueError(f"Missing {key}")
        return -1
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or int(number) not in range(4):
        raise ValueError(f"{key} must be an integer in 0..3, got {value!r}")
    return int(number)


def _values(row, columns):
    # Fill missing physical sensor values before model input adaptation/normalization.
    values = pd.to_numeric(row[list(columns)], errors="raise").fillna(0.0).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Sensor data contains infinite or invalid numeric values")
    return values


class MultimodalDataset(Dataset):
    """One split with pic/ or images/ and sensor/; explicit paths take priority.

    Rows without an image_path follow the original labelled-data pairing rule.
    Directory membership, sensor rows and all stored class numbers stay intact.
    Missing sensor cells are filled with zero in memory for every split.
    """

    def __init__(self, data_dir, split="train", *, image_size=224, acc_length=1024,
                 imu_length=1, imu_channels=9, for_inference=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if min(image_size, acc_length, imu_length, imu_channels) < 1:
            raise ValueError("Image and sensor dimensions must be positive")
        self.root = Path(data_dir).expanduser().resolve()
        self.split = split
        self.split_dir = self.root / split
        image_dirs = [self.split_dir / name for name in ("pic", "images")
                      if (self.split_dir / name).is_dir()]
        if len(image_dirs) > 1:
            raise ValueError(f"Both pic/ and images/ exist in {self.split_dir}; keep one image directory")
        self.image_dir = (image_dirs[0] if image_dirs else self.split_dir / "pic").resolve()
        self.sensor_dir = self.split_dir / "sensor"
        self.image_size, self.acc_length = image_size, acc_length
        self.imu_length, self.imu_channels = imu_length, imu_channels
        for folder in (self.image_dir, self.sensor_dir):
            if not folder.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {folder}")
        files = sorted(self.sensor_dir.rglob("*.csv"), key=_natural_sort_key)
        if not files:
            raise ValueError(f"No sensor CSV files found in {self.sensor_dir}")
        self.rows, self._tables, self._locations = [], [], []
        for path in files:
            frame = _read_csv(path)
            acc_columns = sorted((key for key in frame if re.fullmatch(r"g_\d+", key)),
                                 key=lambda key: int(key[2:]))
            missing = {*STATE_COLUMNS, *IMU_COLUMNS} - set(frame)
            if missing or not acc_columns:
                raise ValueError(f"Missing sensor columns in {path}: {sorted(missing)}; g_* columns required")
            table_index = len(self._tables)
            self._tables.append((frame, acc_columns))
            for index, row in frame.iterrows():
                try:
                    road_name = _text(row, "road_type")
                    road_label = _label(row, "road_label", required=not (for_inference or road_name))
                    if road_name:
                        if road_name not in ROAD_TYPES:
                            raise ValueError(f"Unknown road_type: {road_name!r}")
                        named_label = ROAD_TYPES.index(road_name)
                        if road_label >= 0 and road_label != named_label:
                            raise ValueError("road_label conflicts with road_type")
                        road_label = named_label
                    tire_label = _label(row, "anomaly_class", required=not for_inference)
                    anomaly_type = _text(row, "anomaly_type")
                    validate_tire_label(tire_label, anomaly_type)
                    identity = f"{split}/{path.relative_to(self.sensor_dir).as_posix()}#{index}"
                    image_path = _text(row, "image_path")
                    self.rows.append({
                        "image_path": str(self._image_path(image_path)) if image_path else "",
                        "image_pairing": "explicit" if image_path else "source_order",
                        "road_label": road_label, "anomaly_class": tire_label,
                        "anomaly_type": anomaly_type, "trial_id": _text(row, "trial_id"),
                        "sample_id": _text(row, "sample_id") or identity,
                        "source_row_id": _text(row, "source_row_id"),
                        "group_id": _text(row, "group_id") or _text(row, "trial_id"),
                        "timestamp": next((_text(row, key) for key in (
                            "timestamp", "tms_datetime", "tms_timestamp_raw") if _text(row, key)), ""),
                        "split": split,
                    })
                    self._locations.append((table_index, index))
                except (ValueError, TypeError, FileNotFoundError) as error:
                    raise ValueError(f"{path}, row {index}: {error}") from error
        if not self.rows:
            raise ValueError(f"No sensor records found in {self.sensor_dir}")
        _pair_source_images(self.rows, self.image_dir)

    def _image_path(self, value):
        if not value:
            raise ValueError("Missing image_path; every sensor row needs an explicit image")
        path = Path(value.replace("\\", "/")).expanduser()
        if not path.is_absolute():
            path = (self.image_dir / Path(*path.parts[1:])
                    if path.parts[:1] in (("pic",), ("images",)) else self.image_dir / path)
        path = path.resolve()
        if not path.is_relative_to(self.image_dir):
            raise ValueError(f"image_path must point inside {self.image_dir}: {value!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Sample image does not exist: {path}")
        return path

    def __len__(self):
        return len(self.rows)

    def metadata(self, index):
        return {key: value for key, value in self.rows[index].items() if key != "image_path"}

    def sensor_item(self, index):
        table_index, row_index = self._locations[index]
        frame, acc_columns = self._tables[table_index]
        row = frame.loc[row_index]
        try:
            acc = _values(row, acc_columns)[:self.acc_length]
            acc = np.pad(acc, (0, self.acc_length - len(acc)))
            imu = _values(row, IMU_COLUMNS)
            size = self.imu_length * self.imu_channels
            imu = np.pad(imu[:size], (0, max(0, size - len(imu))))
            return {**self.metadata(index), "tire_acc": torch.from_numpy(acc),
                    "gyro": torch.from_numpy(imu.reshape(self.imu_length, self.imu_channels)),
                    "vehicle_state": torch.from_numpy(_values(row, STATE_COLUMNS))}
        except (ValueError, TypeError) as error:
            raise ValueError(f"Sensor sample {self.rows[index]['sample_id']}: {error}") from error

    def __getitem__(self, index):
        item = self.sensor_item(index)
        with Image.open(self.rows[index]["image_path"]) as image:
            array = np.asarray(image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR), dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        return {**item, "image": (tensor - IMAGENET_MEAN) / IMAGENET_STD}


def collate_samples(samples):
    result = {key: torch.stack([row[key] for row in samples])
              for key in ("image", "tire_acc", "gyro", "vehicle_state")}
    for key in ("road_label", "anomaly_class"):
        result[key] = torch.tensor([row[key] for row in samples], dtype=torch.long)
    for key in ("sample_id", "group_id", "timestamp", "split"):
        result[key] = [str(row.get(key, "")) for row in samples]
    return result


def make_loader(dataset, *, batch_size=32, workers=0, shuffle=False, seed=42):
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    if shuffle and (batch_size < 2 or len(dataset) < 2):
        raise ValueError("Training requires batch_size >= 2 and at least two samples for BatchNorm")
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
        # A singleton training batch cannot pass the road classifier's BatchNorm.
        drop_last=bool(shuffle and len(dataset) % batch_size == 1),
        collate_fn=collate_samples, generator=torch.Generator().manual_seed(seed),
    )
