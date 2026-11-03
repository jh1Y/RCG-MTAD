"""Read existing TTSPD four-state aligned CSVs without changing their labels.

Every CSV row is retained once. Train/val/test membership comes from its split
column or parent split directory. Images use an explicit image_path when present;
otherwise the existing road-wise ordered image association is used. Sensors stay
in physical units; model checkpoints own all fitted normalization parameters.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from .labels import ANOMALY_CLASS_NAMES


ROAD_TYPES = ("积水路面", "破损路面", "水泥路面", "沥青路面")
STATE_COLUMNS = ("pressure_kPa", "temperature_C", "speed_kmh")
IMU_COLUMNS = ("imu_ax_mean", "imu_ay_mean", "imu_az_mean",
               "imu_gx_mean", "imu_gy_mean", "imu_gz_mean")
SPLIT_ALIASES = {"validation": "val", "valid": "val", "0": "train", "1": "val", "2": "test"}


def normalize_split(value):
    value = str(value).strip().lower()
    return SPLIT_ALIASES.get(value, value)


def natural_sort_key(value):
    return [(0, int(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", str(value))]


def _read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            frame = pd.read_csv(path, encoding=encoding,
                                dtype={key: "string" for key in ("sample_id", "source_row_id", "group_id")})
            frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
            if frame.empty:
                raise ValueError(f"Aligned CSV is empty: {path}")
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode aligned CSV: {path}")


def _text(row, key):
    value = row.get(key, "")
    return "" if pd.isna(value) else str(value).strip()


def _label(row, key, *, required, count):
    value = _text(row, key)
    if not value:
        if required:
            raise ValueError(f"Existing data is missing {key}")
        return -1
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or int(number) not in range(count):
        raise ValueError(f"{key} must be an integer in 0..{count - 1}, got {value!r}")
    return int(number)


def _split(row, path, sensor_root, *, for_inference):
    folder_names = {"train", "val", "valid", "validation", "test"}
    folder_splits = {normalize_split(part) for part in path.relative_to(sensor_root).parts[:-1]
                     if part.strip().lower() in folder_names}
    if len(folder_splits) > 1:
        raise ValueError(f"Ambiguous split directories: {path}")
    values = {normalize_split(_text(row, key)) for key in ("dataset_split", "split") if _text(row, key)}
    values.update(folder_splits)
    if len(values) > 1:
        raise ValueError(f"Conflicting split column/directory in {path}")
    split = next(iter(values), "")
    if split and split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown dataset_split value: {split!r}")
    if not split and not for_inference:
        raise ValueError(f"Existing training data needs an explicit train/val/test split: {path}")
    return split or "test"


def _timestamp(row):
    for key in ("tms_datetime", "tms_timestamp_raw", "timestamp"):
        value = _text(row, key)
        if not value:
            continue
        if key == "tms_datetime":
            result = pd.to_datetime(value, errors="coerce")
        else:
            result = pd.to_datetime(value.removesuffix(".0"), format="%Y%m%d%H%M%S", errors="coerce")
        if pd.isna(result):
            raise ValueError(f"Invalid {key}: {value!r}")
        return str(result)
    return ""


def _values(row, columns, *, allow_empty=False):
    values = pd.to_numeric(row[list(columns)], errors="raise").to_numpy(dtype=np.float32)
    if allow_empty:
        values = values[~np.isnan(values)]
    if not np.isfinite(values).all():
        raise ValueError("Existing sensor data contains missing or non-finite values")
    return values


def load_ttspd_samples(data_dir, *, sensor_subdir="sensor", acc_length=1024,
                       imu_length=1, imu_channels=9, recursive_images=False,
                       for_inference=False):
    """Read labeled rows, their existing partitions and raw sensor values.

    No label is inferred from a missing field. Training requires anomaly_class
    (healthy=0, leak=1, overheat=2, imbalance=3); deployment may omit that label.
    Source/sample identifiers are retained solely for alignment and split audits.
    """
    root = Path(data_dir).expanduser().resolve()
    sensor_root = Path(sensor_subdir).expanduser()
    if not sensor_root.is_absolute():
        sensor_root = root / sensor_root
    sensor_root = sensor_root.resolve()
    if not sensor_root.is_dir():
        raise FileNotFoundError(f"Sensor directory does not exist: {sensor_root}")
    files = sorted((path for path in sensor_root.rglob("*.csv")
                    if path.name.lower().endswith("_aligned.csv")
                    and "aligned" in [part.lower() for part in path.relative_to(sensor_root).parts[:-1]]),
                   key=natural_sort_key)
    if not files:
        raise ValueError(f"No existing *_aligned.csv files found in {sensor_root}")
    samples = []
    for path in files:
        frame = _read_csv(path)
        acceleration_columns = sorted((column for column in frame if re.fullmatch(r"g_\d+", column)),
                                      key=lambda column: int(column[2:]))
        required = {*STATE_COLUMNS, *IMU_COLUMNS}
        missing = required - set(frame)
        if missing or not acceleration_columns:
            raise ValueError(f"Missing sensor columns in {path}: {sorted(missing)}; g_* columns required")
        for index, row in frame.iterrows():
            try:
                road_name = _text(row, "road_type") or next((name for name in ROAD_TYPES if name in path.parts), "")
                if road_name and road_name not in ROAD_TYPES:
                    raise ValueError(f"Unknown road_type: {road_name!r}")
                road_label = _label(row, "road_label", required=not bool(road_name), count=len(ROAD_TYPES))
                if road_name:
                    named_label = ROAD_TYPES.index(road_name)
                    if road_label >= 0 and road_label != named_label:
                        raise ValueError("road_label conflicts with road_type")
                    road_label = named_label
                anomaly_class = _label(row, "anomaly_class", required=not for_inference,
                                       count=len(ANOMALY_CLASS_NAMES))
                split = _split(row, path, sensor_root, for_inference=for_inference)
                identity = f"{path.relative_to(sensor_root).as_posix()}#{index}"
                acc = _values(row, acceleration_columns, allow_empty=True)[:acc_length]
                acc = np.pad(acc, (0, acc_length - len(acc)))
                imu = _values(row, IMU_COLUMNS)
                imu_size = imu_length * imu_channels
                imu = np.pad(imu[:imu_size], (0, max(0, imu_size - len(imu))))
                samples.append({
                    "image_path": _text(row, "image_path"),
                    "tire_acc": torch.from_numpy(acc),
                    "gyro": torch.from_numpy(imu.reshape(imu_length, imu_channels)),
                    "vehicle_state": torch.from_numpy(_values(row, STATE_COLUMNS)),
                    "road_label": road_label, "anomaly_class": anomaly_class,
                    "split": split, "sample_id": _text(row, "sample_id") or identity,
                    "source_row_id": _text(row, "source_row_id") or identity,
                    "group_id": _text(row, "group_id") or f"{ROAD_TYPES[road_label]}:{_text(row, 'trial_id')}",
                    "timestamp": _timestamp(row),
                })
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}, row {index}: {error}") from error
    samples.sort(key=lambda row: (row["road_label"], natural_sort_key(row["group_id"]), row["timestamp"]))
    for label, road_name in enumerate(ROAD_TYPES):
        road_rows = [row for row in samples if row["road_label"] == label]
        unassigned = [row for row in road_rows if not row["image_path"]]
        if unassigned:
            folder = root / "images" / road_name
            paths = folder.rglob("*") if recursive_images else folder.glob("*")
            images = sorted((path for path in paths if path.is_file()
                             and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}), key=natural_sort_key)
            if not images:
                raise ValueError(f"No images available for {road_name}: {folder}")
            identities = list(dict.fromkeys(row["source_row_id"] for row in road_rows))
            positions = {identity: index for index, identity in enumerate(identities)}
            for row in unassigned:
                index = min(positions[row["source_row_id"]] * len(images) // len(identities), len(images) - 1)
                row["image_path"] = str(images[index])
    for row in samples:
        image = Path(row["image_path"]).expanduser()
        row["image_path"] = str(image if image.is_absolute() else root / image)
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(f"Sample image does not exist: {row['image_path']}")
    return samples
