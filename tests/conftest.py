import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch


def pytest_sessionstart(session):
    # Small CPU fixtures run faster without launching a large thread pool.
    torch.set_num_threads(1)


@pytest.fixture
def prepared_samples(tmp_path):
    """Already separated directories; no project split/export code is used."""
    rng = np.random.default_rng(42)
    values = {
        "image": rng.integers(0, 256, (24, 16, 16, 3), dtype=np.uint8),
        "tire_acc": rng.normal(size=(24, 32)).astype(np.float32),
        "gyro": rng.normal(size=(24, 9)).astype(np.float32),
        "vehicle_state": np.column_stack((248 + rng.normal(size=24), 25 + rng.normal(size=24), np.full(24, 30))).astype(np.float32),
        "road_label": np.tile([0, 1, 2, 3], 6),
        "anomaly_class": np.tile([0, 0, 1, 1, 2, 2, 3, 3], 3),
        "split": np.repeat(["train", "valid", "test"], 8),
    }
    values["gyro"][:, 6:] = 0
    root = tmp_path / "dataset"
    for split in ("train", "valid", "test"):
        pic = root / split / "pic"
        sensor = root / split / "sensor"
        pic.mkdir(parents=True)
        sensor.mkdir()
        indices = np.flatnonzero(values["split"] == split)
        for index in indices:
            Image.fromarray(values["image"][index]).save(pic / f"{index}.png")
        frame = pd.DataFrame({
            "image_path": [f"pic/{index}.png" for index in indices],
            "sample_id": [f"{index:04d}" for index in indices],
            "road_label": values["road_label"][indices],
            "anomaly_class": values["anomaly_class"][indices],
            **{name: values["vehicle_state"][indices, i] for i, name in enumerate(
                ("pressure_kPa", "temperature_C", "speed_kmh"))},
            **{name: values["gyro"][indices, i] for i, name in enumerate(
                ("imu_ax_mean", "imu_ay_mean", "imu_az_mean", "imu_gx_mean", "imu_gy_mean", "imu_gz_mean"))},
            **{f"g_{i}": values["tire_acc"][indices, i] for i in range(32)},
        })
        frame.to_csv(sensor / "samples.csv", index=False, encoding="utf-8-sig")
    return root, values
