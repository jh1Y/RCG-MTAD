import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from tire_anomaly_system.data.end_to_end import TTSPDEndToEndDataset, partition_dataset


@pytest.fixture
def existing_ttspd(tmp_path):
    road = "沥青路面"
    image_dir = tmp_path / "images" / road
    image_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), (100, 120, 140)).save(image_dir / "1.jpg")
    Image.new("RGB", (32, 32), (200, 220, 240)).save(image_dir / "2.jpg")
    labels = np.tile(np.arange(4), 6)
    splits = np.repeat(["train", "validation", "test"], 8)
    frame = pd.DataFrame({
        "tms_datetime": pd.date_range("2026-01-01", periods=24, freq="s"),
        "road_type": road, "trial_id": "1", "pressure_kPa": 248.0,
        "temperature_C": 25.0, "speed_kmh": 30.0,
        "anomaly_class": labels, "dataset_split": splits,
        "sample_id": [f"sample-{index}" for index in range(24)],
        "source_row_id": [f"{split}-{index // 4}" for index, split in enumerate(splits)],
        "image_path": [f"images/{road}/{1 + index % 2}.jpg" for index in range(24)],
        **{name: float(index + 1) for index, name in enumerate([
            "imu_ax_mean", "imu_ay_mean", "imu_az_mean", "imu_gx_mean", "imu_gy_mean", "imu_gz_mean",
        ])},
        **{f"g_{index}": float(index + 10) for index in range(32)},
    })
    aligned = tmp_path / "sensor" / road / "1" / "aligned"
    aligned.mkdir(parents=True)
    path = aligned / f"{road}_1_aligned.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return tmp_path, path, frame


def test_existing_data_loads_in_physical_units_without_changing_labels(existing_ttspd):
    root, _, frame = existing_ttspd
    dataset = TTSPDEndToEndDataset(root, image_size=16, acc_length=32)
    partitions = partition_dataset(dataset)
    assert len(dataset) == len(frame)
    assert {name: len(indices) for name, indices in partitions.items()} == {"train": 8, "val": 8, "test": 8}
    assert [dataset.metadata(index)["anomaly_class"] for index in range(24)] == frame["anomaly_class"].tolist()
    row = dataset[0]
    torch.testing.assert_close(row["vehicle_state"], torch.tensor([248., 25., 30.]))
    torch.testing.assert_close(row["tire_acc"], torch.arange(10., 42.))
    torch.testing.assert_close(row["gyro"], torch.tensor([[1., 2., 3., 4., 5., 6., 0., 0., 0.]]))
    assert row["image"].shape == (3, 16, 16)
    assert row["image"].mean() < dataset[1]["image"].mean()
    torch.testing.assert_close(dataset.sensor_item(0)["gyro"], row["gyro"])


def test_directory_partitions_and_test_only_loading(existing_ttspd):
    root, path, frame = existing_ttspd
    path.unlink()
    for split in ("train", "validation", "test"):
        folder = root / "sensor" / split / "沥青路面" / "1" / "aligned"
        folder.mkdir(parents=True)
        rows = frame.loc[frame["dataset_split"] == split].drop(columns="dataset_split")
        rows.to_csv(folder / "trial_aligned.csv", index=False, encoding="utf-8-sig")
    dataset = TTSPDEndToEndDataset(root, image_size=16, acc_length=32)
    assert {name: len(indices) for name, indices in partition_dataset(dataset).items()} == {"train": 8, "val": 8, "test": 8}
    test_only = TTSPDEndToEndDataset(root, sensor_subdir="sensor/test",
                                    image_size=16, acc_length=32, for_inference=True)
    assert len(test_only) == 8
    assert all(test_only.metadata(index)["split"] == "test" for index in range(8))


@pytest.mark.parametrize("problem", ["missing_label", "fractional_label", "missing_split", "unknown_split"])
def test_existing_training_data_requires_valid_labels_and_splits(existing_ttspd, problem):
    root, path, frame = existing_ttspd
    if problem == "missing_label":
        frame = frame.drop(columns="anomaly_class")
    elif problem == "fractional_label":
        frame["anomaly_class"] = frame["anomaly_class"].astype(float)
        frame.loc[0, "anomaly_class"] = 1.5
    elif problem == "missing_split":
        frame = frame.drop(columns="dataset_split")
    else:
        frame.loc[0, "dataset_split"] = "unknown"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="anomaly_class|split"):
        TTSPDEndToEndDataset(root, image_size=16, acc_length=32)


def test_split_column_must_agree_with_directory(existing_ttspd):
    root, path, frame = existing_ttspd
    path.unlink()
    folder = root / "sensor" / "train" / "沥青路面" / "1" / "aligned"
    folder.mkdir(parents=True)
    frame.to_csv(folder / "trial_aligned.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Conflicting split"):
        TTSPDEndToEndDataset(root, image_size=16, acc_length=32)


def test_source_overlap_is_rejected_for_existing_records(existing_ttspd):
    root, path, frame = existing_ttspd
    frame.loc[8, "source_row_id"] = frame.loc[0, "source_row_id"]
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="leakage"):
        partition_dataset(TTSPDEndToEndDataset(root, image_size=16, acc_length=32))


def test_ordered_image_fallback_preserves_every_existing_row(existing_ttspd):
    root, path, frame = existing_ttspd
    frame = frame.drop(columns="image_path")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    dataset = TTSPDEndToEndDataset(root, image_size=16, acc_length=32)
    assert len(dataset) == 24
    assert len({row["sample_id"] for row in dataset.rows}) == 24
    for source in frame["source_row_id"].unique():
        assert len({row["image_path"] for row in dataset.rows if row["source_row_id"] == source}) == 1


def test_unlabeled_deployment_rows_stay_unlabeled(existing_ttspd):
    root, path, frame = existing_ttspd
    frame = frame.drop(columns=["anomaly_class", "dataset_split"])
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    dataset = TTSPDEndToEndDataset(root, image_size=16, acc_length=32, for_inference=True)
    assert len(dataset) == 24
    assert all(row["anomaly_class"] == -1 and row["split"] == "test" for row in dataset.rows)
