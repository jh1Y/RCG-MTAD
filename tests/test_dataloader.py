from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from tire_anomaly_system.cli.common import build_datasets
from tire_anomaly_system.cli.infer_model import parse_args as infer_args
from tire_anomaly_system.cli.search_hyperparameters import build_parser, plan_candidates, training_command
from tire_anomaly_system.cli.train_model import parse_args
from tire_anomaly_system.data import MultimodalDataset, make_loader
from tire_anomaly_system.data.labels import ANOMALY_CLASS_NAMES
from tire_anomaly_system.models import RCGMTADConfig


def sensor_csv(root, split="train"):
    return root / split / "sensor" / "samples.csv"


def test_directories_load_once_in_physical_units(prepared_samples):
    root, values = prepared_samples
    for split in ("train", "valid", "test"):
        dataset = MultimodalDataset(root, split, image_size=16, acc_length=32)
        indices = np.flatnonzero(values["split"] == split)
        assert len(dataset) == 8
        assert [row["anomaly_class"] for row in dataset.rows] == values["anomaly_class"][indices].tolist()
        assert all(row["split"] == split for row in dataset.rows)
        row = dataset[0]
        first = indices[0]
        torch.testing.assert_close(row["tire_acc"], torch.tensor(values["tire_acc"][first]))
        torch.testing.assert_close(row["gyro"], torch.tensor(values["gyro"][first:first + 1]))
        torch.testing.assert_close(row["vehicle_state"], torch.tensor(values["vehicle_state"][first]))
        assert row["image"].shape == (3, 16, 16)
    assert MultimodalDataset(root).metadata(0)["sample_id"] == "0000"


def test_split_columns_are_ignored(prepared_samples):
    root, _ = prepared_samples
    path = sensor_csv(root)
    frame = pd.read_csv(path)
    frame["split"] = "test"
    frame["dataset_split"] = "unknown"
    frame.to_csv(path, index=False)
    dataset = MultimodalDataset(root)
    assert len(dataset) == len(frame)
    assert all(row["split"] == "train" for row in dataset.rows)


def test_explicit_images_and_padding(prepared_samples):
    root, _ = prepared_samples
    path = sensor_csv(root)
    frame = pd.read_csv(path)
    frame.loc[0, "image_path"] = "7.png"
    frame.loc[1, "image_path"] = "pic\\0.png"
    frame.loc[0, [f"g_{i}" for i in range(7, 32)]] = np.nan
    frame.to_csv(path, index=False)
    dataset = MultimodalDataset(root, image_size=8, acc_length=40)
    assert dataset.rows[0]["image_path"].endswith("7.png")
    assert dataset.rows[1]["image_path"].endswith("0.png")
    row = dataset[0]
    assert row["image"].shape == (3, 8, 8)
    torch.testing.assert_close(row["tire_acc"][7:], torch.zeros(33))
    with patch("PIL.Image.open", side_effect=AssertionError("sensor loading must not decode images")):
        torch.testing.assert_close(dataset.sensor_item(0)["tire_acc"], row["tire_acc"])


@pytest.mark.parametrize("value", ["", "missing.png", "../test/pic/16.png", "../../test/pic/16.png"])
def test_missing_or_cross_directory_image_is_rejected(prepared_samples, value):
    root, _ = prepared_samples
    path = sensor_csv(root)
    frame = pd.read_csv(path)
    frame.loc[0, "image_path"] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="image_path|Sample image"):
        MultimodalDataset(root)


@pytest.mark.parametrize("problem", ["missing_label", "fractional_label", "road_conflict", "missing_image_column"])
def test_required_fields_and_labels(prepared_samples, problem):
    root, _ = prepared_samples
    path = sensor_csv(root)
    frame = pd.read_csv(path)
    if problem == "missing_label":
        frame = frame.drop(columns="anomaly_class")
    elif problem == "fractional_label":
        frame["anomaly_class"] = frame["anomaly_class"].astype(float)
        frame.loc[0, "anomaly_class"] = 1.5
    elif problem == "road_conflict":
        frame["road_type"] = "沥青路面"
    else:
        frame = frame.drop(columns="image_path")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="anomaly_class|road_label|image_path"):
        MultimodalDataset(root)


def test_batching_keeps_validation_and_test_tail(prepared_samples):
    root, _ = prepared_samples
    datasets = {split: MultimodalDataset(root, split, image_size=8, acc_length=16)
                for split in ("train", "valid", "test")}
    for split in ("valid", "test"):
        loader = make_loader(datasets[split], batch_size=7)
        assert [len(batch["road_label"]) for batch in loader] == [7, 1]
    train_loader = make_loader(datasets["train"], batch_size=7, shuffle=True)
    assert train_loader.drop_last
    with pytest.raises(ValueError, match="BatchNorm"):
        make_loader(datasets["train"], batch_size=1, shuffle=True)


def test_only_requested_directories_are_opened(prepared_samples):
    root, _ = prepared_samples
    args = parse_args(["--data-dir", str(root), "--no-evaluate-test"])
    sensor_csv(root, "test").write_text("not,a,sensor,csv\n1,2,3,4\n", encoding="utf-8")
    datasets = build_datasets(args, RCGMTADConfig(), ("train", "valid"))
    assert set(datasets) == {"train", "valid"}
    with pytest.raises(ValueError, match="Missing sensor columns"):
        build_datasets(args, RCGMTADConfig(), ("test",))


def test_validation_only_and_unlabeled_data(prepared_samples):
    root, _ = prepared_samples
    (root / "train").rename(root / "unused-train")
    (root / "test").rename(root / "unused-test")
    path = sensor_csv(root, "valid")
    frame = pd.read_csv(path).drop(columns=["road_label", "anomaly_class"])
    frame.to_csv(path, index=False)
    args = infer_args(["--data-dir", str(root), "--split", "valid", "--checkpoint", "unused.pt"])
    dataset = build_datasets(args, RCGMTADConfig(), (args.split,), for_inference=True)["valid"]
    assert len(dataset) == 8
    assert all(row["road_label"] == row["anomaly_class"] == -1 for row in dataset.rows)


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_missing_sensor_values_are_zero_filled_in_memory(prepared_samples, split):
    root, values = prepared_samples
    path = sensor_csv(root, split)
    frame = pd.read_csv(path)
    frame.loc[0, "g_3"] = np.nan
    frame.loc[0, ["imu_ax_mean", "imu_ay_mean", "imu_az_mean",
                  "imu_gx_mean", "imu_gy_mean", "imu_gz_mean"]] = np.nan
    frame.loc[0, ["pressure_kPa", "temperature_C"]] = np.nan
    frame.loc[1, [f"g_{i}" for i in range(32)]] = np.nan
    frame.loc[2, ["pressure_kPa", "temperature_C", "speed_kmh"]] = np.nan
    frame.to_csv(path, index=False)
    original = path.read_bytes()
    dataset = MultimodalDataset(root, split, image_size=8, acc_length=40)
    assert len(dataset) == 8
    first = np.flatnonzero(values["split"] == split)[0]
    expected_acc = np.pad(values["tire_acc"][first], (0, 8))
    expected_acc[3] = 0
    row = dataset.sensor_item(0)
    torch.testing.assert_close(row["tire_acc"], torch.tensor(expected_acc))
    torch.testing.assert_close(row["gyro"], torch.zeros(1, 9))
    torch.testing.assert_close(row["vehicle_state"], torch.tensor([0.0, 0.0, 30.0]))
    torch.testing.assert_close(dataset.sensor_item(1)["tire_acc"], torch.zeros(40))
    torch.testing.assert_close(dataset.sensor_item(2)["vehicle_state"], torch.zeros(3))
    batches = list(make_loader(dataset, batch_size=3))
    assert sum(len(batch["road_label"]) for batch in batches) == 8
    assert torch.cat([batch["anomaly_class"] for batch in batches]).tolist() == frame["anomaly_class"].tolist()
    assert all(torch.isfinite(batch[key]).all() for batch in batches
               for key in ("image", "tire_acc", "gyro", "vehicle_state"))
    assert path.read_bytes() == original


@pytest.mark.parametrize("value", [np.inf, -np.inf, "not-a-number"])
def test_invalid_sensors_are_rejected_without_changing_rows(prepared_samples, value):
    root, _ = prepared_samples
    path = sensor_csv(root)
    frame = pd.read_csv(path)
    frame["pressure_kPa"] = frame["pressure_kPa"].astype(object)
    frame.loc[0, "pressure_kPa"] = value
    frame.to_csv(path, index=False)
    dataset = MultimodalDataset(root)
    assert len(dataset) == 8
    with pytest.raises(ValueError, match="Sensor sample"):
        dataset.sensor_item(0)


@pytest.mark.parametrize("stage", ["road", "joint"])
def test_search_commands_use_prepared_directories(prepared_samples, tmp_path, stage):
    root, _ = prepared_samples
    args = build_parser().parse_args(["--stage", stage, "--data-dir", str(root),
                                      "--output-dir", str(tmp_path / "search")])
    candidates, count = plan_candidates(args)
    command = training_command(args, candidates[0], tmp_path / "trial")
    train_args = parse_args(command[4:], stage=stage)
    assert train_args.data_dir == str(root)
    assert not train_args.evaluate_test
    assert count == 324 and len(candidates) == 30


@pytest.mark.parametrize("folder_name", ["pic", "images"])
@pytest.mark.parametrize("path_prefix", ["pic", "images"])
def test_source_image_directory_and_path_aliases(prepared_samples, folder_name, path_prefix):
    root, _ = prepared_samples
    if folder_name == "images":
        (root / "valid" / "pic").rename(root / "valid" / "images")
    path = sensor_csv(root, "valid")
    frame = pd.read_csv(path)
    frame["image_path"] = frame["image_path"].str.replace("pic/", path_prefix + "/", regex=False)
    frame.to_csv(path, index=False)
    dataset = MultimodalDataset(root, "valid", image_size=8)
    assert dataset.image_dir.name == folder_name
    assert len(dataset) == 8 and dataset[0]["image"].shape == (3, 8, 8)


def test_ambiguous_image_directories_are_rejected(prepared_samples):
    root, _ = prepared_samples
    (root / "valid" / "images").mkdir()
    with pytest.raises(ValueError, match="Both pic/ and images/"):
        MultimodalDataset(root, "valid")


def test_original_source_pairing_uses_time_natural_order_and_shared_sources(prepared_samples):
    root, _ = prepared_samples
    path = sensor_csv(root)
    frame = pd.read_csv(path).iloc[:3].copy()
    frame = frame.drop(columns="image_path")
    frame["road_label"] = 3
    frame["source_row_id"] = ["source-1", "source-0", "source-0"]
    frame["tms_datetime"] = ["2026-01-02", "2026-01-01", "2026-01-01"]
    frame["trial_id"] = "1"
    frame.to_csv(path, index=False)
    images = root / "train" / "pic" / "沥青路面"
    images.mkdir()
    (root / "train" / "pic" / "0.png").rename(images / "2.png")
    (root / "train" / "pic" / "1.png").rename(images / "10.png")
    dataset = MultimodalDataset(root, image_size=8)
    assert len(dataset) == 3
    assert [row["image_path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            for row in dataset.rows] == ["10.png", "2.png", "2.png"]
    assert all(row["image_pairing"] == "source_order" for row in dataset.rows)


@pytest.mark.parametrize("anomaly_type", ["pressure_relation_anomaly", "temperature_relation_anomaly", "pressure_leak"])
def test_legacy_class_semantics_cannot_be_renamed(prepared_samples, anomaly_type):
    root, _ = prepared_samples
    path = sensor_csv(root, "valid")
    frame = pd.read_csv(path)
    frame["anomaly_type"] = "none"
    frame.loc[0, "anomaly_type"] = anomaly_type
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incompatible anomaly_type"):
        MultimodalDataset(root, "valid")


def test_canonical_semantics_match_class_numbers(prepared_samples):
    root, _ = prepared_samples
    path = sensor_csv(root, "valid")
    frame = pd.read_csv(path)
    frame["anomaly_type"] = [ANOMALY_CLASS_NAMES[int(value)] for value in frame["anomaly_class"]]
    frame.to_csv(path, index=False)
    dataset = MultimodalDataset(root, "valid")
    assert len(dataset) == 8
    frame.loc[0, "anomaly_type"] = "tire_imbalance"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicts with anomaly_type"):
        MultimodalDataset(root, "valid")


