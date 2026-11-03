from dataclasses import replace
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from tire_anomaly_system.cli.infer_model import infer, parse_args as infer_args
from tire_anomaly_system.cli.train_model import parse_args, train
from tire_anomaly_system.data.end_to_end import RawMultimodalDataset, partition_dataset
from tire_anomaly_system.model_checkpoint import cpu_state, load_model, save_checkpoint
from tire_anomaly_system.models import InputConfig, RCGMTAD, RCGMTADConfig, RTIMFNetConfig, TireClassifierConfig
from tire_anomaly_system.training.engine import fit_normalizers, make_loader, predict


class TinyImageBranch(nn.Module):
    out_dim = 8

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.GELU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, image):
        return self.layers(image).flatten(1)


def tiny_config():
    return RCGMTADConfig(
        road=RTIMFNetConfig(image_size=16, image_fusion_dim=16, dropout=0.0,
                           freeze_backbone=False, train_last_stage=False, gate_temperature=1),
        tire=TireClassifierConfig(acc_length=32, road_embedding_dim=16, hidden_dim=12, dropout=0.0),
        inputs=InputConfig(acc_length=32, road_acc_length=16),
    )


@pytest.fixture
def raw_samples(tmp_path):
    rng = np.random.default_rng(42)
    classes = np.tile([0, 0, 1, 1, 2, 2, 3, 3], 3)
    values = {
        "image": rng.integers(0, 256, (24, 16, 16, 3), dtype=np.uint8),
        "tire_acc": rng.normal(size=(24, 32)).astype(np.float32),
        "gyro": rng.normal(size=(24, 9)).astype(np.float32),
        "vehicle_state": np.column_stack((248 + rng.normal(size=24), 25 + rng.normal(size=24), np.full(24, 30))).astype(np.float32),
        "road_label": np.tile([0, 1, 2, 3], 6), "anomaly_class": classes,
        "split": np.repeat(["train", "val", "test"], 8),
        "sample_id": np.array([f"sample-{index}" for index in range(24)]),
        "source_row_id": np.array([f"source-{index}" for index in range(24)]),
    }
    path = tmp_path / "samples.npz"
    np.savez(path, **values)
    return path, values


def fitted_model(raw_samples, *, frozen=False):
    dataset = RawMultimodalDataset(raw_samples[0], image_size=16, acc_length=32)
    partitions = partition_dataset(dataset)
    model = RCGMTAD(replace(tiny_config(), freeze_road_encoder=frozen), image_backbone=TinyImageBranch())
    fit_normalizers(model, dataset, partitions["train"], device=torch.device("cpu"), batch_size=4)
    return model, dataset, partitions


def test_tire_loss_updates_road_encoder_from_raw_inputs(raw_samples):
    model, dataset, parts = fitted_model(raw_samples)
    batch = next(iter(make_loader(dataset, parts["train"], batch_size=4)))
    model.train()
    output = model(batch["image"], batch["tire_acc"], batch["gyro"], batch["vehicle_state"])
    assert output.road.zroad.shape == (4, 16)
    assert output.tire.probabilities.shape == (4, 4)
    torch.testing.assert_close(output.road.probabilities.sum(1), torch.ones(4))
    torch.testing.assert_close(output.tire.probabilities.sum(1), torch.ones(4))
    nn.functional.cross_entropy(output.tire.logits, batch["anomaly_class"]).backward()
    for parameter in (model.road_encoder.classifier[-1].weight,
                      model.road_encoder.image_backbone.layers[0].weight,
                      model.road_encoder.seq_extractor[0].weight,
                      model.road_encoder.scalar_gate_generator[-1].weight,
                      model.tire_classifier.acc_encoder.parameters().__next__()):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_configuration_enforces_the_saved_class_contract():
    config = tiny_config()
    with pytest.raises(ValueError, match="class order"):
        replace(config, tire=replace(config.tire, num_classes=5))
    with pytest.raises(ValueError, match="class order"):
        replace(config, road=replace(config.road, class_names=("a", "b", "c", "d")))


def test_freezing_preserves_road_weights_and_batchnorm(raw_samples):
    model, dataset, parts = fitted_model(raw_samples, frozen=True)
    before = cpu_state(model.road_encoder)
    model.train()
    assert not model.road_encoder.training and model.tire_classifier.training
    batch = next(iter(make_loader(dataset, parts["train"], batch_size=4)))
    output = model(batch["image"], batch["tire_acc"], batch["gyro"], batch["vehicle_state"])
    output.tire.logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.road_encoder.parameters())
    for key, value in before.items():
        torch.testing.assert_close(model.road_encoder.state_dict()[key], value)


def test_only_training_rows_fit_normalization_and_checkpoint_roundtrip(raw_samples, tmp_path):
    model, dataset, parts = fitted_model(raw_samples)
    values = raw_samples[1]
    healthy = (values["split"] == "train") & (values["anomaly_class"] == 0)
    torch.testing.assert_close(model.preprocessor.acc_mean, torch.tensor(values["tire_acc"][healthy, :16].mean(0)))
    torch.testing.assert_close(model.tire_classifier.acc_mean, torch.tensor(values["tire_acc"][:8].mean(0)))
    loader = make_loader(dataset, parts["test"], batch_size=3)
    before = cpu_state(model)
    predictions = predict(model, loader, torch.device("cpu"))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    path = tmp_path / "complete.pt"
    save_checkpoint(path, model)
    restored, payload = load_model(path, image_backbone=TinyImageBranch())
    actual = predict(restored, loader, torch.device("cpu"))
    assert actual == predictions
    assert payload["class_names"] == ["healthy", "tire_leak", "tire_overheat", "tire_imbalance"]
    payload["class_names"].reverse()
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="class order"):
        load_model(tmp_path / "bad.pt", image_backbone=TinyImageBranch())


def test_split_leakage_and_missing_classes_are_rejected(raw_samples, tmp_path):
    values = dict(raw_samples[1])
    values["source_row_id"] = values["source_row_id"].copy()
    values["source_row_id"][8] = values["source_row_id"][0]
    np.savez(tmp_path / "leak.npz", **values)
    with pytest.raises(ValueError, match="leakage"):
        partition_dataset(RawMultimodalDataset(tmp_path / "leak.npz", acc_length=32))
    values = dict(raw_samples[1])
    values["anomaly_class"] = np.zeros(24, dtype=np.int64)
    np.savez(tmp_path / "healthy.npz", **values)
    dataset = RawMultimodalDataset(tmp_path / "healthy.npz", acc_length=32)
    with pytest.raises(ValueError, match="all four"):
        partition_dataset(dataset)


@pytest.mark.parametrize("stage", ["road", "joint"])
def test_both_training_stages_require_existing_four_state_labels(raw_samples, tmp_path, stage):
    values = {key: value for key, value in raw_samples[1].items() if key != "anomaly_class"}
    path = tmp_path / "missing-labels.npz"
    np.savez(path, **values)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config().to_dict()), encoding="utf-8")
    args = parse_args(["--samples", str(path), "--config", str(config_path),
                       "--output-dir", str(tmp_path / "output")], stage=stage)
    with pytest.raises(ValueError, match="missing or invalid labels"):
        train(args, stage=stage, image_backbone=TinyImageBranch())


def test_archive_labels_cannot_be_silently_truncated(raw_samples, tmp_path):
    values = dict(raw_samples[1])
    values["anomaly_class"] = values["anomaly_class"].astype(float)
    values["anomaly_class"][0] = 1.5
    path = tmp_path / "invalid-label.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError, match="anomaly_class must be an integer"):
        partition_dataset(RawMultimodalDataset(path, image_size=16, acc_length=32))


def test_two_stage_training_and_unlabeled_inference(raw_samples, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config().to_dict()), encoding="utf-8")
    common = ["--samples", str(raw_samples[0]), "--config", str(config_path),
              "--epochs", "2", "--batch-size", "4", "--device", "cpu"]
    road_args = parse_args([*common, "--output-dir", str(tmp_path / "road")], stage="road")
    road_model, road_summary = train(road_args, stage="road", image_backbone=TinyImageBranch())
    road_path = tmp_path / "road" / "road_encoder.pt"
    assert road_path.is_file() and road_summary["stage"] == "road"
    args = parse_args([*common, "--road-checkpoint", str(road_path), "--output-dir", str(tmp_path / "joint")])
    model, summary = train(args, image_backbone=TinyImageBranch())
    assert summary["stage"] == "joint" and summary["selection_partition"] == "val"
    assert summary["test"]["tire"]["sample_count"] == 8
    for key, value in road_model.preprocessor.state_dict().items():
        torch.testing.assert_close(model.preprocessor.state_dict()[key], value)
    # Both branches are updated by the joint training stage.
    assert not torch.equal(road_model.road_encoder.classifier[-1].weight,
                           model.road_encoder.classifier[-1].weight)
    checkpoint = tmp_path / "joint" / "rcg_mtad.pt"
    rows, metrics = infer(infer_args(["--samples", str(raw_samples[0]), "--checkpoint", str(checkpoint),
                                     "--device", "cpu", "--output-dir", str(tmp_path / "inference")]),
                          image_backbone=TinyImageBranch())
    assert len(rows) == 8 and metrics["tire"]["sample_count"] == 8
    with (tmp_path / "inference" / "predictions.csv").open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 8
    unlabeled = {key: value[:1] for key, value in raw_samples[1].items()
                 if key not in {"split", "anomaly_class", "road_label"}}
    np.savez(tmp_path / "unlabeled.npz", **unlabeled)
    deployed, metrics = infer(infer_args(["--samples", str(tmp_path / "unlabeled.npz"),
                                         "--checkpoint", str(checkpoint), "--device", "cpu",
                                         "--split", "all", "--batch-size", "1",
                                         "--output-dir", str(tmp_path / "deployment")]),
                              image_backbone=TinyImageBranch())
    assert len(deployed) == 1 and metrics["tire"]["sample_count"] == 0


def test_manifest_paths_and_short_windows(raw_samples, tmp_path):
    from PIL import Image
    Image.fromarray(raw_samples[1]["image"][0]).save(tmp_path / "image.png")
    np.save(tmp_path / "acc.npy", np.ones(7, dtype=np.float32))
    np.save(tmp_path / "imu.npy", np.zeros(9, dtype=np.float32))
    manifest = tmp_path / "samples.csv"
    manifest.write_text("image_path,tire_acc_path,gyro_path,tire_pressure,tire_temperature,speed,road_label,anomaly_class,split\n"
                        "image.png,acc.npy,imu.npy,250,25,30,1,0,test\n", encoding="utf-8")
    dataset = RawMultimodalDataset(manifest, image_size=16, acc_length=32)
    row = dataset[0]
    assert row["anomaly_class"] == 0 and row["tire_acc"].shape == (32,)
    torch.testing.assert_close(row["tire_acc"][:7], torch.ones(7))
    torch.testing.assert_close(row["tire_acc"][7:], torch.zeros(25))
    torch.testing.assert_close(dataset.sensor_item(0)["tire_acc"], row["tire_acc"])


def test_legacy_initialization_preserves_config_and_resets_old_class_semantics(raw_samples, tmp_path):
    source, _, _ = fitted_model(raw_samples)
    torch.save({"model_state_dict": cpu_state(source.road_encoder)}, tmp_path / "legacy_road.pt")
    torch.save({"model_state_dict": cpu_state(source.tire_classifier),
                "class_names": ["healthy", "old_pressure", "old_temperature", "old_leak"]},
               tmp_path / "legacy_tire.pt")
    config = replace(tiny_config(), road=replace(tiny_config().road, freeze_backbone=True))
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    common = ["--samples", str(raw_samples[0]), "--config", str(tmp_path / "config.json"),
              "--road-checkpoint", str(tmp_path / "legacy_road.pt"),
              "--initial-tire-checkpoint", str(tmp_path / "legacy_tire.pt"),
              "--epochs", "1", "--batch-size", "4", "--device", "cpu",
              "--output-dir", str(tmp_path / "initialized")]
    model, summary = train(parse_args(common), image_backbone=TinyImageBranch())
    assert summary["initialization"]["tire_classifier_reinitialized"]
    assert summary["initialization"]["tire_loaded_keys"] > 0
    assert model.config.road.freeze_backbone
    assert not model.road_encoder.image_backbone.training
    assert not any(parameter.requires_grad for parameter in model.road_encoder.image_backbone.parameters())
    torch.save({"model_state_dict": {"unrelated": torch.zeros(1)}}, tmp_path / "empty_tire.pt")
    common[common.index(str(tmp_path / "legacy_tire.pt"))] = str(tmp_path / "empty_tire.pt")
    with pytest.raises(ValueError, match="no compatible"):
        train(parse_args(common), image_backbone=TinyImageBranch())


def test_real_mobilenet_forward_backward_and_checkpoint(raw_samples, tmp_path):
    pytest.importorskip("timm")
    config = replace(tiny_config(), road=replace(tiny_config().road, image_size=64))
    dataset = RawMultimodalDataset(raw_samples[0], image_size=64, acc_length=32)
    parts = partition_dataset(dataset)
    model = RCGMTAD(config)
    fit_normalizers(model, dataset, parts["train"], device=torch.device("cpu"), batch_size=4)
    batch = next(iter(make_loader(dataset, parts["train"], batch_size=4)))
    model.train()
    output = model(batch["image"], batch["tire_acc"], batch["gyro"], batch["vehicle_state"])
    loss = nn.functional.cross_entropy(output.tire.logits, batch["anomaly_class"]) + nn.functional.cross_entropy(output.road.logits, batch["road_label"])
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.road_encoder.image_backbone.parameters())
    model.eval()
    expected = model(batch["image"], batch["tire_acc"], batch["gyro"], batch["vehicle_state"]).tire.logits
    save_checkpoint(tmp_path / "real.pt", model)
    restored, _ = load_model(tmp_path / "real.pt")
    torch.testing.assert_close(restored(batch["image"], batch["tire_acc"], batch["gyro"], batch["vehicle_state"]).tire.logits, expected)
