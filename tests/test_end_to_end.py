from dataclasses import replace
import csv
import json

import pandas as pd
import pytest
import torch
from torch import nn

from tire_anomaly_system.cli.infer_model import infer, parse_args as infer_args
from tire_anomaly_system.cli.train_model import parse_args, train
from tire_anomaly_system.data import MultimodalDataset, make_loader
from tire_anomaly_system.model_checkpoint import cpu_state, load_model, save_checkpoint
from tire_anomaly_system.models import InputConfig, RCGMTAD, RCGMTADConfig, RTIMFNetConfig, TireClassifierConfig
from tire_anomaly_system.training.engine import fit_normalizers, predict


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


def fitted_model(prepared_samples, *, frozen=False):
    datasets = {split: MultimodalDataset(prepared_samples[0], split, image_size=16, acc_length=32)
                for split in ("train", "valid", "test")}
    model = RCGMTAD(replace(tiny_config(), freeze_road_encoder=frozen), image_backbone=TinyImageBranch())
    fit_normalizers(model, datasets["train"], device=torch.device("cpu"), batch_size=4)
    return model, datasets


def test_tire_loss_updates_road_encoder_from_raw_inputs(prepared_samples):
    model, datasets = fitted_model(prepared_samples)
    batch = next(iter(make_loader(datasets["train"], batch_size=4)))
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


def test_freezing_preserves_road_weights_and_batchnorm(prepared_samples):
    model, datasets = fitted_model(prepared_samples, frozen=True)
    before = cpu_state(model.road_encoder)
    model.train()
    assert not model.road_encoder.training and model.tire_classifier.training
    batch = next(iter(make_loader(datasets["train"], batch_size=4)))
    output = model(batch["image"], batch["tire_acc"], batch["gyro"], batch["vehicle_state"])
    output.tire.logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.road_encoder.parameters())
    for key, value in before.items():
        torch.testing.assert_close(model.road_encoder.state_dict()[key], value)


def test_only_training_rows_fit_normalization_and_checkpoint_roundtrip(prepared_samples, tmp_path):
    model, datasets = fitted_model(prepared_samples)
    values = prepared_samples[1]
    healthy = (values["split"] == "train") & (values["anomaly_class"] == 0)
    torch.testing.assert_close(model.preprocessor.acc_mean, torch.tensor(values["tire_acc"][healthy, :16].mean(0)))
    torch.testing.assert_close(model.tire_classifier.acc_mean, torch.tensor(values["tire_acc"][:8].mean(0)))
    loader = make_loader(datasets["test"], batch_size=3)
    before = cpu_state(model)
    predictions = predict(model, loader, torch.device("cpu"))
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    path = tmp_path / "complete.pt"
    save_checkpoint(path, model)
    restored, payload = load_model(path, image_backbone=TinyImageBranch())
    assert predict(restored, loader, torch.device("cpu")) == predictions
    assert payload["class_names"] == ["healthy", "tire_leak", "tire_overheat", "tire_imbalance"]
    payload["class_names"].reverse()
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="class order"):
        load_model(tmp_path / "bad.pt", image_backbone=TinyImageBranch())


def test_two_stage_training_and_unlabeled_inference(prepared_samples, tmp_path):
    root, _ = prepared_samples
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config().to_dict()), encoding="utf-8")
    common = ["--data-dir", str(root), "--config", str(config_path),
              "--epochs", "2", "--batch-size", "4", "--device", "cpu"]
    road_args = parse_args([*common, "--output-dir", str(tmp_path / "road")], stage="road")
    road_model, road_summary = train(road_args, stage="road", image_backbone=TinyImageBranch())
    road_path = tmp_path / "road" / "road_encoder.pt"
    assert road_path.is_file() and road_summary["stage"] == "road"
    args = parse_args([*common, "--road-checkpoint", str(road_path), "--output-dir", str(tmp_path / "joint")])
    model, summary = train(args, image_backbone=TinyImageBranch())
    assert summary["stage"] == "joint" and summary["selection_partition"] == "valid"
    assert summary["partition_sizes"] == {"train": 8, "valid": 8, "test": 8}
    assert summary["test"]["tire"]["sample_count"] == 8
    for key, value in road_model.preprocessor.state_dict().items():
        torch.testing.assert_close(model.preprocessor.state_dict()[key], value)
    assert not torch.equal(road_model.road_encoder.classifier[-1].weight,
                           model.road_encoder.classifier[-1].weight)
    checkpoint = tmp_path / "joint" / "rcg_mtad.pt"
    rows, metrics = infer(infer_args(["--data-dir", str(root), "--checkpoint", str(checkpoint),
                                     "--device", "cpu", "--output-dir", str(tmp_path / "inference")]),
                          image_backbone=TinyImageBranch())
    assert len(rows) == 8 and metrics["tire"]["sample_count"] == 8
    with (tmp_path / "inference" / "predictions.csv").open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 8
    path = root / "test" / "sensor" / "samples.csv"
    frame = pd.read_csv(path).iloc[:1].drop(columns=["road_label", "anomaly_class"])
    frame.to_csv(path, index=False)
    deployed, metrics = infer(infer_args(["--data-dir", str(root), "--checkpoint", str(checkpoint),
                                         "--device", "cpu", "--batch-size", "1",
                                         "--output-dir", str(tmp_path / "deployment")]),
                              image_backbone=TinyImageBranch())
    assert len(deployed) == 1 and metrics["tire"]["sample_count"] == 0


def test_legacy_initialization_preserves_config_and_resets_old_class_semantics(prepared_samples, tmp_path):
    source, _ = fitted_model(prepared_samples)
    torch.save({"model_state_dict": cpu_state(source.road_encoder)}, tmp_path / "legacy_road.pt")
    torch.save({"model_state_dict": cpu_state(source.tire_classifier),
                "class_names": ["healthy", "old_pressure", "old_temperature", "old_leak"]},
               tmp_path / "legacy_tire.pt")
    config = replace(tiny_config(), road=replace(tiny_config().road, freeze_backbone=True))
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    common = ["--data-dir", str(prepared_samples[0]), "--config", str(tmp_path / "config.json"),
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


def test_real_mobilenet_forward_backward_and_checkpoint(prepared_samples, tmp_path):
    pytest.importorskip("timm")
    config = replace(tiny_config(), road=replace(tiny_config().road, image_size=64))
    dataset = MultimodalDataset(prepared_samples[0], "train", image_size=64, acc_length=32)
    model = RCGMTAD(config)
    fit_normalizers(model, dataset, device=torch.device("cpu"), batch_size=4)
    batch = next(iter(make_loader(dataset, batch_size=4)))
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
