"""Self-contained checkpoints for road pretraining and the complete RCG-MTAD."""

from dataclasses import asdict, replace
from pathlib import Path

import torch

from .models import RCGMTAD, RCGMTADConfig
from .models.road_encoder import SOURCE_ROAD_CLASSES
from .data import ANOMALY_CLASS_NAMES

MODEL_SCHEMA = "rcg-mtad-end-to-end-v1"
ROAD_SCHEMA = "rcg-mtad-road-encoder-v1"


def read_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return payload


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_checkpoint(path, model, *, stage="joint", training=None):
    payload = {
        "schema_version": ROAD_SCHEMA if stage == "road" else MODEL_SCHEMA,
        "road_class_names": list(SOURCE_ROAD_CLASSES),
        "training": training or {},
    }
    if stage == "road":
        payload.update({
            "road_config": asdict(model.config.road),
            "input_config": asdict(model.config.inputs),
            "model_state_dict": cpu_state(model.road_encoder),
            "preprocessor_state_dict": cpu_state(model.preprocessor),
        })
    else:
        payload.update({
            "model_config": model.config.to_dict(),
            "class_names": list(ANOMALY_CLASS_NAMES),
            "model_state_dict": cpu_state(model),
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model(path, *, device="cpu", image_backbone=None):
    payload = read_checkpoint(path)
    if payload.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("Inference requires an RCG-MTAD end-to-end checkpoint")
    if tuple(payload.get("class_names", ())) != ANOMALY_CLASS_NAMES:
        raise ValueError("Incompatible tire class order in checkpoint")
    if tuple(payload.get("road_class_names", ())) != SOURCE_ROAD_CLASSES:
        raise ValueError("Incompatible road class order in checkpoint")
    config = RCGMTADConfig.from_dict(payload["model_config"])
    # All weights are already in this file; never download ImageNet weights.
    config = replace(config, road=replace(config.road, pretrained=False))
    model = RCGMTAD(config, image_backbone=image_backbone)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device)
    return model, payload
