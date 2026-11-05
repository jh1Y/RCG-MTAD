"""Configuration and raw-data options used by all model entry points."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from tire_anomaly_system.checkpoint import inspect_legacy_architecture, load_legacy_checkpoint
from tire_anomaly_system.data import MultimodalDataset
from tire_anomaly_system.model_checkpoint import ROAD_SCHEMA, read_checkpoint
from tire_anomaly_system.models import InputConfig, RCGMTADConfig, SOURCE_ROAD_CLASSES


def add_data_arguments(parser):
    parser.add_argument("--data-dir", default="Datasets",
                        help="Dataset root with train/valid/test; each has pic/ (or images/) and sensor/")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)


def build_datasets(args, config, splits, *, for_inference=False):
    if config.inputs.state_dim != 3:
        raise ValueError("Sensor CSVs provide pressure, temperature and speed (3 scalars)")
    return {split: MultimodalDataset(
        args.data_dir, split,
        image_size=config.road.image_size, acc_length=config.inputs.acc_length,
        imu_channels=config.inputs.imu_channels, imu_length=config.inputs.imu_length,
        for_inference=for_inference,
    ) for split in splits}


def training_config(args):
    config = RCGMTADConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else RCGMTADConfig()
    if args.road_checkpoint:
        payload = read_checkpoint(args.road_checkpoint)
        if payload.get("schema_version") == ROAD_SCHEMA:
            if tuple(payload.get("road_class_names", ())) != SOURCE_ROAD_CLASSES:
                raise ValueError("Incompatible road class order")
            cfg = RCGMTADConfig.from_dict({"road": payload["road_config"]})
            road = cfg.road
            inputs = InputConfig(**{**payload["input_config"], "acc_length": config.inputs.acc_length})
        else:
            inferred = inspect_legacy_architecture(args.road_checkpoint)
            # Legacy source constructor used gate_temperature=1; this scalar
            # cannot be recovered from weights and must not default to 2.
            fields = {key: value for key, value in inferred.items() if key != "state_dict_keys"}
            road = replace(config.road, **fields, gate_temperature=1.0,
                           imu_channel_indices=None, pretrained=False)
            inputs = replace(config.inputs, imu_channels=road.imu_channels, state_dim=road.scalar_dim)
        config = RCGMTADConfig(
            road=replace(road, pretrained=False, freeze_backbone=config.road.freeze_backbone,
                         train_last_stage=config.road.train_last_stage),
            tire=replace(config.tire, road_embedding_dim=road.image_fusion_dim,
                         road_classes=road.num_classes, gyro_channels=len(inputs.gyro_indices),
                         gyro_length=inputs.imu_length),
            inputs=inputs, freeze_road_encoder=config.freeze_road_encoder,
        )
    if args.acc_length is not None:
        config = replace(config, inputs=replace(config.inputs, acc_length=args.acc_length),
                         tire=replace(config.tire, acc_length=args.acc_length))
    if args.image_size is not None:
        config = replace(config, road=replace(config.road, image_size=args.image_size))
    if args.freeze_road_encoder is not None:
        config = replace(config, freeze_road_encoder=args.freeze_road_encoder)
    return config


def initialize_road(model, path):
    if not path:
        return False
    payload = read_checkpoint(path)
    if payload.get("schema_version") == ROAD_SCHEMA:
        model.road_encoder.load_state_dict(payload["model_state_dict"], strict=True)
        model.preprocessor.load_state_dict(payload["preprocessor_state_dict"], strict=True)
        return True
    report = load_legacy_checkpoint(model.road_encoder, path, strict=True)
    print(json.dumps({"legacy_road_keys_loaded": report.loaded_keys,
                      "road_normalizer": "fit on current healthy training samples"}))
    return False
