"""Run images and physical sensor inputs through one saved end-to-end model."""

import argparse
import json
from pathlib import Path

from torch.utils.data import ConcatDataset

from tire_anomaly_system.data import make_loader
from tire_anomaly_system.model_checkpoint import load_model
from tire_anomaly_system.training.engine import predict, prediction_metrics, resolve_device, write_predictions
from .common import add_data_arguments, build_datasets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Infer road type and tire state with RCG-MTAD")
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="test")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    return args


def infer(args, *, image_backbone=None):
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device=device, image_backbone=image_backbone)
    splits = ("train", "valid", "test") if args.split == "all" else (args.split,)
    datasets = build_datasets(args, model.config, splits, for_inference=True)
    input_count = sum(len(dataset) for dataset in datasets.values())
    dataset = ConcatDataset(list(datasets.values())) if args.split == "all" else datasets[args.split]
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.num_workers)
    predictions = predict(model, loader, device)
    output = Path(args.output_dir)
    write_predictions(output / "predictions.csv", predictions)
    summary = {"sample_count": len(predictions), "split": args.split,
               "input_sample_count": input_count, "missing_sensor_fill_value": 0.0,
               "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
               "checkpoint_schema": checkpoint["schema_version"],
               "model_config": model.config.to_dict(),
               **prediction_metrics(predictions),
               "pressure_temperature_calibration": model.tire_classifier.pressure_temperature_calibration()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return predictions, summary


def main():
    infer(parse_args())


if __name__ == "__main__":
    main()
