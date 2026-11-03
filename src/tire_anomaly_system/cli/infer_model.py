"""Run images and physical sensor inputs through one saved end-to-end model."""

import argparse
import json
from pathlib import Path

from tire_anomaly_system.data.end_to_end import normalize_split
from tire_anomaly_system.model_checkpoint import load_model
from tire_anomaly_system.training.engine import make_loader, predict, prediction_metrics, resolve_device, write_predictions
from .common import add_data_arguments, build_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Infer road type and tire state with RCG-MTAD")
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    return args


def infer(args, *, image_backbone=None):
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device=device, image_backbone=image_backbone)
    dataset = build_dataset(args, model.config, for_inference=True)
    indices = [index for index in range(len(dataset)) if args.split == "all"
               or normalize_split(dataset.metadata(index).get("split", "")) == args.split]
    if not indices:
        raise ValueError(f"No samples in split {args.split!r}; use --split all for unlabeled deployment data")
    loader = make_loader(dataset, indices, batch_size=args.batch_size, workers=args.num_workers)
    predictions = predict(model, loader, device)
    output = Path(args.output_dir)
    write_predictions(output / "predictions.csv", predictions)
    summary = {"sample_count": len(predictions), "split": args.split,
               **prediction_metrics(predictions),
               "pressure_temperature_calibration": model.tire_classifier.pressure_temperature_calibration()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return predictions, summary


def main():
    infer(parse_args())


if __name__ == "__main__":
    main()
