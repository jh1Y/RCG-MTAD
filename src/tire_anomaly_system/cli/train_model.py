"""Train the full road-conditioned tire model directly from images and sensors."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from tire_anomaly_system.data import ANOMALY_CLASS_NAMES, make_loader
from tire_anomaly_system.model_checkpoint import cpu_state, read_checkpoint, save_checkpoint
from tire_anomaly_system.models import RCGMTAD
from tire_anomaly_system.training.engine import (
    fit_normalizers, predict, prediction_metrics,
    resolve_device, run_epoch, write_predictions,
)
from .common import add_data_arguments, build_datasets, initialize_road, training_config


def parse_args(argv=None, *, stage="joint"):
    parser = argparse.ArgumentParser(description="Train the road encoder" if stage == "road" else "Train RCG-MTAD end to end")
    add_data_arguments(parser)
    parser.add_argument("--config", help="JSON with road, tire, inputs and freeze_road_encoder settings")
    parser.add_argument("--road-checkpoint", help="Standalone RCG-MTAD road or legacy RTI-MFNet checkpoint")
    parser.add_argument("--initial-tire-checkpoint", help="Optional source four-class classifier initialization")
    parser.add_argument("--output-dir", default=f"outputs/{'road_encoder' if stage == 'road' else 'rcg_mtad'}")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--road-lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--road-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--acc-length", type=int)
    parser.add_argument("--freeze-road-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable during hyperparameter selection; rank candidates using validation only")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience) < 1 or args.batch_size < 2 or args.num_workers < 0:
        parser.error("epochs/patience must be positive, batch-size >= 2, num-workers >= 0")
    if min(args.lr, args.road_lr) <= 0 or min(args.weight_decay, args.road_loss_weight) < 0:
        parser.error("learning rates must be positive; decay and road loss weight nonnegative")
    if not all(math.isfinite(value) for value in (
        args.lr, args.road_lr, args.min_lr, args.weight_decay, args.road_loss_weight,
    )):
        parser.error("learning rates, decay and road loss weight must be finite")
    initial_rates = [args.lr] if stage == "road" else [args.lr, args.road_lr]
    if args.min_lr < 0 or args.min_lr > min(initial_rates):
        parser.error("min-lr must be nonnegative and no larger than any initial learning rate")
    if args.lr_scheduler == "constant" and args.min_lr != 0:
        parser.error("min-lr requires --lr-scheduler cosine")
    if stage == "road" and args.initial_tire_checkpoint:
        parser.error("--initial-tire-checkpoint is used only for joint training")
    return args


def class_weights(labels, device):
    counts = torch.bincount(torch.tensor(labels), minlength=4).float()
    return (counts.sum() / (4 * counts.clamp_min(1))).to(device)


def train(args, *, stage="joint", image_backbone=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    config = training_config(args)
    if stage == "road" and config.freeze_road_encoder:
        raise ValueError("Road pretraining requires a trainable road encoder")
    if config.freeze_road_encoder and not args.road_checkpoint:
        raise ValueError("Freezing the road encoder requires --road-checkpoint")
    splits = ("train", "valid", "test") if args.evaluate_test else ("train", "valid")
    datasets = build_datasets(args, config, splits)
    model = RCGMTAD(config, image_backbone=image_backbone)
    keep_road_normalizer = initialize_road(model, args.road_checkpoint)
    initialization = {}
    if args.initial_tire_checkpoint:
        payload = read_checkpoint(args.initial_tire_checkpoint)
        state = payload.get("model_state_dict", {})
        compatible_classes = tuple(payload.get("class_names", ())) == ANOMALY_CLASS_NAMES
        if not compatible_classes:
            state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        report = model.tire_classifier.load_compatible_state_dict(state)
        if not report["loaded"]:
            raise ValueError("The initial tire checkpoint has no compatible classifier parameters")
        initialization = {"tire_loaded_keys": len(report["loaded"]),
                          "tire_classifier_reinitialized": not compatible_classes}
        print(json.dumps(initialization))
    fit_normalizers(model, datasets["train"], device=device,
                    batch_size=args.batch_size, workers=args.num_workers, stage=stage,
                    keep_road_normalizer=keep_road_normalizer)
    loaders = {name: make_loader(dataset, batch_size=args.batch_size,
                                 workers=args.num_workers, shuffle=name == "train", seed=args.seed)
               for name, dataset in datasets.items()}
    rows = [datasets["train"].metadata(index) for index in range(len(datasets["train"]))]
    road_criterion = torch.nn.CrossEntropyLoss(weight=class_weights([row["road_label"] for row in rows], device))
    tire_criterion = torch.nn.CrossEntropyLoss(weight=class_weights([row["anomaly_class"] for row in rows], device)) if stage == "joint" else None
    groups = [{"params": [p for p in model.road_encoder.parameters() if p.requires_grad],
               "name": "road",
               "lr": args.lr if stage == "road" else args.road_lr}]
    if stage == "joint":
        groups.append({"params": model.tire_classifier.parameters(), "lr": args.lr, "name": "tire"})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    # With E epochs, the first epoch uses the initial LR and the last uses min_lr.
    # Constant scheduling preserves the previous training-entry behavior.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - 1, 1), eta_min=args.min_lr,
    ) if args.lr_scheduler == "cosine" else None
    history, best_state, best_epoch, stale = [], None, 0, 0
    best_score = (-1.0, float("-inf"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        learning_rates = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
        kwargs = dict(stage=stage, road_criterion=road_criterion, tire_criterion=tire_criterion,
                      road_loss_weight=args.road_loss_weight)
        train_metrics = run_epoch(model, loaders["train"], device, optimizer=optimizer, **kwargs)
        val_metrics = run_epoch(model, loaders["valid"], device, **kwargs)
        record = {"epoch": epoch, "learning_rates": learning_rates,
                  "train": train_metrics, "valid": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        score = (val_metrics["road" if stage == "road" else "tire"]["macro_f1"], -val_metrics["loss"])
        if score > best_score:
            best_state, best_score, best_epoch, stale = cpu_state(model), score, epoch, 0
            save_checkpoint(output / ("road_encoder.pt" if stage == "road" else "rcg_mtad.pt"),
                            model, stage=stage, training={"epoch": epoch, "seed": args.seed, "validation": val_metrics})
        else:
            stale += 1
            if stale >= args.patience:
                break
        if scheduler is not None and epoch < args.epochs:
            scheduler.step()
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    summary = {"stage": stage, "best_epoch": best_epoch, "seed": args.seed,
               "epochs_completed": len(history), "validation": history[best_epoch - 1]["valid"],
               "test_evaluated": "test" in loaders,
               "selection_partition": "valid", "partition_sizes": {key: len(value) for key, value in datasets.items()},
               "model_config": config.to_dict(), "training_arguments": vars(args),
               "initialization": initialization, "history": history}
    # The test partition participates only after validation has selected weights.
    if "test" in loaders:
        if stage == "joint":
            predictions = predict(model, loaders["test"], device)
            write_predictions(output / "predictions.csv", predictions)
            summary["test"] = prediction_metrics(predictions)
        else:
            summary["test"] = run_epoch(model, loaders["test"], device, stage="road", road_criterion=road_criterion)
    if stage == "joint":
        summary["pressure_temperature_calibration"] = model.tire_classifier.pressure_temperature_calibration()
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(output / ("road_encoder.pt" if stage == "road" else "rcg_mtad.pt")),
                      "best_epoch": best_epoch}, ensure_ascii=False))
    return model, summary


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
