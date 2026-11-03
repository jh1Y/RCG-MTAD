"""Validation-only hyperparameter selection, launched by scripts/search_*.sh.

This coordinator uses only the standard library. Each trial runs in a separate
training process; importing this module never builds or trains a model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys


SCHEMA = "rcg-mtad-hyperparameter-search-v1"


@dataclass(frozen=True)
class Candidate:
    epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    min_lr: float


REFERENCES = {
    "road": Candidate(150, 64, 1e-4, 1e-4, 1e-6),
    "joint": Candidate(150, 128, 1e-4, 1e-3, 1e-5),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("road", "joint"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir")
    source.add_argument("--samples")
    parser.add_argument("--data-root")
    parser.add_argument("--sensor-subdir", default="sensor")
    parser.add_argument("--recursive-images", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("random", "grid"), default="random")
    parser.add_argument("--trials", type=int, default=30, help="Number of configurations in random mode")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42, help="Same training seed for every candidate")
    parser.add_argument("--epochs-grid", default="50,100,150,200")
    parser.add_argument("--batch-size-grid", default="32,64,128")
    parser.add_argument("--weight-decay-grid", default="1e-5,1e-4,1e-3")
    parser.add_argument("--lr-grid", default="1e-5,1e-4,1e-3")
    parser.add_argument("--min-lr-grid", default="0,1e-6,1e-5")
    parser.add_argument("--include-reference", action=argparse.BooleanOptionalAction, default=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--road-checkpoint")
    initialization.add_argument("--road-search-result", help="best.json from the independent road search")
    initialization.add_argument("--from-scratch", action="store_true", help="Joint search without road pretraining")
    parser.add_argument("--initial-tire-checkpoint")
    parser.add_argument("--freeze-road-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--road-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Write the plan and commands without training")
    return parser


def _grid(text, kind, low, high, name):
    values = []
    for raw in text.split(","):
        value = kind(raw.strip())
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} candidates must lie in [{low}, {high}]")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} candidates cannot be empty")
    return values


def plan_candidates(args):
    axes = (
        _grid(args.epochs_grid, int, 50, 200, "epochs"),
        _grid(args.batch_size_grid, int, 32, 128, "batch size"),
        _grid(args.weight_decay_grid, float, 1e-5, 1e-3, "weight decay"),
        _grid(args.lr_grid, float, 1e-5, 1e-3, "initial LR"),
        _grid(args.min_lr_grid, float, 0, 1e-5, "minimum LR"),
    )
    candidates = [Candidate(*values) for values in itertools.product(*axes) if values[4] <= values[3]]
    if args.mode == "random" and args.trials < 1:
        raise ValueError("trials must be positive")
    reference = REFERENCES[args.stage]
    prefix = []
    if args.include_reference and reference in candidates:
        candidates.remove(reference)
        prefix = [reference]
    if args.mode == "random":
        random.Random(args.search_seed).shuffle(candidates)
        return (prefix + candidates)[:args.trials], len(prefix) + len(candidates)
    return prefix + candidates, len(prefix) + len(candidates)


def _absolute(value):
    return str(Path(value).expanduser().resolve()) if value else None


def resolve_initialization(args):
    if args.stage == "road" and (args.from_scratch or args.initial_tire_checkpoint or args.freeze_road_encoder):
        raise ValueError("Tire initialization, --from-scratch and road freezing apply only to joint search")
    if args.road_search_result:
        path = Path(args.road_search_result).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Road search result not found: {path}. Run the road search, set ROAD_CHECKPOINT, or set TRAIN_FROM_SCRATCH=1.")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version") != SCHEMA or result.get("stage") != "road":
            raise ValueError("road-search-result must be best.json from a road encoder search")
        args.road_checkpoint = _absolute(result["checkpoint"])
    if args.stage == "joint" and not args.road_checkpoint and not args.from_scratch:
        raise ValueError("Joint search requires road initialization, or explicit --from-scratch")
    if args.freeze_road_encoder and not args.road_checkpoint:
        raise ValueError("Road freezing requires a trained road checkpoint")
    args.road_checkpoint = _absolute(args.road_checkpoint)


def training_command(args, candidate, trial_dir):
    module = "train_road_encoder" if args.stage == "road" else "train_model"
    command = [sys.executable, "-u", "-m", f"tire_anomaly_system.cli.{module}"]
    for name in ("data_dir", "samples", "data_root", "config", "road_checkpoint", "initial_tire_checkpoint"):
        value = getattr(args, name)
        if value:
            command.extend(["--" + name.replace("_", "-"), _absolute(value)])
    command.extend([
        "--sensor-subdir", args.sensor_subdir, "--output-dir", str(trial_dir),
        "--epochs", str(candidate.epochs), "--patience", str(candidate.epochs),
        "--batch-size", str(candidate.batch_size), "--weight-decay", str(candidate.weight_decay),
        "--lr", str(candidate.lr), "--lr-scheduler", "cosine", "--min-lr", str(candidate.min_lr),
        "--seed", str(args.seed), "--device", args.device, "--num-workers", str(args.num_workers),
        "--no-evaluate-test",
    ])
    if args.recursive_images:
        command.append("--recursive-images")
    command.append("--freeze-road-encoder" if args.freeze_road_encoder else "--no-freeze-road-encoder")
    if args.stage == "joint":
        # The table specifies a single initial LR for the complete model.
        command.extend(["--road-lr", str(candidate.lr), "--road-loss-weight", str(args.road_loss_weight)])
    return command


def _file_stamp(value):
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _code_hash():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_result(args, candidate, trial_dir):
    summary = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
    if summary.get("stage") != args.stage or summary.get("test_evaluated") is not False:
        raise ValueError("Trial must use the requested stage and validation-only evaluation")
    if summary.get("epochs_completed") != candidate.epochs:
        raise ValueError("Trial did not complete the requested epoch budget")
    validation = summary["validation"]
    score = float(validation["road" if args.stage == "road" else "tire"]["macro_f1"])
    loss = float(validation["loss"])
    if not math.isfinite(score) or not math.isfinite(loss):
        raise ValueError("Trial validation metrics are not finite")
    checkpoint = trial_dir / ("road_encoder.pt" if args.stage == "road" else "rcg_mtad.pt")
    if not checkpoint.is_file():
        raise ValueError(f"Missing selected trial checkpoint: {checkpoint}")
    return {"val_macro_f1": score, "val_loss": loss, "best_epoch": summary["best_epoch"],
            "checkpoint": str(checkpoint), "metrics": str(trial_dir / "metrics.json")}


def _write_results(root, stage, results):
    successful = sorted((row for row in results if row["status"] == "completed"),
                        key=lambda row: (-row["val_macro_f1"], row["val_loss"], row["trial_id"]))
    rows = successful + [row for row in results if row["status"] != "completed"]
    fields = ["trial_id", "status", "epochs", "batch_size", "weight_decay", "lr", "min_lr",
              "is_table_reference", "val_macro_f1", "val_loss", "best_epoch", "checkpoint", "metrics", "log", "error"]
    with (root / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if successful:
        winner = successful[0]
        _write_json(root / "best.json", {
            "schema_version": SCHEMA, "stage": stage, "selection_partition": "val",
            "selection_metric": "road_macro_f1" if stage == "road" else "tire_macro_f1",
            "tie_break": "lower validation total loss, then trial ID",
            "test_evaluated": False, "optimizer": "AdamW", "scheduler": "cosine",
            "parameters": {name: winner[name] for name in Candidate.__dataclass_fields__},
            **{key: winner[key] for key in ("trial_id", "val_macro_f1", "val_loss", "best_epoch", "checkpoint", "metrics")},
        })


def search(args):
    if args.num_workers < 0 or not math.isfinite(args.road_loss_weight) or args.road_loss_weight < 0:
        raise ValueError("num-workers and road-loss-weight must be nonnegative")
    candidates, space_size = plan_candidates(args)
    resolve_initialization(args)
    if not args.dry_run:
        source = Path(args.data_dir or args.samples).expanduser()
        if not (source.is_dir() if args.data_dir else source.is_file()):
            raise ValueError(f"Data source does not exist: {source}")
        for name in ("config", "road_checkpoint", "initial_tire_checkpoint"):
            value = getattr(args, name)
            if value and not Path(value).expanduser().is_file():
                raise ValueError(f"{name} does not exist: {value}")
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = []
    for index, candidate in enumerate(candidates, start=1):
        trial_id = f"trial_{index:04d}"
        plan.append({"trial_id": trial_id, "parameters": asdict(candidate),
                     "command": training_command(args, candidate, root / trial_id)})
    manifest = {
        "schema_version": SCHEMA, "stage": args.stage, "space_size": space_size,
        "mode": args.mode, "search_seed": args.search_seed, "training_seed": args.seed,
        "code_sha256": _code_hash(),
        "config": _file_stamp(args.config), "road_initialization": _file_stamp(args.road_checkpoint),
        "tire_initialization": _file_stamp(args.initial_tire_checkpoint),
        "config_sha256": hashlib.sha256(Path(args.config).expanduser().read_bytes()).hexdigest()
        if args.config and Path(args.config).expanduser().is_file() else None,
        "optimizer": "AdamW", "scheduler": "cosine", "test_evaluated": False, "trials": plan,
    }
    manifest_path = root / "search_manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Search directory already contains a different plan. Choose a new OUTPUT_DIR; RESUME=1 only resumes an identical plan.")
    _write_json(manifest_path, manifest)
    print(f"{args.stage}: {len(plan)} configurations selected from {space_size}; AdamW + cosine; validation only.", flush=True)
    if args.dry_run:
        for trial in plan:
            print(shlex.join(trial["command"]))
        return
    results = []
    for trial, candidate in zip(plan, candidates):
        trial_dir = root / trial["trial_id"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        log = trial_dir / "train.log"
        row = {"trial_id": trial["trial_id"], **asdict(candidate),
               "is_table_reference": candidate == REFERENCES[args.stage], "log": str(log)}
        try:
            cached = None
            metrics_path = trial_dir / "metrics.json"
            if args.resume and metrics_path.is_file():
                try:
                    cached = _read_result(args, candidate, trial_dir)
                except (OSError, ValueError, KeyError):
                    metrics_path.replace(trial_dir / "metrics.previous.json")
            if cached is None:
                print(f"[{trial['trial_id']}] {asdict(candidate)}", flush=True)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write("\n" + shlex.join(trial["command"]) + "\n")
                    stream.flush()
                    process = subprocess.run(trial["command"], stdout=stream, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"Training exited with code {process.returncode}; inspect {log}")
            row.update(status="completed", **(cached if cached is not None else _read_result(args, candidate, trial_dir)))
            print(f"[{trial['trial_id']}] val_macro_f1={row['val_macro_f1']:.6f}", flush=True)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            row.update(status="failed", error=str(error))
            print(f"[{trial['trial_id']}] failed: {error}", file=sys.stderr, flush=True)
            results.append(row)
            _write_results(root, args.stage, results)
            if not args.continue_on_error:
                raise
            continue
        results.append(row)
        _write_results(root, args.stage, results)
    if not any(row["status"] == "completed" for row in results):
        raise RuntimeError("No successful trials; inspect results.csv and per-trial train.log files")
    print(f"Selection complete: {root / 'best.json'}", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        search(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Hyperparameter search failed: {error}\n")


if __name__ == "__main__":
    main()
