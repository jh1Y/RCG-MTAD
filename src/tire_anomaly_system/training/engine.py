"""Shared raw-input training, joint losses and prediction exports."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from tire_anomaly_system.data import ANOMALY_CLASS_NAMES, make_loader
from tire_anomaly_system.models.road_encoder import SOURCE_ROAD_CLASSES


def resolve_device(value):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def batch_inputs(batch, device):
    return tuple(batch[key].to(device) for key in ("image", "tire_acc", "gyro", "vehicle_state"))


def classification_metrics(truth, predicted, class_names):
    truth, predicted = np.asarray(truth, dtype=np.int64), np.asarray(predicted, dtype=np.int64)
    valid = (truth >= 0) & (truth < len(class_names))
    truth, predicted = truth[valid], predicted[valid]
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    support = matrix.sum(1)
    precision = np.divide(matrix.diagonal(), matrix.sum(0), out=np.zeros(len(class_names)), where=matrix.sum(0) > 0)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(len(class_names)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(class_names)), where=(precision + recall) > 0)
    return {
        "sample_count": int(matrix.sum()),
        "accuracy": float(matrix.diagonal().sum() / max(matrix.sum(), 1)),
        "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
    }


@torch.no_grad()
def fit_normalizers(model, dataset, *, device, batch_size=32, workers=0,
                    stage="joint", keep_road_normalizer=False):
    """Fit only from train; P-T calibration is further restricted to healthy rows."""
    sensor_parts = {key: [] for key in ("tire_acc", "gyro", "vehicle_state")}
    labels = []
    # Metadata/sensors can be read without decoding all images twice.
    for index in range(len(dataset)):
        if hasattr(dataset, "sensor_item"):
            row = dataset.sensor_item(int(index))
        else:
            row = dataset[int(index)]
        for key in sensor_parts:
            sensor_parts[key].append(row[key])
        labels.append(int(row.get("anomaly_class", -1)))
    acc, imu, state = (torch.stack(sensor_parts[key]) for key in sensor_parts)
    acc, imu, state = model.preprocessor.prepare(acc, imu, state)
    labels = torch.tensor(labels, dtype=torch.long)
    healthy = labels == 0
    if int(healthy.sum()) < 2:
        raise ValueError("At least two healthy training samples are required")
    if not keep_road_normalizer:
        model.preprocessor.cpu().fit(acc[healthy], imu[healthy], state[healthy])
    elif not bool(model.preprocessor.fitted):
        raise ValueError("Loaded road checkpoint is missing fitted preprocessing")
    model.to(device)
    if stage == "joint":
        previous_mode = model.training
        model.eval()
        embeddings = []
        loader = make_loader(dataset, batch_size=batch_size, workers=workers)
        for batch in loader:
            road = model.encode_road(*batch_inputs(batch, device))
            embeddings.append(road.zroad.detach().cpu())
        cfg = model.config
        model.tire_classifier.cpu().fit_normalizer(
            acc, imu[:, cfg.inputs.gyro_indices, :],
            state[:, [cfg.tire.pressure_state_index, cfg.tire.temperature_state_index]],
            torch.cat(embeddings), labels,
        )
        model.to(device).train(previous_mode)


def run_epoch(model, loader, device, *, stage="joint", optimizer=None,
              road_criterion=None, tire_criterion=None, road_loss_weight=1.0):
    training = optimizer is not None
    model.train(training)
    total, total_road, total_tire, count = 0.0, 0.0, 0.0, 0
    road_truth, road_pred, tire_truth, tire_pred = [], [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            if stage == "road":
                road = model.encode_road(*batch_inputs(batch, device))
                tire = None
            else:
                road, tire = model(*batch_inputs(batch, device))
            road_labels = batch["road_label"].to(device)
            road_loss = road_criterion(road.logits, road_labels)
            tire_loss = road_loss.new_zeros(())
            if tire is not None:
                tire_labels = batch["anomaly_class"].to(device)
                tire_loss = tire_criterion(tire.logits, tire_labels)
            loss = road_loss if stage == "road" else tire_loss + road_loss_weight * road_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Training produced a non-finite loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            size = len(road_labels)
            total += float(loss.detach()) * size
            total_road += float(road_loss.detach()) * size
            total_tire += float(tire_loss.detach()) * size
            count += size
            road_truth.extend(batch["road_label"].tolist())
            road_pred.extend(road.logits.detach().argmax(1).cpu().tolist())
            if tire is not None:
                tire_truth.extend(batch["anomaly_class"].tolist())
                tire_pred.extend(tire.logits.detach().argmax(1).cpu().tolist())
    if not count:
        raise ValueError("No batches were processed")
    result = {"loss": total / count, "road_loss": total_road / count,
              "road": classification_metrics(road_truth, road_pred, SOURCE_ROAD_CLASSES)}
    if stage == "joint":
        result.update({"tire_loss": total_tire / count,
                       "tire": classification_metrics(tire_truth, tire_pred, ANOMALY_CLASS_NAMES)})
    return result


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    results = []
    for batch in loader:
        road, tire = model(*batch_inputs(batch, device))
        rp, tp = road.probabilities.cpu(), tire.probabilities.cpu()
        pt = tire.pressure_temperature_features.cpu()
        for index in range(len(rp)):
            ri, ti = int(rp[index].argmax()), int(tp[index].argmax())
            row = {key: batch[key][index] for key in ("sample_id", "group_id", "timestamp")}
            row.update({
                "true_road": int(batch["road_label"][index]),
                "true_class": int(batch["anomaly_class"][index]),
                "predicted_road": ri, "road_name": SOURCE_ROAD_CLASSES[ri],
                "road_confidence": float(rp[index, ri]),
                "predicted_class": ti, "tire_state": ANOMALY_CLASS_NAMES[ti],
                "tire_confidence": float(tp[index, ti]),
            })
            row.update({f"road_prob_{name}": float(rp[index, i]) for i, name in enumerate(SOURCE_ROAD_CLASSES)})
            row.update({f"prob_{name}": float(tp[index, i]) for i, name in enumerate(ANOMALY_CLASS_NAMES)})
            cfg = model.config.tire
            pt_names = {
                "coupled": ("temperature_corrected_pressure_kpa", "expected_pressure_kpa", "pressure_temperature_residual_z", "relative_pressure_temperature_residual"),
                "raw": ("pressure_kpa", "temperature_c"), "pressure_only": ("pressure_kpa",),
                "temperature_only": ("temperature_c",), "corrected_pressure": ("temperature_corrected_pressure_kpa",),
                "residual": ("pressure_temperature_residual_z", "relative_pressure_temperature_residual"),
            }[cfg.pressure_temperature_mode]
            row.update({name: float(pt[index, i]) for i, name in enumerate(pt_names) if i < pt.shape[1]})
            results.append(row)
    return results


def prediction_metrics(rows):
    return {
        "road": classification_metrics([row["true_road"] for row in rows], [row["predicted_road"] for row in rows], SOURCE_ROAD_CLASSES),
        "tire": classification_metrics([row["true_class"] for row in rows], [row["predicted_class"] for row in rows], ANOMALY_CLASS_NAMES),
    }


def write_predictions(path, rows):
    if not rows:
        raise ValueError("No predictions to export")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
