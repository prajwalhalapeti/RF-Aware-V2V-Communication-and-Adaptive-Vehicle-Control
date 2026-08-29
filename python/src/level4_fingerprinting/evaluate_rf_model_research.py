#!/usr/bin/env python3
"""
Research-oriented evaluation for the existing multi-head V2V RF model.

Important:
- This script measures accuracy on the HDF5 file you provide.
- It is a true held-out test only when that HDF5 file, or its source frames,
  were not used during training.
- It does not use packet-error or coherence-bandwidth outputs because the
  current training loss can default those targets to zero when metric_targets
  are absent.

Example:
python -u evaluate_rf_model_research.py \
  --model results/checkpoints/best.pt \
  --hdf5 data/processed/synthetic_v2v_finite.h5 \
  --device cpu \
  --batch-size 128 \
  --max-samples 0 \
  --out-dir results/rf_model_evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch

from src.level4_fingerprinting.model import build_model


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_normalize_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (2, 1024):
        raise ValueError(f"Expected [N,2,1024], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Batch contains NaN or Inf")

    x = x - x.mean(axis=2, keepdims=True)
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
    if not np.isfinite(rms).all() or np.any(rms < 1.0e-8):
        raise ValueError("Batch contains zero/invalid RMS frames")

    x = x / rms
    if not np.isfinite(x).all():
        raise ValueError("Normalized batch contains NaN or Inf")
    return x.astype(np.float32)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def multiclass_scores(cm: np.ndarray) -> dict[str, float]:
    support = cm.sum(axis=1)
    f1s = []
    recalls = []
    precisions = []

    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])

        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)

        if support[c] > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    accuracy = float(np.trace(cm) / max(cm.sum(), 1))
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "macro_recall_balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
    }


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = y_true.astype(bool)
    pred = y_pred.astype(bool)

    tp = int(np.sum(truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    truth = y_true.astype(bool)
    n_pos = int(truth.sum())
    n_neg = int((~truth).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop

    rank_sum_pos = float(ranks[truth].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denom) if denom > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def chunks(indices: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 evaluates all rows; otherwise uses evenly spaced rows.",
    )
    parser.add_argument("--latency-samples", type=int, default=300)
    parser.add_argument("--out-dir", default="results/rf_model_evaluation")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    hdf5_path = Path(args.hdf5).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    checkpoint = torch.load(model_path, map_location="cpu")

    model = build_model(
        num_devices=int(checkpoint["num_devices"]),
        num_channel_domains=int(checkpoint["num_channel_domains"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    with h5py.File(hdf5_path, "r") as h5:
        iq_key = next((k for k in ("x", "iq", "iq_samples") if k in h5), None)
        if iq_key is None:
            raise KeyError(f"No I/Q dataset found. Keys: {list(h5.keys())}")

        required = ["device_id", "snr_db", "delay_spread"]
        for key in required:
            if key not in h5:
                raise KeyError(f"Missing required key '{key}'")

        nlos_key = (
            "nlos_probability"
            if "nlos_probability" in h5
            else "nlos" if "nlos" in h5 else None
        )
        if nlos_key is None:
            raise KeyError("Missing nlos_probability/nlos")

        n_rows = int(h5[iq_key].shape[0])
        if args.max_samples > 0 and args.max_samples < n_rows:
            indices = np.unique(
                np.linspace(0, n_rows - 1, args.max_samples, dtype=np.int64)
            )
        else:
            indices = np.arange(n_rows, dtype=np.int64)

        true_device_parts = []
        pred_device_parts = []
        confidence_parts = []
        true_nlos_parts = []
        pred_nlos_prob_parts = []
        true_snr_parts = []
        pred_snr_parts = []
        true_delay_parts = []
        pred_delay_parts = []
        valid_indices_parts = []

        failed = []

        for idx_batch in chunks(indices, args.batch_size):
            try:
                raw = np.asarray(h5[iq_key][idx_batch], dtype=np.float32)
                iq = safe_normalize_batch(raw)
                tensor = torch.as_tensor(iq, dtype=torch.float32, device=device)

                with torch.inference_mode():
                    outputs = model(tensor)

                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                        raise ValueError(f"Non-finite model output: {name}")

                probs = torch.softmax(outputs["device_logits"], dim=-1)
                confidence, pred_device = torch.max(probs, dim=-1)
                metric = outputs["metric_regression"].detach().cpu().numpy()
                pred_nlos_prob = (
                    torch.sigmoid(outputs["nlos_logit"]).detach().cpu().numpy()
                )

                pred_snr = metric[:, 0] * 40.0
                pred_delay = np.power(
                    10.0, np.clip(metric[:, 2], -12.0, -5.0)
                )

                true_device_parts.append(np.asarray(h5["device_id"][idx_batch], dtype=np.int64))
                pred_device_parts.append(pred_device.detach().cpu().numpy().astype(np.int64))
                confidence_parts.append(confidence.detach().cpu().numpy().astype(np.float64))
                true_nlos_parts.append(np.asarray(h5[nlos_key][idx_batch], dtype=np.float64))
                pred_nlos_prob_parts.append(np.asarray(pred_nlos_prob, dtype=np.float64))
                true_snr_parts.append(np.asarray(h5["snr_db"][idx_batch], dtype=np.float64))
                pred_snr_parts.append(np.asarray(pred_snr, dtype=np.float64))
                true_delay_parts.append(np.asarray(h5["delay_spread"][idx_batch], dtype=np.float64))
                pred_delay_parts.append(np.asarray(pred_delay, dtype=np.float64))
                valid_indices_parts.append(idx_batch)

            except Exception as exc:
                failed.append({"batch_start": int(idx_batch[0]), "error": str(exc)})

        if not valid_indices_parts:
            raise RuntimeError(f"No valid evaluation batches. First failures: {failed[:3]}")

        valid_indices = np.concatenate(valid_indices_parts)
        y_device = np.concatenate(true_device_parts)
        p_device = np.concatenate(pred_device_parts)
        confidence = np.concatenate(confidence_parts)
        y_nlos_prob = np.concatenate(true_nlos_parts)
        p_nlos_prob = np.concatenate(pred_nlos_prob_parts)
        y_snr = np.concatenate(true_snr_parts)
        p_snr = np.concatenate(pred_snr_parts)
        y_delay = np.concatenate(true_delay_parts)
        p_delay = np.concatenate(pred_delay_parts)

        n_devices = int(checkpoint["num_devices"])
        cm = confusion_matrix(y_device, p_device, n_devices)
        device_metrics = multiclass_scores(cm)

        y_nlos = y_nlos_prob >= 0.5
        p_nlos = p_nlos_prob >= 0.5
        nlos_metrics = binary_scores(y_nlos, p_nlos)
        nlos_metrics["roc_auc"] = roc_auc_rank(y_nlos, p_nlos_prob)

        snr_metrics = regression_scores(y_snr, p_snr)

        y_log_delay = np.log10(np.clip(y_delay, 1.0e-12, None))
        p_log_delay = np.log10(np.clip(p_delay, 1.0e-12, None))
        delay_metrics = regression_scores(y_log_delay, p_log_delay)
        delay_metrics["units"] = "log10(seconds)"

        # Single-frame latency benchmark, matching deployment behavior.
        latency_indices = valid_indices[: min(args.latency_samples, len(valid_indices))]
        latency_ms = []
        warmup_count = min(10, len(latency_indices))

        for j, idx in enumerate(latency_indices):
            iq = safe_normalize_batch(
                np.asarray(h5[iq_key][idx : idx + 1], dtype=np.float32)
            )
            tensor = torch.as_tensor(iq, dtype=torch.float32, device=device)

            start = time.perf_counter_ns()
            with torch.inference_mode():
                _ = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6

            if j >= warmup_count:
                latency_ms.append(elapsed_ms)

    summary = {
        "model": str(model_path),
        "hdf5": str(hdf5_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_args": checkpoint.get("args"),
        "device": str(device),
        "evaluated_samples": int(len(valid_indices)),
        "failed_batches": failed,
        "device_identification": device_metrics,
        "mean_device_confidence": float(np.mean(confidence)),
        "nlos_detection": nlos_metrics,
        "snr_regression_db": snr_metrics,
        "delay_spread_regression": delay_metrics,
        "single_frame_latency_ms": {
            "n": len(latency_ms),
            "median": float(np.median(latency_ms)) if latency_ms else float("nan"),
            "p95": float(np.percentile(latency_ms, 95)) if latency_ms else float("nan"),
            "max": float(np.max(latency_ms)) if latency_ms else float("nan"),
        },
        "warning": (
            "These are held-out results only if this HDF5 file/source frames "
            "were excluded from training."
        ),
    }

    with (out_dir / "rf_model_eval_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    np.savetxt(
        out_dir / "device_confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    with (out_dir / "rf_model_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hdf5_index",
                "true_device_id",
                "pred_device_id",
                "device_confidence",
                "true_nlos_probability",
                "pred_nlos_probability",
                "true_snr_db",
                "pred_snr_db",
                "true_delay_spread_s",
                "pred_delay_spread_s",
            ]
        )
        for row in zip(
            valid_indices,
            y_device,
            p_device,
            confidence,
            y_nlos_prob,
            p_nlos_prob,
            y_snr,
            p_snr,
            y_delay,
            p_delay,
        ):
            writer.writerow(row)

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"\n[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
