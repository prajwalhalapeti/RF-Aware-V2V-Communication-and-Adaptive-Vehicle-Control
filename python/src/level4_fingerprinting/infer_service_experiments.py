"""
Deterministic V2V experiment inference server.

Condition codes:
  0 clean
  1 NLOS: same transmitter, NLOS I/Q selected before CNN
  2 jamming: interference injected into clean I/Q before CNN
  3 spoofing: different transmitter I/Q supplied before CNN

The ZeroMQ transaction always completes when the software path is healthy.
Wireless delivery is represented separately by `wireless_packet_ok`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
import zmq

from src.level4_fingerprinting.model import build_model
from src.level5_vehicle.rf_quality import compute_rf_quality


def safe_normalize(iq: np.ndarray) -> np.ndarray:
    x = np.asarray(iq, dtype=np.float32)
    if x.shape != (2, 1024):
        raise ValueError(f"I/Q shape must be [2,1024], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("I/Q contains NaN or Inf")
    x = x - x.mean(axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if not np.isfinite(rms) or rms < 1.0e-8:
        raise ValueError(f"Invalid I/Q RMS: {rms}")
    x = x / rms
    if not np.isfinite(x).all():
        raise ValueError("Normalized I/Q contains NaN or Inf")
    return x.astype(np.float32)


class FrameStore:
    def __init__(self, path: Path) -> None:
        self.h5 = h5py.File(path, "r")
        self.iq_key = next(
            (k for k in ("x", "iq", "iq_samples") if k in self.h5),
            None,
        )
        if self.iq_key is None:
            raise KeyError(f"No I/Q dataset. Keys: {list(self.h5.keys())}")

        self.n = int(self.h5[self.iq_key].shape[0])
        self.device = np.asarray(self.h5["device_id"][:], dtype=np.int64)

        if "nlos_probability" in self.h5:
            self.gt_nlos = np.asarray(
                self.h5["nlos_probability"][:], dtype=np.float64
            )
        elif "nlos" in self.h5:
            self.gt_nlos = np.asarray(self.h5["nlos"][:], dtype=np.float64)
        else:
            raise KeyError("HDF5 must contain nlos_probability or nlos")

        self.gt_snr = (
            np.asarray(self.h5["snr_db"][:], dtype=np.float64)
            if "snr_db" in self.h5
            else np.full(self.n, 20.0)
        )
        self.gt_delay = (
            np.asarray(self.h5["delay_spread"][:], dtype=np.float64)
            if "delay_spread" in self.h5
            else np.full(self.n, 1.0e-7)
        )

        valid_meta = (
            np.isfinite(self.gt_nlos)
            & np.isfinite(self.gt_snr)
            & np.isfinite(self.gt_delay)
        )
        self.valid_meta = valid_meta

    def pool(self, device_id: int, mode: str) -> np.ndarray:
        mask = self.valid_meta.copy()
        if mode == "clean":
            mask &= self.device == device_id
            mask &= self.gt_nlos <= 0.35
        elif mode == "nlos":
            mask &= self.device == device_id
            mask &= self.gt_nlos >= 0.65
        elif mode == "spoof":
            mask &= self.device != device_id
            mask &= self.gt_nlos <= 0.35
        else:
            raise ValueError(mode)
        return np.flatnonzero(mask)

    def read_valid(self, pool: np.ndarray, offset: int) -> tuple[np.ndarray, int]:
        if pool.size == 0:
            raise ValueError("Requested HDF5 pool is empty")
        for trial in range(min(50, pool.size)):
            idx = int(pool[(offset + trial) % pool.size])
            iq = np.asarray(self.h5[self.iq_key][idx], dtype=np.float32)
            try:
                safe_normalize(iq)
                return iq, idx
            except ValueError:
                continue
        raise ValueError("Could not find a finite nonzero I/Q frame in pool")

    def close(self) -> None:
        self.h5.close()


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu")
    for name, tensor in ckpt["model"].items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint contains non-finite tensor: {name}")

    model = build_model(
        num_devices=int(ckpt["num_devices"]),
        num_channel_domains=int(ckpt["num_channel_domains"]),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def add_jamming(
    iq: np.ndarray, severity: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    sev = float(np.clip(severity, 0.0, 1.0))
    target_sjr_db = 18.0 - 16.0 * sev
    signal_power = float(np.mean(iq.astype(np.float64) ** 2))
    jammer_power = signal_power / (10.0 ** (target_sjr_db / 10.0))
    jammer = rng.normal(
        0.0, math.sqrt(max(jammer_power, 1.0e-15)), size=iq.shape
    )
    return (iq + jammer.astype(np.float32)), target_sjr_db


def decode_model(
    model: torch.nn.Module, iq: np.ndarray, device: torch.device
) -> dict[str, float | int]:
    normalized = safe_normalize(iq)
    tensor = torch.as_tensor(
        normalized, dtype=torch.float32, device=device
    ).unsqueeze(0)

    start = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(tensor)
    inference_ms = (time.perf_counter_ns() - start) / 1.0e6

    for name, value in out.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Model output head is non-finite: {name}")

    probs = torch.softmax(out["device_logits"][0], dim=-1)
    conf, pred_id = torch.max(probs, dim=-1)
    metric = out["metric_regression"][0].detach().cpu()
    nlos = float(torch.sigmoid(out["nlos_logit"][0]).detach().cpu())

    snr = float(metric[0]) * 40.0
    per = float(torch.clamp(metric[1], 0.0, 1.0))
    delay = float(
        torch.pow(torch.tensor(10.0), torch.clamp(metric[2], -12.0, -5.0))
    )
    confidence = float(conf.detach().cpu())

    values = [snr, per, delay, nlos, confidence, inference_ms]
    if not np.isfinite(values).all():
        raise ValueError(f"Decoded model metrics are non-finite: {values}")

    return {
        "transmitter_id": int(pred_id.detach().cpu()),
        "device_confidence": confidence,
        "snr_db": snr,
        "packet_error_risk": per,
        "rms_delay_spread_s": delay,
        "nlos_probability": float(np.clip(nlos, 0.0, 1.0)),
        "inference_ms": inference_ms,
    }


def deterministic_rng(seed: int, packet_index: int, condition: int) -> np.random.Generator:
    combined = (
        int(seed) * 1_000_003
        + int(packet_index) * 9_176
        + int(condition) * 104_729
    ) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(combined)


def error_response(sequence: int, message: str, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_ok": 0,
        "sequence": int(sequence),
        "packet_index": int(request.get("packet_index", -1)),
        "actual_hdf5_index": -1,
        "transmitter_id": -1,
        "gt_device_id": -1,
        "expected_device_id": int(request.get("expected_device_id", -1)),
        "device_confidence": 0.0,
        "snr_db": -60.0,
        "packet_error_risk": 1.0,
        "rms_delay_spread_s": 1.0e-6,
        "nlos_probability": 0.5,
        "gt_nlos": 0.0,
        "rf_quality": 0.0,
        "wireless_packet_ok": 0,
        "inference_ms": 0.0,
        "attack_code": int(request.get("attack_code", 0)),
        "attack_active": int(request.get("attack_active", 0)),
        "severity": float(request.get("severity", 0.0)),
        "error": message[:300],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model).expanduser().resolve(), device)
    store = FrameStore(Path(args.hdf5).expanduser().resolve())

    warmup = torch.ones((1, 2, 1024), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(3):
            model(warmup)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://127.0.0.1:{args.port}")

    sequence = 0
    print(
        f"[READY] port={args.port} frames={store.n} device={device}",
        flush=True,
    )

    try:
        while True:
            request = socket.recv_json()
            try:
                packet_index = int(request["packet_index"])
                condition = int(request.get("attack_code", 0))
                active = int(request.get("attack_active", 0)) != 0
                severity = float(np.clip(request.get("severity", 0.0), 0.0, 1.0))
                run_seed = int(request.get("run_seed", 1))
                expected_device = int(request.get("expected_device_id", 0))
                rng = deterministic_rng(run_seed, packet_index, condition)

                mode = "clean"
                if active and condition == 1:
                    mode = "nlos"
                elif active and condition == 3:
                    mode = "spoof"

                pool = store.pool(expected_device, mode)
                offset = packet_index + run_seed * 997
                iq, actual_idx = store.read_valid(pool, offset)

                if active and condition == 2:
                    iq, _ = add_jamming(iq, severity, rng)

                pred = decode_model(model, iq, device)
                gt_device = int(store.device[actual_idx])
                gt_nlos = float(np.clip(store.gt_nlos[actual_idx], 0.0, 1.0))

                model_per = float(pred["packet_error_risk"])
                attack_extra_per = 0.0
                if active and condition == 1:
                    attack_extra_per = 0.02 + 0.23 * severity
                elif active and condition == 2:
                    attack_extra_per = 0.05 + 0.65 * severity

                effective_per = 1.0 - (1.0 - model_per) * (1.0 - attack_extra_per)
                effective_per = float(np.clip(effective_per, 0.0, 0.999))

                rf_quality = compute_rf_quality(
                    snr_db=float(pred["snr_db"]),
                    packet_error_risk=effective_per,
                    nlos_probability=float(pred["nlos_probability"]),
                    rms_delay_spread_s=float(pred["rms_delay_spread_s"]),
                    device_confidence=float(pred["device_confidence"]),
                )
                if not np.isfinite(rf_quality):
                    raise ValueError("Computed RF quality is non-finite")

                wireless_ok = int(rng.random() >= effective_per)

                response = {
                    "server_ok": 1,
                    "sequence": sequence,
                    "packet_index": packet_index,
                    "actual_hdf5_index": actual_idx,
                    "transmitter_id": int(pred["transmitter_id"]),
                    "gt_device_id": gt_device,
                    "expected_device_id": expected_device,
                    "device_confidence": float(pred["device_confidence"]),
                    "snr_db": float(pred["snr_db"]),
                    "packet_error_risk": effective_per,
                    "rms_delay_spread_s": float(pred["rms_delay_spread_s"]),
                    "nlos_probability": float(pred["nlos_probability"]),
                    "gt_nlos": gt_nlos,
                    "rf_quality": float(np.clip(rf_quality, 0.0, 1.0)),
                    "wireless_packet_ok": wireless_ok,
                    "inference_ms": float(pred["inference_ms"]),
                    "attack_code": condition if active else 0,
                    "attack_active": int(active),
                    "severity": severity,
                }

                socket.send_json(response, allow_nan=False)

                if args.log_every > 0 and sequence % args.log_every == 0:
                    print(
                        "[REQ] "
                        f"seq={sequence} pkt={packet_index} idx={actual_idx} "
                        f"pred={response['transmitter_id']} gt={gt_device} "
                        f"q={response['rf_quality']:.3f} "
                        f"nlos={response['nlos_probability']:.3f} "
                        f"wok={wireless_ok} "
                        f"lat={response['inference_ms']:.2f}ms",
                        flush=True,
                    )
                sequence += 1

            except Exception as exc:
                socket.send_json(
                    error_response(sequence, str(exc), request),
                    allow_nan=False,
                )
                print(f"[ERROR] seq={sequence}: {exc}", flush=True)
                sequence += 1

    except KeyboardInterrupt:
        print("\n[STOP]", flush=True)
    finally:
        store.close()
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()