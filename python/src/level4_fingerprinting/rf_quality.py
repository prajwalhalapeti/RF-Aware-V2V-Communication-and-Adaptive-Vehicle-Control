from __future__ import annotations

import math
from typing import Mapping


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def compute_rf_quality(
    snr_db: float,
    packet_error_risk: float,
    nlos_probability: float,
    rms_delay_spread_s: float,
    device_confidence: float,
) -> float:
    delay_ns = float(rms_delay_spread_s) * 1.0e9
    score = (
        0.16 * (float(snr_db) - 12.0)
        - 2.00 * float(packet_error_risk)
        - 1.65 * float(nlos_probability)
        - 0.018 * delay_ns
        + 1.25 * float(device_confidence)
    )
    return max(0.0, min(1.0, sigmoid(score)))


def compute_rf_quality_from_metrics(metrics: Mapping[str, float]) -> float:
    return compute_rf_quality(
        snr_db=float(metrics.get("snr_db", 0.0)),
        packet_error_risk=float(metrics.get("packet_error_risk", 0.0)),
        nlos_probability=float(metrics.get("nlos_probability", metrics.get("nlos", 0.0))),
        rms_delay_spread_s=float(
            metrics.get("rms_delay_spread_s", metrics.get("delay_spread", 0.0))
        ),
        device_confidence=float(metrics.get("device_confidence", 0.0)),
    )
