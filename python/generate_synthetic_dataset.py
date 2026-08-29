"""
src/generate_synthetic_dataset.py
===================================
Step 6 — Synthetic Data Generator.

Pipeline per frame:
    raw I/Q  -->  Level 1 (path loss, random distance 2-62 ft)
             -->  Level 2 (USRP hardware impairment, random device profile)
             -->  Level 3 (multipath TDL + Rayleigh/Rician fading + AWGN)
             -->  x  [2, 1024] float32  (final corrupted frame)

Balancing contract:
    Every (device_id, channel_type, distance_bucket) combination is visited
    with near-uniform frequency via a pre-built combinatorial index that is
    shuffled once and then cycled/repeated to reach the target frame count.
    This decouples device fingerprint (constant per device) from channel /
    distance artifacts (variable), which is the entire point of the dataset.

Output: data/processed/synthetic_v2v.h5
    x                    [N, 2, 1024]  float32
    device_id            [N]           int32
    distance_ft           [N]           float32
    channel_type          [N]           string (variable-length utf-8)
    snr_db                [N]           float32
    delay_spread          [N]           float32   (RMS delay spread, seconds)
    nlos_probability      [N]           float32
    impairment_profile_id [N]           string (variable-length utf-8)

Usage:
    python src/generate_synthetic_dataset.py --n-frames 50000
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make `src` importable when run as `python src/generate_synthetic_dataset.py`
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.level1_physics.path_loss import apply_distance_attenuation
from src.level2_rf_frontend.usrp_profiles import apply_usrp_frontend
from src.level3_channel.tapped_delay_line import (
    apply_tapped_delay_line,
    delays_to_sample_offsets,
    gains_db_to_linear,
    load_channel_config,
)
from src.level3_channel.rayleigh import apply_rayleigh_fading, add_awgn
from src.level3_channel.rician import apply_rician_fading

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLES_PER_FRAME = 1024
DISTANCE_MIN_FT = 2.0
DISTANCE_MAX_FT = 62.0
DISTANCE_BUCKETS_FT = [2.0, 12.0, 22.0, 32.0, 42.0, 52.0, 62.0]  # 7 strata across the range

DEFAULT_INPUT_H5 = os.path.join("data", "processed", "oracle.h5")
DEFAULT_OUTPUT_H5 = os.path.join("data", "processed", "synthetic_v2v.h5")
DEFAULT_CHANNEL_CONFIG = os.path.join("src", "level3_channel", "channel_config.yaml")

# RAYLEIGH_K_FACTOR_THRESHOLD: environments at/below this K are treated as
# near-Rayleigh (NLOS-dominant) and routed to apply_rayleigh_fading instead
# of apply_rician_fading, per the contract's "Rayleigh or Rician depending
# on environment" instruction. Urban_NLOS (K=0.5) falls below this; the
# other two environments are routed through Rician.
RAYLEIGH_K_FACTOR_THRESHOLD = 1.0

CHUNK_SIZE = 2048  # rows per HDF5 flush


# ===========================================================================
# USRP profile discovery
# ===========================================================================

def discover_usrp_profile_ids() -> list[str]:
    """
    Import the profile registry from usrp_profiles.py and return the list
    of available profile_id strings. Falls back to a documented attribute
    name search so this script doesn't hard-fail if the module exposes the
    registry under a slightly different name.
    """
    import src.level2_rf_frontend.usrp_profiles as usrp_mod

    for attr_name in ("USRP_PROFILES", "PROFILES", "PROFILE_IDS", "USRP_PROFILE_IDS"):
        if hasattr(usrp_mod, attr_name):
            registry = getattr(usrp_mod, attr_name)
            if isinstance(registry, dict):
                return sorted(registry.keys())
            if isinstance(registry, (list, tuple)):
                return sorted(registry)

    raise AttributeError(
        "Could not find a profile registry (expected one of "
        "USRP_PROFILES / PROFILES / PROFILE_IDS / USRP_PROFILE_IDS) "
        "in src.level2_rf_frontend.usrp_profiles. "
        "Update discover_usrp_profile_ids() to match the actual export name."
    )


# ===========================================================================
# Channel environment helpers
# ===========================================================================

def compute_rms_delay_spread(delays_s: list[float], gains_db: list[float]) -> float:
    """
    Compute the RMS delay spread of a multipath profile:

        tau_rms = sqrt( sum(P_i * (tau_i - tau_mean)^2) / sum(P_i) )

    where P_i is linear power (gain^2) of tap i and tau_i its delay in seconds.
    """
    delays = np.asarray(delays_s, dtype=np.float64)
    powers = (10.0 ** (np.asarray(gains_db, dtype=np.float64) / 10.0))  # power, not amplitude
    tau_mean = np.sum(powers * delays) / np.sum(powers)
    tau_rms = np.sqrt(np.sum(powers * (delays - tau_mean) ** 2) / np.sum(powers))
    return float(tau_rms)


class ChannelEnvironment:
    """Precomputed, ready-to-apply channel parameters for one named environment."""

    def __init__(self, name: str, cfg_entry: dict, sample_rate_hz: float):
        self.name = name
        self.k_factor = float(cfg_entry["k_factor"])
        self.snr_db_nominal = float(cfg_entry["snr_db_nominal"])
        self.delays_s = cfg_entry["delays_s"]
        self.gains_db = cfg_entry["gains_db"]
        self.sample_offsets = delays_to_sample_offsets(self.delays_s, sample_rate_hz)
        self.linear_gains = gains_db_to_linear(self.gains_db)
        self.delay_spread = compute_rms_delay_spread(self.delays_s, self.gains_db)

        # NLOS probability heuristic derived from K-factor: low K -> high NLOS.
        # Bounded to [0, 1] via a simple saturating map; K=0 -> 1.0 (pure NLOS),
        # K>=10 -> ~0.05 (near pure LOS).
        self.nlos_probability = float(np.clip(1.0 / (1.0 + self.k_factor), 0.0, 1.0))

    def apply_fading(self, iq_frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.k_factor <= RAYLEIGH_K_FACTOR_THRESHOLD:
            return apply_rayleigh_fading(iq_frame, rng)
        return apply_rician_fading(iq_frame, self.k_factor, rng)


def load_environments(config_path: str, sample_rate_hz: float) -> dict[str, ChannelEnvironment]:
    cfg = load_channel_config(config_path)
    envs = {}
    for name, entry in cfg["environments"].items():
        envs[name] = ChannelEnvironment(name, entry, sample_rate_hz)
    return envs


# ===========================================================================
# Balanced combinatorial sampling plan
# ===========================================================================

def build_balanced_plan(
    device_ids: list[int],
    channel_names: list[str],
    distance_buckets_ft: list[float],
    n_frames: int,
    rng: np.random.Generator,
) -> list[tuple[int, str, float]]:
    """
    Build a list of (device_id, channel_type, distance_ft) tuples of length
    n_frames such that the full Cartesian product of
    (device, channel, distance_bucket) is cycled through as evenly as
    possible, then shuffled.

    This guarantees:
      * every device appears with every channel type
      * every device appears across the full distance range
      * every channel type is paired with every device
    """
    base_combos = list(itertools.product(device_ids, channel_names, distance_buckets_ft))
    n_base = len(base_combos)

    n_full_cycles = n_frames // n_base
    remainder = n_frames % n_base

    plan: list[tuple[int, str, float]] = []
    for _ in range(n_full_cycles):
        cycle = base_combos.copy()
        rng.shuffle(cycle)
        plan.extend(cycle)

    if remainder > 0:
        tail = base_combos.copy()
        rng.shuffle(tail)
        plan.extend(tail[:remainder])

    rng.shuffle(plan)
    return plan


def jitter_distance(distance_bucket_ft: float, rng: np.random.Generator) -> float:
    """
    Add small uniform jitter around a distance bucket center so the dataset
    doesn't collapse onto exactly 7 discrete distance values, while staying
    within [DISTANCE_MIN_FT, DISTANCE_MAX_FT].
    """
    jitter_range = 4.0  # +/- 4 ft
    jittered = distance_bucket_ft + rng.uniform(-jitter_range, jitter_range)
    return float(np.clip(jittered, DISTANCE_MIN_FT, DISTANCE_MAX_FT))


# ===========================================================================
# Source frame loading (cycles through oracle.h5 if N exceeds available frames)
# ===========================================================================

class SourceFrameProvider:
    """Lazily reads raw I/Q frames from oracle.h5, cycling if exhausted."""

    def __init__(self, h5_path: str, rng: np.random.Generator):
        self.h5_path = h5_path
        self.rng = rng
        self._file = h5py.File(h5_path, "r")
        self._iq = self._file["iq"]
        self._n_available = self._iq.shape[0]
        self._order = self.rng.permutation(self._n_available)
        self._cursor = 0
        log.info("SourceFrameProvider: %d raw frames available in %s",
                  self._n_available, h5_path)

    def next_frame(self) -> np.ndarray:
        if self._cursor >= self._n_available:
            self._order = self.rng.permutation(self._n_available)
            self._cursor = 0
        idx = self._order[self._cursor]
        self._cursor += 1
        return np.array(self._iq[idx], dtype=np.float32)  # [2, 1024]

    def close(self):
        self._file.close()


# ===========================================================================
# Per-frame pipeline
# ===========================================================================

def generate_one_frame(
    raw_frame: np.ndarray,
    device_id: int,
    profile_id: str,
    channel_env: ChannelEnvironment,
    distance_ft: float,
    rng: np.random.Generator,
) -> dict:
    """
    Run one raw I/Q frame through Level 1 -> Level 2 -> Level 3 and return
    the final frame plus the full metadata contract dict.
    """
    # ---- Level 1: distance-based path loss ----
    frame = apply_distance_attenuation(raw_frame, distance_ft)

    # ---- Level 2: USRP hardware impairment (bakes in device fingerprint) ----
    frame = apply_usrp_frontend(frame, profile_id)

    # ---- Level 3: multipath TDL ----
    frame = apply_tapped_delay_line(frame, channel_env.sample_offsets, channel_env.linear_gains)

    # ---- Level 3: fading (Rayleigh or Rician depending on environment K) ----
    frame = channel_env.apply_fading(frame, rng)

    # ---- Level 3: AWGN at the environment's nominal SNR with small jitter ----
    snr_db = float(channel_env.snr_db_nominal + rng.normal(0.0, 1.5))
    frame = add_awgn(frame, snr_db, rng)

    return {
        "x": frame.astype(np.float32),
        "device_id": np.int32(device_id),
        "distance_ft": np.float32(distance_ft),
        "channel_type": channel_env.name,
        "snr_db": np.float32(snr_db),
        "delay_spread": np.float32(channel_env.delay_spread),
        "nlos_probability": np.float32(channel_env.nlos_probability),
        "impairment_profile_id": profile_id,
    }


# ===========================================================================
# HDF5 output writer
# ===========================================================================

def create_output_h5(output_path: str, n_frames: int) -> h5py.File:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f = h5py.File(output_path, "w")

    str_dtype = h5py.string_dtype(encoding="utf-8")
    chunk = min(CHUNK_SIZE, n_frames)

    f.create_dataset("x", shape=(n_frames, 2, SAMPLES_PER_FRAME), dtype=np.float32,
                      chunks=(chunk, 2, SAMPLES_PER_FRAME), compression="gzip", compression_opts=4)
    f.create_dataset("device_id", shape=(n_frames,), dtype=np.int32,
                      chunks=(chunk,), compression="gzip", compression_opts=4)
    f.create_dataset("distance_ft", shape=(n_frames,), dtype=np.float32,
                      chunks=(chunk,), compression="gzip", compression_opts=4)
    f.create_dataset("channel_type", shape=(n_frames,), dtype=str_dtype,
                      chunks=(chunk,))
    f.create_dataset("snr_db", shape=(n_frames,), dtype=np.float32,
                      chunks=(chunk,), compression="gzip", compression_opts=4)
    f.create_dataset("delay_spread", shape=(n_frames,), dtype=np.float32,
                      chunks=(chunk,), compression="gzip", compression_opts=4)
    f.create_dataset("nlos_probability", shape=(n_frames,), dtype=np.float32,
                      chunks=(chunk,), compression="gzip", compression_opts=4)
    f.create_dataset("impairment_profile_id", shape=(n_frames,), dtype=str_dtype,
                      chunks=(chunk,))
    return f


# ===========================================================================
# Main generation routine
# ===========================================================================

def generate(
    n_frames: int,
    input_h5: str,
    output_h5: str,
    channel_config_path: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    log.info("Loading channel environments from: %s", channel_config_path)
    cfg = load_channel_config(channel_config_path)
    sample_rate_hz = float(cfg["sample_rate_hz"])
    environments = load_environments(channel_config_path, sample_rate_hz)
    channel_names = sorted(environments.keys())
    log.info("Loaded %d channel environments: %s", len(channel_names), channel_names)

    log.info("Discovering USRP impairment profiles ...")
    profile_ids = discover_usrp_profile_ids()
    log.info("Found %d USRP profiles: %s", len(profile_ids), profile_ids)

    # device_id is the integer index into profile_ids; the profile_id string
    # IS the hardware fingerprint identity, so device count == profile count.
    device_ids = list(range(len(profile_ids)))

    log.info("Building balanced combinatorial sampling plan for %d frames ...", n_frames)
    plan = build_balanced_plan(device_ids, channel_names, DISTANCE_BUCKETS_FT, n_frames, rng)
    log.info("Plan built: %d entries (base combinatorial size = %d)",
              len(plan), len(device_ids) * len(channel_names) * len(DISTANCE_BUCKETS_FT))

    log.info("Opening source frame provider: %s", input_h5)
    source = SourceFrameProvider(input_h5, rng)

    log.info("Creating output HDF5: %s", output_h5)
    out = create_output_h5(output_h5, n_frames)

    # Per-write buffers
    buf_x, buf_dev, buf_dist, buf_chan = [], [], [], []
    buf_snr, buf_delay, buf_nlos, buf_profile = [], [], [], []
    write_cursor = 0

    def flush_buffers():
        nonlocal write_cursor, buf_x, buf_dev, buf_dist, buf_chan
        nonlocal buf_snr, buf_delay, buf_nlos, buf_profile
        n = len(buf_x)
        if n == 0:
            return
        sl = slice(write_cursor, write_cursor + n)
        out["x"][sl] = np.stack(buf_x, axis=0)
        out["device_id"][sl] = np.array(buf_dev, dtype=np.int32)
        out["distance_ft"][sl] = np.array(buf_dist, dtype=np.float32)
        out["channel_type"][sl] = buf_chan
        out["snr_db"][sl] = np.array(buf_snr, dtype=np.float32)
        out["delay_spread"][sl] = np.array(buf_delay, dtype=np.float32)
        out["nlos_probability"][sl] = np.array(buf_nlos, dtype=np.float32)
        out["impairment_profile_id"][sl] = buf_profile
        write_cursor += n
        buf_x, buf_dev, buf_dist, buf_chan = [], [], [], []
        buf_snr, buf_delay, buf_nlos, buf_profile = [], [], [], []

    log.info("Generating %d synthetic frames ...", n_frames)
    for device_id, channel_name, distance_bucket_ft in tqdm(plan, total=n_frames, desc="Synthesizing", unit="frame"):
        raw_frame = source.next_frame()
        profile_id = profile_ids[device_id]
        distance_ft = jitter_distance(distance_bucket_ft, rng)
        env = environments[channel_name]

        sample = generate_one_frame(raw_frame, device_id, profile_id, env, distance_ft, rng)

        buf_x.append(sample["x"])
        buf_dev.append(sample["device_id"])
        buf_dist.append(sample["distance_ft"])
        buf_chan.append(sample["channel_type"])
        buf_snr.append(sample["snr_db"])
        buf_delay.append(sample["delay_spread"])
        buf_nlos.append(sample["nlos_probability"])
        buf_profile.append(sample["impairment_profile_id"])

        if len(buf_x) >= CHUNK_SIZE:
            flush_buffers()
            out.flush()

    flush_buffers()
    out.flush()

    # ---- Metadata attributes for downstream reproducibility ----
    out.attrs["n_frames"] = write_cursor
    out.attrs["device_ids"] = json_dump_safe(profile_ids)
    out.attrs["channel_types"] = json_dump_safe(channel_names)
    out.attrs["distance_buckets_ft"] = json_dump_safe(DISTANCE_BUCKETS_FT)
    out.attrs["seed"] = seed
    out.attrs["source_file"] = input_h5
    out.attrs["channel_config"] = channel_config_path
    out.attrs["sample_rate_hz"] = sample_rate_hz

    out.close()
    source.close()

    log.info("=" * 60)
    log.info("Synthetic dataset generation complete.")
    log.info("  Frames written : %d", write_cursor)
    log.info("  Output         : %s", output_h5)
    log.info("  Output size    : %.2f GB", os.path.getsize(output_h5) / (1024 ** 3))
    log.info("=" * 60)

    print_balance_report(output_h5)


def json_dump_safe(obj) -> str:
    import json
    return json.dumps(obj)


# ===========================================================================
# Post-generation balance verification report
# ===========================================================================

def print_balance_report(output_h5_path: str) -> None:
    """Print a cross-tabulation confirming device x channel x distance balance."""
    with h5py.File(output_h5_path, "r") as f:
        device_id = f["device_id"][:]
        channel_type = np.array([s for s in f["channel_type"][:]])
        distance_ft = f["distance_ft"][:]
        nlos_prob = f["nlos_probability"][:]

    print("\n" + "=" * 62)
    print("  DATA BALANCE VERIFICATION REPORT")
    print("=" * 62)

    print("\n  Frame count per (device_id x channel_type):")
    devices = sorted(set(device_id.tolist()))
    channels = sorted(set(channel_type.tolist()))
    header = "  device_id".ljust(12) + "".join(c.rjust(16) for c in channels)
    print(header)
    for d in devices:
        row = f"  {d}".ljust(12)
        for c in channels:
            count = int(np.sum((device_id == d) & (channel_type == c)))
            row += str(count).rjust(16)
        print(row)

    print("\n  Distance coverage (min / mean / max) per device:")
    for d in devices:
        mask = device_id == d
        d_min, d_mean, d_max = distance_ft[mask].min(), distance_ft[mask].mean(), distance_ft[mask].max()
        print(f"    device {d}:  min={d_min:.1f} ft  mean={d_mean:.1f} ft  max={d_max:.1f} ft")

    nlos_mean = float(np.mean(nlos_prob))
    print(f"\n  Mean NLOS probability across dataset: {nlos_mean:.3f}")
    print("=" * 62 + "\n")


# ===========================================================================
# Entrypoint
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 6 — Generate synthetic V2V dataset (Level 1+2+3 pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-frames", type=int, default=50_000, help="Total number of synthetic frames to generate.")
    p.add_argument("--input", default=DEFAULT_INPUT_H5, help="Path to source oracle.h5.")
    p.add_argument("--output", default=DEFAULT_OUTPUT_H5, help="Path to write synthetic_v2v.h5.")
    p.add_argument("--channel-config", default=DEFAULT_CHANNEL_CONFIG, help="Path to channel_config.yaml.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for full reproducibility.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.input):
        log.error("Input HDF5 not found: %s", args.input)
        sys.exit(1)
    if not os.path.isfile(args.channel_config):
        log.error("Channel config not found: %s", args.channel_config)
        sys.exit(1)

    generate(
        n_frames=args.n_frames,
        input_h5=args.input,
        output_h5=args.output,
        channel_config_path=args.channel_config,
        seed=args.seed,
    )