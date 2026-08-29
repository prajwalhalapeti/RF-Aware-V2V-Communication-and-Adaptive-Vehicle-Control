from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.contracts import (
    HARDWARE_EMBEDDING_DIM,
    IQ_CHANNELS,
    IQ_FRAME_LEN,
    RF_METRIC_NAMES,
)
from src.level4_fingerprinting.augmentations import AugmentConfig, apply_iq_augmentations


IQ_KEY_ALIASES = (
    "iq",
    "IQ",
    "x",
    "X",
    "frames",
    "samples",
    "oracle_iq",
    "raw_iq",
)

LABEL_ALIASES = {
    "frame_id": ("frame_id", "frame_ids", "sequence", "seq"),
    "device_id": ("device_id", "device_ids", "radio_id", "tx_id", "transmitter_id"),
    "distance_ft": ("distance_ft", "distance", "range_ft", "rx_distance_ft"),
    "channel_id": ("channel_id", "channel_ids", "domain_id", "environment_id"),
    "snr_db": ("snr_db", "snr", "SNR"),
    "packet_error_risk": ("packet_error_risk", "per", "PER", "packet_error_rate"),
    "rms_delay_spread_s": (
        "rms_delay_spread_s",
        "delay_spread_s",
        "tau_rms_s",
        "rms_delay_spread",
    ),
    "coherence_bandwidth_hz": (
        "coherence_bandwidth_hz",
        "coherence_bw_hz",
        "bc_hz",
    ),
    "nlos_probability": ("nlos_probability", "p_nlos", "nlos", "is_nlos"),
}


def _is_dataset(obj: Any) -> bool:
    return isinstance(obj, h5py.Dataset)


def _find_dataset_by_alias(h5: h5py.File, aliases: Iterable[str]) -> str | None:
    alias_set = {alias.lower().strip("/") for alias in aliases}

    for alias in aliases:
        if h5.get(alias) is not None and _is_dataset(h5.get(alias)):
            return alias

    matches: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if not _is_dataset(obj):
            return
        base = name.split("/")[-1].lower()
        if base in alias_set:
            matches.append(name)

    h5.visititems(visitor)
    return matches[0] if matches else None


def _open_h5_readonly(path: Path) -> h5py.File:
    try:
        return h5py.File(path, "r", libver="latest", swmr=True)
    except (OSError, ValueError):
        return h5py.File(path, "r")


def _coerce_scalar(value: Any, default: float | int) -> float | int:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    scalar = arr.reshape(-1)[0]
    if isinstance(default, int):
        return int(scalar)
    return float(scalar)


def _complex_or_iq_to_2xn(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)

    if np.iscomplexobj(arr):
        if arr.ndim == 1:
            return np.stack((arr.real, arr.imag), axis=0).astype(np.float32)
        if arr.ndim == 2 and 1 in arr.shape:
            flat = arr.reshape(-1)
            return np.stack((flat.real, flat.imag), axis=0).astype(np.float32)
        raise ValueError(f"complex I/Q frame must be 1-D; got shape {arr.shape}")

    if arr.ndim == 1:
        if arr.shape[0] % 2 != 0:
            raise ValueError(
                "1-D real-valued I/Q frame must contain interleaved I,Q samples "
                f"with even length; got {arr.shape[0]}"
            )
        reshaped = arr.reshape(-1, 2)
        return reshaped.T.astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(f"I/Q frame must be 1-D or 2-D after indexing; got {arr.shape}")

    if arr.shape[0] == IQ_CHANNELS:
        return arr.astype(np.float32)
    if arr.shape[1] == IQ_CHANNELS:
        return arr.T.astype(np.float32)

    raise ValueError(
        f"cannot interpret frame shape {arr.shape}; expected [2, N], [N, 2], "
        "complex [N], or interleaved real [2N]"
    )


def _fit_frame_length(iq: np.ndarray, frame_len: int, start: int | None = None) -> np.ndarray:
    if iq.shape[0] != IQ_CHANNELS:
        raise ValueError(f"expected first I/Q axis of length 2; got {iq.shape}")

    current_len = iq.shape[1]
    if current_len == frame_len:
        return iq.astype(np.float32, copy=False)

    if current_len > frame_len:
        if start is None:
            start = (current_len - frame_len) // 2
        stop = start + frame_len
        return iq[:, start:stop].astype(np.float32, copy=False)

    padded = np.zeros((IQ_CHANNELS, frame_len), dtype=np.float32)
    padded[:, :current_len] = iq
    return padded


def remove_dc_and_power_normalize(
    iq: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    arr = np.asarray(iq, dtype=np.float32)
    if arr.shape[0] != IQ_CHANNELS:
        raise ValueError(f"expected normalized I/Q shape [2, N]; got {arr.shape}")

    arr64 = arr.astype(np.float64, copy=False)
    arr64 = arr64 - arr64.mean()
    avg_power = np.mean(arr64**2)
    scale = math.sqrt(float(avg_power) + float(eps))
    return (arr64 / scale).astype(np.float32, copy=False)


def _derive_channel_id(distance_ft: float, nlos_probability: float) -> int:
    if math.isfinite(distance_ft):
        if distance_ft <= 2.0:
            return 0
        if distance_ft <= 32.0:
            return 1
        return 2
    if math.isfinite(nlos_probability):
        return int(float(nlos_probability) >= 0.5)
    return 0


def _derive_delay_spread_s(nlos_probability: float) -> float:
    if not math.isfinite(nlos_probability):
        return 45.0e-9
    p = min(1.0, max(0.0, float(nlos_probability)))
    return 10.0e-9 + p * 90.0e-9


class OracleHDF5Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        hdf5_path: str | Path,
        iq_key: str | None = None,
        frame_len: int = IQ_FRAME_LEN,
        random_crop: bool = False,
        normalize_power: bool = True,
        augment: bool = False,
        augmentation: AugmentConfig | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser().resolve()
        self.iq_key = iq_key
        self.frame_len = int(frame_len)
        self.random_crop = bool(random_crop)
        self.normalize_power = bool(normalize_power)
        self.augmentation = augmentation or AugmentConfig(enabled=augment)
        self.eps = float(eps)
        self._h5: h5py.File | None = None
        self._resolved_iq_key: str | None = None
        self._resolved_label_keys: dict[str, str | None] = {}

        if self.frame_len <= 0:
            raise ValueError(f"frame_len must be positive; got {self.frame_len}")
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with _open_h5_readonly(self.hdf5_path) as h5:
            self._resolved_iq_key = self._resolve_iq_key(h5)
            iq_ds = h5[self._resolved_iq_key]
            if len(iq_ds.shape) == 0:
                raise ValueError(f"I/Q dataset {self._resolved_iq_key} is scalar")
            self._length = int(iq_ds.shape[0])
            if self._length <= 0:
                raise ValueError(f"I/Q dataset {self._resolved_iq_key} is empty")
            self._resolved_label_keys = {
                name: _find_dataset_by_alias(h5, aliases)
                for name, aliases in LABEL_ALIASES.items()
            }

    def _resolve_iq_key(self, h5: h5py.File) -> str:
        if self.iq_key is not None:
            if h5.get(self.iq_key) is None:
                raise KeyError(f"explicit iq_key not found in HDF5: {self.iq_key}")
            if not _is_dataset(h5.get(self.iq_key)):
                raise TypeError(f"explicit iq_key is not a dataset: {self.iq_key}")
            return self.iq_key

        resolved = _find_dataset_by_alias(h5, IQ_KEY_ALIASES)
        if resolved is None:
            available: list[str] = []

            def visitor(name: str, obj: Any) -> None:
                if _is_dataset(obj):
                    available.append(name)

            h5.visititems(visitor)
            raise KeyError(
                "could not locate an I/Q dataset. Pass iq_key explicitly. "
                f"Available datasets: {available}"
            )
        return resolved

    def __len__(self) -> int:
        return self._length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _h5_file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = _open_h5_readonly(self.hdf5_path)
        return self._h5

    def _read_label(
        self,
        h5: h5py.File,
        name: str,
        index: int,
        default: float | int,
    ) -> float | int:
        key = self._resolved_label_keys.get(name)
        if key is None:
            return default
        return _coerce_scalar(h5[key][index], default)

    def _crop_start(self, current_len: int, index: int) -> int | None:
        if current_len <= self.frame_len:
            return None
        max_start = current_len - self.frame_len
        if not self.random_crop:
            return max_start // 2
        rng = np.random.default_rng(seed=(index + 1) * 1_000_003)
        return int(rng.integers(0, max_start + 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = self._length + index
        if index < 0 or index >= self._length:
            raise IndexError(f"index {index} out of range for length {self._length}")

        h5 = self._h5_file()
        assert self._resolved_iq_key is not None
        raw_frame = h5[self._resolved_iq_key][index]
        iq = _complex_or_iq_to_2xn(raw_frame)
        start = self._crop_start(iq.shape[1], index)
        iq = _fit_frame_length(iq, self.frame_len, start=start)
        if self.normalize_power:
            iq = remove_dc_and_power_normalize(iq, eps=self.eps)

        frame_id = self._read_label(h5, "frame_id", index, index)
        device_id = self._read_label(h5, "device_id", index, -1)
        distance_ft = self._read_label(h5, "distance_ft", index, math.nan)
        channel_id = self._read_label(h5, "channel_id", index, -1)
        snr_db = self._read_label(h5, "snr_db", index, math.nan)
        packet_error_risk = self._read_label(h5, "packet_error_risk", index, math.nan)
        rms_delay_spread_s = self._read_label(h5, "rms_delay_spread_s", index, math.nan)
        coherence_bandwidth_hz = self._read_label(
            h5, "coherence_bandwidth_hz", index, math.nan
        )
        nlos_probability = self._read_label(h5, "nlos_probability", index, math.nan)
        if not math.isfinite(float(nlos_probability)) and h5.get("nlos") is not None:
            nlos_probability = _coerce_scalar(h5["nlos"][index], math.nan)

        if not math.isfinite(float(rms_delay_spread_s)):
            rms_delay_spread_s = _derive_delay_spread_s(float(nlos_probability))

        if int(channel_id) < 0:
            channel_id = _derive_channel_id(float(distance_ft), float(nlos_probability))

        if not math.isfinite(float(coherence_bandwidth_hz)) and math.isfinite(
            float(rms_delay_spread_s)
        ):
            tau = float(rms_delay_spread_s)
            coherence_bandwidth_hz = 1.0 / (5.0 * tau) if tau > 0.0 else math.nan

        metric_values = np.asarray(
            [
                snr_db,
                packet_error_risk,
                rms_delay_spread_s,
                coherence_bandwidth_hz,
                nlos_probability,
            ],
            dtype=np.float32,
        )
        metric_valid = np.isfinite(metric_values)
        metric_values = np.nan_to_num(
            metric_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        iq_tensor = torch.from_numpy(iq)
        iq_tensor = apply_iq_augmentations(iq_tensor, self.augmentation)
        metric_targets = torch.from_numpy(metric_values)
        metric_valid_mask = torch.from_numpy(metric_valid)

        rf_metrics = {
            "timestamp_ns": torch.tensor(0, dtype=torch.long),
            "sequence": torch.tensor(int(frame_id), dtype=torch.long),
            "transmitter_id": torch.tensor(int(device_id), dtype=torch.long),
            "device_confidence": torch.tensor(0.0, dtype=torch.float32),
            "snr_db": metric_targets[0],
            "packet_error_risk": metric_targets[1],
            "rms_delay_spread_s": metric_targets[2],
            "coherence_bandwidth_hz": metric_targets[3],
            "nlos_probability": metric_targets[4],
            "hardware_embedding": torch.zeros(
                HARDWARE_EMBEDDING_DIM,
                dtype=torch.float32,
            ),
        }

        return {
            "iq": iq_tensor,
            "device_id": torch.tensor(int(device_id), dtype=torch.long),
            "distance_ft": torch.tensor(float(distance_ft), dtype=torch.float32),
            "channel_id": torch.tensor(int(channel_id), dtype=torch.long),
            "snr_db": torch.tensor(float(snr_db), dtype=torch.float32),
            "delay_spread": torch.tensor(float(rms_delay_spread_s), dtype=torch.float32),
            "nlos": torch.tensor(float(nlos_probability), dtype=torch.float32),
            "metric_names": RF_METRIC_NAMES,
            "metric_targets": metric_targets,
            "metric_valid_mask": metric_valid_mask,
            "rf_metrics": rf_metrics,
        }


def make_oracle_dataloader(
    hdf5_path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    iq_key: str | None = None,
    random_crop: bool = False,
    augment: bool = False,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    dataset = OracleHDF5Dataset(
        hdf5_path=hdf5_path,
        iq_key=iq_key,
        random_crop=random_crop,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test streamed ORACLE HDF5 reads.")
    parser.add_argument("--hdf5", required=True, help="Path to ORACLE HDF5 file.")
    parser.add_argument("--iq-key", default=None, help="Optional explicit I/Q dataset key.")
    parser.add_argument("--index", type=int, default=0, help="Frame index to inspect.")
    args = parser.parse_args()

    dataset = OracleHDF5Dataset(args.hdf5, iq_key=args.iq_key)
    sample = dataset[args.index]
    print(f"dataset_len={len(dataset)}")
    print(f"iq_shape={tuple(sample['iq'].shape)}")
    print(f"iq_dtype={sample['iq'].dtype}")
    print(f"device_id={int(sample['device_id'])}")
    print(f"channel_id={int(sample['channel_id'])}")
    print(f"metric_names={sample['metric_names']}")
    print(f"metric_targets={sample['metric_targets'].tolist()}")
    print(f"metric_valid_mask={sample['metric_valid_mask'].tolist()}")
    dataset.close()


if __name__ == "__main__":
    _main()
