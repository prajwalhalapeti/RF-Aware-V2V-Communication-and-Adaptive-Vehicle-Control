#!/usr/bin/env python3
"""
Create a finite-only copy of an RF HDF5 dataset.

This script DOES NOT use nan_to_num and DOES NOT alter valid samples.
It removes complete records whose I/Q frame contains NaN or Inf, while
preserving all row-aligned datasets and file attributes.

Usage:
    python -u filter_finite_hdf5.py \
      --input data/processed/synthetic_v2v.h5 \
      --output data/processed/synthetic_v2v_finite.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def choose_iq_key(h5: h5py.File) -> str:
    for key in ("x", "iq", "iq_samples"):
        if key in h5:
            return key
    raise KeyError(f"No I/Q dataset found. Keys: {list(h5.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    args = parser.parse_args()

    src_path = Path(args.input).expanduser().resolve()
    dst_path = Path(args.output).expanduser().resolve()

    if src_path == dst_path:
        raise ValueError("Input and output paths must be different.")
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        iq_key = choose_iq_key(src)
        iq = src[iq_key]
        n_rows = int(iq.shape[0])

        print(f"[SCAN] input={src_path}")
        print(f"[SCAN] iq_key={iq_key} rows={n_rows} shape={iq.shape}")

        valid_parts: list[np.ndarray] = []
        invalid_count = 0

        for start in range(0, n_rows, args.chunk_rows):
            stop = min(start + args.chunk_rows, n_rows)
            block = np.asarray(iq[start:stop])
            finite = np.isfinite(block).reshape(block.shape[0], -1).all(axis=1)
            valid_parts.append(np.flatnonzero(finite) + start)
            invalid_count += int((~finite).sum())

            if start == 0 or stop == n_rows or stop % 5000 == 0:
                print(
                    f"[SCAN] checked={stop}/{n_rows} "
                    f"invalid_so_far={invalid_count}",
                    flush=True,
                )

        valid_indices = (
            np.concatenate(valid_parts)
            if valid_parts
            else np.empty((0,), dtype=np.int64)
        )
        n_valid = int(valid_indices.size)

        print(f"[RESULT] valid={n_valid} invalid={invalid_count} total={n_rows}")
        if n_valid == 0:
            raise RuntimeError("No finite I/Q records remain.")

        with h5py.File(dst_path, "w") as dst:
            for key, obj in src.items():
                if not isinstance(obj, h5py.Dataset):
                    src.copy(key, dst)
                    continue

                row_aligned = obj.ndim >= 1 and int(obj.shape[0]) == n_rows

                if not row_aligned:
                    src.copy(key, dst)
                    continue

                out_shape = (n_valid, *obj.shape[1:])

                create_kwargs = {"shape": out_shape, "dtype": obj.dtype}

                if obj.chunks is not None:
                    first_chunk = min(max(1, obj.chunks[0]), n_valid)
                    create_kwargs["chunks"] = (first_chunk, *obj.chunks[1:])
                if obj.compression is not None:
                    create_kwargs["compression"] = obj.compression
                    create_kwargs["compression_opts"] = obj.compression_opts
                if obj.shuffle:
                    create_kwargs["shuffle"] = True
                if obj.fletcher32:
                    create_kwargs["fletcher32"] = True

                out_ds = dst.create_dataset(key, **create_kwargs)

                write_cursor = 0
                for start in range(0, n_valid, args.chunk_rows):
                    stop = min(start + args.chunk_rows, n_valid)
                    idx = valid_indices[start:stop]
                    out_ds[write_cursor:write_cursor + len(idx)] = obj[idx]
                    write_cursor += len(idx)

                for attr_name, attr_value in obj.attrs.items():
                    out_ds.attrs[attr_name] = attr_value

                print(f"[COPY] {key}: {obj.shape} -> {out_shape}", flush=True)

            for attr_name, attr_value in src.attrs.items():
                dst.attrs[attr_name] = attr_value

            dst.attrs["finite_filter_source"] = str(src_path)
            dst.attrs["finite_filter_original_rows"] = n_rows
            dst.attrs["finite_filter_removed_rows"] = invalid_count
            dst.attrs["n_frames"] = n_valid

    print(f"\n[DONE] wrote finite-only dataset:\n{dst_path}")


if __name__ == "__main__":
    main()