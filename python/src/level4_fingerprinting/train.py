from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split

from src.level4_fingerprinting.dataset import OracleHDF5Dataset
from src.level4_fingerprinting.losses import LossWeights, compute_fingerprinting_losses
from src.level4_fingerprinting.model import build_model


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def chunked_max(dataset: h5py.Dataset, chunk: int = 1_000_000) -> int:
    max_value = 0
    for start in range(0, int(dataset.shape[0]), chunk):
        stop = min(start + chunk, int(dataset.shape[0]))
        max_value = max(max_value, int(np.asarray(dataset[start:stop]).max()))
    return max_value


def infer_counts(hdf5_path: Path) -> tuple[int, int]:
    with h5py.File(hdf5_path, "r") as h5:
        num_devices = chunked_max(h5["device_id"]) + 1
        if "channel_id" in h5:
            num_channels = chunked_max(h5["channel_id"]) + 1
        elif "distance_ft" in h5:
            observed = np.asarray(h5["distance_ft"][: min(10_000, h5["distance_ft"].shape[0])])
            num_channels = max(1, min(3, len(set(observed.tolist()))))
        elif "nlos" in h5 or "nlos_probability" in h5:
            num_channels = 2
        else:
            num_channels = 1
    return num_devices, num_channels


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    return value


def make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    # Apple Silicon + h5py is fastest and most stable with a single process.
    # pin_memory only helps CUDA; on MPS unified memory it adds overhead.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=drop_last,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
) -> dict[str, float]:
    model.eval()
    total = 0
    device_correct = 0
    nlos_correct = 0
    loss_total = 0.0
    batches = 0

    for batch in islice(loader, max_steps):
        batch = move_to_device(batch, device)
        outputs = model(batch["iq"])
        losses = compute_fingerprinting_losses(outputs, batch)
        loss_total += float(losses["total"].detach().cpu())

        pred = outputs["device_logits"].argmax(dim=-1)
        device_correct += int((pred == batch["device_id"]).sum().item())
        nlos_pred = (torch.sigmoid(outputs["nlos_logit"]) >= 0.5).long()
        nlos_true = (batch["nlos"] >= 0.5).long()
        nlos_correct += int((nlos_pred == nlos_true).sum().item())
        total += int(batch["device_id"].numel())
        batches += 1

    return {
        "loss": loss_total / max(1, batches),
        "device_acc": device_correct / max(1, total),
        "nlos_acc": nlos_correct / max(1, total),
        "batches": float(batches),
        "samples": float(total),
    }


def train_one_bounded_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights,
    steps_per_epoch: int,
    log_every: int,
) -> dict[str, float]:
    model.train()
    train_loss = 0.0
    skipped_nan = 0
    batches = 0
    samples = 0
    start_time = time.perf_counter()

    for step, batch in enumerate(islice(loader, steps_per_epoch), start=1):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["iq"])
        losses = compute_fingerprinting_losses(outputs, batch, weights)
        total_loss = losses["total"]

        if not torch.isfinite(total_loss):
            skipped_nan += 1
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss += float(total_loss.detach().cpu())
        batches += 1
        samples += int(batch["device_id"].numel())

        if log_every > 0 and step % log_every == 0:
            elapsed = time.perf_counter() - start_time
            rate = samples / max(elapsed, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "train_step",
                        "step": step,
                        "loss": train_loss / max(1, batches),
                        "samples_per_second": rate,
                        "skipped_nan": skipped_nan,
                    },
                    sort_keys=True,
                )
            )

    elapsed = time.perf_counter() - start_time
    return {
        "loss": train_loss / max(1, batches),
        "batches": float(batches),
        "samples": float(samples),
        "samples_per_second": samples / max(elapsed, 1.0e-9),
        "skipped_nan": float(skipped_nan),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    num_devices: int,
    num_channels: int,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "num_devices": num_devices,
            "num_channel_domains": num_channels,
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default="data/processed/oracle.h5")
    parser.add_argument("--out-dir", default="results/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps_per_epoch <= 0:
        raise ValueError("--steps-per-epoch must be positive")
    if args.validation_steps <= 0:
        raise ValueError("--validation-steps must be positive")

    torch.manual_seed(args.seed)
    hdf5_path = Path(args.hdf5).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    num_devices, num_channels = infer_counts(hdf5_path)
    base_dataset = OracleHDF5Dataset(hdf5_path=hdf5_path, augment=True)

    if args.limit_samples > 0:
        sample_count = min(args.limit_samples, len(base_dataset))
        dataset: torch.utils.data.Dataset = Subset(base_dataset, list(range(sample_count)))
    else:
        dataset = base_dataset

    val_len = max(args.batch_size, int(len(dataset) * args.val_fraction))
    val_len = min(val_len, len(dataset) - args.batch_size)
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError("training split is empty; reduce --val-fraction or increase data")

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        drop_last=True,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        drop_last=False,
    )

    model = build_model(
        num_devices=num_devices,
        num_channel_domains=num_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    weights = LossWeights()
    best_score = -float("inf")
    history: list[dict[str, float]] = []

    print(
        json.dumps(
            {
                "phase": "setup",
                "device": str(device),
                "num_workers": 0,
                "pin_memory": device.type == "cuda",
                "batch_size": args.batch_size,
                "lr": args.lr,
                "steps_per_epoch": args.steps_per_epoch,
                "validation_steps": args.validation_steps,
                "train_len": train_len,
                "val_len": val_len,
                "num_devices": num_devices,
                "num_channels": num_channels,
            },
            sort_keys=True,
        )
    )

    for epoch in range(1, args.epochs + 1):
        train = train_one_bounded_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            weights=weights,
            steps_per_epoch=args.steps_per_epoch,
            log_every=args.log_every,
        )
        scheduler.step()
        val = validate(
            model=model,
            loader=val_loader,
            device=device,
            max_steps=args.validation_steps,
        )

        score = val["device_acc"] + 0.25 * val["nlos_acc"] - 0.05 * val["loss"]
        row = {
            "epoch": float(epoch),
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train["loss"],
            "train_batches": train["batches"],
            "train_samples": train["samples"],
            "train_samples_per_second": train["samples_per_second"],
            "skipped_nan": train["skipped_nan"],
            "val_loss": val["loss"],
            "val_device_acc": val["device_acc"],
            "val_nlos_acc": val["nlos_acc"],
            "val_batches": val["batches"],
            "val_samples": val["samples"],
            "score": score,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        save_checkpoint(
            out_dir / "last.pt",
            model,
            num_devices,
            num_channels,
            epoch,
            row,
            args,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                out_dir / "best.pt",
                model,
                num_devices,
                num_channels,
                epoch,
                row,
                args,
            )

        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
    