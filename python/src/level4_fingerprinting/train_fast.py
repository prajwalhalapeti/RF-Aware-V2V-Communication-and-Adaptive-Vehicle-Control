"""Minimal fast trainer — M2-tuned, ~5min sanity run.
Usage: python train_fast.py --steps-per-epoch 300 --epochs 3
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from src.level4_fingerprinting.dataset import OracleHDF5Dataset
from src.level4_fingerprinting.losses import LossWeights, compute_fingerprinting_losses
from src.level4_fingerprinting.model import build_model


def to_dev(v, d):
    if isinstance(v, torch.Tensor): return v.to(d)
    if isinstance(v, dict): return {k: to_dev(x, d) for k, x in v.items()}
    return v


@torch.no_grad()
def validate(model, loader, dev):
    model.eval(); correct = total = 0; loss_sum = n = 0
    for b in loader:
        b = to_dev(b, dev)
        out = model(b["iq"])
        loss_sum += float(compute_fingerprinting_losses(out, b)["total"]); n += 1
        correct += int((out["device_logits"].argmax(-1) == b["device_id"]).sum())
        total += b["device_id"].numel()
    return loss_sum / max(n, 1), correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default="data/processed/oracle.h5")
    p.add_argument("--out-dir", default="results/checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=300)  # caps runtime
    p.add_argument("--batch-size", type=int, default=256)        # bigger batch = fewer Python-loop hops
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-samples", type=int, default=40_000)  # caps dataset size for a quick run
    args = p.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds = OracleHDF5Dataset(hdf5_path=args.hdf5, augment=True)
    if args.limit_samples > 0:
        ds = Subset(ds, range(min(args.limit_samples, len(ds))))
    val_n = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = random_split(ds, [len(ds) - val_n, val_n], generator=torch.Generator().manual_seed(42))

    dl_kw = dict(num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **dl_kw)

    model = build_model(num_devices=16, num_channel_domains=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    weights = LossWeights()

    print(f"[setup] device={dev} train={len(train_ds)} val={len(val_ds)} steps/epoch={args.steps_per_epoch}")

    for ep in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; n = 0
        bar = tqdm(train_loader, total=args.steps_per_epoch, desc=f"ep{ep}", unit="b")
        for b in bar:
            if n >= args.steps_per_epoch: break
            b = to_dev(b, dev)
            opt.zero_grad(set_to_none=True)
            out = model(b["iq"])
            loss = compute_fingerprinting_losses(out, b, weights)["total"]
            loss.backward(); opt.step()
            loss_sum += float(loss); n += 1
            bar.set_postfix(loss=f"{loss_sum/n:.4f}")

        val_loss, val_acc = validate(model, val_loader, dev)
        row = {"epoch": ep, "train_loss": loss_sum / max(n, 1), "val_loss": val_loss, "val_device_acc": val_acc}
        print(json.dumps(row))
        torch.save({"model": model.state_dict(), "epoch": ep, "metrics": row}, out_dir / "last.pt")


if __name__ == "__main__":
    main()