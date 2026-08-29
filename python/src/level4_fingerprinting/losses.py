from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LossWeights:
    device: float = 1.0
    contrastive: float = 0.20
    channel_adv: float = 0.25
    metric: float = 0.50
    nlos: float = 0.50
    orthogonality: float = 0.05


def normalized_metric_targets(batch: dict[str, Any]) -> Tensor:
    snr = batch["snr_db"].float() / 40.0
    delay = torch.log10(batch["delay_spread"].float().clamp_min(1.0e-12))
    nlos = batch["nlos"].float().clamp(0.0, 1.0)

    if "metric_targets" in batch and isinstance(batch["metric_targets"], Tensor):
        metric_targets = batch["metric_targets"].float()
        packet_error = metric_targets[:, 1].clamp(0.0, 1.0)
        coherence = torch.log10(metric_targets[:, 3].float().clamp_min(1.0)) / 8.0
    else:
        packet_error = torch.zeros_like(snr)
        coherence = torch.zeros_like(snr)

    return torch.stack((snr, packet_error, delay, coherence, nlos), dim=-1)


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    temperature: float = 0.10,
) -> Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    z = F.normalize(embeddings, dim=-1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positives = torch.eq(labels, labels.T).float().to(z.device)
    eye = torch.eye(z.shape[0], device=z.device)
    positives = positives * (1.0 - eye)

    exp_logits = torch.exp(logits) * (1.0 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    denom = positives.sum(dim=1)
    valid = denom > 0
    if not torch.any(valid):
        return embeddings.new_tensor(0.0)
    return -((positives * log_prob).sum(dim=1)[valid] / denom[valid]).mean()


def latent_orthogonality_loss(z_h: Tensor, z_c: Tensor) -> Tensor:
    if z_h.shape[0] < 2:
        return z_h.new_tensor(0.0)
    zh = z_h - z_h.mean(dim=0, keepdim=True)
    zc = z_c - z_c.mean(dim=0, keepdim=True)
    cross_cov = zh.T @ zc / float(z_h.shape[0] - 1)
    return torch.mean(cross_cov * cross_cov)


def compute_fingerprinting_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    weights: LossWeights = LossWeights(),
) -> dict[str, Tensor]:
    device = F.cross_entropy(outputs["device_logits"], batch["device_id"])
    contrastive = supervised_contrastive_loss(outputs["z_h"], batch["device_id"])

    if "channel_domain_logits" in outputs:
        channel_adv = F.cross_entropy(outputs["channel_domain_logits"], batch["channel_id"])
    else:
        channel_adv = outputs["z_h"].new_tensor(0.0)

    metric = F.smooth_l1_loss(outputs["metric_regression"], normalized_metric_targets(batch))
    nlos = F.binary_cross_entropy_with_logits(outputs["nlos_logit"], batch["nlos"].float())
    orthogonality = latent_orthogonality_loss(outputs["z_h"], outputs["z_c"])

    total = (
        weights.device * device
        + weights.contrastive * contrastive
        + weights.channel_adv * channel_adv
        + weights.metric * metric
        + weights.nlos * nlos
        + weights.orthogonality * orthogonality
    )

    return {
        "total": total,
        "device": device.detach(),
        "contrastive": contrastive.detach(),
        "channel_adv": channel_adv.detach(),
        "metric": metric.detach(),
        "nlos": nlos.detach(),
        "orthogonality": orthogonality.detach(),
    }
