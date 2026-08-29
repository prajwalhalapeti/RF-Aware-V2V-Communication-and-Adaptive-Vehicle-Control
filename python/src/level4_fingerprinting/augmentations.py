from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AugmentConfig:
    enabled: bool = False
    max_time_shift: int = 8
    amplitude_jitter_std: float = 0.03
    max_phase_rotation_rad: float = 0.08
    additive_noise_std: float = 0.005


def apply_time_shift(iq: Tensor, max_shift: int) -> Tensor:
    if max_shift <= 0:
        return iq
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(iq, shifts=shift, dims=-1)


def apply_amplitude_jitter(iq: Tensor, jitter_std: float) -> Tensor:
    if jitter_std <= 0.0:
        return iq
    gain = 1.0 + torch.randn((), dtype=iq.dtype, device=iq.device) * jitter_std
    return iq * torch.clamp(gain, 0.85, 1.15)


def apply_phase_rotation(iq: Tensor, max_phase_rad: float) -> Tensor:
    if max_phase_rad <= 0.0:
        return iq
    phase = (2.0 * torch.rand((), dtype=iq.dtype, device=iq.device) - 1.0) * max_phase_rad
    c = torch.cos(phase)
    s = torch.sin(phase)
    i = iq[0].clone()
    q = iq[1].clone()
    return torch.stack((c * i - s * q, s * i + c * q), dim=0)


def apply_additive_noise(iq: Tensor, noise_std: float) -> Tensor:
    if noise_std <= 0.0:
        return iq
    return iq + torch.randn_like(iq) * noise_std


def apply_iq_augmentations(iq: Tensor, config: AugmentConfig) -> Tensor:
    if not config.enabled:
        return iq
    out = apply_time_shift(iq, config.max_time_shift)
    out = apply_amplitude_jitter(out, config.amplitude_jitter_std)
    out = apply_phase_rotation(out, config.max_phase_rotation_rad)
    out = apply_additive_noise(out, config.additive_noise_std)
    return out