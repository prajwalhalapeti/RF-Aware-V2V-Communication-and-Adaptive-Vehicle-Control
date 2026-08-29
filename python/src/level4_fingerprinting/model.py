from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.common.contracts import (
    CHANNEL_EMBEDDING_DIM,
    HARDWARE_EMBEDDING_DIM,
    IQ_CHANNELS,
    IQ_FRAME_LEN,
    RF_METRIC_NAMES,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    num_devices: int
    num_metrics: int = len(RF_METRIC_NAMES)
    num_channel_domains: int | None = None
    hardware_embedding_dim: int = HARDWARE_EMBEDDING_DIM
    channel_embedding_dim: int = CHANNEL_EMBEDDING_DIM
    dropout: float = 0.10


class ConvNormAct1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct1D(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.dropout = nn.Dropout(p=dropout)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = F.silu(out + residual)
        return out


class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int, attention_hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, attention_hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_hidden, 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.score(x)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return pooled, weights.squeeze(1)


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_value: float) -> Tensor:
        ctx.lambda_value = float(lambda_value)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_value * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_value: float = 1.0) -> None:
        super().__init__()
        self.lambda_value = float(lambda_value)

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFn.apply(x, self.lambda_value)


class RFMultiHeadNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_devices <= 0:
            raise ValueError(f"num_devices must be positive; got {config.num_devices}")
        if config.num_metrics <= 0:
            raise ValueError(f"num_metrics must be positive; got {config.num_metrics}")

        self.config = config
        self.stem = ConvNormAct1D(
            in_channels=IQ_CHANNELS,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.block64 = ResidualBlock1D(
            in_channels=64,
            out_channels=64,
            stride=1,
            dropout=config.dropout,
        )
        self.block128 = ResidualBlock1D(
            in_channels=64,
            out_channels=128,
            stride=2,
            dropout=config.dropout,
        )
        self.block256 = ResidualBlock1D(
            in_channels=128,
            out_channels=256,
            stride=2,
            dropout=config.dropout,
        )
        self.context = ResidualBlock1D(
            in_channels=256,
            out_channels=256,
            stride=1,
            dropout=config.dropout,
        )
        self.attention_pool = AttentionPooling1D(channels=256, attention_hidden=128)

        self.hardware_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(256, config.hardware_embedding_dim),
        )
        self.device_classifier = nn.Linear(
            config.hardware_embedding_dim,
            config.num_devices,
        )

        self.channel_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(128, config.channel_embedding_dim),
        )
        self.metric_regressor = nn.Sequential(
            nn.Linear(config.channel_embedding_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(64, config.num_metrics),
        )
        self.nlos_classifier = nn.Sequential(
            nn.Linear(config.channel_embedding_dim, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(32, 1),
        )

        if config.num_channel_domains is not None:
            if config.num_channel_domains <= 0:
                raise ValueError(
                    "num_channel_domains must be positive when provided; "
                    f"got {config.num_channel_domains}"
                )
            self.gradient_reversal = GradientReversal(lambda_value=1.0)
            self.channel_domain_classifier: nn.Module | None = nn.Sequential(
                nn.Linear(config.hardware_embedding_dim, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, config.num_channel_domains),
            )
        else:
            self.gradient_reversal = None
            self.channel_domain_classifier = None

    @staticmethod
    def _assert_input_shape(x: Tensor) -> None:
        if torch.jit.is_tracing():
            return
        if x.ndim != 3:
            raise ValueError(f"expected input [B, 2, 1024]; got ndim={x.ndim}")
        expected = (IQ_CHANNELS, IQ_FRAME_LEN)
        actual = (x.shape[1], x.shape[2])
        if actual != expected:
            raise ValueError(f"expected input [B, {expected[0]}, {expected[1]}]; got {tuple(x.shape)}")

    def encode_features(self, x: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        self._assert_input_shape(x)
        shapes: dict[str, tuple[int, ...]] = {"input": tuple(x.shape)}

        x = self.stem(x)
        shapes["stem"] = tuple(x.shape)

        x = self.block64(x)
        shapes["block64"] = tuple(x.shape)

        x = self.block128(x)
        shapes["block128"] = tuple(x.shape)

        x = self.block256(x)
        shapes["block256"] = tuple(x.shape)

        x = self.context(x)
        shapes["context"] = tuple(x.shape)

        pooled, attention_weights = self.attention_pool(x)
        shapes["attention_pooled"] = tuple(pooled.shape)
        shapes["attention_weights"] = tuple(attention_weights.shape)
        return pooled, shapes

    def forward(self, x: Tensor, return_shapes: bool = False) -> dict[str, Tensor | dict[str, tuple[int, ...]]]:
        pooled, shapes = self.encode_features(x)

        z_h_raw = self.hardware_projection(pooled)
        z_h = F.normalize(z_h_raw, p=2, dim=-1)
        device_logits = self.device_classifier(z_h)

        z_c = self.channel_projection(pooled)
        metric_regression = self.metric_regressor(z_c)
        nlos_logit = self.nlos_classifier(z_c).squeeze(-1)

        outputs: dict[str, Tensor | dict[str, tuple[int, ...]]] = {
            "z_h": z_h,
            "z_c": z_c,
            "device_logits": device_logits,
            "metric_regression": metric_regression,
            "nlos_logit": nlos_logit,
        }

        if self.channel_domain_classifier is not None and self.gradient_reversal is not None:
            reversed_z_h = self.gradient_reversal(z_h)
            outputs["channel_domain_logits"] = self.channel_domain_classifier(reversed_z_h)

        if return_shapes:
            outputs["shapes"] = shapes

        return outputs


def build_model(
    num_devices: int,
    num_metrics: int = len(RF_METRIC_NAMES),
    num_channel_domains: int | None = None,
    dropout: float = 0.10,
) -> RFMultiHeadNet:
    return RFMultiHeadNet(
        ModelConfig(
            num_devices=num_devices,
            num_metrics=num_metrics,
            num_channel_domains=num_channel_domains,
            dropout=dropout,
        )
    )


def _smoke_test() -> None:
    torch.manual_seed(7)
    model = build_model(num_devices=16, num_channel_domains=4)
    model.eval()
    x = torch.randn(2, IQ_CHANNELS, IQ_FRAME_LEN)
    with torch.no_grad():
        out = model(x, return_shapes=True)
    print(out["shapes"])
    print("z_h", tuple(out["z_h"].shape))
    print("z_c", tuple(out["z_c"].shape))
    print("device_logits", tuple(out["device_logits"].shape))
    print("metric_regression", tuple(out["metric_regression"].shape))
    print("nlos_logit", tuple(out["nlos_logit"].shape))
    if "channel_domain_logits" in out:
        print("channel_domain_logits", tuple(out["channel_domain_logits"].shape))


if __name__ == "__main__":
    _smoke_test()
