from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(16, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet50InstanceNet(nn.Module):
    """Same encoder/decoder topology for random, supervised and DINO weights."""

    def __init__(self, initialization: str, weights_path: Path | None = None) -> None:
        super().__init__()
        if initialization not in {"random", "supervised_imagenet", "dino_self_supervised"}:
            raise ValueError(f"Unknown initialization: {initialization}")
        backbone = resnet50(weights=None)
        self.load_report: dict[str, object] = {"initialization": initialization, "weights_path": None}
        if initialization != "random":
            if weights_path is None or not weights_path.is_file():
                raise FileNotFoundError(f"Missing weights for {initialization}: {weights_path}")
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            message = backbone.load_state_dict(state, strict=False)
            allowed_missing = {"fc.weight", "fc.bias"}
            if set(message.missing_keys) - allowed_missing or message.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected weight mismatch: missing={message.missing_keys}, "
                    f"unexpected={message.unexpected_keys}"
                )
            self.load_report = {
                "initialization": initialization,
                "weights_path": str(weights_path.resolve()),
                "missing_keys": list(message.missing_keys),
                "unexpected_keys": list(message.unexpected_keys),
            }

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up4 = ConvBlock(2048 + 1024, 256)
        self.up3 = ConvBlock(256 + 512, 192)
        self.up2 = ConvBlock(192 + 256, 128)
        self.up1 = ConvBlock(128 + 64, 64)
        self.full = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 3, 1)

    @property
    def encoder_parameters(self):
        modules = (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        for module in modules:
            yield from module.parameters()

    @property
    def decoder_parameters(self):
        modules = (self.up4, self.up3, self.up2, self.up1, self.full, self.head)
        for module in modules:
            yield from module.parameters()

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.stem(x)
        c2 = self.layer1(self.pool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = self.up4(torch.cat([self._up(c5, c4), c4], dim=1))
        x = self.up3(torch.cat([self._up(x, c3), c3], dim=1))
        x = self.up2(torch.cat([self._up(x, c2), c2], dim=1))
        x = self.up1(torch.cat([self._up(x, c1), c1], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(self.full(x))


__all__ = ["ResNet50InstanceNet"]
