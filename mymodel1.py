#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""模型定义：仅保留分类支路，不再依赖扰动标签。"""

from typing import Tuple
import torch
import torch.nn as nn

from gconv import PlainFirstLayer


class Conv2dBlock(nn.Module):
    """1×k 的时序卷积块。"""

    def __init__(self, in_ch: int, out_ch: int, k: int, stride: int = 1, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), stride=(1, stride),
                      padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_backbone() -> nn.Sequential:
    """原始网络主干的纯 CNN 堆叠，输出 1024 维特征。"""
    layers = []
    layers += [Conv2dBlock(32,  64, 11, pool=True)]
    layers += [Conv2dBlock(64, 128,  9, pool=True)]
    layers += [Conv2dBlock(128, 256, 7,  pool=True)]
    layers += [Conv2dBlock(256, 256, 7,  pool=True)]
    layers += [Conv2dBlock(256, 512, 5,  pool=True)]
    layers += [Conv2dBlock(512, 512, 5,  pool=True)]
    layers += [Conv2dBlock(512, 1024, 3, pool=True)]
    return nn.Sequential(*layers)


class GroupConvFirstLayer(nn.Module):
    """普通卷积实现的第一层，不依赖扰动参数。"""

    def __init__(self, in_ch: int = 2, out_ch: int = 32, k: int = 5, fs: float = 50e6):
        super().__init__()
        self.layer = nn.Sequential(
            PlainFirstLayer(in_ch=in_ch, out_ch=out_ch, k=k, fs=fs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class PerturbAwareNet(nn.Module):
    """保留原 Backbone/Head 的分类模型。"""

    def __init__(self, n_classes: int, fs: float = 50e6):
        super().__init__()
        self.first = GroupConvFirstLayer(in_ch=2, out_ch=32, k=5, fs=fs)
        self.features = _make_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.first(x)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        feat = x
        logits = self.classifier(x)
        return logits, feat


class CNNTransformerNet(nn.Module):
    """CNN + Transformer 结合的分类模型。"""

    def __init__(self, n_classes: int, fs: float = 50e6, *, d_model: int = 256,
                 nhead: int = 4, num_encoder_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        backbone = _make_backbone()
        self.cnn_stem = nn.Sequential(*list(backbone.children())[:3])
        self._stem_channels = 256

        self.n_tokens = 256
        self.seq_down = nn.AdaptiveAvgPool1d(self.n_tokens)

        self.channel_to_model = nn.Linear(self._stem_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = max(d_model // 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.first(x)
        x = self.cnn_stem(x)
        b, c, _, l = x.shape
        x = x.view(b, c, l)
        x = self.seq_down(x)
        x = x.permute(2, 0, 1).contiguous()
        x = self.channel_to_model(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).contiguous()
        feat = self.seq_pool(x).squeeze(-1)
        logits = self.classifier(feat)
        return logits, feat


def create(name: str, num_classes: int, fs: float = 50e6, **kwargs) -> nn.Module:
    key = (name or "").strip().lower()
    if key == "perturbawarenet":
        return PerturbAwareNet(n_classes=num_classes, fs=fs)
    if key == "cnn_transformer":
        return CNNTransformerNet(n_classes=num_classes, fs=fs, **kwargs)
    raise KeyError(f"Unsupported model name: {name}")
