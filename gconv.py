#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""群等变卷积层实现。

该模块提供用于 IQ 信号的一维群等变卷积算子，不再依赖任何扰动标签或参数。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_complex_kernel(weight: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """对卷积核的 IQ 通道施加复数相位旋转。

    参数:
        weight: [OC, IC, 1, k] 结构的核权重，要求前两个输入通道为 I/Q。
        angle:  标量张量或 Python 标量，表示旋转角度（弧度）。
    返回:
        旋转后的权重张量，形状与 ``weight`` 相同。
    """
    if weight.size(1) < 2:
        return weight

    angle = torch.as_tensor(angle, dtype=weight.dtype, device=weight.device)
    c = torch.cos(angle).view(1, 1, 1, 1)
    s = torch.sin(angle).view(1, 1, 1, 1)

    real = weight[:, 0:1, :, :]
    imag = weight[:, 1:2, :, :]

    rotated_real = real * c - imag * s
    rotated_imag = real * s + imag * c

    rotated = weight.clone()
    rotated[:, 0:1, :, :] = rotated_real
    rotated[:, 1:2, :, :] = rotated_imag
    if weight.size(1) > 2:
        rotated[:, 2:, :, :] = weight[:, 2:, :, :]
    return rotated


class GroupConv1xK(nn.Module):
    """一维 ``1×k`` 群等变卷积。

    通过对基础卷积核施加一组预定义的复数相位旋转来构造群元素，实现对
    IQ 平面上相位旋转群的等变性。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        n_group_elements: int = 4,
        bias: bool = False,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_group_elements < 1:
            raise ValueError("n_group_elements must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_group_elements = n_group_elements
        self.padding = kernel_size // 2 if padding is None else padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        angles = torch.arange(n_group_elements, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / n_group_elements)
        self.register_buffer("angles", angles)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernels = [
            rotate_complex_kernel(self.weight, angle.to(dtype=self.weight.dtype))
            for angle in self.angles
        ]
        weight = torch.cat(kernels, dim=0)
        bias: Optional[torch.Tensor]
        if self.bias is not None:
            bias = self.bias.repeat(self.n_group_elements)
        else:
            bias = None
        return F.conv2d(x, weight, bias=bias, stride=1, padding=(0, self.padding))


class GroupConvFirstLayer(nn.Module):
    """网络的第一层：群等变卷积 + BN + ReLU。"""

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 32,
        k: int = 5,
        *,
        n_group_elements: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if out_ch % n_group_elements != 0:
            raise ValueError("out_ch must be divisible by n_group_elements")

        base_channels = out_ch // n_group_elements
        self.group_conv = GroupConv1xK(
            in_channels=in_ch,
            out_channels=base_channels,
            kernel_size=k,
            n_group_elements=n_group_elements,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(base_channels * n_group_elements)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.group_conv(x)
        x = self.bn(x)
        return self.act(x)


class PlainFirstLayer(nn.Module):
    """最简单的第一层：标准卷积 + BN + ReLU。"""

    def __init__(self, in_ch: int = 2, out_ch: int = 32, k: int = 5) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
