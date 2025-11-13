#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""针对新的 ADS-B 数据集读取工具：仅包含信号与类别标签。"""

import os
import os.path as osp
import json
import numpy as np
import h5py

SIGNAL_KEY = "signal"
FS_KEY = "fs"
TRAIN_LABEL_KEY = "trainlabel"
ALT_LABEL_KEY = "label"


def _check(path: str) -> str:
    path = osp.abspath(path)
    print(f"[DATA] {path}")
    if not osp.exists(path):
        raise FileNotFoundError(path)
    return path


def _signal_legacy_to_p4(arr: np.ndarray) -> np.ndarray:
    """支持多种布局，将信号转换为 [N,2,1,L]。"""
    if arr.ndim != 3:
        raise ValueError(f"signal 需要为 3 维张量，实际形状 {arr.shape}")

    if arr.shape[0] == 2:  # 旧格式: (2, L, N)
        I = arr[0].astype(np.float32).T  # (N, L)
        Q = arr[1].astype(np.float32).T
    elif arr.shape[1] == 2:  # 可能是 (N, 2, L)
        I = arr[:, 0, :].astype(np.float32)
        Q = arr[:, 1, :].astype(np.float32)
    elif arr.shape[2] == 2:  # 可能是 (N, L, 2)
        I = arr[:, :, 0].astype(np.float32)
        Q = arr[:, :, 1].astype(np.float32)
    else:
        raise ValueError(f"无法解析 signal 的形状 {arr.shape}")

    return np.stack([I, Q], axis=1)[:, :, None, :]


def load_adsb_aug5_strict(path: str, shuffle: bool = False, seed: int = 42):
    """读取新的 ADS-B 数据集。

    返回:
        X: [N, 2, 1, L]
        Y: [N]
        fs: 采样率 (float)
    """
    path = _check(path)
    with h5py.File(path, "r") as hf:
        keys = sorted(hf.keys())
        print(json.dumps({"keys": keys}, ensure_ascii=False))

        if SIGNAL_KEY not in hf:
            raise KeyError(f"缺少字段: {SIGNAL_KEY}")
        if FS_KEY not in hf:
            raise KeyError(f"缺少字段: {FS_KEY}")

        signal = np.array(hf[SIGNAL_KEY])
        fs_arr = np.array(hf[FS_KEY])

        if TRAIN_LABEL_KEY in hf:
            y_raw = np.array(hf[TRAIN_LABEL_KEY])
        elif ALT_LABEL_KEY in hf:
            y_raw = np.array(hf[ALT_LABEL_KEY])
        else:
            raise KeyError(f"缺少字段: {TRAIN_LABEL_KEY} / {ALT_LABEL_KEY}")

    X = _signal_legacy_to_p4(signal)
    Y = y_raw.reshape(-1).astype(np.int64)
    fs = float(np.array(fs_arr).reshape(-1)[0])

    if shuffle:
        idx = np.random.RandomState(seed).permutation(X.shape[0])
        X = X[idx]
        Y = Y[idx]

    L = X.shape[-1]
    print(json.dumps({
        "summary": f"N={X.shape[0]}, L={L}, X{X.shape}, Y{Y.shape}, fs={fs}"
    }, ensure_ascii=False))
    return X, Y, fs


if __name__ == "__main__":
    p = r"E:\数据集\ADS-B_Train_100_20dB.mat"
    X, Y, fs = load_adsb_aug5_strict(p, shuffle=False)
    print("加载完成", X.shape, Y.shape, fs)
