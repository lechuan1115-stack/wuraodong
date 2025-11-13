#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""单模型训练脚本（仅分类任务）。"""

import os
import os.path as osp
import time
import json
import csv
import datetime
from types import SimpleNamespace
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mydata_read
import mymodel1


class EarlyStopper:
    def __init__(self, mode="max", min_delta=0.0, patience=10, warmup=0):
        self.mode = mode
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.warmup = int(warmup)
        self.best = None
        self.num_bad = 0

    def _is_better(self, current, best):
        if best is None:
            return True
        if self.mode == "max":
            return (current - best) > self.min_delta
        else:
            return (best - current) > self.min_delta

    def step(self, current, epoch_idx):
        if epoch_idx < self.warmup:
            return False
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad >= self.patience


class NumpySignalDataset(Dataset):
    """把 numpy 数组打包成 PyTorch Dataset。"""

    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def _mixup(x, y, alpha=0.0):
    if not alpha or alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], perm, lam


def _unpack_model_outputs(output):
    if isinstance(output, (tuple, list)):
        logits = output[0] if len(output) > 0 else None
        feat = output[1] if len(output) > 1 else None
        return logits, feat
    return output, None


def train_one_epoch(model, optimizer, loader, device, cfg, epoch_idx):
    model.train()
    ce = nn.CrossEntropyLoss(weight=getattr(cfg, 'class_weight', None))
    losses = AverageMeter()
    correct = 0.0
    total = 0

    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device)

        x_mix, y_a, y_b, perm, lam = _mixup(x, y, cfg.mixup_alpha)
        logits, _ = _unpack_model_outputs(model(x_mix))

        if y_b is not None:
            cls_loss = lam * ce(logits, y_a) + (1.0 - lam) * ce(logits, y_b)
        else:
            cls_loss = ce(logits, y)

        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(1)
            if y_b is not None:
                acc = lam * (pred == y_a).float() + (1.0 - lam) * (pred == y_b).float()
                correct += acc.sum().item()
            else:
                correct += (pred == y).float().sum().item()
            total += y.size(0)
        losses.update(cls_loss.item(), y.size(0))

        if (it + 1) % cfg.print_freq == 0:
            print(f"  iter {it + 1:04d} | loss {losses.avg:.4f}")

    acc = 100.0 * correct / max(1, total)
    return acc, losses.avg


@torch.no_grad()
def _forward_eval(model, batch, device):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device)
    logits, feat = _unpack_model_outputs(model(x))
    return logits, feat, y


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    ce = nn.CrossEntropyLoss(weight=getattr(cfg, 'class_weight', None))
    losses = AverageMeter()
    correct = 0
    total = 0
    for batch in loader:
        logits, feat, y = _forward_eval(model, batch, device)
        loss = ce(logits, y)
        losses.update(loss.item(), y.size(0))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total), losses.avg


@torch.no_grad()
def test_and_visualize(model, loader, device, cfg, save_dir, class_names=None):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        logits, feat, y = _forward_eval(model, batch, device)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    top1 = (y_true == y_pred).mean() * 100.0

    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    except Exception:
        cm, report = None, None

    if cm is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        if class_names is not None:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        plt.savefig(osp.join(save_dir, 'confusion_matrix.png'), dpi=200)
        plt.close(fig)
        with open(osp.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report or '')

    return dict(top1=top1)


def plot_curves(histories, out_path):
    if not histories:
        return
    epochs = [h['epoch'] for h in histories]
    tr = [h['train_acc'] for h in histories]
    va = [h['val_acc'] for h in histories]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, tr, label='Train')
    plt.plot(epochs, va, label='Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training / Validation Curves')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run(cfg, device):
    print("程序：分类训练（无扰动标签）")
    print(f"数据路径：{cfg.data}")
    print(f"保存目录：{cfg.save_root}")
    print(f"模型：{cfg.model_name}")
    os.makedirs(cfg.save_root, exist_ok=True)

    X, Y, fs = mydata_read.load_adsb_aug5_strict(cfg.data, shuffle=False, seed=cfg.seed)
    Y = Y.astype(np.int64)

    uniq = np.unique(Y)
    print("[CHECK] classes in data =", uniq.tolist())
    assert len(uniq) == cfg.class_num, f"class_num({cfg.class_num}) 与数据({len(uniq)})不一致"

    by_cls = defaultdict(list)
    for i, y in enumerate(Y.tolist()):
        by_cls[y].append(i)

    _used_seed = cfg.split_seed if cfg.split_seed is not None else (int(time.time() * 1000) % (2 ** 32 - 1))
    rng = np.random.RandomState(_used_seed)

    tr, va, te = [], [], []
    val_ratio = float(getattr(cfg, "val_ratio", 0.10))
    test_ratio = float(getattr(cfg, "test_ratio", 0.10))
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and (val_ratio + test_ratio) < 1, "val/test 比例不合法"

    for c, ids in by_cls.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n = len(ids)

        n_va = int(round(val_ratio * n))
        n_te = int(round(test_ratio * n))
        n_tr = n - n_va - n_te

        if n >= 3:
            n_tr = max(n_tr, 1)
            n_va = max(n_va, 1)
            n_te = max(n - n_tr - n_va, 1)
        cut1 = n_tr
        cut2 = n_tr + n_va
        tr.extend(ids[:cut1])
        va.extend(ids[cut1:cut2])
        te.extend(ids[cut2:])

    print(f"[SPLIT] seed={_used_seed}  sizes: train={len(tr)}  val={len(va)}  test={len(te)}")
    st_tr, st_va, st_te = set(tr), set(va), set(te)
    print("[SPLIT] overlaps:", len(st_tr & st_va), len(st_tr & st_te), len(st_va & st_te))

    dataset = NumpySignalDataset(X, Y)
    train_set = Subset(dataset, tr)
    val_set = Subset(dataset, va)
    test_set = Subset(dataset, te)

    if getattr(cfg, "save_split", True):
        split_dir = osp.join(cfg.save_root, "split_indices")
        os.makedirs(split_dir, exist_ok=True)
        np.save(osp.join(split_dir, "train_idx.npy"), np.array(tr))
        np.save(osp.join(split_dir, "val_idx.npy"), np.array(va))
        np.save(osp.join(split_dir, "test_idx.npy"), np.array(te))
        with open(osp.join(split_dir, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"split_seed={_used_seed}\nval_ratio={val_ratio}\ntest_ratio={test_ratio}\n")

    pin = device.type == 'cuda'
    trainloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=pin)
    valloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=pin)
    testloader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=pin)

    model = mymodel1.create(name=cfg.model_name, num_classes=cfg.class_num, fs=fs)
    model = model.to(device)

    cnt = Counter([Y[i] for i in tr])
    freq = np.array([cnt.get(c, 1) for c in range(cfg.class_num)], dtype=np.float32)
    class_weight = 1.0 / np.maximum(freq, 1.0)
    class_weight *= (cfg.class_num / class_weight.sum())
    class_weight = torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    cfg.class_weight = class_weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    history = []
    best_val = -1e9
    best_state = None
    t0 = time.time()

    stopper = EarlyStopper(
        mode=cfg.es_mode,
        min_delta=cfg.es_min_delta,
        patience=cfg.es_patience,
        warmup=cfg.es_warmup
    ) if getattr(cfg, "early_stop", False) else None

    for epoch in range(cfg.max_epoch):
        print(f"==> Epoch {epoch + 1}/{cfg.max_epoch}")
        a_tr, l_tr = train_one_epoch(model, optimizer, trainloader, device, cfg, epoch_idx=epoch)
        a_va, l_va = evaluate(model, valloader, device, cfg)
        print(f"Train_Acc: {a_tr:.2f}%  Val_Acc: {a_va:.2f}%  (loss {l_tr:.4f}/{l_va:.4f})")

        history.append(dict(epoch=epoch + 1, train_acc=float(a_tr), val_acc=float(a_va)))

        if a_va > best_val:
            best_val = a_va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if stopper is not None:
            current_metric = a_va if cfg.es_metric == "val_acc" else l_va
            should_stop = stopper.step(current_metric, epoch_idx=epoch)
            if should_stop:
                best_str = (f"{stopper.best:.4f}" if isinstance(stopper.best, (int, float)) else str(stopper.best))
                print(
                    f"[EARLY STOP] No improvement in {cfg.es_patience} epochs "
                    f"(metric={cfg.es_metric}, best={best_str}). "
                    f"Stop at epoch {epoch + 1}."
                )
                break

    elapsed = time.time() - t0
    print("训练耗时：", str(datetime.timedelta(seconds=int(elapsed))))

    ckpt = osp.join(cfg.save_root, f"{cfg.model_name}_best.pt")
    torch.save(best_state, ckpt)
    print("[CKPT] 保存：", ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    acc_val, _ = evaluate(model, valloader, device, cfg)
    test_stats = test_and_visualize(model, testloader, device, cfg, save_dir=cfg.save_root)

    plot_curves(history, osp.join(cfg.save_root, "train_val_curves.png"))

    metrics = dict(val_acc=float(acc_val), test_top1=float(test_stats['top1']), train_time_sec=float(elapsed))
    csv_path = osp.join(cfg.save_root, "summary.csv")
    write_header = (not osp.exists(csv_path))
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["val_acc", "test_top1", "train_time_sec", "timestamp"])
        if write_header:
            w.writeheader()
        w.writerow({**metrics, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    with open(osp.join(cfg.save_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)

    print(f"Val_Acc: {metrics['val_acc']:.2f}%  TestTop1: {metrics['test_top1']:.2f}%")
    print("[DONE] 结果已写入：", cfg.save_root)


def main():
    configs = [
        dict(
            data=r"E:\数据集\ADS-B_Train_100_20dB.mat",
            save_root="./runs_perturbawarenet_20dB",
            model_name="perturbawarenet",
        ),
        dict(
            data=r"E:\数据集\ADS-B_Train_100_20dB.mat",
            save_root="./runs_cnn_transformer_20dB",
            model_name="cnn_transformer",
        ),
    ]

    base_cfg = dict(
        class_num=100,
        batch_size=32,
        workers=0,
        lr=1e-4,
        wd=1e-4,
        max_epoch=80,
        gpu='0',
        seed=42,
        print_freq=10,
        val_ratio=0.10,
        test_ratio=0.10,
        split_seed=None,
        save_split=True,
        early_stop=True,
        es_metric="val_acc",
        es_mode="max",
        es_min_delta=0.1,
        es_patience=8,
        es_warmup=5,
        mixup_alpha=0.2,
    )

    for i, exp in enumerate(configs):
        print(f"\n==================== 实验 {i + 1}/{len(configs)} ====================")
        print(f"数据: {exp['data']}")
        cfg = SimpleNamespace(**{**base_cfg, **exp})

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(cfg.seed)

        run(cfg, device)

    print("\n 所有模型训练完成！")


if __name__ == '__main__':
    main()
