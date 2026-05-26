# -*- coding: utf-8 -*-
"""
train_group_stance.py
群体立场分类实验 - 训练主脚本
支持命令行参数 --model M0/M1/M2/M3
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 添加父目录到路径
TASK5_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK5_DIR)

from data_builder import GroupStanceDataset
from models.group_stance_classifier import StanceClassifier

# sklearn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# ─────────────────────── 全局配置 ───────────────────────
SEED = 2023
RESULTS_DIR = os.path.join(TASK5_DIR, 'results')
CKPT_DIR    = os.path.join(RESULTS_DIR, 'model_checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────── 早停监控 ───────────────────────
class EarlyStopMonitor:
    """监控 val_F1，patience 轮无提升则停止"""

    def __init__(self, patience=20):
        self.patience    = patience
        self.best_val_f1 = -1.0
        self.counter     = 0
        self.should_stop = False

    def update(self, val_f1):
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.counter     = 0
            return True   # 有提升，返回 True 表示可以保存
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ─────────────────────── 评估函数 ───────────────────────
def evaluate(model, loader, criterion, device):
    """
    评估模型，返回 (auc, f1_macro, acc, loss)
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs      = torch.softmax(all_logits, dim=1).numpy()
    preds      = np.argmax(probs, axis=1)

    avg_loss = total_loss / len(all_labels)

    # F1 macro
    f1 = f1_score(all_labels, preds, average='macro', zero_division=0)

    # AUC（对二分类取正类概率）
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(all_labels, probs[:, 1])
    else:
        auc = 0.5

    acc = accuracy_score(all_labels, preds)

    return float(auc), float(f1), float(acc), float(avg_loss)


# ─────────────────────── 训练主函数 ───────────────────────
def train_model(model_type='M2', verbose=True):
    """
    训练指定类型的模型，返回 (train_metrics, val_metrics, test_metrics)。
    每项 metrics = {'auc': ..., 'f1': ..., 'acc': ..., 'loss': ...}
    """
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 构建数据集 ──
    ds = GroupStanceDataset(noise_std=0.01, random_state=SEED)
    train_X, train_y, val_X, val_y, test_X, test_y = ds.build_features(model_type)

    if verbose:
        print(f"\n=== 训练模型: {model_type} ===")
        print(f"  设备: {device}")
        print(f"  特征维度: {train_X.shape[1]}")
        print(f"  训练集: {train_X.shape[0]}  验证集: {val_X.shape[0]}  测试集: {test_X.shape[0]}")
        print(f"  标签分布 - 训练集: dem={( train_y==0).sum()} rep={(train_y==1).sum()}")

    # ── 转 Tensor ──
    def to_loader(X, y, batch_size=32, shuffle=False):
        ds_ = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(ds_, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_X, train_y, batch_size=32, shuffle=True)
    val_loader   = to_loader(val_X,   val_y,   batch_size=64, shuffle=False)
    test_loader  = to_loader(test_X,  test_y,  batch_size=64, shuffle=False)

    # ── 类别权重 ──
    n_dem = 4; n_rep = 3; total = 7
    w_dem = total / (2 * n_dem)   # 0.875
    w_rep = total / (2 * n_rep)   # 1.167
    class_weights = torch.tensor([w_dem, w_rep], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 模型 ──
    input_dim = train_X.shape[1]
    model     = StanceClassifier(input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    monitor   = EarlyStopMonitor(patience=20)

    ckpt_path = os.path.join(CKPT_DIR, f'{model_type}_best.pth')

    # ── 训练循环 ──
    best_val_f1     = -1.0
    best_state_dict = None
    MAX_EPOCHS      = 300

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证
        val_auc, val_f1, val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        improved = monitor.update(val_f1)
        if improved:
            best_val_f1     = val_f1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 10 == 0:
            tr_auc, tr_f1, tr_acc, tr_loss = evaluate(model, train_loader, criterion, device)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.3f}  F1: {tr_f1:.3f}  AUC: {tr_auc:.3f} | "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  F1: {val_f1:.3f}  AUC: {val_auc:.3f} | "
                  f"lr: {cur_lr:.6f}")

        if monitor.should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val F1: {best_val_f1:.3f})")
            break

    # ── 保存最优模型 ──
    if best_state_dict is not None:
        torch.save(best_state_dict, ckpt_path)
        model.load_state_dict(best_state_dict)
        if verbose:
            print(f"  最优模型已保存: {ckpt_path}")

    # ── 最终评估 ──
    train_auc, train_f1, train_acc, train_loss = evaluate(model, train_loader, criterion, device)
    val_auc,   val_f1,   val_acc,   val_loss   = evaluate(model, val_loader,   criterion, device)
    test_auc,  test_f1,  test_acc,  test_loss  = evaluate(model, test_loader,  criterion, device)

    if verbose:
        print(f"\n=== {model_type} 最终结果 ===")
        print(f"Train: loss={train_loss:.3f} acc={train_acc:.2f} f1={train_f1:.2f} auc={train_auc:.2f}")
        print(f"Val:   loss={val_loss:.3f} acc={val_acc:.2f} f1={val_f1:.2f} auc={val_auc:.2f}")
        print(f"Test:  loss={test_loss:.3f} acc={test_acc:.2f} f1={test_f1:.2f} auc={test_auc:.2f}")

    train_metrics = {'auc': train_auc, 'f1': train_f1, 'acc': train_acc, 'loss': train_loss}
    val_metrics   = {'auc': val_auc,   'f1': val_f1,   'acc': val_acc,   'loss': val_loss}
    test_metrics  = {'auc': test_auc,  'f1': test_f1,  'acc': test_acc,  'loss': test_loss}

    return train_metrics, val_metrics, test_metrics


# ─────────────────────── CLI 入口 ───────────────────────
def main():
    parser = argparse.ArgumentParser(description='群体立场分类训练脚本')
    parser.add_argument('--model', type=str, default='M2',
                        choices=['M0', 'M1', 'M2', 'M3'],
                        help='特征模式: M0(81d) M1(64d) M2(74d) M3(155d)')
    args = parser.parse_args()

    train_m, val_m, test_m = train_model(args.model, verbose=True)

    # 保存结果
    eval_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[args.model] = {
        'train': train_m,
        'val':   val_m,
        'test':  test_m,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(eval_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {eval_path}")


if __name__ == '__main__':
    main()
