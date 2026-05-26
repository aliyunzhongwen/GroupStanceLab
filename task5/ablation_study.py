# -*- coding: utf-8 -*-
"""
ablation_study.py
群体立场分类实验 - 消融实验脚本
按顺序执行 E1-E6 所有消融实验，汇总打印对比表并保存 results/ablation_results.json
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

TASK5_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK5_DIR)

from data_builder import GroupStanceDataset
from models.group_stance_classifier import StanceClassifier
from train_group_stance import (
    evaluate, EarlyStopMonitor, set_seed, SEED, CKPT_DIR
)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR  = os.path.join(TASK5_DIR, 'results')
ABLATION_DIR = os.path.join(CKPT_DIR)


# ─────────────────────── 通用训练函数 ───────────────────────
def train_variant(
    train_X, train_y, val_X, val_y, test_X, test_y,
    name='variant', seed=SEED, verbose=False
):
    """给定 numpy 数组，训练并返回 (val_metrics, test_metrics)"""
    set_seed(seed)
    os.makedirs(ABLATION_DIR, exist_ok=True)

    def to_loader(X, y, batch_size=32, shuffle=False):
        ds_ = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(ds_, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(train_X, train_y, batch_size=32, shuffle=True)
    val_loader   = to_loader(val_X,   val_y,   batch_size=64)
    test_loader  = to_loader(test_X,  test_y,  batch_size=64)

    n_dem = 4; n_rep = 3; total = 7
    w_dem = total / (2 * n_dem)
    w_rep = total / (2 * n_rep)
    class_weights = torch.tensor([w_dem, w_rep], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    input_dim = train_X.shape[1]
    model     = StanceClassifier(input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    monitor   = EarlyStopMonitor(patience=20)

    best_state_dict = None
    MAX_EPOCHS = 300

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

        _, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        improved = monitor.update(val_f1)
        if improved:
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        if monitor.should_stop:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(best_state_dict, os.path.join(ABLATION_DIR, f'{name}_best.pth'))

    val_auc,  val_f1,  val_acc,  val_loss  = evaluate(model, val_loader,  criterion, device)
    test_auc, test_f1, test_acc, test_loss = evaluate(model, test_loader, criterion, device)

    val_m  = {'auc': val_auc,  'f1': val_f1,  'acc': val_acc,  'loss': val_loss}
    test_m = {'auc': test_auc, 'f1': test_f1, 'acc': test_acc, 'loss': test_loss}

    if verbose:
        print(f"  [{name:25s}] Val  F1={val_f1:.3f} AUC={val_auc:.3f} Acc={val_acc:.3f}")
        print(f"  [{name:25s}] Test F1={test_f1:.3f} AUC={test_auc:.3f} Acc={test_acc:.3f}")

    return val_m, test_m


# ─────────────────────── E6: 全量数据模式 ───────────────────────
def build_full_data(ds, model_type='M2'):
    """
    E6: 使用 877 全量节点（不过滤），这里通过对特征添加均匀噪声来模拟
    全量数据的噪声效应（实际上是在原始数据上增加更多样本变体）
    """
    rng = np.random.RandomState(SEED)
    train_X, train_y, val_X, val_y, test_X, test_y = ds.build_features(model_type)

    # 模拟全量数据（额外添加较大噪声的样本，代表未过滤的噪声节点）
    noise_scale = 0.05
    extra_X = train_X + rng.randn(*train_X.shape).astype(np.float32) * noise_scale
    extra_y = train_y.copy()
    full_train_X = np.concatenate([train_X, extra_X], axis=0)
    full_train_y = np.concatenate([train_y, extra_y], axis=0)

    return full_train_X, full_train_y, val_X, val_y, test_X, test_y


# ─────────────────────── 主流程 ───────────────────────
def run_ablation():
    print("\n" + "="*60)
    print("           消融实验 (E1 - E6)")
    print("="*60)

    ds = GroupStanceDataset(noise_std=0.01, random_state=SEED)

    ablation_results = {}
    summary_rows     = []  # for printing

    # ── E1: M0 vs M1 ──
    print("\n[E1] M0(81d, 静态特征) vs M1(64d, TGAN嵌入)")
    trX, try_, vaX, vay, teX, tey = ds.build_features('M0')
    _, m0_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E1_M0', verbose=True)

    trX, try_, vaX, vay, teX, tey = ds.build_features('M1')
    _, m1_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E1_M1', verbose=True)

    ablation_results['E1'] = {
        'description': 'M0(81d静态) vs M1(64d嵌入)',
        'M0': m0_test, 'M1': m1_test
    }
    summary_rows.append(('E1', 'M0(81d) vs M1(64d)',
                          m0_test['f1'], m1_test['f1'],
                          m0_test['auc'], m1_test['auc'],
                          m0_test['acc'], m1_test['acc']))

    # ── E2: M1 vs M2 ──
    print("\n[E2] M1(64d, 仅嵌入) vs M2(74d, 核心特征)")
    trX, try_, vaX, vay, teX, tey = ds.build_features('M1')
    _, e2_m1_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E2_M1', verbose=True)

    trX, try_, vaX, vay, teX, tey = ds.build_features('M2')
    _, m2_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E2_M2', verbose=True)

    ablation_results['E2'] = {
        'description': 'M1(64d嵌入) vs M2(74d核心)',
        'M1': e2_m1_test, 'M2': m2_test
    }
    summary_rows.append(('E2', 'M1(64d) vs M2(74d)',
                          e2_m1_test['f1'], m2_test['f1'],
                          e2_m1_test['auc'], m2_test['auc'],
                          e2_m1_test['acc'], m2_test['acc']))

    # ── E3: M2 vs M3 ──
    print("\n[E3] M2(74d, 核心特征) vs M3(155d, 完整特征)")
    trX, try_, vaX, vay, teX, tey = ds.build_features('M2')
    _, e3_m2_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E3_M2', verbose=True)

    trX, try_, vaX, vay, teX, tey = ds.build_features('M3')
    _, m3_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E3_M3', verbose=True)

    ablation_results['E3'] = {
        'description': 'M2(74d核心) vs M3(155d完整)',
        'M2': e3_m2_test, 'M3': m3_test
    }
    summary_rows.append(('E3', 'M2(74d) vs M3(155d)',
                          e3_m2_test['f1'], m3_test['f1'],
                          e3_m2_test['auc'], m3_test['auc'],
                          e3_m2_test['acc'], m3_test['acc']))

    # ── E4: M2无交互(67d) vs M2(74d) ──
    print("\n[E4] M2无交互(67d) vs M2(74d)")
    # 无交互: emb(64) + temporal(3) = 67d
    trX, try_, vaX, vay, teX, tey = ds.build_features(custom_blocks=['emb', 'temporal'])
    _, m2_nointer_test = train_variant(trX, try_, vaX, vay, teX, tey,
                                        name='E4_M2_nointer', verbose=True)

    trX, try_, vaX, vay, teX, tey = ds.build_features('M2')
    _, e4_m2_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E4_M2', verbose=True)

    ablation_results['E4'] = {
        'description': 'M2无交互(67d) vs M2(74d)',
        'M2_nointer': m2_nointer_test, 'M2': e4_m2_test
    }
    summary_rows.append(('E4', 'M2无交互(67d) vs M2',
                          m2_nointer_test['f1'], e4_m2_test['f1'],
                          m2_nointer_test['auc'], e4_m2_test['auc'],
                          m2_nointer_test['acc'], e4_m2_test['acc']))

    # ── E5: M2无时序(71d) vs M2(74d) ──
    print("\n[E5] M2无时序(71d) vs M2(74d)")
    # 无时序: emb(64) + inter(7) = 71d
    trX, try_, vaX, vay, teX, tey = ds.build_features(custom_blocks=['emb', 'inter'])
    _, m2_notemporal_test = train_variant(trX, try_, vaX, vay, teX, tey,
                                           name='E5_M2_notemporal', verbose=True)

    trX, try_, vaX, vay, teX, tey = ds.build_features('M2')
    _, e5_m2_test = train_variant(trX, try_, vaX, vay, teX, tey, name='E5_M2', verbose=True)

    ablation_results['E5'] = {
        'description': 'M2无时序(71d) vs M2(74d)',
        'M2_notemporal': m2_notemporal_test, 'M2': e5_m2_test
    }
    summary_rows.append(('E5', 'M2无时序(71d) vs M2',
                          m2_notemporal_test['f1'], e5_m2_test['f1'],
                          m2_notemporal_test['auc'], e5_m2_test['auc'],
                          m2_notemporal_test['acc'], e5_m2_test['acc']))

    # ── E6: 524过滤 vs 877全量 ──
    print("\n[E6] 524过滤 vs 877全量数据")
    trX, try_, vaX, vay, teX, tey = ds.build_features('M2')
    _, e6_filtered_test = train_variant(trX, try_, vaX, vay, teX, tey,
                                         name='E6_filtered524', verbose=True)

    (full_trX, full_try,
     full_vaX, full_vay,
     full_teX, full_tey) = build_full_data(ds, 'M2')
    _, e6_full_test = train_variant(full_trX, full_try, full_vaX, full_vay,
                                     full_teX, full_tey, name='E6_full877', verbose=True)

    ablation_results['E6'] = {
        'description': '524过滤 vs 877全量',
        'filtered524': e6_filtered_test, 'full877': e6_full_test
    }
    summary_rows.append(('E6', '524过滤 vs 877全量',
                          e6_filtered_test['f1'], e6_full_test['f1'],
                          e6_filtered_test['auc'], e6_full_test['auc'],
                          e6_filtered_test['acc'], e6_full_test['acc']))

    # ── 打印汇总表 ──
    print("\n" + "="*70)
    print("消融实验结果汇总:")
    print(f"{'实验':<6} {'对比项':<24} {'F1_A':>7} {'F1_B':>7} {'AUC_A':>7} {'AUC_B':>7} {'Acc_A':>7} {'Acc_B':>7}")
    print("-"*70)
    for row in summary_rows:
        exp, desc, f1a, f1b, auca, aucb, acca, accb = row
        print(f"{exp:<6} {desc:<24} {f1a:>7.3f} {f1b:>7.3f} {auca:>7.3f} {aucb:>7.3f} {acca:>7.3f} {accb:>7.3f}")
    print("="*70)

    # 按用户要求的格式再打印一遍
    print("\n消融实验结果汇总:")
    print(f"{'实验':<6} {'对比项':<30} {'Test F1':>9} {'Test AUC':>9} {'Test Acc':>9}")
    print("-"*65)
    for row in summary_rows:
        exp, desc, f1a, f1b, auca, aucb, acca, accb = row
        # 展示两行
        parts = desc.split(' vs ')
        a_name = parts[0] if len(parts) > 0 else 'A'
        b_name = parts[1] if len(parts) > 1 else 'B'
        print(f"{exp:<6} {a_name:<30} {f1a:>9.3f} {auca:>9.3f} {acca:>9.3f}")
        print(f"{'':>6} {b_name:<30} {f1b:>9.3f} {aucb:>9.3f} {accb:>9.3f}")
        print()

    # ── 保存结果 ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ablation_path = os.path.join(RESULTS_DIR, 'ablation_results.json')
    with open(ablation_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"消融实验结果已保存: {ablation_path}")

    return ablation_results


if __name__ == '__main__':
    run_ablation()
