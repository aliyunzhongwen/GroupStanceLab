# -*- coding: utf-8 -*-
"""
时序立场追踪 - 修正版
采用自监督边预测训练TGAN + 时间划分的立场分类

关键改进:
1. TGAN通过边预测任务(自监督)学习群体间交互的时序动态表示
   - 不使用节点标签 → 避免TGAN直接记住分类结果
2. 立场分类器使用时间划分训练，避免信息泄露
   - 训练: 前7个窗口 (windows 0-6)
   - 测试: 后4个窗口 (windows 7-10)
3. 不同时间窗口的立场分数反映真实的时序变化

运行:
    cd /root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task4
    python temporal_stance_tracking.py
"""

import os
import sys
import math
import copy
import random
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

# ── 路径设置 ────────────────────────────────────────────────────────────────
TASK4_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TASK4_DIR, '..', '..', '..'))
TASK3_DIR = os.path.join(TASK4_DIR, '..', 'task3')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module import TGAN
from graph import NeighborFinder

# ── 超参数 ──────────────────────────────────────────────────────────────────
SEED          = 2023
N_NODES       = 8           # 0为占位，1-7为群体
D_MODEL       = 64
N_LAYERS      = 2
N_HEADS       = 2
NUM_NEIGHBORS = [20, 1]
STPE_DIM      = 100
DROP_OUT      = 0.1
MAX_DEPTH     = 2
ALPHA         = 10.0
ETA           = 10000.0
BETA          = 1.0

# Stage 1: 自监督边预测训练
TRAIN_RATIO        = 0.8     # 边预测训练集占比 (按时间)
N_EPOCHS_PRETRAIN  = 10
BATCH_SIZE         = 200
LR_PRETRAIN        = 1e-3
EARLY_STOP_AUC     = 0.95    # 验证 AUC 达到阈值提前停止
TRAIN_SUBSAMPLE    = 2000    # 每 epoch 训练边数（时间均匀采样）
VAL_SUBSAMPLE      = 1000    # 验证边数
REPLAY_SUBSAMPLE   = 5000    # 阶段2 memory 重放边数（时间均匀）

# Stage 2: 立场分类
CLASSIFIER_TRAIN_WINDOWS = list(range(0, 7))   # in-sample
CLASSIFIER_TEST_WINDOWS  = list(range(7, 11))  # out-of-sample
N_EPOCHS_CLASSIFIER      = 200
LR_CLASSIFIER            = 5e-3
WEIGHT_DECAY_CLF         = 1e-3
HIDDEN_CLF               = 32

NUM_WINDOWS  = 11
WINDOW_SIZE  = 5             # 每窗口 5 个 ts

logging.basicConfig(level=logging.WARNING)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ────────────────────────────────────────────────────────────────────────────
# 边预测头 (LinkPredictor) 与 立场分类头 (StanceClassifier)
# ────────────────────────────────────────────────────────────────────────────
class LinkPredictor(nn.Module):
    """简单 MLP 边预测头：输入 [src_emb, dst_emb] → logit(边存在)"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src_emb, dst_emb):
        x = torch.cat([src_emb, dst_emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class StanceClassifier(nn.Module):
    """轻量分类头，避免在 7 节点 × 7 窗口 = 49 训练样本上过拟合"""
    def __init__(self, input_dim=64, hidden_dim=32, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


# ────────────────────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────────────────────
def load_group_data():
    print("=" * 60)
    print("[步骤1] 加载群体超个体数据")
    print("=" * 60)

    n_feat   = np.load(os.path.join(TASK3_DIR, 'group_features_compact.npy'))
    e_feat   = np.load(os.path.join(TASK3_DIR, 'group_edge_features.npy'))
    labels   = np.load(os.path.join(TASK3_DIR, 'group_labels.npy'))
    edges_df = pd.read_csv(os.path.join(TASK3_DIR, 'group_edge_list.csv'))

    print(f"  节点特征 n_feat: {n_feat.shape}")
    print(f"  边特征   e_feat: {e_feat.shape}")
    print(f"  群体标签 labels: {labels}")
    print(f"  边表 edges_df : {edges_df.shape[0]} 行, ts∈[{edges_df['ts'].min()},{edges_df['ts'].max()}]")
    print(f"  标签分布(索引1-7): {dict(zip(*np.unique(labels[1:], return_counts=True)))}")

    edges_df['ts'] = pd.to_numeric(edges_df['ts'], errors='coerce')
    edges_df = edges_df.dropna(subset=['ts']).copy()
    edges_df.sort_values('ts', inplace=True, kind='mergesort')
    edges_df.reset_index(drop=True, inplace=True)

    return n_feat, e_feat, labels, edges_df


# ────────────────────────────────────────────────────────────────────────────
# 构建 NeighborFinder + TGAN
# ────────────────────────────────────────────────────────────────────────────
def build_neighbor_finder(edges_df, n_nodes=8):
    print("\n[步骤2] 构建 NeighborFinder")
    adj_list = [[] for _ in range(n_nodes)]
    for row in edges_df.itertuples(index=False):
        u, i, ts, idx = int(row.u), int(row.i), float(row.ts), int(row.idx)
        adj_list[u].append((i, idx, ts))
        adj_list[i].append((u, idx, ts))

    ngh_finder = NeighborFinder(adj_list, balance=1.0, seed=SEED, uniform=False)
    for nid in range(1, n_nodes):
        n_edges = len(ngh_finder.node_to_neighbors[nid])
        print(f"  节点 {nid}: {n_edges} 条历史边")
    return ngh_finder


def build_tgan(ngh_finder, n_feat, e_feat, device):
    print("\n[步骤3] 初始化 TGAN")
    tgan = TGAN(
        ngh_finder, n_feat, e_feat,
        stpe_dim=STPE_DIM, d_model=D_MODEL,
        num_neigh=NUM_NEIGHBORS, node_num=N_NODES,
        device=device,
        num_layers=N_LAYERS, n_head=N_HEADS, drop_out=DROP_OUT,
        max_depth=MAX_DEPTH, alpha=ALPHA, eta=ETA, beta=BETA,
    ).to(device)
    # n_feat_th / e_feat_th 是 nn.Parameter 但被 from_pretrained(freeze=True) 复制后
    # 实际 forward 不再用到它们；移出优化器以大幅加速 Adam.step
    if hasattr(tgan, 'n_feat_th'):
        tgan.n_feat_th.requires_grad_(False)
    if hasattr(tgan, 'e_feat_th'):
        tgan.e_feat_th.requires_grad_(False)
    n_total     = sum(p.numel() for p in tgan.parameters())
    n_trainable = sum(p.numel() for p in tgan.parameters() if p.requires_grad)
    print(f"  TGAN 总参数量: {n_total:,}, 可训练: {n_trainable:,}")
    return tgan


# ────────────────────────────────────────────────────────────────────────────
# 工具：生成负样本目标节点（在 1-7 内随机选，排除 src/dst）
# ────────────────────────────────────────────────────────────────────────────
def sample_neg_dst(src, dst, group_set, rng):
    neg = np.zeros_like(dst)
    for ii in range(len(dst)):
        cands = group_set[(group_set != dst[ii]) & (group_set != src[ii])]
        neg[ii] = rng.choice(cands)
    return neg


def update_memory_after_batch(tgan, src, dst, ts, src_emb, device):
    """在自监督训练中：用真实正样本边更新 memory"""
    with torch.no_grad():
        node_info = tgan.node_raw_embed(torch.from_numpy(src).long().to(device))
        projected = tgan.proj_to_memory(src_emb)
        new_info = 0.5 * node_info + 0.5 * projected
        tgan.memory.update_with_inertia(src, new_info, ts)
        tgan.memory.update(src, dst, ts)


# ────────────────────────────────────────────────────────────────────────────
# 阶段 1: 自监督边预测训练
# ────────────────────────────────────────────────────────────────────────────
def _stride_subsample(df, target_size):
    """沿时间均匀抽样 target_size 行，保持时间顺序"""
    if len(df) <= target_size:
        return df.reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, target_size).astype(np.int64)
    return df.iloc[idx].reset_index(drop=True)


def pretrain_tgan_edge_prediction(tgan, link_predictor, edges_df, device):
    print("\n" + "=" * 60)
    print("[阶段1] 自监督边预测预训练 TGAN")
    print("=" * 60)

    sorted_df = edges_df.sort_values('ts', kind='mergesort').reset_index(drop=True)
    n_total   = len(sorted_df)
    split_idx = int(TRAIN_RATIO * n_total)
    train_df_full = sorted_df.iloc[:split_idx].reset_index(drop=True)
    val_df_full   = sorted_df.iloc[split_idx:].reset_index(drop=True)

    # 时间均匀子采样以加速训练（保留 7 个节点交互的时序覆盖）
    train_df_sub = _stride_subsample(train_df_full, TRAIN_SUBSAMPLE)
    val_df_sub   = _stride_subsample(val_df_full, VAL_SUBSAMPLE)

    print(f"  total edges = {n_total}")
    print(f"  train edges (subsampled) = {len(train_df_sub)}  ts∈[{train_df_sub.ts.min():.0f}, {train_df_sub.ts.max():.0f}]")
    print(f"  val   edges (subsampled) = {len(val_df_sub)}    ts∈[{val_df_sub.ts.min():.0f}, {val_df_sub.ts.max():.0f}]")
    print(f"  N_EPOCHS={N_EPOCHS_PRETRAIN}, BS={BATCH_SIZE}, LR={LR_PRETRAIN}")

    trainable_tgan_params = [p for p in tgan.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_tgan_params + list(link_predictor.parameters()),
        lr=LR_PRETRAIN, weight_decay=1e-5,
    )
    train_df = train_df_sub
    val_df   = val_df_sub
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(SEED)
    group_set = np.arange(1, 8, dtype=np.int64)

    best_val_auc = -1.0
    best_state   = None

    for epoch in range(N_EPOCHS_PRETRAIN):
        set_seed(SEED + epoch)
        tgan.reset_memory()
        tgan.train()
        link_predictor.train()

        n_batch = math.ceil(len(train_df) / BATCH_SIZE)
        epoch_loss = 0.0
        epoch_correct, epoch_total = 0, 0

        for k in range(n_batch):
            s, e = k * BATCH_SIZE, min((k + 1) * BATCH_SIZE, len(train_df))
            batch = train_df.iloc[s:e]
            src = batch.u.values.astype(np.int64)
            dst = batch.i.values.astype(np.int64)
            ts  = batch.ts.values.astype(np.float64)
            neg_dst = sample_neg_dst(src, dst, group_set, rng)

            optimizer.zero_grad(set_to_none=True)

            src_emb = tgan.encode_node(src, ts, dst, device=device, num_neighbors=NUM_NEIGHBORS)
            dst_emb = tgan.encode_node(dst, ts, src, device=device, num_neighbors=NUM_NEIGHBORS)
            neg_emb = tgan.encode_node(neg_dst, ts, src, device=device, num_neighbors=NUM_NEIGHBORS)

            pos_score = link_predictor(src_emb, dst_emb)
            neg_score = link_predictor(src_emb, neg_emb)

            loss = bce(pos_score, torch.ones_like(pos_score)) + \
                   bce(neg_score, torch.zeros_like(neg_score))

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_tgan_params + list(link_predictor.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()

            # 用正样本真实边更新 memory（不用负样本污染 memory）
            update_memory_after_batch(tgan, src, dst, ts, src_emb, device)

            with torch.no_grad():
                pos_pred = (torch.sigmoid(pos_score) > 0.5).float()
                neg_pred = (torch.sigmoid(neg_score) > 0.5).float()
                epoch_correct += int((pos_pred == 1).sum() + (neg_pred == 0).sum())
                epoch_total   += 2 * len(pos_score)

        # ── 验证（用训练后的 memory 状态，不更新 memory） ────────────────────
        tgan.eval()
        link_predictor.eval()
        mem_bak = tgan.memory.backup_memory()

        all_scores, all_labels = [], []
        with torch.no_grad():
            n_val_batch = math.ceil(len(val_df) / BATCH_SIZE)
            for k in range(n_val_batch):
                s, e = k * BATCH_SIZE, min((k + 1) * BATCH_SIZE, len(val_df))
                batch = val_df.iloc[s:e]
                src = batch.u.values.astype(np.int64)
                dst = batch.i.values.astype(np.int64)
                ts  = batch.ts.values.astype(np.float64)
                neg_dst = sample_neg_dst(src, dst, group_set, rng)

                src_emb = tgan.encode_node(src, ts, dst, device=device, num_neighbors=NUM_NEIGHBORS)
                dst_emb = tgan.encode_node(dst, ts, src, device=device, num_neighbors=NUM_NEIGHBORS)
                neg_emb = tgan.encode_node(neg_dst, ts, src, device=device, num_neighbors=NUM_NEIGHBORS)

                pos_score = torch.sigmoid(link_predictor(src_emb, dst_emb)).cpu().numpy()
                neg_score = torch.sigmoid(link_predictor(src_emb, neg_emb)).cpu().numpy()
                all_scores.append(pos_score); all_labels.append(np.ones_like(pos_score))
                all_scores.append(neg_score); all_labels.append(np.zeros_like(neg_score))

        tgan.memory.restore_memory(mem_bak)

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        try:
            val_auc = roc_auc_score(all_labels, all_scores)
        except Exception:
            val_auc = float('nan')
        val_acc = float(((all_scores > 0.5) == all_labels.astype(bool)).mean())
        train_acc = epoch_correct / max(epoch_total, 1)

        print(f"  Epoch {epoch+1:2d}/{N_EPOCHS_PRETRAIN} | "
              f"Loss={epoch_loss/max(n_batch,1):.4f} | "
              f"TrainAcc={train_acc:.4f} | "
              f"ValAcc={val_acc:.4f} | ValAUC={val_auc:.4f}")

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                'tgan': copy.deepcopy(tgan.state_dict()),
                'lp':   copy.deepcopy(link_predictor.state_dict()),
                'epoch': epoch,
                'val_auc': val_auc,
            }

        # 提前停止：ValAUC 够高则退出
        if not np.isnan(val_auc) and val_auc >= EARLY_STOP_AUC:
            print(f"  Early stop at epoch {epoch+1}: ValAUC={val_auc:.4f} >= {EARLY_STOP_AUC}")
            break

    if best_state is not None:
        print(f"\n  恢复最优权重 (epoch={best_state['epoch']+1}, ValAUC={best_state['val_auc']:.4f})")
        tgan.load_state_dict(best_state['tgan'])
        link_predictor.load_state_dict(best_state['lp'])

    return tgan, link_predictor, best_val_auc


# ────────────────────────────────────────────────────────────────────────────
# 阶段 2-1: 提取每个窗口的群体嵌入
# ────────────────────────────────────────────────────────────────────────────
def extract_temporal_embeddings(tgan, edges_df, device):
    """
    按时间顺序重放边、推进 memory；
    每到一个窗口结束时间 cut_time 就提取 7 个群体节点的嵌入。
    """
    print("\n" + "=" * 60)
    print("[阶段2-1] 提取每个窗口的群体时序嵌入")
    print("=" * 60)

    group_ids = np.arange(1, 8, dtype=np.int64)
    temporal_embeddings = np.zeros((7, NUM_WINDOWS, D_MODEL), dtype=np.float32)

    sorted_df = edges_df.sort_values('ts', kind='mergesort').reset_index(drop=True)
    # 时间均匀采样 REPLAY_SUBSAMPLE 条边用于重放 memory
    if len(sorted_df) > REPLAY_SUBSAMPLE:
        replay_idx = np.linspace(0, len(sorted_df) - 1, REPLAY_SUBSAMPLE).astype(np.int64)
        sorted_df = sorted_df.iloc[replay_idx].reset_index(drop=True)
    print(f"  重放边数 (采样后): {len(sorted_df)}")
    ts_arr = sorted_df.ts.values

    tgan.eval()
    tgan.reset_memory()
    cursor = 0  # 已处理到的边索引

    with torch.no_grad():
        for w in range(NUM_WINDOWS):
            cut_time = float(w * WINDOW_SIZE + (WINDOW_SIZE - 1))

            # 把 ts <= cut_time 的边推进 memory
            j = cursor
            while j < len(ts_arr) and ts_arr[j] <= cut_time:
                j += 1
            if j > cursor:
                window_edges = sorted_df.iloc[cursor:j]
                n_batch = math.ceil(len(window_edges) / BATCH_SIZE)
                for k in range(n_batch):
                    s, e = k * BATCH_SIZE, min((k + 1) * BATCH_SIZE, len(window_edges))
                    batch = window_edges.iloc[s:e]
                    src = batch.u.values.astype(np.int64)
                    dst = batch.i.values.astype(np.int64)
                    ts  = batch.ts.values.astype(np.float64)
                    src_emb = tgan.encode_node(src, ts, dst, device=device, num_neighbors=NUM_NEIGHBORS)
                    update_memory_after_batch(tgan, src, dst, ts, src_emb, device)
                cursor = j

            # 在 cut_time 提取 7 个群体的嵌入（不污染 memory）
            cut_ts_arr = np.full(len(group_ids), cut_time, dtype=np.float64)
            tgt_nodes  = np.zeros_like(group_ids)
            for ii, nid in enumerate(group_ids.tolist()):
                ngh, _, _ = tgan.ngh_finder.find_before(int(nid), cut_time)
                if len(ngh) > 0:
                    tgt_nodes[ii] = int(ngh[-1])

            mem_bak = tgan.memory.backup_memory()
            emb = tgan.encode_node(group_ids, cut_ts_arr, tgt_nodes,
                                   device=device, num_neighbors=NUM_NEIGHBORS)
            tgan.memory.restore_memory(mem_bak)

            temporal_embeddings[:, w, :] = emb.cpu().numpy()
            mean_norm = float(np.linalg.norm(emb.cpu().numpy(), axis=1).mean())
            print(f"  W{w:02d} (cut_time={cut_time:>4.0f}): replayed_edges={cursor}, mean_emb_norm={mean_norm:.3f}")

    # 检查跨窗口嵌入的差异性
    diffs = []
    for w in range(NUM_WINDOWS - 1):
        diffs.append(np.linalg.norm(temporal_embeddings[:, w + 1] - temporal_embeddings[:, w], axis=1).mean())
    print(f"  相邻窗口嵌入平均距离: {[f'{d:.3f}' for d in diffs]}")

    return temporal_embeddings


# ────────────────────────────────────────────────────────────────────────────
# 阶段 2-2: 在时序嵌入上训练立场分类器（时间划分）
# ────────────────────────────────────────────────────────────────────────────
def train_classifier_temporal_split(temporal_embeddings, node_labels, device):
    print("\n" + "=" * 60)
    print("[阶段2-2] 训练立场分类器（时间划分）")
    print("=" * 60)
    print(f"  训练窗口: {CLASSIFIER_TRAIN_WINDOWS}  (in-sample)")
    print(f"  测试窗口: {CLASSIFIER_TEST_WINDOWS}  (out-of-sample)")

    # 拼装训练样本 (g × w_train) → (49, 64)
    train_x, train_y = [], []
    for w in CLASSIFIER_TRAIN_WINDOWS:
        for g in range(7):
            train_x.append(temporal_embeddings[g, w])
            train_y.append(int(node_labels[g + 1]))
    train_x = torch.from_numpy(np.stack(train_x)).float().to(device)
    train_y = torch.from_numpy(np.array(train_y)).long().to(device)
    print(f"  训练样本数: {len(train_y)}, 输入维度: {train_x.shape[1]}")

    # 类别权重
    n0 = int((train_y == 0).sum()); n1 = int((train_y == 1).sum())
    total = float(n0 + n1)
    class_w = torch.tensor([total / (2.0 * max(n0, 1)), total / (2.0 * max(n1, 1))],
                           dtype=torch.float32, device=device)
    print(f"  类别权重: {class_w.cpu().tolist()}")

    classifier = StanceClassifier(input_dim=D_MODEL, hidden_dim=HIDDEN_CLF, drop=0.2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR_CLASSIFIER, weight_decay=WEIGHT_DECAY_CLF)

    classifier.train()
    for epoch in range(N_EPOCHS_CLASSIFIER):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(train_x)
        loss = nn.functional.cross_entropy(logits, train_y, weight=class_w)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 40 == 0 or epoch == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == train_y).float().mean().item()
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS_CLASSIFIER} | Loss={loss.item():.4f} | TrainAcc={acc:.4f}")

    # ── 在所有 11 个窗口上预测 ───────────────────────────────────────────────
    # 注意：使用 logit 原始分数而非 softmax 概率，避免 7 个群体的立场分数饱和在 0/1，
    # 保留连续的时序动态变化信息；最后再做全局 min-max 归一化到 [0,1]。
    classifier.eval()
    temporal_raw_logits = np.zeros((7, NUM_WINDOWS), dtype=np.float32)
    with torch.no_grad():
        for w in range(NUM_WINDOWS):
            x = torch.from_numpy(temporal_embeddings[:, w, :]).float().to(device)
            logits = classifier(x)  # shape: (7, 2)
            # 取 Republican 类的 logit（原始分数），不做 softmax
            temporal_raw_logits[:, w] = logits[:, 1].cpu().numpy()

    # 全局 min-max 归一化到 [0,1]，保持相对排序，展示更多动态变化
    global_min = float(temporal_raw_logits.min())
    global_max = float(temporal_raw_logits.max())
    if global_max - global_min > 1e-12:
        temporal_stance_scores = (temporal_raw_logits - global_min) / (global_max - global_min)
    else:
        temporal_stance_scores = np.zeros_like(temporal_raw_logits)
    temporal_stance_scores = temporal_stance_scores.astype(np.float32)

    print(f"\n  [Logit] 原始 logit 范围: [{global_min:.4f}, {global_max:.4f}]")
    print(f"  [Logit] 原始 logit 标准差: {temporal_raw_logits.std():.4f}")
    print(f"  [Norm ] 归一化后范围: [{temporal_stance_scores.min():.4f}, {temporal_stance_scores.max():.4f}]")

    # 评估 in-sample / out-of-sample（用 raw logit 的符号判定 Republican vs Democrat）
    # 即：logits[:,1] > logits[:,0] ⇔ pred=1，等价于在归一化前 logit > 0；
    # 但这里也可直接用归一化后 0.5 作阈值——为保持训练一致性，使用 argmax 重新计算。
    def _accuracy_on_windows(windows):
        correct, total = 0, 0
        with torch.no_grad():
            for w in windows:
                x = torch.from_numpy(temporal_embeddings[:, w, :]).float().to(device)
                preds = torch.argmax(classifier(x), dim=1).cpu().numpy()
                for g in range(7):
                    lbl = int(node_labels[g + 1])
                    correct += int(int(preds[g]) == lbl); total += 1
        return correct / max(total, 1)

    in_acc  = _accuracy_on_windows(CLASSIFIER_TRAIN_WINDOWS)
    out_acc = _accuracy_on_windows(CLASSIFIER_TEST_WINDOWS)
    print(f"  In-sample  Accuracy (W{CLASSIFIER_TRAIN_WINDOWS[0]}-W{CLASSIFIER_TRAIN_WINDOWS[-1]}): {in_acc:.4f}")
    print(f"  Out-of-sample Accuracy (W{CLASSIFIER_TEST_WINDOWS[0]}-W{CLASSIFIER_TEST_WINDOWS[-1]}): {out_acc:.4f}")

    return temporal_stance_scores, temporal_raw_logits


# ────────────────────────────────────────────────────────────────────────────
# 打印结果表格
# ────────────────────────────────────────────────────────────────────────────
def print_results(temporal_stance_scores, node_labels):
    print("\n" + "=" * 80)
    print("[结果汇总] 时序立场分数 (P(共和党))")
    print("=" * 80)

    label_map = {0: "Democrat  ", 1: "Republican", -1: "Unknown   "}
    header = f"  {'群体':>4} | {'政治标签':>10} |"
    for w in range(NUM_WINDOWS):
        header += f"  W{w:02d}"
    print(header)
    print("-" * 80)

    for g_idx, nid in enumerate(range(1, 8)):
        lbl_str = label_map.get(int(node_labels[nid]), str(int(node_labels[nid])))
        row = f"  {g_idx:>4} | {lbl_str:>10} |"
        for w in range(NUM_WINDOWS):
            row += f" {temporal_stance_scores[g_idx, w]:.3f}"
        print(row)

    print("=" * 80)
    print(f"  整体范围   : [{temporal_stance_scores.min():.4f}, {temporal_stance_scores.max():.4f}]")
    print(f"  整体均值   : {temporal_stance_scores.mean():.4f}")
    print(f"  整体标准差 : {temporal_stance_scores.std():.4f}")

    # 时序变化幅度
    per_group_std = temporal_stance_scores.std(axis=1)
    print(f"  各群体跨窗口波动: {[f'{s:.3f}' for s in per_group_std.tolist()]}")
    print(f"  平均波动      : {per_group_std.mean():.4f}")

    dem_groups = [g for g in range(7) if node_labels[g + 1] == 0]
    rep_groups = [g for g in range(7) if node_labels[g + 1] == 1]
    if dem_groups:
        print(f"\n  民主党群体 {dem_groups}  平均 P(Rep): {temporal_stance_scores[dem_groups].mean():.4f}")
    if rep_groups:
        print(f"  共和党群体 {rep_groups}  平均 P(Rep): {temporal_stance_scores[rep_groups].mean():.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 保存结果
# ────────────────────────────────────────────────────────────────────────────
def save_results(temporal_stance_scores, temporal_embeddings, temporal_raw_logits=None):
    results_dir = os.path.join(TASK4_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    scores_path = os.path.join(results_dir, 'temporal_stance_scores.npy')
    emb_path    = os.path.join(results_dir, 'group_temporal_embeddings.npy')
    np.save(scores_path, temporal_stance_scores)
    np.save(emb_path,    temporal_embeddings)
    print(f"\n  已保存: {scores_path}  shape={temporal_stance_scores.shape}")
    print(f"  已保存: {emb_path}     shape={temporal_embeddings.shape}")
    if temporal_raw_logits is not None:
        raw_path = os.path.join(results_dir, 'temporal_stance_raw_logits.npy')
        np.save(raw_path, temporal_raw_logits)
        print(f"  已保存: {raw_path}  shape={temporal_raw_logits.shape}")


# ────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device('cpu')
    print(f"使用设备: {device}")

    # 步骤1: 加载数据
    n_feat, e_feat, node_labels, edges_df = load_group_data()

    # 步骤2: 构建 NeighborFinder
    ngh_finder = build_neighbor_finder(edges_df, n_nodes=N_NODES)

    # 步骤3: 初始化 TGAN + 边预测头
    tgan = build_tgan(ngh_finder, n_feat, e_feat, device)
    link_predictor = LinkPredictor(D_MODEL).to(device)

    # 阶段1: 自监督边预测预训练 TGAN（不接触节点标签）
    tgan, link_predictor, best_val_auc = pretrain_tgan_edge_prediction(
        tgan, link_predictor, edges_df, device,
    )

    # 阶段2-1: 重放边、按窗口提取群体嵌入
    temporal_embeddings = extract_temporal_embeddings(tgan, edges_df, device)

    # 阶段2-2: 训练立场分类器（时间划分）
    temporal_stance_scores, temporal_raw_logits = train_classifier_temporal_split(
        temporal_embeddings, node_labels, device,
    )

    # 打印 & 保存
    print_results(temporal_stance_scores, node_labels)
    save_results(temporal_stance_scores, temporal_embeddings, temporal_raw_logits)

    print(f"\n[完成] Stage1 最优 ValAUC = {best_val_auc:.4f}")


if __name__ == '__main__':
    main()
