# -*- coding: utf-8 -*-
"""
data_builder.py
群体立场分类实验 - 数据构建模块
构建 GroupStanceDataset，支持 M0/M1/M2/M3 四种特征模式
"""

import os
import sys
import json
import numpy as np

# ─────────────────────── 路径配置 ───────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # → CorDGT/lab3/GroupStanceAnalysis

TASK4_DIR = os.path.join(BASE_DIR, 'lab3', 'GroupStanceAnalysis', 'task4', 'results')
TASK3_DIR = os.path.join(BASE_DIR, 'lab3', 'GroupStanceAnalysis', 'task3')

# ─────────────────────── 数据加载 ───────────────────────
def load_all_data():
    """加载所有原始数据，返回字典"""
    data = {}

    # group_temporal_embeddings: (7, 11, 64)
    path = os.path.join(TASK4_DIR, 'group_temporal_embeddings.npy')
    if os.path.exists(path):
        data['gte'] = np.load(path).astype(np.float32)
    else:
        print(f"[WARN] 未找到 group_temporal_embeddings.npy，使用随机初始化")
        data['gte'] = np.random.randn(7, 11, 64).astype(np.float32)

    # interaction_windows: (11, 7, 7)
    path = os.path.join(TASK4_DIR, 'interaction_windows.npy')
    if os.path.exists(path):
        data['iw'] = np.load(path).astype(np.float32)
    else:
        print(f"[WARN] 未找到 interaction_windows.npy，使用零矩阵")
        data['iw'] = np.zeros((11, 7, 7), dtype=np.float32)

    # temporal_stance_scores: (7, 11)
    path = os.path.join(TASK4_DIR, 'temporal_stance_scores.npy')
    if os.path.exists(path):
        data['tss'] = np.load(path).astype(np.float32)
    else:
        print(f"[WARN] 未找到 temporal_stance_scores.npy，使用零矩阵")
        data['tss'] = np.zeros((7, 11), dtype=np.float32)

    # influence_model_results → fj_alpha: (7,)
    path = os.path.join(TASK4_DIR, 'influence_model_results.json')
    if os.path.exists(path):
        with open(path) as f:
            infl = json.load(f)
        fj = infl.get('friedkin_johnsen', {})
        alpha = fj.get('alpha', [0.5] * 7)
        data['fj_alpha'] = np.array(alpha, dtype=np.float32)
    else:
        print(f"[WARN] 未找到 influence_model_results.json，alpha 默认 0.5")
        data['fj_alpha'] = np.full(7, 0.5, dtype=np.float32)

    # group_labels: (8,) → 取 [1:8] 得群体 0-6 的标签
    path = os.path.join(TASK3_DIR, 'group_labels.npy')
    if os.path.exists(path):
        gl = np.load(path)
        data['group_labels'] = gl[1:8].astype(np.int64)  # shape (7,)
    else:
        print(f"[WARN] 未找到 group_labels.npy，使用 [1,0,0,1,1,0,0]")
        data['group_labels'] = np.array([1, 0, 0, 1, 1, 0, 0], dtype=np.int64)

    # group_features_compact: (8, 81) → 取 [1:8] 得群体 0-6 的静态特征
    path = os.path.join(TASK3_DIR, 'group_features_compact.npy')
    if os.path.exists(path):
        gfc = np.load(path).astype(np.float32)
        data['static_feat'] = gfc[1:8, :]  # shape (7, 81)
    else:
        print(f"[WARN] 未找到 group_features_compact.npy，使用零矩阵")
        data['static_feat'] = np.zeros((7, 81), dtype=np.float32)

    return data


# ─────────────────────── 特征构建 ───────────────────────
def _build_sample_features(g, w, data, model_type):
    """
    为群体 g 在窗口 w 构建特征向量。
    model_type: 'M0' | 'M1' | 'M2' | 'M3'
    custom_blocks: dict，用于消融实验中替换特征块
    """
    gte       = data['gte']       # (7, 11, 64)
    iw        = data['iw']        # (11, 7, 7)
    tss       = data['tss']       # (7, 11)
    fj_alpha  = data['fj_alpha']  # (7,)
    static_f  = data['static_feat']  # (7, 81)

    # Block 1: TGAN 嵌入 (64d)
    emb = gte[g, w, :]  # (64,)

    # Block 2: 交互特征 (7d)
    inter = iw[w, g, :]  # (7,)

    # Block 3: 时序变化特征 (3d)
    vel = tss[g, w] - tss[g, max(0, w - 1)]
    hist_len = max(1, w)
    vol = float(np.std(tss[g, :hist_len]))
    alpha = float(fj_alpha[g])
    temporal = np.array([vel, vol, alpha], dtype=np.float32)

    # Block 4: 静态特征 (81d)
    static = static_f[g, :]  # (81,)

    if model_type == 'M0':
        return static                                          # (81,)
    elif model_type == 'M1':
        return emb                                             # (64,)
    elif model_type == 'M2':
        return np.concatenate([emb, inter, temporal])          # (74,)
    elif model_type == 'M3':
        return np.concatenate([emb, inter, temporal, static])  # (155,)
    else:
        raise ValueError(f"未知模型类型: {model_type}，支持 M0/M1/M2/M3")


def _build_sample_features_custom(g, w, data, blocks):
    """
    消融实验专用：自定义选择特征块
    blocks: list of str，可选 'emb'/'inter'/'temporal'/'static'
    """
    gte       = data['gte']
    iw        = data['iw']
    tss       = data['tss']
    fj_alpha  = data['fj_alpha']
    static_f  = data['static_feat']

    parts = []
    if 'emb' in blocks:
        parts.append(gte[g, w, :])                          # 64d
    if 'inter' in blocks:
        parts.append(iw[w, g, :])                            # 7d
    if 'temporal' in blocks:
        vel = tss[g, w] - tss[g, max(0, w - 1)]
        hist_len = max(1, w)
        vol = float(np.std(tss[g, :hist_len]))
        alpha = float(fj_alpha[g])
        parts.append(np.array([vel, vol, alpha], dtype=np.float32))  # 3d
    if 'vel_only' in blocks:
        vel = tss[g, w] - tss[g, max(0, w - 1)]
        parts.append(np.array([vel], dtype=np.float32))      # 1d
    if 'static' in blocks:
        parts.append(static_f[g, :])                         # 81d

    return np.concatenate(parts).astype(np.float32)


# ─────────────────────── 数据集类 ───────────────────────
class GroupStanceDataset:
    """
    群体立场分类数据集。
    支持 M0/M1/M2/M3 特征模式，以及自定义特征块（消融实验）。
    """

    # 数据划分窗口索引
    TRAIN_WINDOWS = list(range(0, 6))    # W0~W5
    VAL_WINDOWS   = list(range(6, 8))    # W6~W7
    TEST_WINDOWS  = list(range(8, 11))   # W8~W10

    N_GROUPS  = 7
    N_WINDOWS = 11

    def __init__(self, noise_std=0.01, random_state=2023):
        self.noise_std    = noise_std
        self.random_state = random_state
        self.data         = load_all_data()
        self.labels       = self.data['group_labels']  # (7,)

    def build_features(self, model_type='M2', custom_blocks=None):
        """
        构建所有样本的特征矩阵和标签向量。

        Args:
            model_type:    'M0'|'M1'|'M2'|'M3'
            custom_blocks: list[str]，消融实验时替换特征块组合

        Returns:
            train_X, train_y, val_X, val_y, test_X, test_y
        """
        rng = np.random.RandomState(self.random_state)

        def _build(windows, augment=False):
            X_list, y_list = [], []
            for w in windows:
                for g in range(self.N_GROUPS):
                    if custom_blocks is not None:
                        feat = _build_sample_features_custom(g, w, self.data, custom_blocks)
                    else:
                        feat = _build_sample_features(g, w, self.data, model_type)
                    X_list.append(feat)
                    y_list.append(self.labels[g])

            X = np.stack(X_list, axis=0).astype(np.float32)  # (N, D)
            y = np.array(y_list, dtype=np.int64)              # (N,)

            if augment and self.noise_std > 0:
                # ×5 高斯噪声增广
                aug_X = [X]
                aug_y = [y]
                for _ in range(4):
                    noise = rng.randn(*X.shape).astype(np.float32) * self.noise_std
                    aug_X.append(X + noise)
                    aug_y.append(y.copy())
                X = np.concatenate(aug_X, axis=0)
                y = np.concatenate(aug_y, axis=0)
                # shuffle
                idx = rng.permutation(len(X))
                X, y = X[idx], y[idx]

            return X, y

        train_X, train_y = _build(self.TRAIN_WINDOWS, augment=True)
        val_X,   val_y   = _build(self.VAL_WINDOWS,   augment=False)
        test_X,  test_y  = _build(self.TEST_WINDOWS,  augment=False)

        return train_X, train_y, val_X, val_y, test_X, test_y

    def get_split(self, window_indices, model_type='M2', custom_blocks=None):
        """
        获取指定窗口索引对应的样本（无增广）。
        """
        X_list, y_list = [], []
        for w in window_indices:
            for g in range(self.N_GROUPS):
                if custom_blocks is not None:
                    feat = _build_sample_features_custom(g, w, self.data, custom_blocks)
                else:
                    feat = _build_sample_features(g, w, self.data, model_type)
                X_list.append(feat)
                y_list.append(self.labels[g])
        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def get_input_dim(self, model_type='M2', custom_blocks=None):
        """返回特征维度"""
        if custom_blocks is not None:
            feat = _build_sample_features_custom(0, 0, self.data, custom_blocks)
            return feat.shape[0]
        dim_map = {'M0': 81, 'M1': 64, 'M2': 74, 'M3': 155}
        return dim_map.get(model_type, 74)


# ─────────────────────── 快速验证 ───────────────────────
if __name__ == '__main__':
    print("=== GroupStanceDataset 验证 ===")
    ds = GroupStanceDataset()
    print(f"标签 (7个群体): {ds.labels}")
    print(f"  民主党(0): {(ds.labels == 0).sum()} 个")
    print(f"  共和党(1): {(ds.labels == 1).sum()} 个")
    print()

    for mt in ['M0', 'M1', 'M2', 'M3']:
        tr_X, tr_y, va_X, va_y, te_X, te_y = ds.build_features(mt)
        print(f"[{mt}] train={tr_X.shape} val={va_X.shape} test={te_X.shape} "
              f"dim={tr_X.shape[1]}")
    print()

    # 消融：无交互 (67d)
    X_nointer, y_nointer, *_ = ds.build_features.__func__(
        ds, 'M2') if False else (None, None)
    # 用 custom_blocks 测试
    tr_X2, _, _, _, _, _ = ds.build_features(
        custom_blocks=['emb', 'temporal'])
    print(f"[无交互 消融] dim={tr_X2.shape[1]}")  # 应为 67

    tr_X3, _, _, _, _, _ = ds.build_features(
        custom_blocks=['emb', 'inter'])
    print(f"[无时序 消融] dim={tr_X3.shape[1]}")  # 应为 71
