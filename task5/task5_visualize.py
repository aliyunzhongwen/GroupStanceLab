# -*- coding: utf-8 -*-
"""
task5_visualize.py
群体立场分类实验 - 可视化脚本
生成：confusion_matrix.png / roc_curve.png / ablation_bar.png / prediction_timeline.png
"""

import os
import sys
import json
import warnings
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

TASK5_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK5_DIR)

from data_builder import GroupStanceDataset
from models.group_stance_classifier import StanceClassifier
from train_group_stance import SEED, CKPT_DIR

from sklearn.metrics import (confusion_matrix, roc_curve, auc as sk_auc,
                               f1_score, roc_auc_score, accuracy_score)

VIZ_DIR     = os.path.join(TASK5_DIR, 'results', 'visualizations')
RESULTS_DIR = os.path.join(TASK5_DIR, 'results')

# ─────────────────────── 中文字体配置 ───────────────────────
def setup_chinese_font():
    """搜索系统中文字体，返回可用字体名称"""
    candidates = []
    try:
        for f in fm.findSystemFonts():
            fl = f.lower()
            if any(k in fl for k in [
                'notosanscjk', 'noto_sans_cjk',
                'wenquanyi', 'wqy',
                'simhei', 'simsun', 'microsoftyahei', 'microsoft yahei',
                'droid sans fallback', 'droidsansfallback'
            ]):
                candidates.append(f)
    except Exception:
        pass

    if candidates:
        font_path = candidates[0]
        try:
            prop = fm.FontProperties(fname=font_path)
            font_name = prop.get_name()
            matplotlib.rcParams['font.family'] = font_name
            plt.rcParams['font.family'] = font_name
            print(f"[字体] 使用: {font_path}")
            return font_path
        except Exception as e:
            print(f"[字体] 加载失败: {e}")

    # 回退：使用 sans-serif
    print("[字体] 未找到中文字体，标题将使用英文")
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    return None


def get_font_prop(font_path=None):
    if font_path and os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return fm.FontProperties()


# ─────────────────────── 加载模型 ───────────────────────
def load_model(model_type, input_dim):
    """从 checkpoint 加载模型"""
    ckpt_path = os.path.join(CKPT_DIR, f'{model_type}_best.pth')
    model = StanceClassifier(input_dim)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state)
        print(f"[模型] 加载 {model_type}: {ckpt_path}")
    else:
        print(f"[警告] 未找到 {ckpt_path}，使用随机初始化")
    model.eval()
    return model


def get_probs_labels(model, X, y):
    """推理得到概率和标签"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs  = torch.softmax(logits, dim=1).numpy()
    return probs, y


# ─────────────────────── 图1: 混淆矩阵 ───────────────────────
def plot_confusion_matrix(ds, font_path=None):
    fp = get_font_prop(font_path)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # 加载 M2 模型
    train_X, train_y, val_X, val_y, test_X, test_y = ds.build_features('M2')
    model = load_model('M2', test_X.shape[1])

    probs, labels = get_probs_labels(model, test_X, test_y)
    preds = np.argmax(probs, axis=1)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    class_names = ['民主党', '共和党']
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontproperties=fp, fontsize=13)
    ax.set_yticklabels(class_names, fontproperties=fp, fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_title('群体立场分类混淆矩阵 (M2, 测试集)', fontproperties=fp, fontsize=14, pad=12)
    ax.set_xlabel('预测标签', fontproperties=fp, fontsize=12)
    ax.set_ylabel('真实标签', fontproperties=fp, fontsize=12)

    # 添加性能注释
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    ax.annotate(f'F1={f1:.3f}  Acc={acc:.3f}',
                xy=(0.5, -0.12), xycoords='axes fraction',
                ha='center', fontsize=11,
                fontproperties=fp)

    plt.tight_layout()
    out_path = os.path.join(VIZ_DIR, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] {out_path}")


# ─────────────────────── 图2: ROC 曲线对比 ───────────────────────
def plot_roc_curves(ds, font_path=None):
    fp = get_font_prop(font_path)
    os.makedirs(VIZ_DIR, exist_ok=True)

    dim_map = {'M0': 81, 'M1': 64, 'M2': 74, 'M3': 155}
    colors  = {'M0': '#e74c3c', 'M1': '#3498db', 'M2': '#2ecc71', 'M3': '#9b59b6'}

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    for mt in ['M0', 'M1', 'M2', 'M3']:
        _, _, _, _, test_X, test_y = ds.build_features(mt)
        model = load_model(mt, test_X.shape[1])
        probs, labels = get_probs_labels(model, test_X, test_y)

        n_pos = (labels == 1).sum()
        n_neg = (labels == 0).sum()
        if n_pos > 0 and n_neg > 0:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = sk_auc(fpr, tpr)
        else:
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.5, 0.5])
            roc_auc = 0.5

        ax.plot(fpr, tpr, color=colors[mt], lw=2,
                label=f'{mt} (dim={dim_map[mt]}, AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='随机基线')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('假正率 (FPR)', fontproperties=fp, fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontproperties=fp, fontsize=12)
    ax.set_title('各特征模型 ROC 曲线对比', fontproperties=fp, fontsize=14, pad=12)
    ax.legend(prop=fp if font_path else None, fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    plt.rcParams['axes.unicode_minus'] = False

    plt.tight_layout()
    out_path = os.path.join(VIZ_DIR, 'roc_curve.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] {out_path}")


# ─────────────────────── 图3: 消融实验柱状图 ───────────────────────
def plot_ablation_bar(font_path=None):
    fp = get_font_prop(font_path)
    os.makedirs(VIZ_DIR, exist_ok=True)

    ablation_path = os.path.join(RESULTS_DIR, 'ablation_results.json')
    if not os.path.exists(ablation_path):
        print(f"[警告] 未找到 {ablation_path}，跳过消融柱状图")
        return

    with open(ablation_path) as f:
        ablation = json.load(f)

    # 提取每个实验的 A/B Test F1
    exp_labels = []
    f1_a_list, f1_b_list = [], []
    name_a_list, name_b_list = [], []

    exp_order = [
        ('E1', 'M0', 'M1',          'M0(81d)', 'M1(64d)'),
        ('E2', 'M1', 'M2',          'M1(64d)', 'M2(74d)'),
        ('E3', 'M2', 'M3',          'M2(74d)', 'M3(155d)'),
        ('E4', 'M2_nointer', 'M2',  '无交互(67d)', 'M2(74d)'),
        ('E5', 'M2_notemporal', 'M2','无时序(71d)', 'M2(74d)'),
        ('E6', 'filtered524', 'full877', '524过滤', '877全量'),
    ]

    for exp_id, key_a, key_b, label_a, label_b in exp_order:
        if exp_id not in ablation:
            continue
        exp_data = ablation[exp_id]
        f1_a = exp_data.get(key_a, {}).get('f1', 0.0)
        f1_b = exp_data.get(key_b, {}).get('f1', 0.0)
        exp_labels.append(exp_id)
        f1_a_list.append(f1_a)
        f1_b_list.append(f1_b)
        name_a_list.append(label_a)
        name_b_list.append(label_b)

    if not exp_labels:
        print("[警告] 消融实验数据为空，跳过")
        return

    n    = len(exp_labels)
    x    = np.arange(n)
    w    = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)

    bars_a = ax.bar(x - w/2, f1_a_list, w, label='对照组', color='#5b8dd9', alpha=0.88)
    bars_b = ax.bar(x + w/2, f1_b_list, w, label='实验组', color='#e8734a', alpha=0.88)

    # 数值标注
    for bar in bars_a:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.571, color='gray', linestyle='--', alpha=0.7, lw=1.2, label='多数类基线')
    ax.axhline(y=0.500, color='darkgray', linestyle=':', alpha=0.7, lw=1.2, label='随机基线')

    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontproperties=fp, fontsize=12)
    ax.set_ylabel('Test F1 (Macro)', fontproperties=fp, fontsize=12)
    ax.set_title('消融实验结果对比', fontproperties=fp, fontsize=14, pad=12)
    ax.set_ylim([0, 1.15])
    ax.legend(prop=fp if font_path else None, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # x 轴下方添加对比说明
    for i, (na, nb) in enumerate(zip(name_a_list, name_b_list)):
        ax.annotate(f'{na}\nvs\n{nb}',
                    xy=(x[i], -0.22), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=7,
                    fontproperties=fp)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out_path = os.path.join(VIZ_DIR, 'ablation_bar.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] {out_path}")


# ─────────────────────── 图4: 预测概率时序图 ───────────────────────
def plot_prediction_timeline(ds, font_path=None):
    fp = get_font_prop(font_path)
    os.makedirs(VIZ_DIR, exist_ok=True)

    _, _, _, _, test_X, test_y = ds.build_features('M2')
    model = load_model('M2', test_X.shape[1])
    probs, labels = get_probs_labels(model, test_X, test_y)

    # test_X 按 (window, group) 排列：W8、W9、W10，每个窗口 7 个群体
    N_GROUPS  = 7
    TEST_WINS = [8, 9, 10]
    group_probs = np.zeros((N_GROUPS, len(TEST_WINS)))

    idx = 0
    for wi, w in enumerate(TEST_WINS):
        for g in range(N_GROUPS):
            group_probs[g, wi] = probs[idx, 1]  # 共和党概率
            idx += 1

    group_colors = [
        '#e74c3c', '#3498db', '#2ecc71', '#e67e22',
        '#9b59b6', '#1abc9c', '#f39c12'
    ]
    group_labels_str = [f'群体{g}' for g in range(N_GROUPS)]
    true_labels = ds.labels  # (7,) 真实标签

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    for g in range(N_GROUPS):
        label_str = group_labels_str[g]
        party     = '(共)' if true_labels[g] == 1 else '(民)'
        style     = '-' if true_labels[g] == 1 else '--'
        ax.plot(TEST_WINS, group_probs[g], marker='o', color=group_colors[g],
                linestyle=style, linewidth=2, markersize=7,
                label=f'{label_str}{party}')

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, lw=1.2)
    ax.fill_between(TEST_WINS, 0.5, 1.0, alpha=0.04, color='red')
    ax.fill_between(TEST_WINS, 0.0, 0.5, alpha=0.04, color='blue')

    ax.set_xticks(TEST_WINS)
    ax.set_xticklabels([f'W{w}' for w in TEST_WINS], fontproperties=fp, fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('测试窗口', fontproperties=fp, fontsize=12)
    ax.set_ylabel('预测为共和党的概率', fontproperties=fp, fontsize=12)
    ax.set_title('各群体测试窗口立场预测概率时序图', fontproperties=fp, fontsize=14, pad=12)
    ax.legend(prop=fp if font_path else None, fontsize=9,
              loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)

    # 添加文字说明
    ax.text(0.98, 0.52, '→ 共和党倾向', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='red', alpha=0.7,
            fontproperties=fp)
    ax.text(0.98, 0.45, '→ 民主党倾向', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='blue', alpha=0.7,
            fontproperties=fp)

    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    out_path = os.path.join(VIZ_DIR, 'prediction_timeline.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] {out_path}")


# ─────────────────────── 主函数 ───────────────────────
def main():
    print("\n=== Task5 可视化生成 ===")
    font_path = setup_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False

    ds = GroupStanceDataset(noise_std=0.01, random_state=SEED)

    print("\n[1/4] 生成混淆矩阵...")
    plot_confusion_matrix(ds, font_path)

    print("\n[2/4] 生成 ROC 曲线...")
    plot_roc_curves(ds, font_path)

    print("\n[3/4] 生成消融实验柱状图...")
    plot_ablation_bar(font_path)

    print("\n[4/4] 生成预测概率时序图...")
    plot_prediction_timeline(ds, font_path)

    print(f"\n所有可视化已保存到: {VIZ_DIR}")


if __name__ == '__main__':
    main()
