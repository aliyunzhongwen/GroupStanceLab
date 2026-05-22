# -*- coding: utf-8 -*-
"""
Task 4 可视化（论文发表级别）
====================================================================

字体策略：
    优先使用宋体 (SimSun)；若系统无宋体，则按以下顺序回退：
        SimSun → STSong → FangSong → Noto Serif CJK SC → WenQuanYi 系列
    西文/数字优先 Times New Roman，回退 Liberation Serif → DejaVu Serif。

生成 7 张论文级图表：
    1. 演化轨迹图        evolution_trajectory.png
    2. 烟花图            firework_diagram.png
    3. 演化热力图        evolution_heatmap.png
    4. 交互-演化联动图   interaction_evolution_coupling.png
    5. 因果影响网络图    causal_influence_network.png
    6. 小提琴图          stance_violin_plot.png
    7. 极化时间线图      polarization_timeline.png

运行：
    cd /root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task4
    python task4_visualize.py
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyArrowPatch


# ══════════════════════════════════════════════════════════════════════════════
# 字体与全局样式
# ══════════════════════════════════════════════════════════════════════════════
def setup_fonts():
    """检测系统字体并配置：中文宋体 + 西文 Times New Roman。"""
    available = {f.name for f in fontManager.ttflist}

    # 注意：matplotlib 加载 .ttc 时仅注册一个 face 名，因此
    # /usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc 在系统中
    # 实际登记为 'Noto Serif CJK JP'，但其字形完整支持简中 (SC)。
    cn_candidates = [
        'SimSun', 'STSong', 'FangSong', 'NSimSun',
        'Noto Serif CJK SC', 'Noto Serif CJK JP',
        'Source Han Serif SC', 'Source Han Serif CN',
        'Noto Sans CJK SC', 'Noto Sans CJK JP',
        'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
    ]
    en_candidates = [
        'Times New Roman', 'Liberation Serif',
        'Nimbus Roman', 'DejaVu Serif',
    ]
    cn_font = next((f for f in cn_candidates if f in available), None)
    en_font = next((f for f in en_candidates if f in available), 'DejaVu Serif')

    # matplotlib 在缺字时会按 font.serif 顺序逐个查找，因此中文放在前面，
    # 西文/数字放在后面，缺中文字符时也能找到对应字形。
    serif_chain = []
    if cn_font:
        serif_chain.append(cn_font)
    serif_chain.extend([en_font, 'DejaVu Serif'])

    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = serif_chain
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    print(f"[字体] 中文: {cn_font or '未找到，使用回退字体'}  |  西文: {en_font}")
    return cn_font, en_font


setup_fonts()

# 论文级别样式
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#666666',
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})


# ══════════════════════════════════════════════════════════════════════════════
# 路径与配色
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)
TASK3_DIR = os.path.join(LAB_DIR, 'task3')
TASK2_DIR = os.path.join(LAB_DIR, 'task2')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# 学术配色（共和党红色系；民主党蓝/青色系）
# 索引顺序与 task3 group_labels[1:] 一致：群体 0,3,4 共和党；1,2,5,6 民主党
COLORS = [
    '#C0392B',  # G0 共和党
    '#2980B9',  # G1 民主党
    '#3498DB',  # G2 民主党
    '#E74C3C',  # G3 共和党
    '#D35400',  # G4 共和党
    '#1ABC9C',  # G5 民主党
    '#2C3E50',  # G6 民主党
]
REP_COLOR = '#C0392B'
DEM_COLOR = '#2980B9'

# 短标签（图内紧凑显示）与全名（图例/标题使用）
GROUP_NAMES_SHORT = ['群体0', '群体1', '群体2', '群体3', '群体4', '群体5', '群体6']
GROUP_NAMES_FULL = [
    '群体0(共)', '群体1(民)', '群体2(民)', '群体3(共)',
    '群体4(共)', '群体5(民)', '群体6(民)',
]


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    data = {}
    stance_path = os.path.join(RESULTS_DIR, 'temporal_stance_scores.npy')
    if not os.path.exists(stance_path):
        print(f"[警告] 缺少 {stance_path}, 多数图表无法生成")
        return data
    data['stance_scores'] = np.load(stance_path)

    iw_path = os.path.join(RESULTS_DIR, 'interaction_windows.npy')
    if os.path.exists(iw_path):
        data['interaction_windows'] = np.load(iw_path)

    emb_path = os.path.join(RESULTS_DIR, 'group_temporal_embeddings.npy')
    if os.path.exists(emb_path):
        data['embeddings'] = np.load(emb_path)  # (7, 11, 64)

    gl_path = os.path.join(TASK3_DIR, 'group_labels.npy')
    if os.path.exists(gl_path):
        data['group_labels'] = np.load(gl_path)

    ga_path = os.path.join(TASK2_DIR, 'group_assignments_balanced.npy')
    if os.path.exists(ga_path):
        data['group_assignments'] = np.load(ga_path)

    for name in ['evolution_metrics', 'granger_results', 'change_points']:
        fpath = os.path.join(RESULTS_DIR, f'{name}.json')
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data[name] = json.load(f)

    print("数据加载完成")
    return data


def get_party_labels(data):
    """返回长度 7 的 0/1 党派标签（1=共和党, 0=民主党）。"""
    gl = data.get('group_labels')
    if gl is not None and len(gl) >= 8:
        return gl[1:]
    # 回退：依据短名推断
    return np.array([1, 0, 0, 1, 1, 0, 0])


# ══════════════════════════════════════════════════════════════════════════════
# 1. 演化轨迹图
# ══════════════════════════════════════════════════════════════════════════════
def plot_evolution_trajectory(data):
    print("[1/7] 演化轨迹图")
    stance = data['stance_scores']  # (7, 11)
    n_windows = stance.shape[1]
    party = get_party_labels(data)

    # 嵌入方差作为不确定性带
    band = None
    if 'embeddings' in data:
        emb = data['embeddings']  # (7, 11, 64)
        # 取嵌入最后一维归一化后的标准差作为代理不确定性
        std = emb.std(axis=2)  # (7, 11)
        std = std / (std.max() + 1e-8) * 0.04  # 缩放到约 ±0.04 区间
        band = std

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(n_windows)

    for g in range(7):
        ls = '-' if party[g] == 1 else '--'
        marker = 'o' if party[g] == 1 else 's'
        ax.plot(x, stance[g], color=COLORS[g], marker=marker,
                markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                linewidth=2.0, linestyle=ls, label=GROUP_NAMES_FULL[g],
                alpha=0.95, zorder=3)
        if band is not None:
            ax.fill_between(x, stance[g] - band[g], stance[g] + band[g],
                            color=COLORS[g], alpha=0.12, zorder=2)

    ax.axhline(y=0.5, color='#7F8C8D', linestyle=':', linewidth=1.0,
               alpha=0.8, label='中性 (0.5)', zorder=1)
    # 党派区域阴影
    ax.axhspan(0.5, 1.0, color=REP_COLOR, alpha=0.04, zorder=0)
    ax.axhspan(0.0, 0.5, color=DEM_COLOR, alpha=0.04, zorder=0)

    ax.set_xlabel('时间窗口', fontsize=12)
    ax.set_ylabel('立场强度  P(共和党)', fontsize=12)
    ax.set_title('群体立场演化轨迹', fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'W{t}' for t in range(n_windows)])
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.3, n_windows - 0.7)
    ax.text(n_windows - 0.8, 0.97, '共和党区域', fontsize=9, ha='right',
            color=REP_COLOR, alpha=0.7, style='italic')
    ax.text(n_windows - 0.8, 0.03, '民主党区域', fontsize=9, ha='right',
            color=DEM_COLOR, alpha=0.7, style='italic')
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                    fontsize=10, title='群体', title_fontsize=11)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    path = os.path.join(VIS_DIR, 'evolution_trajectory.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 烟花图（群体间交互辐射图）
# ══════════════════════════════════════════════════════════════════════════════
def plot_firework_diagram(data):
    print("[2/7] 烟花图")
    iw = data.get('interaction_windows')
    if iw is None:
        print("  [跳过] 缺少 interaction_windows")
        return
    ga = data.get('group_assignments')
    party = get_party_labels(data)

    w = min(5, iw.shape[0] - 1)
    iw_w = iw[w]

    sizes = (np.array([(ga == g).sum() for g in range(7)])
             if ga is not None else np.ones(7) * 100)

    # 按党派分块的环形排列：共和党聚在上半，民主党聚在下半，更直观
    rep_idx = [g for g in range(7) if party[g] == 1]
    dem_idx = [g for g in range(7) if party[g] == 0]
    order = rep_idx + dem_idx
    angles = np.zeros(7)
    n_rep, n_dem = len(rep_idx), len(dem_idx)
    for k, g in enumerate(rep_idx):
        angles[g] = np.pi - np.pi * (k + 1) / (n_rep + 1) + np.pi  # 上半圆
    for k, g in enumerate(dem_idx):
        angles[g] = -np.pi * (k + 1) / (n_dem + 1)  # 下半圆
    # 简单一点：全部均匀环形
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 2
    pos_x = np.cos(angles)
    pos_y = np.sin(angles)

    fig, ax = plt.subplots(figsize=(11, 9.5))

    # 边的对数缩放
    flow_max = max(iw_w[a, b] for a in range(7) for b in range(7) if a != b)
    flow_max = max(flow_max, 1.0)

    # 选出 top-K 边做数值标注
    edge_list = []
    for a in range(7):
        for b in range(7):
            if a != b and iw_w[a, b] >= 1:
                edge_list.append((a, b, iw_w[a, b]))
    edge_list.sort(key=lambda x: -x[2])
    annotate_set = {(a, b) for a, b, _ in edge_list[:5]}

    for a, b, flow in edge_list:
        ratio = np.log1p(flow) / np.log1p(flow_max)
        lw = 0.6 + 4.0 * ratio
        alpha = 0.25 + 0.65 * ratio
        # 跨党用紫色；同党用其本色
        if party[a] == party[b]:
            color = REP_COLOR if party[a] == 1 else DEM_COLOR
        else:
            color = '#6C3483'

        rad = 0.15 + 0.05 * ((a + b) % 3)
        arrow = FancyArrowPatch((pos_x[a], pos_y[a]), (pos_x[b], pos_y[b]),
                                connectionstyle=f'arc3,rad={rad}',
                                arrowstyle='-|>',
                                mutation_scale=12 + 8 * ratio,
                                linewidth=lw, color=color, alpha=alpha,
                                shrinkA=18, shrinkB=18, zorder=2)
        ax.add_patch(arrow)

        if (a, b) in annotate_set:
            mid_x = (pos_x[a] + pos_x[b]) / 2 + rad * (pos_y[b] - pos_y[a]) * 0.5
            mid_y = (pos_y[a] + pos_y[b]) / 2 - rad * (pos_x[b] - pos_x[a]) * 0.5
            ax.text(mid_x, mid_y, f'{int(flow)}', fontsize=8.5,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=color, linewidth=0.8, alpha=0.9),
                    zorder=4)

    # 节点
    sz_max = max(sizes.max(), 1)
    for g in range(7):
        node_size = 600 + 1500 * (sizes[g] / sz_max)
        # 外圈光晕
        ax.scatter(pos_x[g], pos_y[g], s=node_size * 1.6,
                   c=COLORS[g], alpha=0.25, zorder=4)
        ax.scatter(pos_x[g], pos_y[g], s=node_size,
                   c=COLORS[g], edgecolors='white', linewidths=2.2, zorder=5)
        ax.text(pos_x[g], pos_y[g], GROUP_NAMES_SHORT[g],
                fontsize=10, fontweight='bold', ha='center', va='center',
                color='white', zorder=6)
        # 群体规模注释
        ax.text(pos_x[g] * 1.20, pos_y[g] * 1.20, f'n={int(sizes[g])}',
                fontsize=8.5, ha='center', va='center',
                color='#34495E', zorder=6)

    ax.set_title(f'群体间交互辐射图 (时间窗口 W{w})',
                 fontsize=14, pad=14)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_handles = [
        mpatches.Patch(color=REP_COLOR, label='共和党群体'),
        mpatches.Patch(color=DEM_COLOR, label='民主党群体'),
        plt.Line2D([], [], color='#6C3483', linewidth=2, label='跨党派交互'),
        plt.Line2D([], [], color=REP_COLOR, linewidth=2, label='党内交互'),
    ]
    leg = ax.legend(handles=legend_handles, loc='upper right',
                    fontsize=10, framealpha=0.92,
                    bbox_to_anchor=(1.02, 1.0))
    leg.get_frame().set_linewidth(0.8)
    ax.text(0, -1.45, '节点大小 ∝ 群体成员数；边宽 ∝ log(交互量)',
            fontsize=9, ha='center', color='#7F8C8D', style='italic')

    fig.tight_layout()
    path = os.path.join(VIS_DIR, 'firework_diagram.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 演化热力图
# ══════════════════════════════════════════════════════════════════════════════
def plot_evolution_heatmap(data):
    print("[3/7] 演化热力图")
    stance = data['stance_scores']
    party = get_party_labels(data)

    rep_groups = [g for g in range(7) if party[g] == 1]
    dem_groups = [g for g in range(7) if party[g] == 0]
    order = rep_groups + dem_groups

    matrix = stance[order]
    row_labels = [GROUP_NAMES_FULL[g] for g in order]
    n_windows = stance.shape[1]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1,
                   interpolation='nearest')

    for i in range(len(order)):
        for j in range(n_windows):
            val = matrix[i, j]
            text_color = 'white' if (val > 0.78 or val < 0.22) else '#1C1C1C'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9.5, color=text_color, fontweight='medium')

    ax.set_xticks(np.arange(n_windows))
    ax.set_xticklabels([f'W{t}' for t in range(n_windows)])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('时间窗口', fontsize=12)
    ax.set_ylabel('群体', fontsize=12)
    ax.set_title('群体立场时序热力图  (颜色编码: P(共和党))',
                 fontsize=14, pad=12)

    # 党派分隔线 + 党派标记
    if rep_groups and dem_groups:
        ax.axhline(y=len(rep_groups) - 0.5, color='#1C1C1C',
                   linewidth=2.0, linestyle='-')
        ax.text(-1.1, (len(rep_groups) - 1) / 2, '共\n和\n党',
                fontsize=11, fontweight='bold', ha='center', va='center',
                color=REP_COLOR,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=REP_COLOR, linewidth=1.0))
        ax.text(-1.1, len(rep_groups) + (len(dem_groups) - 1) / 2,
                '民\n主\n党',
                fontsize=11, fontweight='bold', ha='center', va='center',
                color=DEM_COLOR,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=DEM_COLOR, linewidth=1.0))

    # 关闭网格（heatmap 不需要）
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('P(共和党)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    path = os.path.join(VIS_DIR, 'evolution_heatmap.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 交互-演化联动图
# ══════════════════════════════════════════════════════════════════════════════
def plot_interaction_evolution_coupling(data):
    print("[4/7] 交互-演化联动图")
    stance = data['stance_scores']
    iw = data.get('interaction_windows')
    if iw is None:
        print("  [跳过] 缺少 interaction_windows")
        return
    party = get_party_labels(data)

    pairs = []
    # 跨党最大交互对
    best_cross, best_cross_flow = None, -1
    for a in range(7):
        for b in range(a + 1, 7):
            if party[a] != party[b]:
                f = iw[:, a, b].sum() + iw[:, b, a].sum()
                if f > best_cross_flow:
                    best_cross_flow, best_cross = f, (a, b)
    if best_cross:
        pairs.append(best_cross)
    # 同党最大交互对
    best_same, best_same_flow = None, -1
    for a in range(7):
        for b in range(a + 1, 7):
            if party[a] == party[b]:
                f = iw[:, a, b].sum() + iw[:, b, a].sum()
                if f > best_same_flow:
                    best_same_flow, best_same = f, (a, b)
    if best_same:
        pairs.append(best_same)
    # 第二个跨党对（次大）
    for a in range(7):
        for b in range(a + 1, 7):
            if party[a] != party[b] and (a, b) not in pairs:
                pairs.append((a, b))
                break
        if len(pairs) >= 3:
            break

    n_pairs = len(pairs)
    n_windows = stance.shape[1]
    x = np.arange(n_windows)

    fig, axes = plt.subplots(n_pairs, 1, figsize=(13, 3.6 * n_pairs),
                             squeeze=False)

    for idx, (a, b) in enumerate(pairs):
        ax1 = axes[idx, 0]
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

        flow_ab = iw[:, a, b]
        flow_ba = iw[:, b, a]
        bw = 0.38
        ax1.bar(x - bw / 2, flow_ab, bw, color=COLORS[a],
                alpha=0.78, edgecolor='white', linewidth=0.6,
                label=f'{GROUP_NAMES_SHORT[a]} → {GROUP_NAMES_SHORT[b]}')
        ax1.bar(x + bw / 2, flow_ba, bw, color=COLORS[b],
                alpha=0.78, edgecolor='white', linewidth=0.6,
                label=f'{GROUP_NAMES_SHORT[b]} → {GROUP_NAMES_SHORT[a]}')
        ax1.set_ylabel('交互量', fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'W{t}' for t in range(n_windows)])
        ax1.tick_params(axis='y')
        leg1 = ax1.legend(loc='upper left', fontsize=9, framealpha=0.92)
        leg1.get_frame().set_linewidth(0.7)

        ax2 = ax1.twinx()
        ax2.grid(False)
        stance_diff = np.abs(stance[a] - stance[b])
        ax2.plot(x, stance_diff, color='#E67E22', marker='D',
                 linewidth=2.4, markersize=7,
                 markeredgecolor='white', markeredgewidth=1.0,
                 label='|立场差距|', zorder=5)
        ax2.fill_between(x, 0, stance_diff, color='#E67E22',
                         alpha=0.12, zorder=2)
        ax2.set_ylabel('|立场差距|', fontsize=11, color='#B9530F')
        ax2.tick_params(axis='y', labelcolor='#B9530F')
        ax2.set_ylim(0, max(stance_diff.max() * 1.15, 0.05))
        for sp in ['top']:
            ax2.spines[sp].set_visible(False)
        leg2 = ax2.legend(loc='upper right', fontsize=9, framealpha=0.92)
        leg2.get_frame().set_linewidth(0.7)

        pair_type = '跨党派' if party[a] != party[b] else '党内'
        ax1.set_title(
            f'{GROUP_NAMES_FULL[a]}  ↔  {GROUP_NAMES_FULL[b]}   [{pair_type}]',
            fontsize=12, pad=8)

    axes[-1, 0].set_xlabel('时间窗口', fontsize=12)
    fig.suptitle('交互强度与立场差距联动分析',
                 fontsize=15, fontweight='bold', y=1.005)
    fig.tight_layout()

    path = os.path.join(VIS_DIR, 'interaction_evolution_coupling.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 因果影响网络图
# ══════════════════════════════════════════════════════════════════════════════
def plot_causal_influence_network(data):
    print("[5/7] 因果影响网络图")
    granger = data.get('granger_results')
    if granger is None:
        print("  [跳过] 缺少 granger_results.json")
        return
    edges = granger.get('edges', [])
    if not edges:
        print("  [跳过] granger_results 中无边数据")
        return
    try:
        import networkx as nx
    except ImportError:
        print("  [跳过] 需要安装 networkx: pip install networkx")
        return

    party = get_party_labels(data)
    G = nx.DiGraph()
    for g in range(7):
        G.add_node(g, party=int(party[g]))

    sig_edges = []
    for threshold in [0.05, 0.1, 0.2]:
        sig_edges = [e for e in edges if e['p_value'] < threshold]
        if len(sig_edges) >= 3:
            break
    if not sig_edges:
        sig_edges = sorted(edges, key=lambda e: e['p_value'])[:5]

    for e in sig_edges:
        G.add_edge(e['source'], e['target'],
                   weight=-np.log(max(e['p_value'], 1e-10)),
                   p_value=e['p_value'])

    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(11, 9.5))

    # 节点
    for g in G.nodes():
        ax.scatter(pos[g][0], pos[g][1], s=2000,
                   c=COLORS[g], alpha=0.22, zorder=3)
        ax.scatter(pos[g][0], pos[g][1], s=1100,
                   c=COLORS[g], edgecolors='white', linewidths=2.5, zorder=4)
        ax.text(pos[g][0], pos[g][1], GROUP_NAMES_SHORT[g],
                fontsize=10.5, fontweight='bold', ha='center', va='center',
                color='white', zorder=5)

    # 边按显著性渐变染色
    if G.edges():
        p_vals = [G[u][v]['p_value'] for u, v in G.edges()]
        # 使用反向 p 值映射颜色（越小越深）
        norm = mcolors.Normalize(vmin=0.0, vmax=0.2)
        cmap = plt.cm.plasma_r

        max_w = max(G[u][v]['weight'] for u, v in G.edges())
        for u, v in G.edges():
            p = G[u][v]['p_value']
            w = G[u][v]['weight']
            color = cmap(norm(min(p, 0.2)))
            lw = 1.2 + 3.5 * (w / max_w)
            style = '-' if p < 0.05 else '--'
            arrow = FancyArrowPatch(pos[u], pos[v],
                                    connectionstyle='arc3,rad=0.16',
                                    arrowstyle='-|>',
                                    mutation_scale=20,
                                    linewidth=lw, color=color,
                                    linestyle=style, alpha=0.92,
                                    shrinkA=22, shrinkB=22, zorder=2)
            ax.add_patch(arrow)

            # p 值标注
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
            offset = 0.09
            mx += -dy * offset
            my += dx * offset
            ax.text(mx, my, f'p={p:.3f}', fontsize=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, linewidth=0.8, alpha=0.92),
                    zorder=5)

        # 颜色条表显著性
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
        cbar.set_label('p 值（越小越显著）', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    handles = [
        plt.Line2D([], [], color='#444', linestyle='-', linewidth=2.5,
                   label='p < 0.05  显著'),
        plt.Line2D([], [], color='#888', linestyle='--', linewidth=2.0,
                   label='p ≥ 0.05  边缘显著'),
        mpatches.Patch(color=REP_COLOR, label='共和党'),
        mpatches.Patch(color=DEM_COLOR, label='民主党'),
    ]
    leg = ax.legend(handles=handles, loc='upper left',
                    fontsize=9.5, framealpha=0.92,
                    bbox_to_anchor=(-0.02, 1.02))
    leg.get_frame().set_linewidth(0.8)

    ax.set_title('Granger 因果影响网络', fontsize=14, pad=14)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    path = os.path.join(VIS_DIR, 'causal_influence_network.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 小提琴图
# ══════════════════════════════════════════════════════════════════════════════
def plot_stance_violin(data):
    print("[6/7] 小提琴图")
    stance = data['stance_scores']
    party = get_party_labels(data)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(7)
    parts = ax.violinplot([stance[g] for g in range(7)],
                          positions=positions,
                          widths=0.78,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)

    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(COLORS[i])
        body.set_alpha(0.55)
        body.set_edgecolor(COLORS[i])
        body.set_linewidth(1.4)

    # 散点（每个时间窗口的实际值）
    for g in range(7):
        jitter = (np.random.RandomState(g).rand(stance.shape[1]) - 0.5) * 0.15
        ax.scatter(np.full(stance.shape[1], g) + jitter, stance[g],
                   s=22, color=COLORS[g], edgecolors='white',
                   linewidth=0.6, alpha=0.9, zorder=3)

    # 均值与中位数
    means = stance.mean(axis=1)
    medians = np.median(stance, axis=1)
    ax.scatter(positions, means, marker='D', s=60, color='black',
               edgecolors='white', linewidth=1.0, zorder=5, label='均值')
    ax.scatter(positions, medians, marker='_', s=180, color='#E67E22',
               linewidth=3.0, zorder=5, label='中位数')

    ax.axhline(y=0.5, color='#7F8C8D', linestyle=':', linewidth=1.0,
               alpha=0.8)
    ax.axhspan(0.5, 1.0, color=REP_COLOR, alpha=0.04, zorder=0)
    ax.axhspan(0.0, 0.5, color=DEM_COLOR, alpha=0.04, zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(GROUP_NAMES_FULL, fontsize=10.5)
    ax.set_xlabel('群体', fontsize=12)
    ax.set_ylabel('立场分数  P(共和党)', fontsize=12)
    ax.set_title('群体立场分布  (跨 11 个时间窗口)', fontsize=14, pad=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    handles = [
        mpatches.Patch(color=REP_COLOR, alpha=0.55, label='共和党群体'),
        mpatches.Patch(color=DEM_COLOR, alpha=0.55, label='民主党群体'),
        plt.Line2D([], [], color='black', marker='D', linestyle='None',
                   markersize=7, label='均值'),
        plt.Line2D([], [], color='#E67E22', marker='_', linestyle='None',
                   markersize=14, markeredgewidth=3, label='中位数'),
    ]
    leg = ax.legend(handles=handles, loc='center left',
                    bbox_to_anchor=(1.02, 0.5), fontsize=10)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    path = os.path.join(VIS_DIR, 'stance_violin_plot.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 极化时间线图
# ══════════════════════════════════════════════════════════════════════════════
def plot_polarization_timeline(data):
    print("[7/7] 极化时间线图")
    metrics = data.get('evolution_metrics')
    ei_series, details = [], []
    if metrics and 'ei_index' in metrics:
        ei_data = metrics['ei_index']
        ei_series = ei_data.get('ei_index', [])
        details = ei_data.get('details', [])
    if not ei_series:
        dns_path = os.path.join(RESULTS_DIR, 'dynamic_network_stats.json')
        if os.path.exists(dns_path):
            with open(dns_path, 'r', encoding='utf-8') as f:
                dns = json.load(f)
            entry = dns.get('ei_index_series', [])
            ei_series = [e['EI'] for e in entry] if entry else []
            details = entry
    if not ei_series:
        print("  [跳过] 缺少 E-I Index 数据")
        return

    n_windows = len(ei_series)
    x = np.arange(n_windows)
    ei_arr = np.asarray(ei_series, dtype=float)

    cp_data = data.get('change_points')
    change_points = cp_data.get('all_change_point_windows', []) if cp_data else []

    iw = data.get('interaction_windows')
    if iw is not None:
        total_interactions = iw.sum(axis=(1, 2))
    else:
        total_interactions = (np.array([d.get('total', 0) for d in details])
                              if details else None)

    fig, ax1 = plt.subplots(figsize=(13, 6))

    # 次坐标轴：交互量柱（先画在底层）
    ax2 = None
    if total_interactions is not None and len(total_interactions) == n_windows:
        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.bar(x, total_interactions, color='#95A5A6',
                alpha=0.32, width=0.7, edgecolor='white',
                linewidth=0.6, label='总交互量', zorder=1)
        ax2.set_ylabel('总交互量', fontsize=11, color='#566573')
        ax2.tick_params(axis='y', labelcolor='#566573')
        for sp in ['top']:
            ax2.spines[sp].set_visible(False)

    # 主轴：E-I 折线及阴影
    ax1.fill_between(x, ei_arr, 0,
                     where=(ei_arr < 0), interpolate=True,
                     color=DEM_COLOR, alpha=0.18, label='同质性 (E-I < 0)',
                     zorder=2)
    ax1.fill_between(x, ei_arr, 0,
                     where=(ei_arr >= 0), interpolate=True,
                     color=REP_COLOR, alpha=0.18, label='异质性 (E-I ≥ 0)',
                     zorder=2)
    ax1.plot(x, ei_arr, color='#6C3483', marker='o', linewidth=2.6,
             markersize=8, markeredgecolor='white', markeredgewidth=1.2,
             label='E-I 指数', zorder=4)
    ax1.axhline(y=0, color='#34495E', linestyle='--', linewidth=1.0,
                alpha=0.6, zorder=3)

    ax1.set_xlabel('时间窗口', fontsize=12)
    ax1.set_ylabel('E-I 指数', fontsize=12, color='#6C3483')
    ax1.tick_params(axis='y', labelcolor='#6C3483')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'W{t}' for t in range(n_windows)])

    # 突变点标注
    y_top = ax1.get_ylim()[1] if ax1.get_ylim()[1] > ei_arr.max() else ei_arr.max() * 1.1
    for cp in change_points:
        if 0 <= cp < n_windows:
            ax1.axvline(x=cp, color='#E67E22', linestyle=':',
                        linewidth=1.8, alpha=0.85, zorder=3)
            ax1.annotate(f'突变点\nW{cp}',
                         xy=(cp, ei_arr[cp]),
                         xytext=(cp, y_top),
                         fontsize=9, color='#B9530F', ha='center', va='top',
                         arrowprops=dict(arrowstyle='->', color='#E67E22',
                                         lw=1.0, alpha=0.8),
                         bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                   edgecolor='#E67E22', linewidth=0.8,
                                   alpha=0.9))

    ax1.set_title('群体极化指数时序变化  (E-I Index)',
                  fontsize=14, pad=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    for sp in ['top']:
        ax1.spines[sp].set_visible(False)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        leg = ax1.legend(lines1 + lines2, labels1 + labels2,
                         loc='upper left', fontsize=10, framealpha=0.92,
                         ncol=2)
    else:
        leg = ax1.legend(loc='upper left', fontsize=10, framealpha=0.92)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    path = os.path.join(VIS_DIR, 'polarization_timeline.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Task 4 可视化生成（论文级别）")
    print("=" * 60)

    data = load_data()
    if 'stance_scores' not in data:
        print("[错误] 缺少核心数据 temporal_stance_scores.npy，无法生成图表")
        return

    plot_evolution_trajectory(data)
    plot_firework_diagram(data)
    plot_evolution_heatmap(data)
    plot_interaction_evolution_coupling(data)
    plot_causal_influence_network(data)
    plot_stance_violin(data)
    plot_polarization_timeline(data)

    print("\n" + "=" * 60)
    print("  全部可视化图表生成完成！")
    print(f"  保存目录: {VIS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
