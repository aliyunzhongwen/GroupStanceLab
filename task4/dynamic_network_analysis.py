"""
动态网络演化分析
核心问题：群体间交互网络的结构如何随时间变化？是否存在突变点？
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# ─── 路径设置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
TASK3_DIR  = os.path.join(BASE_DIR, "task3")
TASK4_DIR  = SCRIPT_DIR
RESULTS    = os.path.join(TASK4_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)

NUM_GROUPS  = 7
NUM_WINDOWS = 11
TS_PER_WIN  = 5


# ════════════════════════════════════════════════════════════════════════════
# 工具：加载或重建 interaction_windows.npy
# ════════════════════════════════════════════════════════════════════════════
def load_or_build_interaction_windows() -> np.ndarray:
    iw_path = os.path.join(RESULTS, "interaction_windows.npy")
    if os.path.exists(iw_path):
        print(f"  加载已有 interaction_windows.npy ...")
        return np.load(iw_path)

    print("  interaction_windows.npy 不存在，自行计算...")
    edges = pd.read_csv(os.path.join(TASK3_DIR, "group_edge_list.csv"))
    iw = np.zeros((NUM_WINDOWS, NUM_GROUPS, NUM_GROUPS), dtype=np.float64)
    for t in range(NUM_WINDOWS):
        ts_min, ts_max = t * TS_PER_WIN, t * TS_PER_WIN + TS_PER_WIN - 1
        sub = edges[(edges["ts"] >= ts_min) & (edges["ts"] <= ts_max)]
        for _, row in sub.iterrows():
            a, b = int(row["u"]) - 1, int(row["i"]) - 1
            if a != b:
                iw[t, a, b] += 1
    np.save(iw_path, iw)
    print(f"  已保存 → {iw_path}")
    return iw


# ════════════════════════════════════════════════════════════════════════════
# 1. 构建时序图序列
# ════════════════════════════════════════════════════════════════════════════
def build_temporal_graphs(iw: np.ndarray) -> list:
    """
    对每个时间窗口构建 NetworkX 有向加权图。
    节点 0-6，边权重 = 该窗口内交互次数。
    """
    print("[1/5] 构建时序有向图...")
    graphs = []
    for t in range(NUM_WINDOWS):
        G = nx.DiGraph()
        G.add_nodes_from(range(NUM_GROUPS))
        for a in range(NUM_GROUPS):
            for b in range(NUM_GROUPS):
                w = iw[t, a, b]
                if w > 0 and a != b:
                    G.add_edge(a, b, weight=float(w))
        graphs.append(G)
        print(f"    窗口{t:2d}: 节点={G.number_of_nodes():2d}  "
              f"边={G.number_of_edges():3d}  "
              f"总权重={sum(d['weight'] for _,_,d in G.edges(data=True)):8.0f}")
    return graphs


# ════════════════════════════════════════════════════════════════════════════
# 2. 计算每个窗口的网络拓扑指标
# ════════════════════════════════════════════════════════════════════════════
def compute_topology_metrics(graphs: list, labels: np.ndarray) -> list:
    """
    计算每个窗口的 8 个拓扑指标。
    labels: 群体0-6的政治标签（0=民主党, 1=共和党）。
    """
    print("[2/5] 计算网络拓扑指标...")
    metrics_list = []

    for t, G in enumerate(graphs):
        m = {"window": t}

        # ── 密度 ──────────────────────────────────────────────────
        m["density"] = nx.density(G)

        # ── 总边权重 & 平均加权度 ──────────────────────────────────
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        m["total_weight"]       = float(sum(weights)) if weights else 0.0
        m["avg_weighted_degree"] = (
            float(np.mean([G.degree(n, weight="weight") for n in G.nodes()]))
        )

        # ── 互惠性（双向边比例）──────────────────────────────────
        if G.number_of_edges() > 0:
            reciprocal = sum(1 for u, v in G.edges() if G.has_edge(v, u))
            m["reciprocity"] = float(reciprocal / G.number_of_edges())
        else:
            m["reciprocity"] = 0.0

        # ── 加权聚类系数（无向近似）──────────────────────────────
        UG = nx.Graph()
        UG.add_nodes_from(range(NUM_GROUPS))
        for u, v, d in G.edges(data=True):
            if UG.has_edge(u, v):
                UG[u][v]["weight"] += d["weight"]
            else:
                UG.add_edge(u, v, weight=d["weight"])
        try:
            clust = nx.average_clustering(UG, weight="weight")
        except Exception:
            clust = 0.0
        m["clustering"] = float(clust)

        # ── 强连通分量数 ──────────────────────────────────────────
        m["n_strongly_connected"] = nx.number_strongly_connected_components(G)

        # ── 度同配性 ─────────────────────────────────────────────
        try:
            assort = nx.degree_assortativity_coefficient(G)
            m["degree_assortativity"] = float(assort) if not np.isnan(assort) else 0.0
        except Exception:
            m["degree_assortativity"] = 0.0

        # ── 模块度（按政治立场）──────────────────────────────────
        communities = [
            {n for n in range(NUM_GROUPS) if labels[n] == 0},  # 民主党
            {n for n in range(NUM_GROUPS) if labels[n] == 1},  # 共和党
        ]
        communities = [c for c in communities if len(c) > 0]
        try:
            mod = nx.community.modularity(UG, communities, weight="weight")
            m["modularity"] = float(mod) if not np.isnan(mod) else 0.0
        except Exception:
            m["modularity"] = 0.0

        metrics_list.append(m)
        print(f"    W{t:2d}: density={m['density']:.3f}  "
              f"reciprocity={m['reciprocity']:.3f}  "
              f"modularity={m['modularity']:+.3f}  "
              f"SCC={m['n_strongly_connected']}")

    return metrics_list


# ════════════════════════════════════════════════════════════════════════════
# 3. 突变点检测
# ════════════════════════════════════════════════════════════════════════════
def detect_change_points(metrics_list: list) -> dict:
    """
    对每个指标计算时间序列，突变定义:
      |指标[t] - 指标[t-1]| > 2 * std(全序列差分)
    """
    print("[3/5] 突变点检测...")
    metric_names = [
        "density", "total_weight", "avg_weighted_degree",
        "reciprocity", "clustering", "n_strongly_connected",
        "degree_assortativity", "modularity",
    ]

    change_points = {}
    all_cps = set()

    for name in metric_names:
        series = np.array([m[name] for m in metrics_list], dtype=float)
        diffs  = np.diff(series)
        thresh = 2.0 * np.std(diffs) if diffs.std() > 1e-10 else np.inf

        cps = []
        for t in range(1, NUM_WINDOWS):
            diff = abs(series[t] - series[t - 1])
            if diff > thresh:
                cps.append({
                    "window": t,
                    "prev_value": float(series[t - 1]),
                    "curr_value": float(series[t]),
                    "abs_change": float(diff),
                    "threshold":  float(thresh),
                })
                all_cps.add(t)

        change_points[name] = {
            "series":        series.tolist(),
            "diffs":         diffs.tolist(),
            "threshold":     float(thresh),
            "change_points": cps,
        }
        if cps:
            print(f"    {name:<26}: 突变点窗口 {[cp['window'] for cp in cps]}")
        else:
            print(f"    {name:<26}: 无突变点")

    change_points["all_change_point_windows"] = sorted(all_cps)
    return change_points


# ════════════════════════════════════════════════════════════════════════════
# 4. 突变前后立场分析
# ════════════════════════════════════════════════════════════════════════════
def analyze_stance_around_changepoints(change_points: dict,
                                        stance_scores: np.ndarray,
                                        labels: np.ndarray) -> dict:
    """
    对每个突变点，比较突变前后1-2个窗口的群体立场变化。
    """
    print("[4/5] 突变前后立场分析...")
    label_name = {0: "民主党", 1: "共和党"}
    cp_windows = change_points["all_change_point_windows"]

    analysis = {}
    for t in cp_windows:
        pre_windows  = [max(0, t-2), max(0, t-1)]
        post_windows = [min(NUM_WINDOWS-1, t), min(NUM_WINDOWS-1, t+1)]

        pre_mean  = stance_scores[:, pre_windows].mean(axis=1)   # (7,)
        post_mean = stance_scores[:, post_windows].mean(axis=1)  # (7,)
        delta     = post_mean - pre_mean                         # (7,)

        group_details = {}
        for g in range(NUM_GROUPS):
            group_details[f"group{g}"] = {
                "label":       label_name.get(int(labels[g]), "?"),
                "pre_mean":    float(pre_mean[g]),
                "post_mean":   float(post_mean[g]),
                "delta":       float(delta[g]),
                "direction":   "↑" if delta[g] > 0.005 else ("↓" if delta[g] < -0.005 else "—"),
            }

        # 民主党 vs 共和党 平均变化
        dem_delta = float(delta[labels == 0].mean()) if (labels == 0).any() else 0.0
        rep_delta = float(delta[labels == 1].mean()) if (labels == 1).any() else 0.0

        analysis[f"window_{t}"] = {
            "change_point_window":  t,
            "pre_windows_used":     pre_windows,
            "post_windows_used":    post_windows,
            "group_details":        group_details,
            "avg_delta_democrat":   dem_delta,
            "avg_delta_republican": rep_delta,
            "polarization_change":  float(rep_delta - dem_delta),
        }
        print(f"    突变点 W{t}: 民主Δ={dem_delta:+.4f}  共和Δ={rep_delta:+.4f}  "
              f"极化Δ={rep_delta - dem_delta:+.4f}")

    return analysis


# ════════════════════════════════════════════════════════════════════════════
# 5. E-I Index & 核心-边缘结构
# ════════════════════════════════════════════════════════════════════════════
def compute_ei_index(iw: np.ndarray, labels: np.ndarray) -> list:
    """
    EI[t] = (跨立场交互数 - 同立场交互数) / 总交互数
    范围 [-1, 1]，-1=完全隔离，1=完全跨派交互
    """
    ei_series = []
    for t in range(NUM_WINDOWS):
        cross = same = 0.0
        for a in range(NUM_GROUPS):
            for b in range(NUM_GROUPS):
                if a == b:
                    continue
                w = iw[t, a, b]
                if labels[a] == labels[b]:
                    same  += w
                else:
                    cross += w
        total = cross + same
        ei = float((cross - same) / total) if total > 0 else 0.0
        ei_series.append({"window": t, "EI": ei,
                           "cross_interactions": float(cross),
                           "same_interactions":  float(same),
                           "total":              float(total)})
    return ei_series


def compute_core_periphery(graphs: list) -> list:
    """
    简化核心-边缘分析：按加权出入度之和确定核心节点（前3名）。
    """
    cp_series = []
    for t, G in enumerate(graphs):
        strength = {n: G.degree(n, weight="weight") for n in G.nodes()}
        sorted_nodes = sorted(strength.items(), key=lambda x: x[1], reverse=True)
        core    = [n for n, _ in sorted_nodes[:3]]
        periph  = [n for n, _ in sorted_nodes[3:]]
        cp_series.append({
            "window":       t,
            "core_nodes":   core,
            "periphery_nodes": periph,
            "core_strength":   [float(strength[n]) for n in core],
        })
    return cp_series


# ════════════════════════════════════════════════════════════════════════════
# 打印时序指标表格
# ════════════════════════════════════════════════════════════════════════════
def print_metrics_table(metrics_list: list, ei_series: list,
                         change_points: dict, stance_scores: np.ndarray,
                         labels: np.ndarray):
    print("\n" + "═"*90)
    print("  动态网络演化分析报告")
    print("═"*90)

    label_name = {0: "民主党", 1: "共和党"}
    cp_windows = set(change_points["all_change_point_windows"])

    # ── 拓扑指标时序表 ─────────────────────────────────────────────
    print("\n[时序网络拓扑指标]")
    print(f"  {'W':>3} {'密度':>7} {'总权重':>10} {'均加权度':>9} "
          f"{'互惠性':>8} {'聚类':>7} {'SCC':>4} {'同配':>8} {'模块度':>8} "
          f"{'EI指数':>8} {'突变*'}")
    print("  " + "-"*87)
    for m, ei in zip(metrics_list, ei_series):
        t   = m["window"]
        flag = " ←突变" if t in cp_windows else ""
        print(f"  {t:3d} {m['density']:7.3f} {m['total_weight']:10.0f} "
              f"{m['avg_weighted_degree']:9.1f} {m['reciprocity']:8.3f} "
              f"{m['clustering']:7.3f} {m['n_strongly_connected']:4d} "
              f"{m['degree_assortativity']:8.3f} {m['modularity']:8.3f} "
              f"{ei['EI']:8.3f}{flag}")

    # ── 突变点汇总 ─────────────────────────────────────────────────
    print("\n[突变点汇总]")
    if cp_windows:
        for name, val in change_points.items():
            if name == "all_change_point_windows":
                continue
            cps = val.get("change_points", [])
            if cps:
                for cp in cps:
                    print(f"  指标={name:<26}  W{cp['window']:2d}: "
                          f"{cp['prev_value']:+.4f} → {cp['curr_value']:+.4f}  "
                          f"Δ={cp['abs_change']:.4f}  阈值={cp['threshold']:.4f}")
    else:
        print("  未检测到显著突变点")

    # ── EI 指数时序 ────────────────────────────────────────────────
    print("\n[E-I Index 时序（>0 = 跨派交互为主，<0 = 同派交互为主）]")
    for ei in ei_series:
        bar_len = int((ei["EI"] + 1) * 15)
        bar = "▓" * bar_len
        print(f"  W{ei['window']:2d} EI={ei['EI']:+.3f}  [{bar:<30}]  "
              f"跨派={ei['cross_interactions']:8.0f}  同派={ei['same_interactions']:8.0f}")

    # ── 立场趋势 ──────────────────────────────────────────────────
    print("\n[各群体立场趋势（P 共和党）]")
    print(f"  {'群体':<14}", end="")
    for t in range(NUM_WINDOWS):
        flag = "*" if t in cp_windows else " "
        print(f" W{t}{flag}", end="")
    print()
    for g in range(NUM_GROUPS):
        lbl = label_name.get(int(labels[g]), "?")
        print(f"  G{g}({lbl:4s})      ", end="")
        for t in range(NUM_WINDOWS):
            print(f" {stance_scores[g, t]:.2f}", end="")
        print()

    print("\n" + "═"*90)


# ════════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  动态网络演化分析  (Dynamic Network Evolution Analysis)")
    print("=" * 70)

    # ── 加载数据 ──────────────────────────────────────────────────
    print("\n[数据加载]")
    stance_scores = np.load(
        os.path.join(RESULTS, "temporal_stance_scores.npy"))   # (7,11)
    group_labels_raw = np.load(
        os.path.join(TASK3_DIR, "group_labels.npy"))           # (8,)
    labels = group_labels_raw[1:]                              # (7,) 群体0-6

    print(f"    stance_scores: {stance_scores.shape}")
    print(f"    group_labels:  {group_labels_raw}")

    # ── 加载/重建时序交互窗口 ──────────────────────────────────────
    print("\n[加载时序交互窗口]")
    iw = load_or_build_interaction_windows()                   # (11,7,7)
    print(f"    interaction_windows: {iw.shape}")

    # ── Step 1: 构建时序图 ────────────────────────────────────────
    graphs = build_temporal_graphs(iw)

    # ── Step 2: 拓扑指标 ──────────────────────────────────────────
    metrics_list = compute_topology_metrics(graphs, labels)

    # ── Step 3: 突变点检测 ────────────────────────────────────────
    change_points = detect_change_points(metrics_list)

    # ── Step 4: 突变前后立场分析 ──────────────────────────────────
    cp_stance = analyze_stance_around_changepoints(
        change_points, stance_scores, labels)

    # ── Step 5: E-I Index & 核心-边缘 ────────────────────────────
    print("[5/5] 计算 E-I Index 与核心-边缘结构...")
    ei_series  = compute_ei_index(iw, labels)
    cp_series  = compute_core_periphery(graphs)

    for ei in ei_series:
        print(f"    W{ei['window']:2d}  EI={ei['EI']:+.3f}")

    # ── 保存结果 ──────────────────────────────────────────────────
    print("\n[保存结果]")

    stats_out = {
        "meta": {
            "num_groups":    NUM_GROUPS,
            "num_windows":   NUM_WINDOWS,
            "group_labels":  {f"group{g}": int(labels[g])
                              for g in range(NUM_GROUPS)},
        },
        "window_metrics":   metrics_list,
        "ei_index_series":  ei_series,
        "core_periphery":   cp_series,
    }
    stats_path = os.path.join(RESULTS, "dynamic_network_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)
    print(f"    已保存 → {stats_path}")

    cp_out = {
        "change_points_by_metric": {
            k: v for k, v in change_points.items()
            if k != "all_change_point_windows"
        },
        "all_change_point_windows": change_points["all_change_point_windows"],
        "stance_analysis_at_changepoints": cp_stance,
    }
    cp_path = os.path.join(RESULTS, "change_points.json")
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(cp_out, f, indent=2, ensure_ascii=False)
    print(f"    已保存 → {cp_path}")

    # ── 打印报告 ──────────────────────────────────────────────────
    print_metrics_table(metrics_list, ei_series, change_points,
                        stance_scores, labels)

    print("\n分析完成。")
