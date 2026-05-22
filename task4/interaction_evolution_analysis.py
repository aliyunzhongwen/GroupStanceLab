"""
交互-演化关联分析
核心问题：群体A对群体B的交互是否导致B的立场发生变化？
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# ─── 路径设置 ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)          # GroupStanceAnalysis/
TASK3_DIR  = os.path.join(BASE_DIR, "task3")
TASK4_DIR  = SCRIPT_DIR
RESULTS    = os.path.join(TASK4_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)

NUM_GROUPS  = 7
NUM_WINDOWS = 11
TS_PER_WIN  = 5   # 每个窗口包含5个时间切片


# ════════════════════════════════════════════════════════════════════════════
# 1. 按时间窗口统计群体间交互量
# ════════════════════════════════════════════════════════════════════════════
def build_interaction_windows(edges: pd.DataFrame) -> np.ndarray:
    """
    将边表按时间窗口聚合为 (11, 7, 7) 的交互矩阵。
    u/i 取值 1-7 → 映射到群体索引 0-6。
    窗口 t 对应 ts in [t*5, t*5+4]。
    """
    print("[1/6] 按时间窗口统计群体间交互量...")
    iw = np.zeros((NUM_WINDOWS, NUM_GROUPS, NUM_GROUPS), dtype=np.float64)

    for t in range(NUM_WINDOWS):
        ts_min = t * TS_PER_WIN
        ts_max = ts_min + TS_PER_WIN - 1
        mask   = (edges["ts"] >= ts_min) & (edges["ts"] <= ts_max)
        sub    = edges[mask]
        for _, row in sub.iterrows():
            a = int(row["u"]) - 1   # 1-7 → 0-6
            b = int(row["i"]) - 1
            if a != b:              # 忽略自环
                iw[t, a, b] += 1

    print(f"    interaction_windows shape: {iw.shape}")
    print(f"    总交互事件数: {iw.sum():.0f}")
    return iw


# ════════════════════════════════════════════════════════════════════════════
# 2. 计算立场变化量
# ════════════════════════════════════════════════════════════════════════════
def compute_delta_stance(stance_scores: np.ndarray) -> np.ndarray:
    """
    delta_stance[g][t] = stance_scores[g][t+1] - stance_scores[g][t]
    shape = (7, 10)
    """
    print("[2/6] 计算立场变化量...")
    delta = stance_scores[:, 1:] - stance_scores[:, :-1]   # (7, 10)
    print(f"    delta_stance shape: {delta.shape}")
    return delta


# ════════════════════════════════════════════════════════════════════════════
# 3. 交互-演化相关性分析
# ════════════════════════════════════════════════════════════════════════════
def correlation_analysis(iw: np.ndarray, delta: np.ndarray) -> dict:
    """
    对每对群体 (A, B), A≠B:
      X = iw[0:10, A, B]   (A→B 在窗口0-9的交互量)
      Y = delta[B][0:10]   (B 在窗口0-9到下一窗口的立场变化)
    计算 Pearson + Spearman 相关系数。
    """
    print("[3/6] 计算交互-演化相关性...")
    results = {}
    for a in range(NUM_GROUPS):
        for b in range(NUM_GROUPS):
            if a == b:
                continue
            X = iw[:10, a, b].astype(float)
            Y = delta[b, :10].astype(float)

            # Pearson（要求线性关系）
            if X.std() < 1e-10 or Y.std() < 1e-10:
                pearson_r, pearson_p = 0.0, 1.0
            else:
                pearson_r, pearson_p = stats.pearsonr(X, Y)

            # Spearman（秩相关，对非线性更鲁棒）
            spearman_r, spearman_p = stats.spearmanr(X, Y)

            key = f"group{a}_to_group{b}"
            results[key] = {
                "source_group": int(a),
                "target_group": int(b),
                "pearson_r":    float(pearson_r),
                "pearson_p":    float(pearson_p),
                "spearman_r":   float(spearman_r),
                "spearman_p":   float(spearman_p),
                "mean_interaction": float(X.mean()),
                "std_interaction":  float(X.std()),
                "significant_pearson":  bool(pearson_p  < 0.05),
                "significant_spearman": bool(spearman_p < 0.05),
            }
    print(f"    共分析 {len(results)} 对群体关系")
    return results


# ════════════════════════════════════════════════════════════════════════════
# 4. 区分同立场交互 vs 跨立场交互
# ════════════════════════════════════════════════════════════════════════════
def split_by_stance(corr_results: dict, group_labels: np.ndarray) -> dict:
    """
    group_labels 索引0占位(-1)，索引1-7为群体0-6的标签。
    """
    print("[4/6] 区分同立场 vs 跨立场交互...")
    # 群体0-6的标签
    labels = group_labels[1:]   # shape=(7,)

    same_pearson, cross_pearson  = [], []
    same_spearman, cross_spearman = [], []

    for key, val in corr_results.items():
        a, b  = val["source_group"], val["target_group"]
        la, lb = labels[a], labels[b]
        if la == lb:
            same_pearson.append(val["pearson_r"])
            same_spearman.append(val["spearman_r"])
        else:
            cross_pearson.append(val["pearson_r"])
            cross_spearman.append(val["spearman_r"])

    summary = {
        "same_stance": {
            "n_pairs":          len(same_pearson),
            "mean_pearson_r":   float(np.mean(same_pearson))  if same_pearson  else 0.0,
            "mean_spearman_r":  float(np.mean(same_spearman)) if same_spearman else 0.0,
            "std_pearson_r":    float(np.std(same_pearson))   if same_pearson  else 0.0,
        },
        "cross_stance": {
            "n_pairs":          len(cross_pearson),
            "mean_pearson_r":   float(np.mean(cross_pearson))  if cross_pearson  else 0.0,
            "mean_spearman_r":  float(np.mean(cross_spearman)) if cross_spearman else 0.0,
            "std_pearson_r":    float(np.std(cross_pearson))   if cross_pearson  else 0.0,
        },
    }
    return summary, labels


# ════════════════════════════════════════════════════════════════════════════
# 5. 假设验证
# ════════════════════════════════════════════════════════════════════════════
def hypothesis_testing(corr_results: dict, labels: np.ndarray,
                        iw: np.ndarray, delta: np.ndarray) -> dict:
    """
    H1: 跨立场交互导致立场趋同（民主党群体接受共和党影响后 P(共和党)上升；反之下降）
    H2: 同立场交互强化回音室（交互量增加 → 立场更极端）
    """
    print("[5/6] 假设验证...")
    hypotheses = {}

    # ── H1: 跨立场交互 ──────────────────────────────────────────
    # 方向性分析：共和党(1)群体 A → 民主党(0)群体 B
    #   若有影响，则 B 的 P(共和党) 应该上升 → pearson_r > 0
    # 民主党(0)群体 A → 共和党(1)群体 B
    #   若有影响，则 B 的 P(共和党) 应该下降 → pearson_r < 0

    rep_to_dem_r  = []  # 共和 → 民主
    dem_to_rep_r  = []  # 民主 → 共和

    for key, val in corr_results.items():
        a, b  = val["source_group"], val["target_group"]
        la, lb = labels[a], labels[b]
        if la == 1 and lb == 0:          # 共和→民主
            rep_to_dem_r.append(val["pearson_r"])
        elif la == 0 and lb == 1:        # 民主→共和
            dem_to_rep_r.append(val["pearson_r"])

    # 单样本 t 检验：均值是否显著异于0
    def t_test_zero(arr, label):
        if len(arr) < 2:
            return {"mean": float(np.mean(arr)) if arr else 0.0,
                    "t": 0.0, "p": 1.0, "significant": False}
        t, p = stats.ttest_1samp(arr, 0)
        return {"mean": float(np.mean(arr)), "t": float(t),
                "p": float(p), "significant": bool(p < 0.05),
                "direction": "positive" if np.mean(arr) > 0 else "negative",
                "interpretation": label}

    h1_rep_to_dem = t_test_zero(
        rep_to_dem_r,
        "共和党→民主党交互增加 → 民主党立场向共和党靠拢(预期正相关)"
    )
    h1_dem_to_rep = t_test_zero(
        dem_to_rep_r,
        "民主党→共和党交互增加 → 共和党立场向民主党靠拢(预期负相关)"
    )

    # 综合H1判断
    h1_supported = (
        (h1_rep_to_dem["significant"] and h1_rep_to_dem.get("direction") == "positive") or
        (h1_dem_to_rep["significant"] and h1_dem_to_rep.get("direction") == "negative")
    )

    hypotheses["H1_cross_stance_convergence"] = {
        "hypothesis": "跨立场交互导致目标群体立场向来源群体靠拢（立场趋同）",
        "republican_to_democrat": h1_rep_to_dem,
        "democrat_to_republican": h1_dem_to_rep,
        "supported":              bool(h1_supported),
    }

    # ── H2: 同立场交互强化回音室 ──────────────────────────────
    # 同立场交互 → 立场更极端
    # 共和党群体: P(共和党) 应升高 → pearson_r > 0
    # 民主党群体: P(共和党) 应降低 → pearson_r < 0
    same_rep_r = []  # 共和 → 共和，目标是共和党
    same_dem_r = []  # 民主 → 民主，目标是民主党

    for key, val in corr_results.items():
        a, b  = val["source_group"], val["target_group"]
        la, lb = labels[a], labels[b]
        if la == 1 and lb == 1:
            same_rep_r.append(val["pearson_r"])
        elif la == 0 and lb == 0:
            same_dem_r.append(val["pearson_r"])

    h2_same_rep = t_test_zero(
        same_rep_r,
        "同立场（共和）交互增加 → P(共和党)升高（更极端，预期正相关）"
    )
    h2_same_dem = t_test_zero(
        same_dem_r,
        "同立场（民主）交互增加 → P(共和党)降低（更极端，预期负相关）"
    )

    h2_supported = (
        (h2_same_rep["significant"] and h2_same_rep.get("direction") == "positive") or
        (h2_same_dem["significant"] and h2_same_dem.get("direction") == "negative")
    )

    hypotheses["H2_echo_chamber"] = {
        "hypothesis": "同立场交互强化回音室（立场更极端）",
        "same_republican": h2_same_rep,
        "same_democrat":   h2_same_dem,
        "supported":       bool(h2_supported),
    }

    return hypotheses


# ════════════════════════════════════════════════════════════════════════════
# 打印分析表格
# ════════════════════════════════════════════════════════════════════════════
def print_results(stance_scores, delta, iw, corr_results, stance_summary,
                   hypotheses, labels):
    print("\n" + "═"*70)
    print("  交互-演化关联分析报告")
    print("═"*70)

    label_name = {0: "民主党", 1: "共和党"}
    group_info = [f"G{g}({label_name.get(labels[g], '?')})" for g in range(NUM_GROUPS)]

    # ── 立场分数概览 ──────────────────────────────────────────────
    print("\n[立场分数矩阵 P(共和党)]  行=群体, 列=时间窗口(0-10)")
    header = " " * 6 + "".join(f"W{t:2d}" for t in range(NUM_WINDOWS))
    print(header)
    for g in range(NUM_GROUPS):
        row = f"{group_info[g]:10s} " + "".join(f"{v:.2f} " for v in stance_scores[g])
        print(row)

    # ── 立场变化量概览 ─────────────────────────────────────────────
    print("\n[立场变化量 Δ(P 共和党)]  行=群体, 列=窗口t→t+1")
    header2 = " " * 10 + "".join(f" Δ{t}-{t+1}" for t in range(10))
    print(header2)
    for g in range(NUM_GROUPS):
        row = f"{group_info[g]:10s} " + "".join(f" {v:+.3f}" for v in delta[g])
        print(row)

    # ── 总交互量热图（文字版）──────────────────────────────────────
    print("\n[总交互矩阵（累计）]  行=来源, 列=目标")
    total_iw = iw.sum(axis=0)
    print(" " * 10 + "".join(f" {group_info[b]:>12s}" for b in range(NUM_GROUPS)))
    for a in range(NUM_GROUPS):
        row = f"{group_info[a]:10s} " + "".join(f" {total_iw[a,b]:12.0f}" for b in range(NUM_GROUPS))
        print(row)

    # ── 相关性 Top-10 ──────────────────────────────────────────────
    print("\n[交互→立场变化 Pearson相关系数 Top-10（绝对值）]")
    sorted_pairs = sorted(corr_results.items(),
                          key=lambda x: abs(x[1]["pearson_r"]), reverse=True)
    print(f"  {'对':<22} {'Pearson r':>10} {'p':>8} {'Spearman r':>12} {'p':>8} {'显著*'}")
    print("  " + "-"*68)
    for k, v in sorted_pairs[:10]:
        a, b = v["source_group"], v["target_group"]
        sig = "✓" if v["significant_pearson"] else " "
        print(f"  {group_info[a]}→{group_info[b]:<16} "
              f"{v['pearson_r']:>10.4f} {v['pearson_p']:>8.4f} "
              f"{v['spearman_r']:>12.4f} {v['spearman_p']:>8.4f}  {sig}")

    # ── 同/跨立场汇总 ──────────────────────────────────────────────
    print("\n[同立场 vs 跨立场 交互-演化相关性汇总]")
    for stance_type, d in stance_summary.items():
        print(f"  {stance_type:<14}: n={d['n_pairs']:2d}  "
              f"mean Pearson r={d['mean_pearson_r']:+.4f}  "
              f"mean Spearman r={d['mean_spearman_r']:+.4f}  "
              f"std={d['std_pearson_r']:.4f}")

    # ── 假设验证 ──────────────────────────────────────────────────
    print("\n[假设验证结果]")
    for hname, hval in hypotheses.items():
        sup = "✓ 支持" if hval["supported"] else "✗ 不支持"
        print(f"\n  {hname} [{sup}]")
        print(f"  假设: {hval['hypothesis']}")
        for sub_key in [k for k in hval.keys() if k not in ("hypothesis", "supported")]:
            sv = hval[sub_key]
            if isinstance(sv, dict):
                sig = "显著" if sv.get("significant") else "不显著"
                print(f"    {sub_key:<30}: mean r={sv.get('mean',0):+.4f}  "
                      f"t={sv.get('t',0):.3f}  p={sv.get('p',1):.4f}  [{sig}]")

    print("\n" + "═"*70)


# ════════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  交互-演化关联分析  (Interaction-Evolution Correlation Analysis)")
    print("=" * 70)

    # ── 加载数据 ──────────────────────────────────────────────────
    print("\n[数据加载]")
    stance_scores = np.load(
        os.path.join(RESULTS, "temporal_stance_scores.npy"))       # (7,11)
    group_labels  = np.load(
        os.path.join(TASK3_DIR, "group_labels.npy"))               # (8,)
    edges = pd.read_csv(
        os.path.join(TASK3_DIR, "group_edge_list.csv"))

    print(f"    stance_scores: {stance_scores.shape}")
    print(f"    group_labels:  {group_labels}")
    print(f"    edges:         {edges.shape}")

    # ── Step 1: 时序交互窗口 ──────────────────────────────────────
    iw_path = os.path.join(RESULTS, "interaction_windows.npy")
    if os.path.exists(iw_path):
        print("[1/6] 加载已有 interaction_windows.npy ...")
        iw = np.load(iw_path)
    else:
        iw = build_interaction_windows(edges)
        np.save(iw_path, iw)
        print(f"    已保存 → {iw_path}")

    # ── Step 2: 立场变化量 ────────────────────────────────────────
    delta = compute_delta_stance(stance_scores)

    # ── Step 3: 相关性分析 ────────────────────────────────────────
    corr_results = correlation_analysis(iw, delta)

    # ── Step 4: 同/跨立场分组 ────────────────────────────────────
    stance_summary, labels = split_by_stance(corr_results, group_labels)

    # ── Step 5: 假设验证 ──────────────────────────────────────────
    hypotheses = hypothesis_testing(corr_results, labels, iw, delta)

    # ── Step 6: 保存结果 ──────────────────────────────────────────
    print("[6/6] 保存结果...")
    out = {
        "meta": {
            "num_groups":   NUM_GROUPS,
            "num_windows":  NUM_WINDOWS,
            "ts_per_window": TS_PER_WIN,
            "group_labels": {f"group{g}": int(labels[g])
                             for g in range(NUM_GROUPS)},
        },
        "stance_summary":       stance_summary,
        "hypotheses":           hypotheses,
        "pairwise_correlations": corr_results,
    }
    out_path = os.path.join(RESULTS, "interaction_evolution_correlation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"    已保存 → {out_path}")

    # ── 打印详细报告 ──────────────────────────────────────────────
    print_results(stance_scores, delta, iw, corr_results,
                  stance_summary, hypotheses, labels)

    print("\n分析完成。")
