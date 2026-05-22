"""
Granger因果分析
检验群体间交互是否"引起"(Granger-cause)群体立场变化
"""

import os
import json
import warnings
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)


def load_data():
    stance_scores = np.load(
        os.path.join(BASE_DIR, 'results', 'temporal_stance_scores.npy')
    )  # (7, 11)
    interaction_windows = np.load(
        os.path.join(BASE_DIR, 'results', 'interaction_windows.npy')
    )  # (11, 7, 7)
    group_labels = np.load(
        os.path.join(LAB_DIR, 'task3', 'group_labels.npy')
    )  # (8,), 索引1-7对应群体0-6
    return stance_scores, interaction_windows, group_labels


def get_party_label(group_idx, group_labels):
    """根据group_labels获取群体党派标签"""
    return int(group_labels[group_idx + 1])


def granger_test_pair(y_series, x_series, maxlag=2):
    """
    检验x是否Granger-causes y
    y_series: 被影响变量的时序 (T,)
    x_series: 可能的因果变量的时序 (T,)
    返回: (min_p_value, f_statistic, significant)
    """
    # 数据长度检查
    if len(y_series) < maxlag + 3:
        return 1.0, 0.0, False

    # 检查方差是否为零
    if np.std(y_series) < 1e-10 or np.std(x_series) < 1e-10:
        return 1.0, 0.0, False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 构建数据: [y, x]，检验x是否Granger-causes y
            data = np.column_stack([y_series, x_series])
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # 取各lag的F-test p值中最小的
            p_values = []
            f_stats = []
            for lag in range(1, maxlag + 1):
                ssr_ftest = result[lag][0]['ssr_ftest']
                p_values.append(ssr_ftest[1])
                f_stats.append(ssr_ftest[0])

            min_idx = np.argmin(p_values)
            min_p = p_values[min_idx]
            best_f = f_stats[min_idx]
            significant = min_p < 0.05

            return float(min_p), float(best_f), significant

    except Exception as e:
        # 数据不足或其他问题
        return 1.0, 0.0, False


def build_causality_network(stance_scores, interaction_windows, group_labels):
    """
    对每对群体(A, B)进行Granger因果检验:
    检验A→B的交互量时序是否Granger-causes B的立场变化
    """
    n_groups = stance_scores.shape[0]
    edges = []
    adjacency = np.zeros((n_groups, n_groups))  # 因果邻接矩阵

    for a in range(n_groups):
        for b in range(n_groups):
            if a == b:
                continue

            # X: A→B的交互量时序
            x_series = interaction_windows[:, a, b]
            # Y: B的立场时序
            y_series = stance_scores[b, :]

            min_p, f_stat, significant = granger_test_pair(
                y_series, x_series, maxlag=2
            )

            edge_info = {
                'source': int(a),
                'target': int(b),
                'source_party': get_party_label(a, group_labels),
                'target_party': get_party_label(b, group_labels),
                'same_party': bool(get_party_label(a, group_labels) == get_party_label(b, group_labels)),
                'p_value': float(min_p),
                'f_statistic': float(f_stat),
                'significant': bool(significant),
                'interaction_total': float(interaction_windows[:, a, b].sum())
            }
            edges.append(edge_info)

            if significant:
                adjacency[a, b] = 1

    return edges, adjacency


def analyze_causality_network(edges, adjacency, group_labels):
    """分析因果网络"""
    n_groups = adjacency.shape[0]

    # 出度和入度
    out_degree = adjacency.sum(axis=1)  # 影响他人的数量
    in_degree = adjacency.sum(axis=0)   # 被他人影响的数量

    # 核心影响群体
    max_out = int(out_degree.max())
    most_influential = [int(i) for i in range(n_groups) if out_degree[i] == max_out]
    max_in = int(in_degree.max())
    most_influenced = [int(i) for i in range(n_groups) if in_degree[i] == max_in]

    # 因果网络密度
    n_possible = n_groups * (n_groups - 1)
    n_actual = int(adjacency.sum())
    density = n_actual / n_possible if n_possible > 0 else 0

    # 同立场 vs 跨立场因果
    same_party_causal = 0
    cross_party_causal = 0
    same_party_total = 0
    cross_party_total = 0

    for edge in edges:
        if edge['same_party']:
            same_party_total += 1
            if edge['significant']:
                same_party_causal += 1
        else:
            cross_party_total += 1
            if edge['significant']:
                cross_party_causal += 1

    # 双向因果对
    bidirectional = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if adjacency[i, j] == 1 and adjacency[j, i] == 1:
                bidirectional.append((int(i), int(j)))

    analysis = {
        'out_degree': [float(x) for x in out_degree],
        'in_degree': [float(x) for x in in_degree],
        'most_influential': most_influential,
        'most_influenced': most_influenced,
        'network_density': float(density),
        'n_significant_edges': int(n_actual),
        'n_possible_edges': int(n_possible),
        'same_party_causal': int(same_party_causal),
        'same_party_total': int(same_party_total),
        'cross_party_causal': int(cross_party_causal),
        'cross_party_total': int(cross_party_total),
        'bidirectional_pairs': bidirectional
    }

    return analysis


def print_results(edges, analysis, group_labels):
    """打印详细结果"""
    n_groups = len(analysis['out_degree'])

    print("=" * 70)
    print("Granger因果分析结果")
    print("=" * 70)

    # 显著因果边
    significant_edges = [e for e in edges if e['significant']]

    print(f"\n【显著因果边列表】 (p < 0.05)")
    print("-" * 70)
    if significant_edges:
        print(f"{'方向':<12} {'党派':<8} {'p值':>10} {'F统计量':>10} {'交互量':>10}")
        print("-" * 70)
        for e in sorted(significant_edges, key=lambda x: x['p_value']):
            direction = f"{e['source']}→{e['target']}"
            party = "同党" if e['same_party'] else "跨党"
            print(f"{direction:<12} {party:<8} {e['p_value']:>10.4f} "
                  f"{e['f_statistic']:>10.3f} {e['interaction_total']:>10.0f}")
    else:
        print("  未发现显著的Granger因果关系")

    # 因果网络统计
    print(f"\n【因果网络统计】")
    print("-" * 40)
    print(f"  网络密度: {analysis['network_density']:.3f}")
    print(f"  显著边数: {analysis['n_significant_edges']} / {analysis['n_possible_edges']}")

    # 出度和入度
    print(f"\n【各群体出度(影响他人)与入度(被影响)】")
    print("-" * 50)
    print(f"{'群体':<6} {'出度':>6} {'入度':>6} {'党派':>8}")
    print("-" * 50)
    for i in range(n_groups):
        party = "共和党" if get_party_label(i, group_labels) == 1 else "民主党"
        print(f"  {i:<4} {int(analysis['out_degree'][i]):>6} "
              f"{int(analysis['in_degree'][i]):>6} {party:>8}")

    # 核心影响群体
    print(f"\n【核心影响群体】")
    print(f"  最具影响力(出度最高): 群体{analysis['most_influential']}, "
          f"出度={int(max(analysis['out_degree']))}")
    print(f"  最易被影响(入度最高): 群体{analysis['most_influenced']}, "
          f"入度={int(max(analysis['in_degree']))}")

    # 同党 vs 跨党
    print(f"\n【同党派 vs 跨党派因果对比】")
    print("-" * 50)
    s_total = analysis['same_party_total']
    c_total = analysis['cross_party_total']
    s_causal = analysis['same_party_causal']
    c_causal = analysis['cross_party_causal']
    s_pct = s_causal / s_total * 100 if s_total > 0 else 0
    c_pct = c_causal / c_total * 100 if c_total > 0 else 0
    print(f"  同党派: {s_causal}/{s_total} 显著 ({s_pct:.1f}%)")
    print(f"  跨党派: {c_causal}/{c_total} 显著 ({c_pct:.1f}%)")

    # 双向因果
    if analysis['bidirectional_pairs']:
        print(f"\n【双向Granger因果群体对】")
        for pair in analysis['bidirectional_pairs']:
            party_a = "共和党" if get_party_label(pair[0], group_labels) == 1 else "民主党"
            party_b = "共和党" if get_party_label(pair[1], group_labels) == 1 else "民主党"
            print(f"  群体{pair[0]}({party_a}) ↔ 群体{pair[1]}({party_b})")
    else:
        print(f"\n  未发现双向Granger因果群体对")

    # 所有边的详细结果（含非显著）
    print(f"\n【全部因果检验结果】")
    print("-" * 80)
    print(f"{'方向':<12} {'党派':<6} {'p值':>10} {'F统计量':>10} {'显著':>6} {'交互量':>10}")
    print("-" * 80)
    for e in sorted(edges, key=lambda x: x['p_value']):
        direction = f"{e['source']}→{e['target']}"
        party = "同党" if e['same_party'] else "跨党"
        sig = "是" if e['significant'] else "否"
        print(f"{direction:<12} {party:<6} {e['p_value']:>10.4f} "
              f"{e['f_statistic']:>10.3f} {sig:>6} {e['interaction_total']:>10.0f}")


def main():
    stance_scores, interaction_windows, group_labels = load_data()
    print(f"数据加载完成: stance_scores {stance_scores.shape}, "
          f"interaction_windows {interaction_windows.shape}")
    print(f"群体标签(索引1-7对应群体0-6): {group_labels[1:]}")

    print("\n正在进行Granger因果检验...")
    print("(maxlag=2, 显著性水平α=0.05)")

    edges, adjacency = build_causality_network(
        stance_scores, interaction_windows, group_labels
    )

    analysis = analyze_causality_network(edges, adjacency, group_labels)

    print_results(edges, analysis, group_labels)

    # 保存结果
    output = {
        'edges': edges,
        'adjacency_matrix': adjacency.tolist(),
        'analysis': analysis
    }

    output_path = os.path.join(BASE_DIR, 'results', 'granger_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
