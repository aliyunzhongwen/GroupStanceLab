# -*- coding: utf-8 -*-
"""
演化分析综合评价指标计算

计算以下指标:
1. E-I Index 时序 (群体极化指数)
2. 极小化指标 (同立场群体合并合理性)
3. 极大化指标 (跨立场群体分裂合理性)
4. 立场稳定性指数
5. 演化预测准确率汇总
6. 综合输出

运行方式:
    cd /root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task4
    python evolution_metrics.py
"""

import os
import json
import numpy as np

# ── 路径设置 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)
TASK3_DIR = os.path.join(LAB_DIR, 'task3')
TASK2_DIR = os.path.join(LAB_DIR, 'task2')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def load_data():
    """加载所有必要数据"""
    print("=" * 60)
    print("[步骤0] 加载数据")
    print("=" * 60)

    # 时序立场分数 (7, 11)
    stance_scores = np.load(os.path.join(RESULTS_DIR, 'temporal_stance_scores.npy'))
    print(f"  temporal_stance_scores: {stance_scores.shape}")

    # 时序交互窗口 (11, 7, 7)
    interaction_windows = np.load(os.path.join(RESULTS_DIR, 'interaction_windows.npy'))
    print(f"  interaction_windows: {interaction_windows.shape}")

    # 群体标签 (8,) — 索引0占位(-1), 索引1-7对应群体0-6
    group_labels = np.load(os.path.join(TASK3_DIR, 'group_labels.npy'))
    print(f"  group_labels: {group_labels}")

    # 群体特征 (8, 81)
    group_features = np.load(os.path.join(TASK3_DIR, 'group_features_compact.npy'))
    print(f"  group_features_compact: {group_features.shape}")

    # 总体交互矩阵 (7, 7)
    interaction_matrix = np.load(os.path.join(TASK3_DIR, 'interaction_matrix.npy'))
    print(f"  interaction_matrix: {interaction_matrix.shape}")

    # 群体分配 (878,)
    group_assignments = np.load(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'))
    print(f"  group_assignments: {group_assignments.shape}")

    return (stance_scores, interaction_windows, group_labels,
            group_features, interaction_matrix, group_assignments)


# ══════════════════════════════════════════════════════════════════════════════
# 1. E-I Index 时序 (群体极化指数)
# ══════════════════════════════════════════════════════════════════════════════
def compute_ei_index(interaction_windows, group_labels):
    """
    对每个时间窗口t计算E-I Index:
      EI[t] = (外部交互数 - 内部交互数) / 总交互数
      - 内部交互: 同立场群体间的交互
      - 外部交互: 跨立场群体间的交互
      范围[-1, 1], -1=完全同质交互, +1=完全异质交互
    """
    print("\n[步骤1] 计算E-I Index时序")

    n_windows = interaction_windows.shape[0]
    n_groups = interaction_windows.shape[1]
    labels = group_labels[1:]  # 群体0-6的标签, shape=(7,)

    ei_series = []
    details = []

    for t in range(n_windows):
        iw_t = interaction_windows[t]  # (7, 7)
        same_interactions = 0.0
        cross_interactions = 0.0

        for a in range(n_groups):
            for b in range(n_groups):
                if a == b:
                    continue
                val = iw_t[a, b]
                if labels[a] == labels[b]:
                    same_interactions += val
                else:
                    cross_interactions += val

        total = same_interactions + cross_interactions
        if total > 0:
            ei = (cross_interactions - same_interactions) / total
        else:
            ei = 0.0

        ei_series.append(ei)
        details.append({
            'window': int(t),
            'EI': float(ei),
            'same_interactions': float(same_interactions),
            'cross_interactions': float(cross_interactions),
            'total': float(total)
        })

    ei_array = np.array(ei_series)

    # 趋势分析
    from scipy import stats as sp_stats
    x = np.arange(n_windows)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, ei_array)
    trend = "上升趋势(极化减弱)" if slope > 0 else "下降趋势(极化增强)"

    print(f"  EI Index: {[f'{v:.4f}' for v in ei_array]}")
    print(f"  趋势: {trend}, slope={slope:.4f}, p={p_value:.4f}")

    return {
        'ei_index': ei_array.tolist(),
        'details': details,
        'trend': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'interpretation': trend
        },
        'mean': float(ei_array.mean()),
        'std': float(ei_array.std()),
        'min': float(ei_array.min()),
        'max': float(ei_array.max())
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. 极小化指标 (Group Merging)
# ══════════════════════════════════════════════════════════════════════════════
def compute_merging_metrics(group_labels, group_features, interaction_matrix,
                            group_assignments):
    """
    评估每对同立场群体的合并合理性。
    对每对(A, B)同立场群体:
      - 交互密度: (interaction[A,B] + interaction[B,A]) / (size_A * size_B)
      - 特征余弦相似度: cosine(feat_A, feat_B)
      - 合并合理性得分: 0.5 * 交互密度标准化 + 0.5 * 余弦相似度
    """
    print("\n[步骤2] 计算极小化指标(同立场群体合并合理性)")

    n_groups = 7
    labels = group_labels[1:]  # (7,)

    # 计算群体大小
    sizes = np.array([(group_assignments == g).sum() for g in range(n_groups)])
    print(f"  群体规模: {sizes.tolist()}")

    # 群体特征: 索引1-7对应群体0-6
    features = group_features[1:]  # (7, 81)

    # 找同立场群体对
    same_stance_pairs = []
    for a in range(n_groups):
        for b in range(a + 1, n_groups):
            if labels[a] == labels[b]:
                same_stance_pairs.append((a, b))

    print(f"  同立场群体对: {same_stance_pairs}")

    # 计算每对的合并合理性
    merging_results = []
    density_values = []

    for a, b in same_stance_pairs:
        # 交互密度
        interaction_flow = float(interaction_matrix[a, b] + interaction_matrix[b, a])
        density = interaction_flow / (sizes[a] * sizes[b]) if (sizes[a] * sizes[b]) > 0 else 0.0
        density_values.append(density)

        # 余弦相似度
        fa, fb = features[a], features[b]
        norm_a = np.linalg.norm(fa)
        norm_b = np.linalg.norm(fb)
        cos_sim = float(np.dot(fa, fb) / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0

        merging_results.append({
            'group_a': int(a),
            'group_b': int(b),
            'party': 'Republican' if labels[a] == 1 else 'Democrat',
            'interaction_a_to_b': float(interaction_matrix[a, b]),
            'interaction_b_to_a': float(interaction_matrix[b, a]),
            'interaction_flow': interaction_flow,
            'size_a': int(sizes[a]),
            'size_b': int(sizes[b]),
            'interaction_density': float(density),
            'cosine_similarity': float(cos_sim),
        })

    # 标准化交互密度到[0,1]
    if len(density_values) > 0:
        min_d = min(density_values)
        max_d = max(density_values)
        range_d = max_d - min_d if max_d > min_d else 1.0

        for i, res in enumerate(merging_results):
            norm_density = (density_values[i] - min_d) / range_d
            res['interaction_density_normalized'] = float(norm_density)
            res['merging_score'] = float(0.5 * norm_density + 0.5 * res['cosine_similarity'])

    # 按合并得分排序
    merging_results.sort(key=lambda x: x['merging_score'], reverse=True)

    print("  合并合理性排名:")
    for res in merging_results:
        print(f"    G{res['group_a']}({res['party'][0]})-G{res['group_b']}({res['party'][0]}): "
              f"score={res['merging_score']:.4f} "
              f"(density_norm={res['interaction_density_normalized']:.4f}, "
              f"cos={res['cosine_similarity']:.4f})")

    return {
        'same_stance_pairs': same_stance_pairs,
        'group_sizes': sizes.tolist(),
        'pair_results': merging_results
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. 极大化指标 (Group Splitting)
# ══════════════════════════════════════════════════════════════════════════════
def compute_splitting_metrics(group_labels, group_features, interaction_matrix):
    """
    评估每对跨立场群体的分裂合理性。
    对每对(A, B)跨立场群体:
      - 交互稀疏度: 1 - (interaction[A,B] + interaction[B,A]) / max_interaction
      - 特征差异度: 1 - cosine(feat_A, feat_B)
      - 分裂合理性得分: 0.5 * 交互稀疏度 + 0.5 * 特征差异度
    """
    print("\n[步骤3] 计算极大化指标(跨立场群体分裂合理性)")

    n_groups = 7
    labels = group_labels[1:]

    features = group_features[1:]  # (7, 81)

    # 找跨立场群体对
    cross_stance_pairs = []
    for a in range(n_groups):
        for b in range(a + 1, n_groups):
            if labels[a] != labels[b]:
                cross_stance_pairs.append((a, b))

    print(f"  跨立场群体对: {cross_stance_pairs}")

    # 计算所有对的最大交互量
    all_flows = []
    for a, b in cross_stance_pairs:
        flow = float(interaction_matrix[a, b] + interaction_matrix[b, a])
        all_flows.append(flow)
    max_interaction = max(all_flows) if all_flows else 1.0

    splitting_results = []

    for idx, (a, b) in enumerate(cross_stance_pairs):
        interaction_flow = all_flows[idx]
        # 交互稀疏度
        sparsity = 1.0 - interaction_flow / max_interaction if max_interaction > 0 else 1.0

        # 特征差异度
        fa, fb = features[a], features[b]
        norm_a = np.linalg.norm(fa)
        norm_b = np.linalg.norm(fb)
        cos_sim = float(np.dot(fa, fb) / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0
        feature_diff = 1.0 - cos_sim

        splitting_results.append({
            'group_a': int(a),
            'group_b': int(b),
            'party_a': 'Republican' if labels[a] == 1 else 'Democrat',
            'party_b': 'Republican' if labels[b] == 1 else 'Democrat',
            'interaction_a_to_b': float(interaction_matrix[a, b]),
            'interaction_b_to_a': float(interaction_matrix[b, a]),
            'interaction_flow': float(interaction_flow),
            'interaction_sparsity': float(sparsity),
            'cosine_similarity': float(cos_sim),
            'feature_difference': float(feature_diff),
            'splitting_score': float(0.5 * sparsity + 0.5 * feature_diff),
        })

    # 按分裂得分排序
    splitting_results.sort(key=lambda x: x['splitting_score'], reverse=True)

    print("  分裂合理性排名:")
    for res in splitting_results:
        print(f"    G{res['group_a']}({res['party_a'][0]})-G{res['group_b']}({res['party_b'][0]}): "
              f"score={res['splitting_score']:.4f} "
              f"(sparsity={res['interaction_sparsity']:.4f}, "
              f"feat_diff={res['feature_difference']:.4f})")

    return {
        'cross_stance_pairs': cross_stance_pairs,
        'max_interaction': float(max_interaction),
        'pair_results': splitting_results
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 立场稳定性指数
# ══════════════════════════════════════════════════════════════════════════════
def compute_stability_index(stance_scores, group_labels):
    """
    对每个群体: stability[g] = var(stance_scores[g, :])
    方差越小越稳定。
    """
    print("\n[步骤4] 计算立场稳定性指数")

    n_groups = stance_scores.shape[0]
    labels = group_labels[1:]

    variances = np.var(stance_scores, axis=1)  # (7,)

    results = []
    for g in range(n_groups):
        party = 'Republican' if labels[g] == 1 else 'Democrat'
        results.append({
            'group': int(g),
            'party': party,
            'variance': float(variances[g]),
            'mean_stance': float(stance_scores[g].mean()),
            'std_stance': float(stance_scores[g].std()),
            'range': float(stance_scores[g].max() - stance_scores[g].min()),
        })

    # 按立场分组统计
    dem_groups = [g for g in range(n_groups) if labels[g] == 0]
    rep_groups = [g for g in range(n_groups) if labels[g] == 1]

    dem_var = variances[dem_groups]
    rep_var = variances[rep_groups]

    group_stats = {
        'democrat': {
            'groups': dem_groups,
            'mean_variance': float(dem_var.mean()),
            'std_variance': float(dem_var.std()),
        },
        'republican': {
            'groups': rep_groups,
            'mean_variance': float(rep_var.mean()),
            'std_variance': float(rep_var.std()),
        }
    }

    # 最稳定和最不稳定的群体
    most_stable = int(np.argmin(variances))
    least_stable = int(np.argmax(variances))

    print(f"  各群体方差: {[f'{v:.6f}' for v in variances]}")
    print(f"  民主党平均方差: {dem_var.mean():.6f}")
    print(f"  共和党平均方差: {rep_var.mean():.6f}")
    print(f"  最稳定群体: G{most_stable} (var={variances[most_stable]:.6f})")
    print(f"  最不稳定群体: G{least_stable} (var={variances[least_stable]:.6f})")

    return {
        'individual_stability': results,
        'group_statistics': group_stats,
        'most_stable_group': most_stable,
        'least_stable_group': least_stable,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. 演化预测准确率汇总
# ══════════════════════════════════════════════════════════════════════════════
def compute_prediction_summary():
    """从已有结果中提取预测指标"""
    print("\n[步骤5] 演化预测准确率汇总")

    prediction_summary = {}

    # 从 influence_model_results.json 提取 MAE/RMSE
    influence_path = os.path.join(RESULTS_DIR, 'influence_model_results.json')
    if os.path.exists(influence_path):
        with open(influence_path, 'r', encoding='utf-8') as f:
            influence_data = json.load(f)

        prediction_summary['influence_models'] = {
            'degroot': {
                'mae': influence_data.get('degroot', {}).get('mae', None),
                'rmse': influence_data.get('degroot', {}).get('rmse', None),
            },
            'friedkin_johnsen': {
                'mae': influence_data.get('friedkin_johnsen', {}).get('mae', None),
                'rmse': influence_data.get('friedkin_johnsen', {}).get('rmse', None),
                'alpha': influence_data.get('friedkin_johnsen', {}).get('alpha', None),
            },
            'gnn': {
                'mae': influence_data.get('gnn', {}).get('mae', None),
                'rmse': influence_data.get('gnn', {}).get('rmse', None),
                'test_mae': influence_data.get('gnn', {}).get('test_mae', None),
                'test_rmse': influence_data.get('gnn', {}).get('test_rmse', None),
            },
        }

        # 找最佳模型
        models_mae = {
            'DeGroot': influence_data.get('degroot', {}).get('mae', float('inf')),
            'Friedkin-Johnsen': influence_data.get('friedkin_johnsen', {}).get('mae', float('inf')),
            'GNN': influence_data.get('gnn', {}).get('mae', float('inf')),
        }
        best_model = min(models_mae, key=models_mae.get)
        prediction_summary['influence_models']['best_model'] = best_model

        print(f"  影响力模型对比:")
        for name, mae in models_mae.items():
            print(f"    {name}: MAE={mae:.4f}")
        print(f"  最佳模型: {best_model}")
    else:
        print("  [跳过] influence_model_results.json 不存在")

    # 从 granger_results.json 提取因果显著性
    granger_path = os.path.join(RESULTS_DIR, 'granger_results.json')
    if os.path.exists(granger_path):
        with open(granger_path, 'r', encoding='utf-8') as f:
            granger_data = json.load(f)

        edges = granger_data.get('edges', [])
        total_edges = len(edges)
        significant_edges = sum(1 for e in edges if e.get('significant', False))
        p_values = [e['p_value'] for e in edges if 'p_value' in e]

        # 按阈值统计
        p_005 = sum(1 for p in p_values if p < 0.05)
        p_010 = sum(1 for p in p_values if p < 0.10)
        p_020 = sum(1 for p in p_values if p < 0.20)

        # 同党 vs 跨党
        same_total = sum(1 for e in edges if e.get('same_party', False))
        cross_total = sum(1 for e in edges if not e.get('same_party', False))
        same_sig = sum(1 for e in edges if e.get('same_party', False) and e.get('significant', False))
        cross_sig = sum(1 for e in edges if not e.get('same_party', False) and e.get('significant', False))

        prediction_summary['granger_causality'] = {
            'total_pairs': total_edges,
            'significant_pairs_p005': p_005,
            'significant_pairs_p010': p_010,
            'significant_pairs_p020': p_020,
            'significance_rate_p005': float(p_005 / total_edges) if total_edges > 0 else 0.0,
            'significance_rate_p010': float(p_010 / total_edges) if total_edges > 0 else 0.0,
            'same_party': {
                'total': same_total,
                'significant': same_sig,
                'rate': float(same_sig / same_total) if same_total > 0 else 0.0,
            },
            'cross_party': {
                'total': cross_total,
                'significant': cross_sig,
                'rate': float(cross_sig / cross_total) if cross_total > 0 else 0.0,
            },
        }

        print(f"  Granger因果分析:")
        print(f"    总对数: {total_edges}")
        print(f"    p<0.05: {p_005}/{total_edges} ({p_005/total_edges*100:.1f}%)")
        print(f"    p<0.10: {p_010}/{total_edges} ({p_010/total_edges*100:.1f}%)")
        print(f"    同党显著: {same_sig}/{same_total}")
        print(f"    跨党显著: {cross_sig}/{cross_total}")
    else:
        print("  [跳过] granger_results.json 不存在")

    return prediction_summary


# ══════════════════════════════════════════════════════════════════════════════
# 6. 综合输出
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(ei_results, merging_results, splitting_results,
                  stability_results, prediction_summary, group_labels):
    """打印汇总表"""
    labels = group_labels[1:]

    print("\n" + "=" * 70)
    print("  演化分析综合评价指标 — 汇总")
    print("=" * 70)

    # E-I Index
    print("\n┌─ E-I Index 时序")
    print(f"│  均值: {ei_results['mean']:.4f} ± {ei_results['std']:.4f}")
    print(f"│  范围: [{ei_results['min']:.4f}, {ei_results['max']:.4f}]")
    print(f"│  趋势: {ei_results['trend']['interpretation']}")

    # 合并指标
    print("\n┌─ 极小化指标(合并合理性)")
    for res in merging_results['pair_results'][:3]:
        print(f"│  G{res['group_a']}-G{res['group_b']}({res['party'][0]}): "
              f"score={res['merging_score']:.4f}")

    # 分裂指标
    print("\n┌─ 极大化指标(分裂合理性)")
    for res in splitting_results['pair_results'][:3]:
        print(f"│  G{res['group_a']}({res['party_a'][0]})-G{res['group_b']}({res['party_b'][0]}): "
              f"score={res['splitting_score']:.4f}")

    # 稳定性
    print("\n┌─ 立场稳定性")
    dem_stats = stability_results['group_statistics']['democrat']
    rep_stats = stability_results['group_statistics']['republican']
    print(f"│  民主党平均方差: {dem_stats['mean_variance']:.6f}")
    print(f"│  共和党平均方差: {rep_stats['mean_variance']:.6f}")
    print(f"│  最稳定: G{stability_results['most_stable_group']}")
    print(f"│  最不稳定: G{stability_results['least_stable_group']}")

    # 预测
    if 'influence_models' in prediction_summary:
        print("\n┌─ 演化预测")
        print(f"│  最佳模型: {prediction_summary['influence_models']['best_model']}")
    if 'granger_causality' in prediction_summary:
        gc = prediction_summary['granger_causality']
        print(f"│  Granger显著率(p<0.05): {gc['significance_rate_p005']*100:.1f}%")

    print("\n" + "=" * 70)


def save_results(ei_results, merging_results, splitting_results,
                 stability_results, prediction_summary):
    """保存所有指标结果"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 综合指标
    evolution_metrics = {
        'ei_index': ei_results,
        'merging_metrics': merging_results,
        'splitting_metrics': splitting_results,
        'stability_index': stability_results,
        'prediction_summary': prediction_summary,
    }

    path1 = os.path.join(RESULTS_DIR, 'evolution_metrics.json')
    with open(path1, 'w', encoding='utf-8') as f:
        json.dump(evolution_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  已保存: {path1}")

    # 极化详情
    path2 = os.path.join(RESULTS_DIR, 'group_polarization_metrics.json')
    with open(path2, 'w', encoding='utf-8') as f:
        json.dump(ei_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  已保存: {path2}")

    # 合并指标
    path3 = os.path.join(RESULTS_DIR, 'group_merging_metrics.json')
    with open(path3, 'w', encoding='utf-8') as f:
        json.dump(merging_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  已保存: {path3}")

    # 分裂指标
    path4 = os.path.join(RESULTS_DIR, 'group_splitting_metrics.json')
    with open(path4, 'w', encoding='utf-8') as f:
        json.dump(splitting_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  已保存: {path4}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # 加载数据
    (stance_scores, interaction_windows, group_labels,
     group_features, interaction_matrix, group_assignments) = load_data()

    # 1. E-I Index
    ei_results = compute_ei_index(interaction_windows, group_labels)

    # 2. 极小化指标
    merging_results = compute_merging_metrics(
        group_labels, group_features, interaction_matrix, group_assignments
    )

    # 3. 极大化指标
    splitting_results = compute_splitting_metrics(
        group_labels, group_features, interaction_matrix
    )

    # 4. 立场稳定性
    stability_results = compute_stability_index(stance_scores, group_labels)

    # 5. 演化预测汇总
    prediction_summary = compute_prediction_summary()

    # 6. 保存
    save_results(ei_results, merging_results, splitting_results,
                 stability_results, prediction_summary)

    # 打印汇总
    print_summary(ei_results, merging_results, splitting_results,
                  stability_results, prediction_summary, group_labels)

    print("\n演化指标计算完成！")


if __name__ == '__main__':
    main()
