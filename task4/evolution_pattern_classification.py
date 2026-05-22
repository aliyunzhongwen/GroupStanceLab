"""
演化模式分类
对每对群体交互，分类为四种模式之一：
- 趋同型(Convergence): |stance_A - stance_B|随时间递减
- 极化型(Polarization): |stance_A - stance_B|随时间递增
- 稳定型(Stability): 差异无显著趋势
- 传染型(Contagion): 单向显著影响
"""

import os
import json
import numpy as np
from scipy import stats

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
    """根据group_labels获取群体党派标签。索引1-7对应群体0-6"""
    return int(group_labels[group_idx + 1])


def classify_evolution_pattern(stance_scores, interaction_windows, group_labels):
    """对每对有显著交互的群体进行演化模式分类"""
    n_groups = stance_scores.shape[0]  # 7
    n_windows = stance_scores.shape[1]  # 11
    time_points = np.arange(n_windows)

    # 计算每对群体的总交互量
    total_interactions = interaction_windows.sum(axis=0)  # (7, 7)
    interaction_threshold = 100

    results = {}
    pattern_counts = {
        'Convergence': 0,
        'Polarization': 0,
        'Contagion': 0,
        'Stability': 0
    }

    # 按党派分类统计
    same_party_patterns = {
        'Convergence': 0, 'Polarization': 0, 'Contagion': 0, 'Stability': 0
    }
    cross_party_patterns = {
        'Convergence': 0, 'Polarization': 0, 'Contagion': 0, 'Stability': 0
    }

    for a in range(n_groups):
        for b in range(n_groups):
            if a == b:
                continue

            # 双向总交互量（A→B + B→A）
            total_ab = total_interactions[a, b] + total_interactions[b, a]
            if total_ab < interaction_threshold:
                continue

            # 1) 计算立场差距序列
            diff = np.abs(stance_scores[a] - stance_scores[b])  # (11,)

            # 2) 线性回归: diff = slope * t + intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                time_points, diff
            )

            # 计算阈值：基于数据自适应
            # 使用差距序列标准差的一定比例作为阈值
            diff_std = np.std(diff)
            threshold = max(0.001, diff_std * 0.05)

            # 3) 传染型检测：单向相关性检验
            # A→B的交互时序与B的立场变化的相关性
            interaction_ab = interaction_windows[:, a, b]  # A→B交互量 (11,)
            delta_stance_b = np.diff(stance_scores[b])  # B的立场变化 (10,)

            # 需要对齐：用t时刻的交互预测t+1时刻的立场变化
            # interaction_ab[:-1] 对应 delta_stance_b
            is_contagion = False
            contagion_direction = None
            ab_corr_p = 1.0
            ba_corr_p = 1.0

            if len(delta_stance_b) > 2:
                # A→B: A对B的交互与B的立场变化相关
                if np.std(interaction_ab[:-1]) > 0 and np.std(delta_stance_b) > 0:
                    corr_ab, ab_corr_p = stats.pearsonr(
                        interaction_ab[:-1], delta_stance_b
                    )

                # B→A: B对A的交互与A的立场变化相关
                interaction_ba = interaction_windows[:, b, a]  # B→A交互量
                delta_stance_a = np.diff(stance_scores[a])
                if np.std(interaction_ba[:-1]) > 0 and np.std(delta_stance_a) > 0:
                    corr_ba, ba_corr_p = stats.pearsonr(
                        interaction_ba[:-1], delta_stance_a
                    )

                # 传染型: 单向显著（一个方向p<0.1，另一方向不显著）
                ab_sig = ab_corr_p < 0.1
                ba_sig = ba_corr_p < 0.1
                if ab_sig and not ba_sig:
                    is_contagion = True
                    contagion_direction = f"{a}→{b}"
                elif ba_sig and not ab_sig:
                    is_contagion = True
                    contagion_direction = f"{b}→{a}"

            # 4) 分类判定
            if is_contagion:
                pattern = 'Contagion'
            elif slope < -threshold and p_value < 0.05:
                pattern = 'Convergence'
            elif slope > threshold and p_value < 0.05:
                pattern = 'Polarization'
            else:
                pattern = 'Stability'

            pattern_counts[pattern] += 1

            # 党派分类
            party_a = get_party_label(a, group_labels)
            party_b = get_party_label(b, group_labels)
            if party_a == party_b:
                same_party_patterns[pattern] += 1
            else:
                cross_party_patterns[pattern] += 1

            pair_key = f"({a},{b})"
            results[pair_key] = {
                'group_A': int(a),
                'group_B': int(b),
                'party_A': party_a,
                'party_B': party_b,
                'same_party': party_a == party_b,
                'total_interaction': int(total_ab),
                'pattern': pattern,
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'diff_mean': float(np.mean(diff)),
                'diff_std': float(np.std(diff)),
                'diff_start': float(diff[0]),
                'diff_end': float(diff[-1]),
                'ab_corr_p': float(ab_corr_p),
                'ba_corr_p': float(ba_corr_p),
                'contagion_direction': contagion_direction
            }

    return results, pattern_counts, same_party_patterns, cross_party_patterns


def print_results(results, pattern_counts, same_party_patterns, cross_party_patterns):
    """打印详细结果"""
    print("=" * 70)
    print("演化模式分类结果")
    print("=" * 70)

    print("\n【各群体对的演化模式】")
    print("-" * 90)
    print(f"{'群体对':<10} {'党派':<8} {'模式':<15} {'斜率':>8} {'p值':>8} "
          f"{'差距起':>6} {'差距终':>6} {'交互量':>8}")
    print("-" * 90)

    for pair_key, info in sorted(results.items()):
        party_str = "同党" if info['same_party'] else "跨党"
        print(f"({info['group_A']},{info['group_B']}){'':<4} "
              f"{party_str:<8} {info['pattern']:<15} "
              f"{info['slope']:>8.4f} {info['p_value']:>8.4f} "
              f"{info['diff_start']:>6.3f} {info['diff_end']:>6.3f} "
              f"{info['total_interaction']:>8d}")

    print("\n【模式分布统计】")
    print("-" * 40)
    total = sum(pattern_counts.values())
    for pattern, count in pattern_counts.items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {pattern:<15}: {count:>3d} ({pct:>5.1f}%) {bar}")

    print(f"\n  总群体对数: {total}")

    print("\n【同党派 vs 跨党派模式对比】")
    print("-" * 60)
    print(f"{'模式':<15} {'同党派':>8} {'跨党派':>8} {'同党占比':>10} {'跨党占比':>10}")
    print("-" * 60)
    same_total = sum(same_party_patterns.values())
    cross_total = sum(cross_party_patterns.values())
    for pattern in ['Convergence', 'Polarization', 'Contagion', 'Stability']:
        s = same_party_patterns[pattern]
        c = cross_party_patterns[pattern]
        s_pct = s / same_total * 100 if same_total > 0 else 0
        c_pct = c / cross_total * 100 if cross_total > 0 else 0
        print(f"{pattern:<15} {s:>8d} {c:>8d} {s_pct:>9.1f}% {c_pct:>9.1f}%")

    # 传染型详情
    contagion_pairs = [info for info in results.values() if info['pattern'] == 'Contagion']
    if contagion_pairs:
        print("\n【传染型详情】")
        print("-" * 60)
        for info in contagion_pairs:
            print(f"  群体({info['group_A']},{info['group_B']}): "
                  f"方向={info['contagion_direction']}, "
                  f"A→B p={info['ab_corr_p']:.4f}, B→A p={info['ba_corr_p']:.4f}")

    # 趋同型详情
    conv_pairs = [info for info in results.values() if info['pattern'] == 'Convergence']
    if conv_pairs:
        print("\n【趋同型详情】")
        for info in conv_pairs:
            print(f"  群体({info['group_A']},{info['group_B']}): "
                  f"差距 {info['diff_start']:.3f}→{info['diff_end']:.3f}, "
                  f"斜率={info['slope']:.4f}, p={info['p_value']:.4f}")

    # 极化型详情
    pol_pairs = [info for info in results.values() if info['pattern'] == 'Polarization']
    if pol_pairs:
        print("\n【极化型详情】")
        for info in pol_pairs:
            print(f"  群体({info['group_A']},{info['group_B']}): "
                  f"差距 {info['diff_start']:.3f}→{info['diff_end']:.3f}, "
                  f"斜率={info['slope']:.4f}, p={info['p_value']:.4f}")


def main():
    stance_scores, interaction_windows, group_labels = load_data()
    print(f"数据加载完成: stance_scores {stance_scores.shape}, "
          f"interaction_windows {interaction_windows.shape}")
    print(f"群体标签(索引1-7对应群体0-6): {group_labels[1:]}")

    results, pattern_counts, same_party_patterns, cross_party_patterns = \
        classify_evolution_pattern(stance_scores, interaction_windows, group_labels)

    print_results(results, pattern_counts, same_party_patterns, cross_party_patterns)

    # 保存结果
    output = {
        'pair_results': results,
        'pattern_counts': pattern_counts,
        'same_party_patterns': same_party_patterns,
        'cross_party_patterns': cross_party_patterns
    }

    output_path = os.path.join(BASE_DIR, 'results', 'evolution_patterns.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
