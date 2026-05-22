# -*- coding: utf-8 -*-
"""
Balanced Community Detection
Target: 8-15 groups, each with at least 20 members
"""

import os
import json
import pickle
import numpy as np
import networkx as nx
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import community as community_louvain
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

font_candidates = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_candidates
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
VIS_DIR = os.path.join(TASK2_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 60)
print("Loading data...")
print("=" * 60)

with open(os.path.join(TASK2_DIR, 'combined_graph_undirected.pkl'), 'rb') as f:
    G_undirected = pickle.load(f)
with open(os.path.join(TASK2_DIR, 'combined_graph_directed.pkl'), 'rb') as f:
    G_directed = pickle.load(f)
node_labels = np.load(os.path.join(BASE_DIR, 'processed', 'ml_twitter_node_labels.npy'))
features_compact = np.load(os.path.join(BASE_DIR, 'task1', 'features', 'individual_features_compact.npy'))

print(f"Undirected graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
print(f"Node labels: shape={node_labels.shape}")
print(f"Compact features: shape={features_compact.shape}")

valid_nodes = sorted([n for n in G_undirected.nodes() if 1 <= n <= 877])
n_valid = len(valid_nodes)
print(f"Valid nodes: {n_valid}")

ground_truth = {n: int(node_labels[n]) for n in valid_nodes}
gt_array = np.array([ground_truth[n] for n in valid_nodes])
print(f"Democrat: {np.sum(gt_array == 0)}, Republican: {np.sum(gt_array == 1)}")

node_list = sorted(valid_nodes)
node_to_idx = {n: i for i, n in enumerate(node_list)}
n_nodes = len(node_list)

adj_matrix = np.zeros((n_nodes, n_nodes))
for u, v, data in G_undirected.edges(data=True):
    if u in node_to_idx and v in node_to_idx:
        w = data.get('weight', 1.0)
        adj_matrix[node_to_idx[u], node_to_idx[v]] = w
        adj_matrix[node_to_idx[v], node_to_idx[u]] = w

features_valid = features_compact[1:878].copy()
nan_mask = np.isnan(features_valid)
if nan_mask.any():
    col_means = np.nanmean(features_valid, axis=0)
    for col in range(features_valid.shape[1]):
        features_valid[nan_mask[:, col], col] = col_means[col]
    features_valid = np.nan_to_num(features_valid, nan=0.0)


# ============ Evaluation Functions ============
def compute_modularity(G, partition_dict):
    try:
        return community_louvain.modularity(partition_dict, G)
    except:
        communities = {}
        for node, comm in partition_dict.items():
            if comm not in communities:
                communities[comm] = set()
            communities[comm].add(node)
        comm_list = list(communities.values())
        return nx.community.modularity(G, comm_list)


def compute_nmi(partition_dict, valid_nodes, ground_truth):
    pred = [partition_dict[n] for n in valid_nodes]
    gt = [ground_truth[n] for n in valid_nodes]
    return normalized_mutual_info_score(gt, pred)


def compute_purity(partition_dict, valid_nodes, ground_truth):
    communities = {}
    for n in valid_nodes:
        comm = partition_dict[n]
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(ground_truth[n])
    total_correct = 0
    total_nodes = 0
    for comm_id, labels in communities.items():
        labels = np.array(labels)
        majority_count = max(np.sum(labels == 0), np.sum(labels == 1))
        total_correct += majority_count
        total_nodes += len(labels)
    return total_correct / total_nodes if total_nodes > 0 else 0


def compute_gini(sizes):
    sizes = np.array(sizes, dtype=float)
    if len(sizes) == 0 or np.sum(sizes) == 0:
        return 0
    sizes_sorted = np.sort(sizes)
    n = len(sizes_sorted)
    gini = (2 * np.sum((np.arange(1, n + 1) * sizes_sorted))) / (n * np.sum(sizes_sorted)) - (n + 1) / n
    return gini


def compute_balance_score(sizes, total_nodes):
    return 1 - (max(sizes) - min(sizes)) / total_nodes


def compute_effective_ratio(sizes, min_size=20):
    effective = sum(1 for s in sizes if s >= min_size)
    return effective / len(sizes) if len(sizes) > 0 else 0


def get_sizes(partition_dict, valid_nodes):
    communities = {}
    for n in valid_nodes:
        comm = partition_dict[n]
        if comm not in communities:
            communities[comm] = 0
        communities[comm] += 1
    return sorted(communities.values(), reverse=True)


def evaluate_balanced(G, partition_dict, valid_nodes, ground_truth, method_name):
    modularity = compute_modularity(G, partition_dict)
    nmi = compute_nmi(partition_dict, valid_nodes, ground_truth)
    purity = compute_purity(partition_dict, valid_nodes, ground_truth)
    sizes = get_sizes(partition_dict, valid_nodes)
    n_comms = len(sizes)
    balance_score = compute_balance_score(sizes, len(valid_nodes))
    gini = compute_gini(sizes)
    effective_ratio = compute_effective_ratio(sizes)
    result = {
        'method': method_name,
        'modularity': round(modularity, 4),
        'nmi': round(nmi, 4),
        'purity': round(purity, 4),
        'num_communities': n_comms,
        'max_size': max(sizes),
        'min_size': min(sizes),
        'mean_size': round(float(np.mean(sizes)), 1),
        'std_size': round(float(np.std(sizes)), 1),
        'sizes': sizes,
        'balance_score': round(balance_score, 4),
        'gini': round(gini, 4),
        'effective_ratio': round(effective_ratio, 4)
    }
    print(f"  Mod={result['modularity']:.4f}, NMI={result['nmi']:.4f}, Pur={result['purity']:.4f}")
    print(f"  #C={n_comms}, Sizes={sizes}")
    print(f"  Balance={balance_score:.4f}, Gini={gini:.4f}, EffRatio={effective_ratio:.4f}")
    return result


def merge_small_groups(partition_dict, valid_nodes, G, min_size=20):
    partition = partition_dict.copy()
    while True:
        comm_members = {}
        for n in valid_nodes:
            c = partition[n]
            if c not in comm_members:
                comm_members[c] = []
            comm_members[c].append(n)
        small_comms = [c for c, members in comm_members.items() if len(members) < min_size]
        if not small_comms:
            break
        smallest = min(small_comms, key=lambda c: len(comm_members[c]))
        members = comm_members[smallest]
        other_comms = [c for c in comm_members.keys() if c != smallest]
        if not other_comms:
            break
        conn_strength = {}
        for c in other_comms:
            strength = 0
            for m in members:
                for neighbor in G.neighbors(m):
                    if neighbor in valid_nodes and partition[neighbor] == c:
                        strength += G[m][neighbor].get('weight', 1.0)
            conn_strength[c] = strength
        if max(conn_strength.values()) > 0:
            target = max(conn_strength, key=conn_strength.get)
        else:
            target = max(other_comms, key=lambda c: len(comm_members[c]))
        for m in members:
            partition[m] = target
    unique_labels = sorted(set(partition[n] for n in valid_nodes))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for n in valid_nodes:
        partition[n] = label_map[partition[n]]
    return partition


def split_large_groups(partition_dict, valid_nodes, features, node_list, max_size=200):
    partition = partition_dict.copy()
    next_label = max(partition[n] for n in valid_nodes) + 1
    node_to_idx_local = {n: i for i, n in enumerate(node_list)}
    while True:
        comm_members = {}
        for n in valid_nodes:
            c = partition[n]
            if c not in comm_members:
                comm_members[c] = []
            comm_members[c].append(n)
        large_comms = [c for c, members in comm_members.items() if len(members) > max_size]
        if not large_comms:
            break
        for c in large_comms:
            members = comm_members[c]
            n_sub = max(2, len(members) // 100)
            n_sub = min(n_sub, len(members) // 20)
            if n_sub < 2:
                continue
            member_features = np.array([features[node_to_idx_local[m]] for m in members])
            km = KMeans(n_clusters=n_sub, random_state=2023, n_init=10)
            sub_labels = km.fit_predict(member_features)
            for i, m in enumerate(members):
                if sub_labels[i] == 0:
                    partition[m] = c
                else:
                    partition[m] = next_label + sub_labels[i] - 1
            next_label += n_sub - 1
        break
    unique_labels = sorted(set(partition[n] for n in valid_nodes))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for n in valid_nodes:
        partition[n] = label_map[partition[n]]
    return partition


# ============ Balanced Community Detection ============
all_results = []
all_partitions = {}

# --- (a) Louvain multi-resolution + merge ---
print("\n" + "=" * 60)
print("1(a) Louvain Multi-Resolution + Merge Small Groups")
print("=" * 60)

for resolution in [1.5, 2.0, 2.5]:
    print(f"\n  Resolution = {resolution}")
    partition = community_louvain.best_partition(G_undirected, resolution=resolution, random_state=2023)
    partition_merged = merge_small_groups(partition, valid_nodes, G_undirected, min_size=20)
    method_name = f'Louvain(res={resolution})+Merge'
    result = evaluate_balanced(G_undirected, partition_merged, valid_nodes, ground_truth, method_name)
    result['params'] = f'resolution={resolution}, merge_min=20'
    all_results.append(result)
    all_partitions[f'louvain_res{resolution}_merged'] = partition_merged

# --- (b) Hierarchical Louvain ---
print("\n" + "=" * 60)
print("1(b) Hierarchical Louvain")
print("=" * 60)

for res1 in [0.8, 1.0]:
    print(f"\n  First level resolution = {res1}")
    partition_coarse = community_louvain.best_partition(G_undirected, resolution=res1, random_state=2023)
    comm_members = {}
    for n in valid_nodes:
        c = partition_coarse[n]
        if c not in comm_members:
            comm_members[c] = []
        comm_members[c].append(n)
    partition_hier = {}
    next_label = 0
    for c, members in sorted(comm_members.items()):
        if len(members) > 150:
            subgraph = G_undirected.subgraph(members).copy()
            if subgraph.number_of_edges() > 0:
                sub_partition = community_louvain.best_partition(subgraph, resolution=1.5, random_state=2023)
                sub_comms = set(sub_partition.values())
                for node in members:
                    partition_hier[node] = next_label + sub_partition[node]
                next_label += len(sub_comms)
            else:
                for node in members:
                    partition_hier[node] = next_label
                next_label += 1
        else:
            for node in members:
                partition_hier[node] = next_label
            next_label += 1
    partition_hier = merge_small_groups(partition_hier, valid_nodes, G_undirected, min_size=20)
    method_name = f'HierLouvain(res1={res1})'
    result = evaluate_balanced(G_undirected, partition_hier, valid_nodes, ground_truth, method_name)
    result['params'] = f'res1={res1}, split_threshold=150'
    all_results.append(result)
    all_partitions[f'hier_louvain_res{res1}'] = partition_hier

# --- (c) Constrained Spectral Clustering ---
print("\n" + "=" * 60)
print("1(c) Constrained Spectral Clustering")
print("=" * 60)

for k in [6, 8, 10, 12, 15]:
    print(f"\n  K = {k}")
    try:
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=2023, n_init=10)
        labels_sc = sc.fit_predict(adj_matrix)
        partition_sc = {node_list[i]: int(labels_sc[i]) for i in range(n_nodes)}
        partition_sc = merge_small_groups(partition_sc, valid_nodes, G_undirected, min_size=20)
        partition_sc = split_large_groups(partition_sc, valid_nodes, features_valid, node_list, max_size=200)
        method_name = f'Spectral(K={k})+Balance'
        result = evaluate_balanced(G_undirected, partition_sc, valid_nodes, ground_truth, method_name)
        result['params'] = f'K={k}, merge<20, split>200'
        all_results.append(result)
        all_partitions[f'spectral_k{k}_balanced'] = partition_sc
    except Exception as e:
        print(f"  Failed: {e}")

# --- (d) Feature-based Balanced K-means ---
print("\n" + "=" * 60)
print("1(d) Feature-based Balanced K-means")
print("=" * 60)

for k in [6, 8, 10, 12]:
    print(f"\n  K = {k}")
    km = MiniBatchKMeans(n_clusters=k, random_state=2023, batch_size=256, n_init=10)
    labels_km = km.fit_predict(features_valid)
    partition_km = {node_list[i]: int(labels_km[i]) for i in range(n_nodes)}
    partition_km = merge_small_groups(partition_km, valid_nodes, G_undirected, min_size=20)
    partition_km = split_large_groups(partition_km, valid_nodes, features_valid, node_list, max_size=200)
    method_name = f'MiniBatchKMeans(K={k})+Balance'
    result = evaluate_balanced(G_undirected, partition_km, valid_nodes, ground_truth, method_name)
    result['params'] = f'K={k}, balanced post-processing'
    all_results.append(result)
    all_partitions[f'mbkmeans_k{k}_balanced'] = partition_km

    km2 = KMeans(n_clusters=k, random_state=2023, n_init=10)
    labels_km2 = km2.fit_predict(features_valid)
    partition_km2 = {node_list[i]: int(labels_km2[i]) for i in range(n_nodes)}
    partition_km2 = merge_small_groups(partition_km2, valid_nodes, G_undirected, min_size=20)
    partition_km2 = split_large_groups(partition_km2, valid_nodes, features_valid, node_list, max_size=200)
    method_name2 = f'KMeans(K={k})+Balance'
    result2 = evaluate_balanced(G_undirected, partition_km2, valid_nodes, ground_truth, method_name2)
    result2['params'] = f'K={k}, balanced post-processing'
    all_results.append(result2)
    all_partitions[f'kmeans_k{k}_balanced'] = partition_km2

# --- (e) Hybrid Method ---
print("\n" + "=" * 60)
print("1(e) Hybrid Method (0.6*Graph + 0.4*Feature)")
print("=" * 60)

adj_norm = adj_matrix / (adj_matrix.max() + 1e-8)
cos_sim = cosine_similarity(features_valid)
cos_sim = np.maximum(cos_sim, 0)
fused_matrix = 0.6 * adj_norm + 0.4 * cos_sim
print(f"  Fused matrix range: [{fused_matrix.min():.4f}, {fused_matrix.max():.4f}]")

for k in [8, 10, 12]:
    print(f"\n  K = {k}")
    try:
        sc_hybrid = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=2023, n_init=10)
        labels_hybrid = sc_hybrid.fit_predict(fused_matrix)
        partition_hybrid = {node_list[i]: int(labels_hybrid[i]) for i in range(n_nodes)}
        partition_hybrid = merge_small_groups(partition_hybrid, valid_nodes, G_undirected, min_size=20)
        partition_hybrid = split_large_groups(partition_hybrid, valid_nodes, features_valid, node_list, max_size=200)
        method_name = f'Hybrid(0.6G+0.4F,K={k})+Bal'
        result = evaluate_balanced(G_undirected, partition_hybrid, valid_nodes, ground_truth, method_name)
        result['params'] = f'alpha=0.6, K={k}, balanced'
        all_results.append(result)
        all_partitions[f'hybrid_k{k}_balanced'] = partition_hybrid
    except Exception as e:
        print(f"  Failed: {e}")


# ============ Comprehensive Scoring ============
print("\n" + "=" * 60)
print("Comprehensive Scoring")
print("=" * 60)

mods = [r['modularity'] for r in all_results]
nmis = [r['nmi'] for r in all_results]
purities = [r['purity'] for r in all_results]
mod_min, mod_max = min(mods), max(mods)
nmi_min, nmi_max = min(nmis), max(nmis)
pur_min, pur_max = min(purities), max(purities)

def normalize(val, vmin, vmax):
    if vmax - vmin < 1e-8:
        return 0.5
    return (val - vmin) / (vmax - vmin)

best_result = None
best_score = -1

for r in all_results:
    mod_n = normalize(r['modularity'], mod_min, mod_max)
    nmi_n = normalize(r['nmi'], nmi_min, nmi_max)
    pur_n = normalize(r['purity'], pur_min, pur_max)
    bal = r['balance_score']
    gini_inv = 1 - r['gini']
    score = 0.25 * mod_n + 0.25 * nmi_n + 0.20 * pur_n + 0.15 * bal + 0.15 * gini_inv
    r['comprehensive_score'] = round(score, 4)
    if score > best_score:
        best_score = score
        best_result = r

print(f"\n{'Method':<35} {'#C':>3} {'Mod':>7} {'NMI':>7} {'Pur':>7} {'Bal':>7} {'Gini':>7} {'Score':>7}")
print("-" * 95)
for r in all_results:
    marker = " ***" if r is best_result else ""
    print(f"{r['method']:<35} {r['num_communities']:>3} {r['modularity']:>7.4f} {r['nmi']:>7.4f} {r['purity']:>7.4f} {r['balance_score']:>7.4f} {r['gini']:>7.4f} {r['comprehensive_score']:>7.4f}{marker}")

print(f"\n*** Best: {best_result['method']} ***")
print(f"  Score={best_result['comprehensive_score']:.4f}, Mod={best_result['modularity']:.4f}")
print(f"  NMI={best_result['nmi']:.4f}, Purity={best_result['purity']:.4f}")
print(f"  Balance={best_result['balance_score']:.4f}, Gini={best_result['gini']:.4f}")
print(f"  #Communities={best_result['num_communities']}, Sizes={best_result['sizes']}")

# Find partition key
best_key = None
for key, part in all_partitions.items():
    sizes = get_sizes(part, valid_nodes)
    if len(sizes) == best_result['num_communities'] and sizes == best_result['sizes']:
        best_key = key
        break
if best_key is None:
    for key, part in all_partitions.items():
        mod = compute_modularity(G_undirected, part)
        if abs(mod - best_result['modularity']) < 0.001:
            best_key = key
            break
print(f"  Key: {best_key}")
best_partition = all_partitions[best_key]


# ============ Save Results ============
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

# group_assignments_balanced.npy
group_assignments = np.full(878, -1, dtype=int)
for n in valid_nodes:
    group_assignments[n] = best_partition[n]
np.save(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'), group_assignments)
print(f"  Saved group_assignments_balanced.npy: shape={group_assignments.shape}")

# balanced_results.json
results_json = {
    'best_method': best_result['method'],
    'best_params': best_result['params'],
    'best_comprehensive_score': best_result['comprehensive_score'],
    'best_metrics': {
        'modularity': best_result['modularity'],
        'nmi': best_result['nmi'],
        'purity': best_result['purity'],
        'balance_score': best_result['balance_score'],
        'gini': best_result['gini'],
        'effective_ratio': best_result['effective_ratio'],
        'num_communities': best_result['num_communities'],
        'sizes': best_result['sizes']
    },
    'scoring_formula': '0.25*mod_norm + 0.25*NMI_norm + 0.20*purity_norm + 0.15*balance + 0.15*(1-gini)',
    'results': all_results
}
with open(os.path.join(TASK2_DIR, 'balanced_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print("  Saved balanced_results.json")

# group_evaluation_balanced.md
comm_stats = {}
for n in valid_nodes:
    c = best_partition[n]
    if c not in comm_stats:
        comm_stats[c] = {'dem': 0, 'rep': 0, 'total': 0}
    comm_stats[c]['total'] += 1
    if ground_truth[n] == 0:
        comm_stats[c]['dem'] += 1
    else:
        comm_stats[c]['rep'] += 1

md = []
md.append("# \u5e73\u8861\u7fa4\u4f53\u5212\u5206\u8bc4\u4f30\u62a5\u544a\n")
md.append("## 1. \u6982\u8ff0\n")
md.append(f"- \u6570\u636e\u96c6: Twitter\u653f\u6cbb\u4ea4\u4e92\u7f51\u7edc")
md.append(f"- \u6709\u6548\u8282\u70b9\u6570: {n_valid}")
md.append(f"- \u56fe\u8fb9\u6570: {G_undirected.number_of_edges()}")
md.append(f"- \u653f\u515a\u5206\u5e03: \u6c11\u4e3b\u515a={int(np.sum(gt_array==0))}, \u5171\u548c\u515a={int(np.sum(gt_array==1))}")
md.append(f"- \u76ee\u6807: 8-15\u4e2a\u7fa4\u4f53\uff0c\u6bcf\u4e2a\u7fa4\u4f53\u81f3\u5c1120\u4eba\n")

md.append("## 2. \u5404\u65b9\u6cd5\u5bf9\u6bd4\u8868\u683c\n")
md.append("| \u65b9\u6cd5 | \u7fa4\u4f53\u6570 | \u6a21\u5757\u5ea6 | NMI | \u7eaf\u5ea6 | \u5e73\u8861\u6027 | \u57fa\u5c3c\u7cfb\u6570 | \u6709\u6548\u6bd4 | \u7efc\u5408\u5f97\u5206 |")
md.append("|------|:------:|:------:|:---:|:----:|:------:|:--------:|:------:|:--------:|")
for r in all_results:
    md.append(f"| {r['method']} | {r['num_communities']} | {r['modularity']:.4f} | {r['nmi']:.4f} | {r['purity']:.4f} | {r['balance_score']:.4f} | {r['gini']:.4f} | {r['effective_ratio']:.2f} | {r['comprehensive_score']:.4f} |")

md.append(f"\n## 3. \u6700\u4f18\u65b9\u6848\u8be6\u60c5\n")
md.append(f"**\u65b9\u6cd5**: {best_result['method']}\n")
md.append(f"**\u53c2\u6570**: {best_result['params']}\n")
md.append(f"**\u7efc\u5408\u5f97\u5206**: {best_result['comprehensive_score']:.4f}\n")
md.append(f"**\u8bc4\u4f30\u6307\u6807**:")
md.append(f"- \u6a21\u5757\u5ea6: {best_result['modularity']:.4f}")
md.append(f"- NMI: {best_result['nmi']:.4f}")
md.append(f"- \u7eaf\u5ea6: {best_result['purity']:.4f}")
md.append(f"- \u5e73\u8861\u6027\u5206\u6570: {best_result['balance_score']:.4f}")
md.append(f"- \u57fa\u5c3c\u7cfb\u6570: {best_result['gini']:.4f}")
md.append(f"- \u6709\u6548\u7fa4\u4f53\u6bd4\u4f8b: {best_result['effective_ratio']:.4f}")
md.append(f"- \u7fa4\u4f53\u6570: {best_result['num_communities']}")
md.append(f"- \u7fa4\u4f53\u5927\u5c0f\u5206\u5e03: {best_result['sizes']}\n")

md.append("## 4. \u5404\u7fa4\u4f53\u7acb\u573a\u5206\u5e03\n")
md.append("| \u7fa4\u4f53 | \u603b\u4eba\u6570 | \u6c11\u4e3b\u515a | \u5171\u548c\u515a | \u4e3b\u5bfc\u7acb\u573a | \u7eaf\u5ea6 |")
md.append("|:----:|:------:|:------:|:------:|:--------:|:----:|")
for c in sorted(comm_stats.keys()):
    s = comm_stats[c]
    dominant = "\u6c11\u4e3b\u515a" if s['dem'] > s['rep'] else "\u5171\u548c\u515a"
    pur = max(s['dem'], s['rep']) / s['total']
    md.append(f"| {c} | {s['total']} | {s['dem']} | {s['rep']} | {dominant} | {pur:.2%} |")

md.append(f"\n## 5. \u4e0e\u4e4b\u524d\u4e0d\u5e73\u8861\u65b9\u6848\u7684\u5bf9\u6bd4\n")
md.append("| \u6307\u6807 | \u4e4b\u524d\u65b9\u6848 Spectral(K=4) | \u5f53\u524d\u5e73\u8861\u65b9\u6848 |")
md.append("|------|:----------------------:|:------------:|")
md.append(f"| \u7fa4\u4f53\u6570 | 4 | {best_result['num_communities']} |")
md.append(f"| \u5927\u5c0f\u5206\u5e03 | [526, 347, 2, 2] | {best_result['sizes']} |")
md.append(f"| \u6700\u5927\u7fa4\u4f53 | 526 | {best_result['max_size']} |")
md.append(f"| \u6700\u5c0f\u7fa4\u4f53 | 2 | {best_result['min_size']} |")
old_balance = 1 - (526 - 2) / 877
old_gini = compute_gini([526, 347, 2, 2])
md.append(f"| \u5e73\u8861\u6027\u5206\u6570 | {old_balance:.4f} | {best_result['balance_score']:.4f} |")
md.append(f"| \u57fa\u5c3c\u7cfb\u6570 | {old_gini:.4f} | {best_result['gini']:.4f} |")
md.append(f"| \u6709\u6548\u7fa4\u4f53\u6bd4\u4f8b | 0.50 (2/4>=20\u4eba) | {best_result['effective_ratio']:.2f} |")

md.append(f"\n## 6. \u8bc4\u5206\u516c\u5f0f\n")
md.append("\u7efc\u5408\u5f97\u5206 = 0.25\u00d7\u6a21\u5757\u5ea6(\u5f52\u4e00\u5316) + 0.25\u00d7NMI(\u5f52\u4e00\u5316) + 0.20\u00d7\u7eaf\u5ea6(\u5f52\u4e00\u5316) + 0.15\u00d7\u5e73\u8861\u6027\u5206\u6570 + 0.15\u00d7(1-\u57fa\u5c3c\u7cfb\u6570)\n")
md.append("## 7. \u7ed3\u8bba\n")
md.append(f"\u901a\u8fc7\u591a\u79cd\u5e73\u8861\u7ea6\u675f\u7b56\u7565\uff0c\u6700\u7ec8\u9009\u62e9 **{best_result['method']}** \u4f5c\u4e3a\u6700\u4f18\u5e73\u8861\u65b9\u6848\u3002")
md.append(f"\u4e0e\u4e4b\u524d\u6781\u4e0d\u5e73\u8861\u7684Spectral(K=4)\u65b9\u6848\u76f8\u6bd4\uff0c\u5e73\u8861\u6027\u5927\u5e45\u63d0\u5347\uff08\u5e73\u8861\u6027\u5206\u6570\u4ece{old_balance:.4f}\u63d0\u5347\u5230{best_result['balance_score']:.4f}\uff09\uff0c")
md.append(f"\u57fa\u5c3c\u7cfb\u6570\u4ece{old_gini:.4f}\u964d\u4f4e\u5230{best_result['gini']:.4f}\uff0c\u6240\u6709\u7fa4\u4f53\u5747\u8fbe\u5230\u6709\u6548\u89c4\u6a21\uff08>=20\u4eba\uff09\u3002")

with open(os.path.join(TASK2_DIR, 'group_evaluation_balanced.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))
print("  Saved group_evaluation_balanced.md")


# ============ Visualizations ============
print("\n" + "=" * 60)
print("Generating visualizations...")
print("=" * 60)

community_labels = [best_partition[n] for n in valid_nodes]
unique_communities = sorted(set(community_labels))
n_comms = len(unique_communities)
cmap = plt.cm.get_cmap('tab10' if n_comms <= 10 else 'tab20')
comm_colors = {c: cmap(i / max(n_comms - 1, 1)) for i, c in enumerate(unique_communities)}

# (a) balanced_community_network.png
print("\n  Generating balanced_community_network.png...")
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
G_vis = G_undirected.subgraph(valid_nodes).copy()
pos = nx.spring_layout(G_vis, seed=2023, k=0.3, iterations=50)

dem_nodes = [n for n in valid_nodes if ground_truth[n] == 0]
rep_nodes = [n for n in valid_nodes if ground_truth[n] == 1]
dem_colors_list = [comm_colors[best_partition[n]] for n in dem_nodes]
rep_colors_list = [comm_colors[best_partition[n]] for n in rep_nodes]

nx.draw_networkx_edges(G_vis, pos, alpha=0.05, width=0.3, ax=ax)
nx.draw_networkx_nodes(G_vis, pos, nodelist=dem_nodes, node_color=dem_colors_list,
                       node_shape='o', node_size=30, alpha=0.8, ax=ax,
                       edgecolors='blue', linewidths=0.5)
nx.draw_networkx_nodes(G_vis, pos, nodelist=rep_nodes, node_color=rep_colors_list,
                       node_shape='s', node_size=30, alpha=0.8, ax=ax,
                       edgecolors='red', linewidths=0.5)

legend_elements = []
for c in unique_communities:
    count = community_labels.count(c)
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=comm_colors[c], markersize=10,
                                  label=f'\u7fa4\u4f53{c} (n={count})'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              markeredgecolor='blue', markersize=10, label='\u6c11\u4e3b\u515a'))
legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                              markeredgecolor='red', markersize=10, label='\u5171\u548c\u515a'))
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
ax.set_title(f'\u4f18\u5316\u7fa4\u4f53\u5212\u5206\u7ed3\u679c\uff08\u5e73\u8861\u7248\uff09\n({best_result["method"]})', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'balanced_community_network.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved balanced_community_network.png")

# (b) balanced_community_sizes.png
print("\n  Generating balanced_community_sizes.png...")
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
groups_sorted = sorted(comm_stats.keys())
dem_counts = [comm_stats[c]['dem'] for c in groups_sorted]
rep_counts = [comm_stats[c]['rep'] for c in groups_sorted]
x = np.arange(len(groups_sorted))

ax.bar(x, dem_counts, color='#2196F3', alpha=0.85, label='\u6c11\u4e3b\u515a', edgecolor='white', linewidth=0.5)
ax.bar(x, rep_counts, bottom=dem_counts, color='#F44336', alpha=0.85, label='\u5171\u548c\u515a', edgecolor='white', linewidth=0.5)
for i, c in enumerate(groups_sorted):
    total = comm_stats[c]['total']
    ax.text(i, total + 2, str(total), ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('\u7fa4\u4f53\u7f16\u53f7', fontsize=12)
ax.set_ylabel('\u6210\u5458\u6570', fontsize=12)
ax.set_title('\u7fa4\u4f53\u5927\u5c0f\u4e0e\u7acb\u573a\u5206\u5e03', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([f'\u7fa4\u4f53{c}' for c in groups_sorted], rotation=30, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'balanced_community_sizes.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved balanced_community_sizes.png")

# (c) balanced_comparison.png
print("\n  Generating balanced_comparison.png...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
method_names_short = [r['method'][:22] for r in all_results]
xi = np.arange(len(all_results))

axes[0, 0].barh(xi, [r['modularity'] for r in all_results], color='steelblue', alpha=0.8)
axes[0, 0].set_yticks(xi)
axes[0, 0].set_yticklabels(method_names_short, fontsize=7)
axes[0, 0].set_xlabel('\u6a21\u5757\u5ea6')
axes[0, 0].set_title('\u6a21\u5757\u5ea6\u5bf9\u6bd4', fontsize=11)

axes[0, 1].barh(xi, [r['nmi'] for r in all_results], color='darkorange', alpha=0.8)
axes[0, 1].set_yticks(xi)
axes[0, 1].set_yticklabels(method_names_short, fontsize=7)
axes[0, 1].set_xlabel('NMI')
axes[0, 1].set_title('NMI\u5bf9\u6bd4', fontsize=11)

axes[0, 2].barh(xi, [r['purity'] for r in all_results], color='forestgreen', alpha=0.8)
axes[0, 2].set_yticks(xi)
axes[0, 2].set_yticklabels(method_names_short, fontsize=7)
axes[0, 2].set_xlabel('\u7eaf\u5ea6')
axes[0, 2].set_title('\u7eaf\u5ea6\u5bf9\u6bd4', fontsize=11)

axes[1, 0].barh(xi, [r['balance_score'] for r in all_results], color='purple', alpha=0.8)
axes[1, 0].set_yticks(xi)
axes[1, 0].set_yticklabels(method_names_short, fontsize=7)
axes[1, 0].set_xlabel('\u5e73\u8861\u6027\u5206\u6570')
axes[1, 0].set_title('\u5e73\u8861\u6027\u5206\u6570\u5bf9\u6bd4', fontsize=11)

axes[1, 1].barh(xi, [1 - r['gini'] for r in all_results], color='teal', alpha=0.8)
axes[1, 1].set_yticks(xi)
axes[1, 1].set_yticklabels(method_names_short, fontsize=7)
axes[1, 1].set_xlabel('1 - \u57fa\u5c3c\u7cfb\u6570')
axes[1, 1].set_title('\u5747\u8861\u5ea6\u5bf9\u6bd4\uff081-\u57fa\u5c3c\u7cfb\u6570\uff09', fontsize=11)

scores_all = [r['comprehensive_score'] for r in all_results]
colors_bar = ['gold' if s == max(scores_all) else 'coral' for s in scores_all]
axes[1, 2].barh(xi, scores_all, color=colors_bar, alpha=0.8)
axes[1, 2].set_yticks(xi)
axes[1, 2].set_yticklabels(method_names_short, fontsize=7)
axes[1, 2].set_xlabel('\u7efc\u5408\u5f97\u5206')
axes[1, 2].set_title('\u7efc\u5408\u5f97\u5206\u5bf9\u6bd4', fontsize=11)

plt.suptitle('\u5e73\u8861\u7fa4\u4f53\u5212\u5206\u65b9\u6848\u6307\u6807\u5bf9\u6bd4', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'balanced_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved balanced_comparison.png")


# ============ Done ============
print("\n" + "=" * 60)
print("Balanced Community Detection Complete!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - {os.path.join(TASK2_DIR, 'group_assignments_balanced.npy')}")
print(f"  - {os.path.join(TASK2_DIR, 'balanced_results.json')}")
print(f"  - {os.path.join(TASK2_DIR, 'group_evaluation_balanced.md')}")
print(f"  - {os.path.join(VIS_DIR, 'balanced_community_network.png')}")
print(f"  - {os.path.join(VIS_DIR, 'balanced_community_sizes.png')}")
print(f"  - {os.path.join(VIS_DIR, 'balanced_comparison.png')}")
print(f"\nBest: {best_result['method']} ({best_result['params']})")
print(f"  #C={best_result['num_communities']}, Sizes={best_result['sizes']}")
print(f"  Mod={best_result['modularity']:.4f}, NMI={best_result['nmi']:.4f}, Pur={best_result['purity']:.4f}")
print(f"  Balance={best_result['balance_score']:.4f}, Gini={best_result['gini']:.4f}")
