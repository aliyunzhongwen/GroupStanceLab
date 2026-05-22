# -*- coding: utf-8 -*-
"""
Task 8: Community Detection and Group Quality Assessment
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
from networkx.algorithms.community import label_propagation_communities
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Chinese font
font_candidates = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_candidates
plt.rcParams['axes.unicode_minus'] = False

# Paths
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
VIS_DIR = os.path.join(TASK2_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# Load data
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
print(f"Directed graph: {G_directed.number_of_nodes()} nodes, {G_directed.number_of_edges()} edges")
print(f"Node labels: shape={node_labels.shape}, values={np.unique(node_labels)}")
print(f"Compact features: shape={features_compact.shape}")

# Valid nodes: 1-877
valid_nodes = sorted([n for n in G_undirected.nodes() if 1 <= n <= 877])
print(f"Valid nodes: {len(valid_nodes)}")

# Ground truth labels (0=Democrat, 1=Republican)
ground_truth = {n: int(node_labels[n]) for n in valid_nodes}
gt_array = np.array([ground_truth[n] for n in valid_nodes])
print(f"Democrat: {np.sum(gt_array == 0)}, Republican: {np.sum(gt_array == 1)}")


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


def compute_size_stats(partition_dict, valid_nodes):
    communities = {}
    for n in valid_nodes:
        comm = partition_dict[n]
        if comm not in communities:
            communities[comm] = 0
        communities[comm] += 1

    sizes = list(communities.values())
    return {
        'num_communities': len(sizes),
        'max_size': max(sizes),
        'min_size': min(sizes),
        'mean_size': float(np.mean(sizes)),
        'std_size': float(np.std(sizes)),
        'sizes': sorted(sizes, reverse=True)
    }


def evaluate_partition(G, partition_dict, valid_nodes, ground_truth, method_name):
    modularity = compute_modularity(G, partition_dict)
    nmi = compute_nmi(partition_dict, valid_nodes, ground_truth)
    purity = compute_purity(partition_dict, valid_nodes, ground_truth)
    size_stats = compute_size_stats(partition_dict, valid_nodes)

    result = {
        'method': method_name,
        'modularity': round(modularity, 4),
        'nmi': round(nmi, 4),
        'purity': round(purity, 4),
        'num_communities': size_stats['num_communities'],
        'max_size': size_stats['max_size'],
        'min_size': size_stats['min_size'],
        'mean_size': round(size_stats['mean_size'], 1),
        'std_size': round(size_stats['std_size'], 1),
        'sizes': size_stats['sizes']
    }

    print(f"  Modularity: {result['modularity']:.4f}")
    print(f"  NMI: {result['nmi']:.4f}")
    print(f"  Purity: {result['purity']:.4f}")
    print(f"  Communities: {result['num_communities']}")
    print(f"  Size: max={result['max_size']}, min={result['min_size']}, mean={result['mean_size']:.1f}, std={result['std_size']:.1f}")

    return result


# ============ 1. Community Detection Algorithms ============
all_results = []
all_partitions = {}

# --- (a) Louvain ---
print("\n" + "=" * 60)
print("1(a) Louvain Algorithm")
print("=" * 60)

louvain_results = []
for resolution in [0.5, 0.8, 1.0, 1.2, 1.5]:
    print(f"\n  Resolution = {resolution}")
    partition = community_louvain.best_partition(G_undirected, resolution=resolution, random_state=2023)
    result = evaluate_partition(G_undirected, partition, valid_nodes, ground_truth, f'Louvain(res={resolution})')
    result['params'] = f'resolution={resolution}'
    louvain_results.append(result)
    all_partitions[f'louvain_res{resolution}'] = partition

# Select best Louvain
best_louvain_idx = 0
best_score = -1
for i, r in enumerate(louvain_results):
    nc = r['num_communities']
    nc_bonus = 1.0 if 4 <= nc <= 10 else 0.5
    score = r['modularity'] * 0.4 + r['nmi'] * 0.3 + r['purity'] * 0.2 + nc_bonus * 0.1
    if score > best_score:
        best_score = score
        best_louvain_idx = i

best_louvain = louvain_results[best_louvain_idx]
print(f"\n  Best Louvain: {best_louvain['method']}, score={best_score:.4f}")
all_results.extend(louvain_results)

# --- (b) Label Propagation ---
print("\n" + "=" * 60)
print("1(b) Label Propagation")
print("=" * 60)

best_lp_mod = -1
best_lp_partition = None
for trial in range(5):
    communities_lp = list(label_propagation_communities(G_undirected))
    partition_lp = {}
    for i, comm in enumerate(communities_lp):
        for node in comm:
            partition_lp[node] = i

    for n in valid_nodes:
        if n not in partition_lp:
            partition_lp[n] = -1

    mod = compute_modularity(G_undirected, partition_lp)
    if mod > best_lp_mod:
        best_lp_mod = mod
        best_lp_partition = partition_lp.copy()

print(f"\n  Best Label Propagation result:")
result_lp = evaluate_partition(G_undirected, best_lp_partition, valid_nodes, ground_truth, 'Label Propagation')
result_lp['params'] = 'default (best of 5 trials)'
all_results.append(result_lp)
all_partitions['label_propagation'] = best_lp_partition

# --- (c) Spectral Clustering ---
print("\n" + "=" * 60)
print("1(c) Spectral Clustering")
print("=" * 60)

node_list = sorted(valid_nodes)
node_to_idx = {n: i for i, n in enumerate(node_list)}
n_nodes = len(node_list)

# Adjacency matrix
adj_matrix = np.zeros((n_nodes, n_nodes))
for u, v, data in G_undirected.edges(data=True):
    if u in node_to_idx and v in node_to_idx:
        w = data.get('weight', 1.0)
        adj_matrix[node_to_idx[u], node_to_idx[v]] = w
        adj_matrix[node_to_idx[v], node_to_idx[u]] = w

print(f"  Adjacency matrix: shape={adj_matrix.shape}, nonzero={np.count_nonzero(adj_matrix)}")

spectral_results = []
for k in [2, 3, 4, 5, 6, 8, 10]:
    print(f"\n  K = {k}")
    try:
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=2023, n_init=10)
        labels_sc = sc.fit_predict(adj_matrix)
        partition_sc = {node_list[i]: int(labels_sc[i]) for i in range(n_nodes)}
        result_sc = evaluate_partition(G_undirected, partition_sc, valid_nodes, ground_truth, f'Spectral(K={k})')
        result_sc['params'] = f'K={k}'
        spectral_results.append(result_sc)
        all_partitions[f'spectral_k{k}'] = partition_sc
    except Exception as e:
        print(f"  Spectral K={k} failed: {e}")

all_results.extend(spectral_results)

# --- (d) K-means ---
print("\n" + "=" * 60)
print("1(d) K-means (feature-based)")
print("=" * 60)

features_valid = features_compact[1:878].copy()
print(f"  Valid features: shape={features_valid.shape}")

nan_mask = np.isnan(features_valid)
if nan_mask.any():
    print(f"  Found NaN values, filling with column means")
    col_means = np.nanmean(features_valid, axis=0)
    for col in range(features_valid.shape[1]):
        features_valid[nan_mask[:, col], col] = col_means[col]
    # If still NaN (entire column is NaN)
    features_valid = np.nan_to_num(features_valid, nan=0.0)

kmeans_results = []
for k in [2, 3, 4, 5, 6, 8, 10]:
    print(f"\n  K = {k}")
    km = KMeans(n_clusters=k, random_state=2023, n_init=10)
    labels_km = km.fit_predict(features_valid)
    partition_km = {node_list[i]: int(labels_km[i]) for i in range(n_nodes)}
    result_km = evaluate_partition(G_undirected, partition_km, valid_nodes, ground_truth, f'KMeans(K={k})')
    result_km['params'] = f'K={k}'
    kmeans_results.append(result_km)
    all_partitions[f'kmeans_k{k}'] = partition_km

all_results.extend(kmeans_results)

# --- (e) Hybrid: Graph + Feature Similarity ---
print("\n" + "=" * 60)
print("1(e) Hybrid: Graph Structure + Feature Similarity")
print("=" * 60)

adj_norm = adj_matrix / (adj_matrix.max() + 1e-8)
cos_sim = cosine_similarity(features_valid)
cos_sim = np.maximum(cos_sim, 0)

hybrid_results = []
for alpha in [0.3, 0.5, 0.7]:
    for k in [2, 3, 4, 5, 6, 8]:
        print(f"\n  alpha={alpha}, K={k}")
        fused_matrix = alpha * adj_norm + (1 - alpha) * cos_sim

        try:
            sc_hybrid = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=2023, n_init=10)
            labels_hybrid = sc_hybrid.fit_predict(fused_matrix)
            partition_hybrid = {node_list[i]: int(labels_hybrid[i]) for i in range(n_nodes)}
            result_hybrid = evaluate_partition(G_undirected, partition_hybrid, valid_nodes, ground_truth,
                                              f'Hybrid(a={alpha},K={k})')
            result_hybrid['params'] = f'alpha={alpha}, K={k}'
            hybrid_results.append(result_hybrid)
            all_partitions[f'hybrid_a{alpha}_k{k}'] = partition_hybrid
        except Exception as e:
            print(f"  Hybrid alpha={alpha}, K={k} failed: {e}")

all_results.extend(hybrid_results)


# ============ 2. Comprehensive Comparison ============
print("\n" + "=" * 60)
print("Comprehensive Comparison and Best Method Selection")
print("=" * 60)

print(f"\n{'Method':<30} {'Params':<20} {'#Comm':>6} {'Mod':>8} {'NMI':>8} {'Purity':>8} {'Max':>6} {'Min':>6}")
print("-" * 100)
for r in all_results:
    print(f"{r['method']:<30} {r['params']:<20} {r['num_communities']:>6} {r['modularity']:>8.4f} {r['nmi']:>8.4f} {r['purity']:>8.4f} {r['max_size']:>6} {r['min_size']:>6}")

# Select best method
print("\n\nSelecting best method...")
best_result = None
best_overall_score = -1

for r in all_results:
    mod_score = r['modularity']
    nmi_score = r['nmi']
    nc = r['num_communities']
    if 4 <= nc <= 10:
        nc_score = 1.0
    elif 2 <= nc <= 3:
        nc_score = 0.7
    elif 11 <= nc <= 15:
        nc_score = 0.6
    else:
        nc_score = 0.3

    purity_score = r['purity']
    overall = mod_score * 0.35 + nmi_score * 0.30 + nc_score * 0.15 + purity_score * 0.20
    r['overall_score'] = round(overall, 4)

    if overall > best_overall_score:
        best_overall_score = overall
        best_result = r

print(f"\nBest method: {best_result['method']}")
print(f"  Overall score: {best_result['overall_score']:.4f}")
print(f"  Modularity: {best_result['modularity']:.4f}")
print(f"  NMI: {best_result['nmi']:.4f}")
print(f"  Purity: {best_result['purity']:.4f}")
print(f"  Communities: {best_result['num_communities']}")
print(f"  Sizes: {best_result['sizes']}")

# Find the partition key for best result
best_key = None
for key, part in all_partitions.items():
    stats = compute_size_stats(part, valid_nodes)
    if stats['num_communities'] == best_result['num_communities']:
        mod = compute_modularity(G_undirected, part)
        if abs(mod - best_result['modularity']) < 0.001:
            best_key = key
            break

print(f"  Partition key: {best_key}")
best_partition = all_partitions[best_key]


# ============ 3. Save Results ============
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

# (1) group_assignments.npy
group_assignments = np.full(878, -1, dtype=int)
for n in valid_nodes:
    group_assignments[n] = best_partition[n]
np.save(os.path.join(TASK2_DIR, 'group_assignments.npy'), group_assignments)
print(f"  Saved group_assignments.npy: shape={group_assignments.shape}")

# (2) all_results.json
with open(os.path.join(TASK2_DIR, 'all_results.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'best_method': best_result['method'],
        'best_params': best_result['params'],
        'best_overall_score': best_result['overall_score'],
        'results': all_results
    }, f, ensure_ascii=False, indent=2)
print("  Saved all_results.json")

# (3) group_evaluation.md
md_lines = []
md_lines.append("# Community Detection and Group Quality Assessment Report\n")
md_lines.append("## 1. Overview\n")
md_lines.append(f"- Dataset: Twitter Political Interaction Network")
md_lines.append(f"- Nodes: {len(valid_nodes)}")
md_lines.append(f"- Edges: {G_undirected.number_of_edges()}")
md_lines.append(f"- Party distribution: Democrat={int(np.sum(gt_array==0))}, Republican={int(np.sum(gt_array==1))}\n")
md_lines.append("## 2. Algorithm Results Comparison\n")
md_lines.append("| Method | Params | #Communities | Modularity | NMI | Purity | Max Size | Min Size |")
md_lines.append("|--------|--------|:------------:|:----------:|:---:|:------:|:--------:|:--------:|")
for r in all_results:
    md_lines.append(f"| {r['method']} | {r['params']} | {r['num_communities']} | {r['modularity']:.4f} | {r['nmi']:.4f} | {r['purity']:.4f} | {r['max_size']} | {r['min_size']} |")
md_lines.append(f"\n## 3. Best Method\n")
md_lines.append(f"**Method**: {best_result['method']}\n")
md_lines.append(f"**Params**: {best_result['params']}\n")
md_lines.append(f"**Metrics**:")
md_lines.append(f"- Modularity: {best_result['modularity']:.4f}")
md_lines.append(f"- NMI: {best_result['nmi']:.4f}")
md_lines.append(f"- Purity: {best_result['purity']:.4f}")
md_lines.append(f"- Communities: {best_result['num_communities']}")
md_lines.append(f"- Overall score: {best_result['overall_score']:.4f}")
md_lines.append(f"- Size distribution: {best_result['sizes']}\n")
md_lines.append("## 4. Selection Criteria\n")
md_lines.append("Score = Modularity*0.35 + NMI*0.30 + Community_count_rationality*0.15 + Purity*0.20\n")
md_lines.append("Priority:")
md_lines.append("1. High modularity (reasonable community structure)")
md_lines.append("2. High NMI (correspondence with party labels)")
md_lines.append("3. Reasonable community count (4-10 preferred)")
md_lines.append("4. High purity (but not requiring 100%)\n")
md_lines.append("## 5. Conclusions\n")
md_lines.append("Key findings:\n")
md_lines.append("1. **Louvain** excels in modularity, auto-determines community count")
md_lines.append("2. **Spectral Clustering** offers flexible control over community count")
md_lines.append("3. **K-means** reflects node distribution in feature space")
md_lines.append("4. **Hybrid method** combines graph structure and node features")
md_lines.append("5. **Label Propagation** is fast but unstable\n")
md_lines.append(f"Final selection: **{best_result['method']}**")

with open(os.path.join(TASK2_DIR, 'group_evaluation.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
print("  Saved group_evaluation.md")


# ============ 4. Visualizations ============
print("\n" + "=" * 60)
print("Generating visualizations...")
print("=" * 60)

# --- (a) community_network.png ---
print("\n  Generating community_network.png...")

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

community_labels = [best_partition[n] for n in valid_nodes]
unique_communities = sorted(set(community_labels))
n_comms = len(unique_communities)
cmap = plt.cm.get_cmap('tab10' if n_comms <= 10 else 'tab20')
comm_colors = {c: cmap(i / max(n_comms - 1, 1)) for i, c in enumerate(unique_communities)}

G_vis = G_undirected.subgraph(valid_nodes).copy()

print("  Computing layout (spring_layout)...")
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
for i, c in enumerate(unique_communities):
    count = community_labels.count(c)
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=comm_colors[c], markersize=10,
                                  label=f'Group {c} (n={count})'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              markeredgecolor='blue', markersize=10, label='Democrat'))
legend_elements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                              markeredgecolor='red', markersize=10, label='Republican'))

ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)
ax.set_title(u'\u6700\u4f18\u7fa4\u4f53\u5212\u5206\u7ed3\u679c - \u7f51\u7edc\u7740\u8272\u56fe\n' + f'({best_result["method"]})', fontsize=14)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'community_network.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved community_network.png")

# --- (b) community_comparison.png ---
print("\n  Generating community_comparison.png...")

representative_results = []
representative_results.append(best_louvain)
representative_results.append(result_lp)
if spectral_results:
    best_spectral = max(spectral_results, key=lambda x: x.get('overall_score', 0))
    representative_results.append(best_spectral)
if kmeans_results:
    best_kmeans = max(kmeans_results, key=lambda x: x.get('overall_score', 0))
    representative_results.append(best_kmeans)
if hybrid_results:
    best_hybrid_r = max(hybrid_results, key=lambda x: x.get('overall_score', 0))
    representative_results.append(best_hybrid_r)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

method_names = [r['method'] for r in representative_results]
short_names = [name[:18] if len(name) > 18 else name for name in method_names]
x = np.arange(len(representative_results))

# Modularity
mods = [r['modularity'] for r in representative_results]
bars1 = axes[0].bar(x, mods, color='steelblue', alpha=0.8)
axes[0].set_title(u'\u6a21\u5757\u5ea6\u5bf9\u6bd4', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
axes[0].set_ylabel(u'\u6a21\u5757\u5ea6')
for bar, val in zip(bars1, mods):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# NMI
nmis = [r['nmi'] for r in representative_results]
bars2 = axes[1].bar(x, nmis, color='darkorange', alpha=0.8)
axes[1].set_title(u'NMI\u5bf9\u6bd4\uff08\u53c2\u8003\u653f\u515a\u6807\u7b7e\uff09', fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
axes[1].set_ylabel('NMI')
for bar, val in zip(bars2, nmis):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Purity
purities = [r['purity'] for r in representative_results]
bars3 = axes[2].bar(x, purities, color='forestgreen', alpha=0.8)
axes[2].set_title(u'\u7fa4\u4f53\u7acb\u573a\u7eaf\u5ea6\u5bf9\u6bd4', fontsize=12)
axes[2].set_xticks(x)
axes[2].set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
axes[2].set_ylabel(u'\u7eaf\u5ea6')
for bar, val in zip(bars3, purities):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle(u'\u793e\u533a\u53d1\u73b0\u7b97\u6cd5\u8bc4\u4f30\u6307\u6807\u5bf9\u6bd4', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'community_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved community_comparison.png")

# --- (c) community_size_distribution.png ---
print("\n  Generating community_size_distribution.png...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sizes = best_result['sizes']
labels_pie = [f'Group {unique_communities[i]}\n(n={s})' for i, s in enumerate(sizes)]
colors_pie = [comm_colors[unique_communities[i]] for i in range(len(sizes))]

ax1.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 9})
ax1.set_title(u'\u7fa4\u4f53\u5927\u5c0f\u5206\u5e03\uff08\u997c\u56fe\uff09', fontsize=12)

ax2.bar(range(len(sizes)), sizes, color=colors_pie, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel(u'\u7fa4\u4f53\u7f16\u53f7', fontsize=11)
ax2.set_ylabel(u'\u8282\u70b9\u6570\u91cf', fontsize=11)
ax2.set_title(u'\u7fa4\u4f53\u5927\u5c0f\u5206\u5e03\uff08\u67f1\u72b6\u56fe\uff09', fontsize=12)
ax2.set_xticks(range(len(sizes)))
ax2.set_xticklabels([f'Group {unique_communities[i]}' for i in range(len(sizes))], rotation=30, ha='right')
for i, s in enumerate(sizes):
    ax2.text(i, s + 2, str(s), ha='center', va='bottom', fontsize=9)

plt.suptitle(u'\u7fa4\u4f53\u5927\u5c0f\u5206\u5e03', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'community_size_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved community_size_distribution.png")


# ============ Done ============
print("\n" + "=" * 60)
print("Task 8 Complete!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - {os.path.join(TASK2_DIR, 'group_assignments.npy')}")
print(f"  - {os.path.join(TASK2_DIR, 'all_results.json')}")
print(f"  - {os.path.join(TASK2_DIR, 'group_evaluation.md')}")
print(f"  - {os.path.join(VIS_DIR, 'community_network.png')}")
print(f"  - {os.path.join(VIS_DIR, 'community_comparison.png')}")
print(f"  - {os.path.join(VIS_DIR, 'community_size_distribution.png')}")
print(f"\nBest method: {best_result['method']} ({best_result['params']})")
print(f"  Communities={best_result['num_communities']}, Mod={best_result['modularity']:.4f}, NMI={best_result['nmi']:.4f}, Purity={best_result['purity']:.4f}")
