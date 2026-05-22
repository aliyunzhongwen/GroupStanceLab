# -*- coding: utf-8 -*-
"""
Optimized group partition visualization.
Generates: balanced_community_network.png, balanced_community_sizes.png,
           balanced_comparison.png, community_radar.png
"""

import os
import json
import pickle
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ============ Font Setup ============
def find_chinese_font():
    preferred = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC',
                 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei',
                 'Microsoft YaHei', 'DejaVu Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in preferred:
        if font in available_fonts:
            return font
    for f in available_fonts:
        if 'CJK' in f or 'Hei' in f or 'Song' in f:
            return f
    return 'DejaVu Sans'

chinese_font = find_chinese_font()
print(f"Using font: {chinese_font}")
plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ Paths ============
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
VIS_DIR = os.path.join(TASK2_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# ============ Load Data ============
print("=" * 60)
print("Loading data...")
print("=" * 60)

with open(os.path.join(TASK2_DIR, 'combined_graph_undirected.pkl'), 'rb') as f:
    G_undirected = pickle.load(f)

group_assignments = np.load(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'))
node_labels = np.load(os.path.join(BASE_DIR, 'processed', 'ml_twitter_node_labels.npy'))
structural_features = np.load(os.path.join(BASE_DIR, 'task1', 'features', 'structural_features.npy'))

print(f"Graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
print(f"Group assignments: shape={group_assignments.shape}")
print(f"Node labels: shape={node_labels.shape}")
print(f"Structural features: shape={structural_features.shape}")

valid_nodes = sorted([n for n in range(1, 878) if group_assignments[n] >= 0])
n_valid = len(valid_nodes)
print(f"Valid nodes: {n_valid}")

labels = group_assignments
unique_groups = sorted(set(labels[n] for n in valid_nodes))
n_groups = len(unique_groups)
print(f"Number of groups: {n_groups}, Groups: {unique_groups}")

with open(os.path.join(TASK2_DIR, 'balanced_results.json'), 'r', encoding='utf-8') as f:
    balanced_results = json.load(f)
all_results = balanced_results['results']

# ============ Color Scheme ============
GROUP_COLORS = [
    '#E24A33', '#348ABD', '#988ED5', '#77B05D',
    '#FBC15E', '#8EBA42', '#FFB5B8', '#56B4E9',
    '#009E73', '#CC79A7',
]
group_color_map = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(unique_groups)}
group_members = {g: [n for n in valid_nodes if labels[n] == g] for g in unique_groups}

# ============ Fig 1: balanced_community_network.png ============
print("\n" + "=" * 60)
print("1. Generating balanced_community_network.png (optimized)...")
print("=" * 60)

G_vis = G_undirected.subgraph(valid_nodes).copy()

# Build super graph for group centers
super_G = nx.Graph()
for g in unique_groups:
    super_G.add_node(g)

for i, g1 in enumerate(unique_groups):
    for g2 in unique_groups[i+1:]:
        weight = 0
        members1 = set(group_members[g1])
        members2 = set(group_members[g2])
        for u, v in G_vis.edges():
            if (u in members1 and v in members2) or (u in members2 and v in members1):
                weight += 1
        if weight > 0:
            super_G.add_edge(g1, g2, weight=weight)

# Layer 1: group center layout
center_pos = nx.spring_layout(super_G, k=3.5, seed=2023, iterations=100)

# Node degree for sizing
degrees = dict(G_vis.degree())
degree_values = np.array([degrees.get(n, 0) for n in valid_nodes])
if degree_values.max() > degree_values.min():
    node_sizes = 30 + (degree_values - degree_values.min()) / (degree_values.max() - degree_values.min()) * 270
else:
    node_sizes = np.full(len(valid_nodes), 80)
node_size_map = {n: node_sizes[i] for i, n in enumerate(valid_nodes)}

# Layer 2: arrange nodes around group centers
pos = {}
max_group_size = max(len(m) for m in group_members.values())
for g in unique_groups:
    members = group_members[g]
    cx, cy = center_pos[g]
    sub_G = G_vis.subgraph(members)
    n_members = len(members)
    scale = 0.15 + 0.10 * (n_members / max_group_size)
    if sub_G.number_of_edges() > 0:
        sub_pos = nx.spring_layout(sub_G, center=(cx, cy), scale=scale, seed=2023, iterations=50)
    else:
        sub_pos = nx.circular_layout(sub_G, center=(cx, cy), scale=scale)
    pos.update(sub_pos)

# Draw
fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='white')
ax.set_facecolor('white')

intra_edges = []
inter_edges = []
for u, v in G_vis.edges():
    if labels[u] == labels[v]:
        intra_edges.append((u, v))
    else:
        inter_edges.append((u, v))

print(f"  Intra-group edges: {len(intra_edges)}, Inter-group edges: {len(inter_edges)}")

# Draw intra-group edges (very faint)
nx.draw_networkx_edges(G_vis, pos, edgelist=intra_edges, alpha=0.03,
                       width=0.3, edge_color='#CCCCCC', ax=ax)

# Draw top inter-group edges (dashed)
if inter_edges:
    inter_weights = [(u, v, G_vis[u][v].get('weight', 1)) for u, v in inter_edges]
    inter_weights.sort(key=lambda x: x[2], reverse=True)
    top_inter = [(u, v) for u, v, w in inter_weights[:min(150, len(inter_weights))]]
    nx.draw_networkx_edges(G_vis, pos, edgelist=top_inter, alpha=0.08,
                           width=0.4, edge_color='#999999', style='dashed', ax=ax)

# Draw nodes by group
for g in unique_groups:
    members = group_members[g]
    dem_members = [n for n in members if node_labels[n] == 0]
    rep_members = [n for n in members if node_labels[n] == 1]
    
    if dem_members:
        sizes_dem = [node_size_map[n] for n in dem_members]
        nx.draw_networkx_nodes(G_vis, pos, nodelist=dem_members,
                               node_color=[group_color_map[g]] * len(dem_members),
                               node_shape='o', node_size=sizes_dem, alpha=0.85, ax=ax,
                               edgecolors='#1565C0', linewidths=1.5)
    
    if rep_members:
        sizes_rep = [node_size_map[n] for n in rep_members]
        nx.draw_networkx_nodes(G_vis, pos, nodelist=rep_members,
                               node_color=[group_color_map[g]] * len(rep_members),
                               node_shape='o', node_size=sizes_rep, alpha=0.85, ax=ax,
                               edgecolors='#C62828', linewidths=1.5)

# Legends
legend1_elements = []
for g in unique_groups:
    count = len(group_members[g])
    legend1_elements.append(Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=group_color_map[g], markersize=12,
                                   markeredgecolor='gray', markeredgewidth=0.5,
                                   label=f'\u7fa4\u4f53{g} (n={count})'))

legend2_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA',
           markeredgecolor='#1565C0', markeredgewidth=2, markersize=12,
           label='\u6c11\u4e3b\u515a'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA',
           markeredgecolor='#C62828', markeredgewidth=2, markersize=12,
           label='\u5171\u548c\u515a'),
]

leg1 = ax.legend(handles=legend1_elements, loc='upper left', fontsize=10,
                 title='\u7fa4\u4f53', title_fontsize=11, framealpha=0.9, edgecolor='#CCCCCC')
ax.add_artist(leg1)
ax.legend(handles=legend2_elements, loc='lower left', fontsize=10,
          title='\u653f\u515a\uff08\u8fb9\u6846\u989c\u8272\uff09', title_fontsize=11,
          framealpha=0.9, edgecolor='#CCCCCC')

ax.set_title('\u7fa4\u4f53\u5212\u5206\u7f51\u7edc\u7ed3\u6784\u56fe\uff087\u7fa4\u4f53\uff0cSpectral+\u5e73\u8861\u4f18\u5316\uff09',
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'balanced_community_network.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Done: balanced_community_network.png")


# ============ Fig 2: balanced_community_sizes.png ============
print("\n" + "=" * 60)
print("2. Generating balanced_community_sizes.png (optimized)...")
print("=" * 60)

comm_stats = {}
for n in valid_nodes:
    g = labels[n]
    if g not in comm_stats:
        comm_stats[g] = {'dem': 0, 'rep': 0, 'total': 0}
    comm_stats[g]['total'] += 1
    if node_labels[n] == 0:
        comm_stats[g]['dem'] += 1
    else:
        comm_stats[g]['rep'] += 1

fig, ax = plt.subplots(1, 1, figsize=(12, 7), facecolor='white')
ax.set_facecolor('white')

groups_sorted = sorted(comm_stats.keys())
dem_counts = [comm_stats[g]['dem'] for g in groups_sorted]
rep_counts = [comm_stats[g]['rep'] for g in groups_sorted]
totals = [comm_stats[g]['total'] for g in groups_sorted]
x = np.arange(len(groups_sorted))
bar_width = 0.6

color_dem = '#4A90D9'
color_rep = '#E74C3C'

bars_dem = ax.bar(x, dem_counts, bar_width, color=color_dem, edgecolor='black',
                  linewidth=0.8, label='\u6c11\u4e3b\u515a', zorder=3)
bars_rep = ax.bar(x, rep_counts, bar_width, bottom=dem_counts, color=color_rep,
                  edgecolor='black', linewidth=0.8, label='\u5171\u548c\u515a', zorder=3)

for i, g in enumerate(groups_sorted):
    total = totals[i]
    dem = dem_counts[i]
    rep = rep_counts[i]
    
    if dem > 15:
        dem_pct = dem / total * 100
        ax.text(i, dem / 2, f'{dem}\n({dem_pct:.0f}%)', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    
    if rep > 15:
        rep_pct = rep / total * 100
        ax.text(i, dem + rep / 2, f'{rep}\n({rep_pct:.0f}%)', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    
    ax.text(i, total + 3, f'{total}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#333333')

ax.set_xlabel('\u7fa4\u4f53\u7f16\u53f7', fontsize=13)
ax.set_ylabel('\u6210\u5458\u6570', fontsize=13)
ax.set_title('\u7fa4\u4f53\u89c4\u6a21\u4e0e\u7acb\u573a\u6784\u6210\u5206\u5e03', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'\u7fa4\u4f53{g}' for g in groups_sorted], fontsize=11)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax.set_ylim(0, max(totals) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'balanced_community_sizes.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Done: balanced_community_sizes.png")


# ============ Fig 3: balanced_comparison.png ============
print("\n" + "=" * 60)
print("3. Generating balanced_comparison.png (optimized, Top-8)...")
print("=" * 60)

results_sorted = sorted(all_results, key=lambda r: r['comprehensive_score'], reverse=True)
top_results = results_sorted[:8]
top_results = list(reversed(top_results))

method_names = [r['method'] for r in top_results]
xi = np.arange(len(top_results))

metric_colors = ['#348ABD', '#E24A33', '#77B05D', '#988ED5', '#56B4E9', '#FBC15E']

fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='white')

metrics = [
    ('modularity', '\u6a21\u5757\u5ea6', metric_colors[0]),
    ('nmi', 'NMI', metric_colors[1]),
    ('purity', '\u7eaf\u5ea6', metric_colors[2]),
    ('balance_score', '\u5e73\u8861\u6027\u5206\u6570', metric_colors[3]),
    (None, '\u5747\u8861\u5ea6\uff081-\u57fa\u5c3c\u7cfb\u6570\uff09', metric_colors[4]),
    ('comprehensive_score', '\u7efc\u5408\u5f97\u5206', metric_colors[5]),
]

for idx, (metric_key, title, color) in enumerate(metrics):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    ax.set_facecolor('#FAFAFA')
    
    if metric_key is None:
        values = [1 - r['gini'] for r in top_results]
    else:
        values = [r[metric_key] for r in top_results]
    
    max_val = max(values)
    colors_bar = []
    for v in values:
        if abs(v - max_val) < 1e-6:
            colors_bar.append('#FFD700')
        else:
            colors_bar.append(color)
    
    bars = ax.barh(xi, values, color=colors_bar, alpha=0.85, edgecolor='white',
                   linewidth=0.5, height=0.7)
    
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8, color='#333333')
    
    ax.set_yticks(xi)
    ax.set_yticklabels(method_names, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('\u5e73\u8861\u7fa4\u4f53\u5212\u5206\u65b9\u6848\u6307\u6807\u5bf9\u6bd4\uff08Top-8\u65b9\u6cd5\uff09',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'balanced_comparison.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Done: balanced_comparison.png")


# ============ Fig 4: community_radar.png ============
print("\n" + "=" * 60)
print("4. Generating community_radar.png...")
print("=" * 60)

# structural_features (878,8): in_degree, out_degree, total_degree, degree_centrality,
#                               pagerank, betweenness, clustering, local_reaching

radar_data = {}
for g in unique_groups:
    members = group_members[g]
    size_norm = len(members) / max(len(group_members[gg]) for gg in unique_groups)
    dem_ratio = sum(1 for n in members if node_labels[n] == 0) / len(members)
    rep_ratio = 1 - dem_ratio
    avg_degree = np.mean([structural_features[n, 2] for n in members])
    avg_pagerank = np.mean([structural_features[n, 4] for n in members])
    radar_data[g] = {
        'size_norm': size_norm,
        'dem_ratio': dem_ratio,
        'rep_ratio': rep_ratio,
        'avg_degree': avg_degree,
        'avg_pagerank': avg_pagerank,
    }

all_degrees = [radar_data[g]['avg_degree'] for g in unique_groups]
all_pageranks = [radar_data[g]['avg_pagerank'] for g in unique_groups]
max_degree = max(all_degrees) if max(all_degrees) > 0 else 1
max_pagerank = max(all_pageranks) if max(all_pageranks) > 0 else 1

for g in unique_groups:
    radar_data[g]['avg_degree_norm'] = radar_data[g]['avg_degree'] / max_degree
    radar_data[g]['avg_pagerank_norm'] = radar_data[g]['avg_pagerank'] / max_pagerank

categories = ['\u7fa4\u4f53\u5927\u5c0f', '\u6c11\u4e3b\u515a\u6bd4\u4f8b',
              '\u5171\u548c\u515a\u6bd4\u4f8b', '\u5e73\u5747\u5ea6\u6570',
              '\u5e73\u5747PageRank']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True), facecolor='white')

for g in unique_groups:
    values = [
        radar_data[g]['size_norm'],
        radar_data[g]['dem_ratio'],
        radar_data[g]['rep_ratio'],
        radar_data[g]['avg_degree_norm'],
        radar_data[g]['avg_pagerank_norm'],
    ]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=group_color_map[g],
            label=f'\u7fa4\u4f53{g} (n={len(group_members[g])})', markersize=5)
    ax.fill(angles, values, alpha=0.1, color=group_color_map[g])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10,
          framealpha=0.9, edgecolor='#CCCCCC')
ax.set_title('\u5404\u7fa4\u4f53\u7279\u5f81\u96f7\u8fbe\u56fe', fontsize=16, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'community_radar.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("  Done: community_radar.png")

# ============ Done ============
print("\n" + "=" * 60)
print("All optimized visualizations generated!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  1. {os.path.join(VIS_DIR, 'balanced_community_network.png')}")
print(f"  2. {os.path.join(VIS_DIR, 'balanced_community_sizes.png')}")
print(f"  3. {os.path.join(VIS_DIR, 'balanced_comparison.png')}")
print(f"  4. {os.path.join(VIS_DIR, 'community_radar.png')}")
