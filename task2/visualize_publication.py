# -*- coding: utf-8 -*-
"""
Publication-quality visualization for group stance analysis.
Style: Nature/Science computational social science papers.
Generates 5 figures for the community detection results.
"""

import os
import json
import pickle
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Publication Style Configuration
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Find CJK font
for fname in fm.findSystemFonts():
    try:
        name = fm.FontProperties(fname=fname).get_name()
        if 'Noto Sans CJK' in name or 'WenQuanYi' in name:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            print(f"Using font: {name}")
            break
    except:
        continue
plt.rcParams['axes.unicode_minus'] = False

# Nature Reviews color palette
COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
DEM_COLOR = '#3C5488'   # Deep blue for Democrats
REP_COLOR = '#E64B35'   # Nature red for Republicans

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Data Loading
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
VIS_DIR = os.path.join(TASK2_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

print("Loading data...")
with open(os.path.join(TASK2_DIR, 'combined_graph_undirected.pkl'), 'rb') as f:
    G_undirected = pickle.load(f)

group_assignments = np.load(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'))
node_labels = np.load(os.path.join(BASE_DIR, 'processed', 'ml_twitter_node_labels.npy'))
structural_features = np.load(os.path.join(BASE_DIR, 'task1', 'features', 'structural_features.npy'))

with open(os.path.join(TASK2_DIR, 'balanced_results.json'), 'r', encoding='utf-8') as f:
    balanced_results = json.load(f)

# Valid nodes (index 1-877)
valid_nodes = sorted([n for n in range(1, 878) if group_assignments[n] >= 0])
unique_groups = sorted(set(group_assignments[n] for n in valid_nodes))
n_groups = len(unique_groups)
group_members = {g: [n for n in valid_nodes if group_assignments[n] == g] for g in unique_groups}
group_color_map = {g: COLORS[i % len(COLORS)] for i, g in enumerate(unique_groups)}

print(f"Graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
print(f"Valid nodes: {len(valid_nodes)}, Groups: {n_groups} ({unique_groups})")

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Figure 1: Publication-quality Network Visualization
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
print("\n[1/5] Generating balanced_community_network.png ...")

G_vis = G_undirected.subgraph(valid_nodes).copy()

# Community-aware layout: compute group centers first
super_G = nx.Graph()
for g in unique_groups:
    super_G.add_node(g)
for i, g1 in enumerate(unique_groups):
    for g2 in unique_groups[i+1:]:
        members1 = set(group_members[g1])
        members2 = set(group_members[g2])
        weight = sum(1 for u, v in G_vis.edges() if (u in members1 and v in members2) or (u in members2 and v in members1))
        if weight > 0:
            super_G.add_edge(g1, g2, weight=weight)

# Group center positions with high separation
center_pos = nx.spring_layout(super_G, k=4.0, seed=42, iterations=200)

# Node positions: spring layout within each community cluster
pos = {}
max_group_size = max(len(m) for m in group_members.values())
for g in unique_groups:
    members = group_members[g]
    cx, cy = center_pos[g]
    sub_G = G_vis.subgraph(members)
    scale = 0.12 + 0.08 * (len(members) / max_group_size)
    if sub_G.number_of_edges() > 0:
        sub_pos = nx.spring_layout(sub_G, center=(cx, cy), scale=scale, seed=42, iterations=100)
    else:
        sub_pos = nx.circular_layout(sub_G, center=(cx, cy), scale=scale)
    pos.update(sub_pos)

# Degree-based node sizing (log scale)
degrees = dict(G_vis.degree())
degree_arr = np.array([degrees.get(n, 0) for n in valid_nodes])
log_degrees = np.log1p(degree_arr)
if log_degrees.max() > log_degrees.min():
    node_sizes = 15 + (log_degrees - log_degrees.min()) / (log_degrees.max() - log_degrees.min()) * 135
else:
    node_sizes = np.full(len(valid_nodes), 50)
node_size_map = {n: node_sizes[i] for i, n in enumerate(valid_nodes)}

# Filter edges: only weight > 2
edges_to_draw = [(u, v) for u, v in G_vis.edges() if G_vis[u][v].get('weight', 1) > 2]

fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='white')
ax.set_facecolor('white')

# Draw edges (very thin, low opacity)
if edges_to_draw:
    nx.draw_networkx_edges(G_vis, pos, edgelist=edges_to_draw,
                           alpha=0.15, width=0.1, edge_color='#cccccc', ax=ax)

# Draw nodes by group
for g in unique_groups:
    members = group_members[g]
    sizes = [node_size_map[n] for n in members]
    nx.draw_networkx_nodes(G_vis, pos, nodelist=members,
                           node_color=[group_color_map[g]] * len(members),
                           node_size=sizes, alpha=0.85, ax=ax,
                           edgecolors='#333333', linewidths=0.3)

# Legend
legend_elements = []
for g in unique_groups:
    count = len(group_members[g])
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=group_color_map[g], markersize=8,
                                  markeredgecolor='#333333', markeredgewidth=0.3,
                                  label=f'Community {g} (n={count})'))

ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          framealpha=0.95, edgecolor='#dddddd', fancybox=False,
          borderpad=0.8, labelspacing=0.6)

# Sub-figure label
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top', ha='left')

ax.axis('off')
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(VIS_DIR, 'balanced_community_network.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("    Done.")

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Figure 2: Stance Composition Horizontal Stacked Bar
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
print("\n[2/5] Generating community_stance_composition.png ...")

# Compute stats
comm_stats = {}
for g in unique_groups:
    members = group_members[g]
    dem = sum(1 for n in members if node_labels[n] == 0)
    rep = sum(1 for n in members if node_labels[n] == 1)
    total = len(members)
    comm_stats[g] = {'dem': dem, 'rep': rep, 'total': total,
                     'dem_pct': dem / total * 100, 'rep_pct': rep / total * 100}

# Sort by group size (large to small)
groups_sorted = sorted(unique_groups, key=lambda g: comm_stats[g]['total'], reverse=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='white')
ax.set_facecolor('white')

y_pos = np.arange(len(groups_sorted))
bar_height = 0.65

for i, g in enumerate(groups_sorted):
    dem_pct = comm_stats[g]['dem_pct']
    rep_pct = comm_stats[g]['rep_pct']
    
    # Democrat portion
    ax.barh(i, dem_pct, height=bar_height, color=DEM_COLOR, edgecolor='white', linewidth=0.5)
    # Republican portion
    ax.barh(i, rep_pct, height=bar_height, left=dem_pct, color=REP_COLOR, edgecolor='white', linewidth=0.5)
    
    # Percentage labels inside bars
    if dem_pct > 12:
        ax.text(dem_pct / 2, i, f'{dem_pct:.0f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    if rep_pct > 12:
        ax.text(dem_pct + rep_pct / 2, i, f'{rep_pct:.0f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    
    # Total count on the right
    ax.text(102, i, f'n={comm_stats[g]["total"]}', ha='left', va='center',
            fontsize=8, color='#555555')

# 50% reference line
ax.axvline(x=50, color='#999999', linestyle='--', linewidth=0.6, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Community {g}' for g in groups_sorted], fontsize=9)
ax.set_xlim(0, 100)
ax.set_xlabel('Proportion (%)', fontsize=10)
ax.set_xticks([0, 25, 50, 75, 100])

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=DEM_COLOR,
           markersize=10, label='Democrat'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=REP_COLOR,
           markersize=10, label='Republican'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
          framealpha=0.95, edgecolor='#dddddd', fancybox=False)

ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top', ha='left')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(VIS_DIR, 'community_stance_composition.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("    Done.")

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Figure 3: Faceted Radar Chart (Small Multiples)
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
print("\n[3/5] Generating community_radar.png ...")

# Compute radar metrics per group
# structural_features: [in_degree, out_degree, total_degree, degree_centrality, pagerank, betweenness, clustering, local_reaching]
radar_metrics = {}
for g in unique_groups:
    members = group_members[g]
    size_norm = len(members) / max(len(group_members[gg]) for gg in unique_groups)
    dem_pct = sum(1 for n in members if node_labels[n] == 0) / len(members)
    in_deg_mean = np.mean([structural_features[n, 0] for n in members])
    out_deg_mean = np.mean([structural_features[n, 1] for n in members])
    pagerank_mean = np.mean([structural_features[n, 4] for n in members])
    clustering_mean = np.mean([structural_features[n, 6] for n in members])
    radar_metrics[g] = [size_norm, dem_pct, in_deg_mean, out_deg_mean, pagerank_mean, clustering_mean]

# Normalize each dimension to [0, 1]
all_values = np.array([radar_metrics[g] for g in unique_groups])
mins = all_values.min(axis=0)
maxs = all_values.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1

for g in unique_groups:
    radar_metrics[g] = [(v - mn) / rng for v, mn, rng in zip(radar_metrics[g], mins, ranges)]

categories = ['Size', 'Democrat %', 'In-degree', 'Out-degree', 'PageRank', 'Clustering']
N_dims = len(categories)
angles = np.linspace(0, 2 * np.pi, N_dims, endpoint=False).tolist()
angles += angles[:1]

# 2x4 grid
fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw=dict(polar=True), facecolor='white')
axes_flat = axes.flatten()

for idx, g in enumerate(unique_groups):
    ax = axes_flat[idx]
    values = radar_metrics[g] + [radar_metrics[g][0]]
    
    ax.plot(angles, values, 'o-', linewidth=1.5, color=group_color_map[g], markersize=3)
    ax.fill(angles, values, alpha=0.25, color=group_color_map[g])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=6)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(f'Community {g}\n(n={len(group_members[g])})', fontsize=9,
                 fontweight='bold', pad=12)
    ax.spines['polar'].set_linewidth(0.5)

# Last subplot: hide if fewer groups than 8
for idx in range(len(unique_groups), 8):
    axes_flat[idx].set_visible(False)

# Sub-figure label on first subplot
axes_flat[0].text(-0.3, 1.2, '(c)', transform=axes_flat[0].transAxes,
                  fontsize=13, fontweight='bold', va='top', ha='left')

plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(VIS_DIR, 'community_radar.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("    Done.")

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Figure 4: Method Comparison Heatmap
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
print("\n[4/5] Generating method_comparison_heatmap.png ...")

all_results = balanced_results['results']
results_sorted = sorted(all_results, key=lambda r: r['comprehensive_score'], reverse=True)
top8 = results_sorted[:8]

# Build matrix
methods = [r['method'] for r in top8]
metric_keys = ['modularity', 'nmi', 'purity', 'balance_score', None, 'comprehensive_score']
metric_labels = ['Modularity', 'NMI', 'Purity', 'Balance', '1-Gini', 'Score']

matrix = np.zeros((len(methods), len(metric_keys)))
for i, r in enumerate(top8):
    for j, key in enumerate(metric_keys):
        if key is None:
            matrix[i, j] = 1 - r['gini']
        else:
            matrix[i, j] = r[key]

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), facecolor='white')
ax.set_facecolor('white')

# Normalize per column for color mapping
col_mins = matrix.min(axis=0)
col_maxs = matrix.max(axis=0)
col_ranges = col_maxs - col_mins
col_ranges[col_ranges == 0] = 1
matrix_norm = (matrix - col_mins) / col_ranges

# Draw heatmap manually for fine control
cmap = plt.cm.RdYlBu_r
for i in range(len(methods)):
    for j in range(len(metric_keys)):
        color = cmap(matrix_norm[i, j])
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color,
                              edgecolor='white', linewidth=1.5)
        ax.add_patch(rect)
        # Text annotation
        val = matrix[i, j]
        # Choose text color based on brightness
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = 'white' if brightness < 0.55 else '#333333'
        fontweight = 'bold' if i == 0 else 'normal'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=9, color=text_color, fontweight=fontweight)

# Highlight best method row
rect_highlight = plt.Rectangle((-0.5, -0.5), len(metric_keys), 1,
                               fill=False, edgecolor='#E64B35', linewidth=2.5)
ax.add_patch(rect_highlight)

ax.set_xlim(-0.5, len(metric_keys) - 0.5)
ax.set_ylim(-0.5, len(methods) - 0.5)
ax.invert_yaxis()

ax.set_xticks(range(len(metric_keys)))
ax.set_xticklabels(metric_labels, fontsize=9, fontweight='bold')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=8.5)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# Colorbar
sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
cbar.set_label('Relative performance', fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.5)

# Best method annotation
ax.text(len(metric_keys) - 0.5, 0, ' \u2190 Best', ha='left', va='center',
        fontsize=8, color='#E64B35', fontweight='bold')

ax.text(0.02, -0.08, '(d)', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top', ha='left')

plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(VIS_DIR, 'method_comparison_heatmap.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("    Done.")

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
# Figure 5: Combined Figure (2x2 layout)
# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
print("\n[5/5] Generating combined_figure.png ...")

from matplotlib.image import imread

fig = plt.figure(figsize=(16, 14), facecolor='white')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2)

# Load saved images
img_paths = [
    os.path.join(VIS_DIR, 'balanced_community_network.png'),
    os.path.join(VIS_DIR, 'community_stance_composition.png'),
    os.path.join(VIS_DIR, 'community_radar.png'),
    os.path.join(VIS_DIR, 'method_comparison_heatmap.png'),
]

positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for (row, col), img_path in zip(positions, img_paths):
    ax = fig.add_subplot(gs[row, col])
    img = imread(img_path)
    ax.imshow(img)
    ax.axis('off')

plt.savefig(os.path.join(VIS_DIR, 'combined_figure.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("    Done.")

# ¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T¨T
print("\n" + "=" * 60)
print("All publication-quality figures generated!")
print("=" * 60)
print(f"\nOutput directory: {VIS_DIR}")
for f in ['balanced_community_network.png', 'community_stance_composition.png',
           'community_radar.png', 'method_comparison_heatmap.png', 'combined_figure.png']:
    fpath = os.path.join(VIS_DIR, f)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  ? {f} ({size_kb:.1f} KB)")
    else:
        print(f"  ? {f} MISSING")
