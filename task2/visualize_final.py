# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import numpy as np
import pickle
import json
import os
import seaborn as sns
from matplotlib.lines import Line2D

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

GROUP_COLORS = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#f39c12', '#16a085', '#d35400']
OUTPUT_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task2/visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading data...')
with open('/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task2/combined_graph_undirected.pkl', 'rb') as f:
    G = pickle.load(f)
group_assignments = np.load('/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task2/group_assignments_balanced.npy')
node_labels = np.load('/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/processed/ml_twitter_node_labels.npy')
structural_features = np.load('/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task1/features/structural_features.npy')
with open('/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task2/balanced_results.json', 'r') as f:
    balanced_results = json.load(f)

nodes = sorted(G.nodes())
node_to_group = {n: int(group_assignments[n]) for n in nodes}
node_to_party = {n: int(node_labels[n]) for n in nodes}
num_groups = 7
group_sizes = {}
group_nodes = {}
for gid in range(num_groups):
    gnodes = [n for n in nodes if node_to_group[n] == gid]
    group_nodes[gid] = gnodes
    group_sizes[gid] = len(gnodes)
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
print(f'Group sizes: {group_sizes}')

# ============================================================
# Fig1: Community Network
# ============================================================
print('\nFig1: Network...')
fig, ax = plt.subplots(1, 1, figsize=(14, 11))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

meta_G = nx.Graph()
for gid in range(num_groups):
    meta_G.add_node(gid)
for u, v, d in G.edges(data=True):
    gu, gv = node_to_group[u], node_to_group[v]
    if gu != gv:
        if meta_G.has_edge(gu, gv):
            meta_G[gu][gv]['weight'] += 1
        else:
            meta_G.add_edge(gu, gv, weight=1)

center_pos = nx.spring_layout(meta_G, k=4, seed=42, iterations=100)
pos = {}
for gid in range(num_groups):
    sub_nodes = group_nodes[gid]
    if not sub_nodes:
        continue
    subG = G.subgraph(sub_nodes)
    sub_pos = nx.spring_layout(subG, k=1.5/np.sqrt(len(sub_nodes)), seed=42+gid, iterations=50)
    cx, cy = center_pos[gid]
    for n in sub_nodes:
        px, py = sub_pos[n]
        pos[n] = (cx + px*0.6, cy + py*0.6)

degrees = dict(G.degree())
deg_array = np.array([degrees[n] for n in nodes])
sizes_arr = 20 + np.log1p(deg_array) * 40

heavy_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1) >= 3]
nx.draw_networkx_edges(G, pos, edgelist=heavy_edges, edge_color='#d0d0d0', alpha=0.1, width=0.3, ax=ax)

for gid in range(num_groups):
    sub_nodes = group_nodes[gid]
    ns = [sizes_arr[nodes.index(n)] for n in sub_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=sub_nodes, node_color=GROUP_COLORS[gid],
                           node_size=ns, alpha=0.8, edgecolors='white', linewidths=0.5, ax=ax)

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLORS[gid],
                          markersize=10, label=f'\u7fa4\u4f53{gid} (n={group_sizes[gid]})')
                   for gid in range(num_groups)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#ccc', fancybox=True)
ax.set_title('\u7fa4\u4f53\u5212\u5206\u7f51\u7edc\u7ed3\u6784\u56fe', fontsize=16, fontweight='bold', pad=15)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'balanced_community_network.png'), dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.close()
print('  OK: balanced_community_network.png')

# ============================================================
# Fig2: Stance Composition
# ============================================================
print('\nFig2: Stance Composition...')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

group_party_data = []
for gid in range(num_groups):
    sub_nodes = group_nodes[gid]
    parties = [node_to_party[n] for n in sub_nodes]
    dem = sum(1 for p in parties if p == 0)
    rep = sum(1 for p in parties if p == 1)
    total = dem + rep
    group_party_data.append((gid, group_sizes[gid], dem/total*100 if total else 0, rep/total*100 if total else 0, total))

group_party_data.sort(key=lambda x: x[1], reverse=True)
y_labels = [f'\u7fa4\u4f53{d[0]} (n={d[1]})' for d in group_party_data]
dem_pcts = [d[2] for d in group_party_data]
rep_pcts = [d[3] for d in group_party_data]
totals = [d[4] for d in group_party_data]
y_pos = np.arange(len(y_labels))

ax.barh(y_pos, dem_pcts, 0.6, color='#2c3e50', label='\u6c11\u4e3b\u515a')
ax.barh(y_pos, rep_pcts, 0.6, left=dem_pcts, color='#e74c3c', label='\u5171\u548c\u515a')

for i, (dp, rp) in enumerate(zip(dem_pcts, rep_pcts)):
    if dp > 10:
        ax.text(dp/2, i, f'{dp:.0f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    if rp > 10:
        ax.text(dp+rp/2, i, f'{rp:.0f}%', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

for i, t in enumerate(totals):
    ax.text(102, i, f'n={t}', ha='left', va='center', fontsize=9, color='#555')

ax.axvline(x=50, color='#888', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=11)
ax.set_xlabel('\u6bd4\u4f8b (%)', fontsize=12)
ax.set_xlim(0, 100)
ax.set_title('\u5404\u7fa4\u4f53\u653f\u6cbb\u7acb\u573a\u6784\u6210', fontsize=14, fontweight='bold', pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'community_stance_composition.png'), dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.close()
print('  OK: community_stance_composition.png')

# ============================================================
# Fig3: Radar Charts
# ============================================================
print('\nFig3: Radar Charts...')
radar_data = []
for gid in range(num_groups):
    sub_nodes = group_nodes[gid]
    parties = [node_to_party[n] for n in sub_nodes]
    dem_pct = sum(1 for p in parties if p == 0) / max(len(parties), 1) * 100
    in_deg = np.mean([structural_features[n, 0] for n in sub_nodes])
    out_deg = np.mean([structural_features[n, 1] for n in sub_nodes])
    pr = np.mean([structural_features[n, 4] for n in sub_nodes])
    clust = np.mean([structural_features[n, 6] for n in sub_nodes])
    radar_data.append([len(sub_nodes), dem_pct, in_deg, out_deg, pr, clust])

radar_data = np.array(radar_data)
r_min, r_max = radar_data.min(0), radar_data.max(0)
r_range = r_max - r_min
r_range[r_range == 0] = 1
radar_norm = (radar_data - r_min) / r_range

categories = ['\u7fa4\u4f53\u89c4\u6a21', '\u6c11\u4e3b\u515a%', '\u5165\u5ea6\u5747\u503c', '\u51fa\u5ea6\u5747\u503c', 'PageRank', '\u805a\u7c7b\u7cfb\u6570']
num_vars = len(categories)
angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 4, figsize=(16, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')

for idx in range(8):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    if idx < num_groups:
        values = radar_norm[idx].tolist() + [radar_norm[idx][0]]
        ax.plot(angles, values, 'o-', linewidth=1.5, color=GROUP_COLORS[idx], alpha=0.8, markersize=4)
        ax.fill(angles, values, color=GROUP_COLORS[idx], alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=7, color='#888')
        ax.set_title(f'\u7fa4\u4f53{idx} (n={group_sizes[idx]})', fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
    elif idx == 7:
        for gid in range(num_groups):
            v = radar_norm[gid].tolist() + [radar_norm[gid][0]]
            ax.plot(angles, v, '-', linewidth=1.2, color=GROUP_COLORS[gid], alpha=0.6, label=f'\u7fa4\u4f53{gid}')
            ax.fill(angles, v, color=GROUP_COLORS[gid], alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=7, color='#888')
        ax.set_title('\u603b\u89c8\u5bf9\u6bd4', fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.3), fontsize=8, ncol=4)

fig.suptitle('\u5404\u7fa4\u4f53\u591a\u7ef4\u7279\u5f81\u753b\u50cf', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'community_radar.png'), dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.close()
print('  OK: community_radar.png')

# ============================================================
# Fig4: Method Comparison Heatmap
# ============================================================
print('\nFig4: Heatmap...')
results_sorted = sorted(balanced_results['results'], key=lambda x: x.get('comprehensive_score', 0), reverse=True)[:8]

method_names = []
matrix = []
keys = ['modularity', 'nmi', 'purity', 'balance_score', 'gini', 'comprehensive_score']
labels_x = ['\u6a21\u5757\u5ea6', 'NMI', '\u7eaf\u5ea6', '\u5e73\u8861\u6027', '\u5747\u8861\u5ea6', '\u7efc\u5408\u5f97\u5206']
best_method = balanced_results['best_method']

for item in results_sorted:
    name = item['method']
    if name == best_method:
        name = '\u2605 ' + name
    method_names.append(name)
    row = [1 - item[k] if k == 'gini' else item[k] for k in keys]
    matrix.append(row)

matrix = np.array(matrix)
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')
sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', xticklabels=labels_x, yticklabels=method_names,
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8, 'label': '\u6307\u6807\u503c'}, ax=ax)
ax.set_title('\u7fa4\u4f53\u5212\u5206\u65b9\u6cd5\u8bc4\u4f30\u5bf9\u6bd4', fontsize=14, fontweight='bold', pad=12)
ax.set_xticklabels(labels_x, fontsize=11, rotation=0)
ax.set_yticklabels(method_names, fontsize=10, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison_heatmap.png'), dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.close()
print('  OK: method_comparison_heatmap.png')

# ============================================================
# Fig5: Community Size Donut
# ============================================================
print('\nFig5: Donut Chart...')
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('white')

s_list = [group_sizes[gid] for gid in range(num_groups)]
l_list = [f'\u7fa4\u4f53{gid}\nn={group_sizes[gid]}' for gid in range(num_groups)]

wedges, texts = ax.pie(s_list, labels=l_list, colors=GROUP_COLORS[:num_groups],
                       wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
                       startangle=90, textprops={'fontsize': 10})
ax.text(0, 0, '7\u7fa4\u4f53\n877\u4eba', ha='center', va='center', fontsize=16, fontweight='bold', color='#2c3e50')
ax.set_title('\u7fa4\u4f53\u89c4\u6a21\u5206\u5e03', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'community_size_pie.png'), dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.close()
print('  OK: community_size_pie.png')

print('\n' + '='*50)
print('ALL DONE!')
print('='*50)
