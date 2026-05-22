# -*- coding: utf-8 -*-
"""
Task 7: Static Interaction Graph Construction
Build weighted interaction graphs from ml_twitter.csv
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Font settings for Chinese
font_candidates = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei']
font_set = False
for font_name in font_candidates:
    try:
        from matplotlib.font_manager import FontProperties, findfont
        fp = FontProperties(family=font_name)
        found = findfont(fp, fallback_to_default=False)
        if found and 'DejaVu' not in found:
            plt.rcParams['font.sans-serif'] = [font_name]
            font_set = True
            print(f"Using font: {font_name}")
            break
    except Exception:
        continue

if not font_set:
    plt.rcParams['font.sans-serif'] = ['DejaVuSans']
    print("Warning: No Chinese font found, using default")

plt.rcParams['axes.unicode_minus'] = False

# Path configuration
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
VIS_DIR = os.path.join(TASK2_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# Load data
print("\n" + "="*60)
print("Loading data...")
print("="*60)

edge_file = os.path.join(PROCESSED_DIR, 'ml_twitter.csv')
label_file = os.path.join(PROCESSED_DIR, 'ml_twitter_node_labels.npy')

df = pd.read_csv(edge_file)
node_labels = np.load(label_file)  # shape (878,), index 0 unused, nodes 1-877

print(f"Edge table size: {len(df)} records")
print(f"Node labels shape: {node_labels.shape}")
print(f"Label distribution (nodes 1-877): {dict(Counter(node_labels[1:]))}")
print(f"Edge type distribution: label=1(retweet): {(df['label']==1).sum()}, label=3(reply): {(df['label']==3).sum()}")


def build_weighted_graph(edges_df, directed=True):
    """Build weighted graph from edge DataFrame, merging duplicate edges as weights"""
    edge_counts = edges_df.groupby(['u', 'i']).size().reset_index(name='weight')
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add all nodes (1-877)
    G.add_nodes_from(range(1, 878))
    
    # Add weighted edges using vectorized approach
    u_arr = edge_counts['u'].values
    i_arr = edge_counts['i'].values
    w_arr = edge_counts['weight'].values
    
    if directed:
        edge_list = [(int(u_arr[k]), int(i_arr[k]), {'weight': int(w_arr[k])}) for k in range(len(u_arr))]
        G.add_edges_from(edge_list)
    else:
        for k in range(len(u_arr)):
            u, i, w = int(u_arr[k]), int(i_arr[k]), int(w_arr[k])
            if G.has_edge(u, i):
                G[u][i]['weight'] += w
            else:
                G.add_edge(u, i, weight=w)
    
    return G


# 1. Combined weighted interaction graph
print("\n" + "="*60)
print("1. Building combined weighted interaction graph")
print("="*60)

G_directed = build_weighted_graph(df, directed=True)
G_undirected = build_weighted_graph(df, directed=False)

with open(os.path.join(TASK2_DIR, 'combined_graph_directed.pkl'), 'wb') as f:
    pickle.dump(G_directed, f)
with open(os.path.join(TASK2_DIR, 'combined_graph_undirected.pkl'), 'wb') as f:
    pickle.dump(G_undirected, f)

print(f"Directed combined graph - Nodes: {G_directed.number_of_nodes()}, Edges: {G_directed.number_of_edges()}")
print(f"Undirected combined graph - Nodes: {G_undirected.number_of_nodes()}, Edges: {G_undirected.number_of_edges()}")

# 2. Retweet graph
print("\n" + "="*60)
print("2. Building retweet graph (label=1)")
print("="*60)

df_retweet = df[df['label'] == 1]
G_retweet = build_weighted_graph(df_retweet, directed=True)

with open(os.path.join(TASK2_DIR, 'retweet_graph.pkl'), 'wb') as f:
    pickle.dump(G_retweet, f)

print(f"Retweet graph - Nodes: {G_retweet.number_of_nodes()}, Edges: {G_retweet.number_of_edges()}")

# 3. Reply graph
print("\n" + "="*60)
print("3. Building reply graph (label=3)")
print("="*60)

df_reply = df[df['label'] == 3]
G_reply = build_weighted_graph(df_reply, directed=True)

with open(os.path.join(TASK2_DIR, 'reply_graph.pkl'), 'wb') as f:
    pickle.dump(G_reply, f)

print(f"Reply graph - Nodes: {G_reply.number_of_nodes()}, Edges: {G_reply.number_of_edges()}")

# 4. Statistics
print("\n" + "="*60)
print("4. Graph Statistics")
print("="*60)

def print_graph_stats(G, name):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_edges > 0:
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        avg_weight = np.mean(weights)
        max_weight = max(weights)
        max_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if d['weight'] == max_weight]
        
        print(f"\n--- {name} ---")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {n_edges}")
        print(f"  Average weight: {avg_weight:.4f}")
        print(f"  Max weight: {max_weight}")
        print(f"  Max weight edge(s): {max_edges[:5]}")
    else:
        print(f"\n--- {name} ---")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {n_edges}")

print_graph_stats(G_directed, "Directed Combined Graph")
print_graph_stats(G_undirected, "Undirected Combined Graph")
print_graph_stats(G_retweet, "Retweet Graph")
print_graph_stats(G_reply, "Reply Graph")

# Connectivity analysis
n_weakly_connected = nx.number_weakly_connected_components(G_directed)
n_strongly_connected = nx.number_strongly_connected_components(G_directed)
n_connected_undirected = nx.number_connected_components(G_undirected)

print(f"\n--- Connectivity Analysis ---")
print(f"  Directed graph weakly connected components: {n_weakly_connected}")
print(f"  Directed graph strongly connected components: {n_strongly_connected}")
print(f"  Undirected graph connected components: {n_connected_undirected}")

# 5. Visualization
print("\n" + "="*60)
print("5. Visualization")
print("="*60)

# 5.1 Network structure plot
print("Drawing network structure...")

degrees = dict(G_undirected.degree())

# Use Top-200 nodes by degree for visualization
top_k = 200
sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
top_nodes = [n for n, d in sorted_nodes[:top_k]]
G_sub = G_undirected.subgraph(top_nodes).copy()

print(f"  Using Top-{top_k} subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

# Color by stance label: 0=Democrat(blue), 1=Republican(red)
color_map = []
for node in G_sub.nodes():
    label = node_labels[node] if node < len(node_labels) else -1
    if label == 0:
        color_map.append('#2196F3')  # Blue - Democrat
    elif label == 1:
        color_map.append('#F44336')  # Red - Republican
    else:
        color_map.append('#9E9E9E')  # Gray - Unknown

# Node size scaled by degree
sub_degrees = dict(G_sub.degree())
max_deg = max(sub_degrees.values()) if sub_degrees else 1
node_sizes = [30 + 200 * (sub_degrees[n] / max_deg) for n in G_sub.nodes()]

fig, ax = plt.subplots(1, 1, figsize=(14, 12))
print("  Computing spring layout...")
pos = nx.spring_layout(G_sub, k=0.3, iterations=30, seed=42)

nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.15, width=0.5,
                       edge_color='gray', arrows=False)

nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_color=color_map,
                       node_size=node_sizes, alpha=0.8, edgecolors='white', linewidths=0.3)

ax.set_title(u"\u7efc\u5408\u4ea4\u4e92\u7f51\u7edc\u7ed3\u6784\u56fe\n(\u5ea6\u6570Top-200\u5b50\u56fe, \u7ea2=\u5171\u548c\u515a, \u84dd=\u6c11\u4e3b\u515a)", fontsize=16)
ax.axis('off')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#F44336', label=u'\u5171\u548c\u515a (Republican)'),
    Patch(facecolor='#2196F3', label=u'\u6c11\u4e3b\u515a (Democrat)'),
    Patch(facecolor='#9E9E9E', label=u'\u672a\u77e5 (Unknown)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'network_structure.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: visualizations/network_structure.png")

# 5.2 Edge weight distribution
print("Drawing edge weight distribution...")

all_weights = [d['weight'] for _, _, d in G_directed.edges(data=True)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Combined graph
axes[0].hist(all_weights, bins=50, color='#4CAF50', alpha=0.8, edgecolor='white')
axes[0].set_title(u"\u7efc\u5408\u4ea4\u4e92\u56fe - \u8fb9\u6743\u91cd\u5206\u5e03", fontsize=13)
axes[0].set_xlabel(u"\u6743\u91cd (\u4ea4\u4e92\u6b21\u6570)", fontsize=11)
axes[0].set_ylabel(u"\u8fb9\u6570", fontsize=11)
axes[0].set_yscale('log')
axes[0].axvline(np.mean(all_weights), color='red', linestyle='--', label=f'mean={np.mean(all_weights):.2f}')
axes[0].legend(fontsize=10)

# Retweet graph
rt_weights = [d['weight'] for _, _, d in G_retweet.edges(data=True)]
if rt_weights:
    axes[1].hist(rt_weights, bins=50, color='#FF9800', alpha=0.8, edgecolor='white')
    axes[1].set_title(u"\u8f6c\u53d1\u56fe - \u8fb9\u6743\u91cd\u5206\u5e03", fontsize=13)
    axes[1].set_xlabel(u"\u6743\u91cd (\u8f6c\u53d1\u6b21\u6570)", fontsize=11)
    axes[1].set_ylabel(u"\u8fb9\u6570", fontsize=11)
    axes[1].set_yscale('log')
    axes[1].axvline(np.mean(rt_weights), color='red', linestyle='--', label=f'mean={np.mean(rt_weights):.2f}')
    axes[1].legend(fontsize=10)

# Reply graph
rp_weights = [d['weight'] for _, _, d in G_reply.edges(data=True)]
if rp_weights:
    axes[2].hist(rp_weights, bins=50, color='#9C27B0', alpha=0.8, edgecolor='white')
    axes[2].set_title(u"\u56de\u590d\u56fe - \u8fb9\u6743\u91cd\u5206\u5e03", fontsize=13)
    axes[2].set_xlabel(u"\u6743\u91cd (\u56de\u590d\u6b21\u6570)", fontsize=11)
    axes[2].set_ylabel(u"\u8fb9\u6570", fontsize=11)
    axes[2].set_yscale('log')
    axes[2].axvline(np.mean(rp_weights), color='red', linestyle='--', label=f'mean={np.mean(rp_weights):.2f}')
    axes[2].legend(fontsize=10)

plt.suptitle(u"\u8fb9\u6743\u91cd\u5206\u5e03", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'edge_weight_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: visualizations/edge_weight_distribution.png")

# Done
print("\n" + "="*60)
print("Static interaction graph construction complete!")
print("="*60)
print(f"\nOutput files:")
print(f"  - {TASK2_DIR}/combined_graph_directed.pkl")
print(f"  - {TASK2_DIR}/combined_graph_undirected.pkl")
print(f"  - {TASK2_DIR}/retweet_graph.pkl")
print(f"  - {TASK2_DIR}/reply_graph.pkl")
print(f"  - {VIS_DIR}/network_structure.png")
print(f"  - {VIS_DIR}/edge_weight_distribution.png")
