# -*- coding: utf-8 -*-
"""
Extract structural features from ml_twitter.csv edge list.
Build a directed graph and compute 8 structural features for nodes 1-877.
Save as (878, n_features) matrix with row 0 as zero placeholder.
"""

import time
import json
import numpy as np
import pandas as pd
import networkx as nx

# ===================== Paths =====================
EDGE_CSV = "/root/CORDGT/CorDGT/processed/ml_twitter.csv"
OUTPUT_NPY = "/root/CORDGT/CorDGT/lab3/task1/features/structural_features.npy"
OUTPUT_NAMES = "/root/CORDGT/CorDGT/lab3/task1/features/structural_feature_names.json"

# ===================== 1. Read edge table, build DiGraph =====================
print("=" * 60)
print("Step 1: Read edge table and build directed graph")
print("=" * 60)

df = pd.read_csv(EDGE_CSV)
print(f"  Edge count: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Source node range: {df['u'].min()} - {df['u'].max()}")
print(f"  Target node range: {df['i'].min()} - {df['i'].max()}")
print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

G = nx.DiGraph()
edges = list(zip(df['u'], df['i']))
G.add_edges_from(edges)
print(f"  DiGraph nodes: {G.number_of_nodes()}")
print(f"  DiGraph edges: {G.number_of_edges()}")

# Ensure all nodes 1-877 exist in graph (including isolated ones)
for node_id in range(1, 878):
    if node_id not in G:
        G.add_node(node_id)
print(f"  After adding isolated nodes, graph nodes: {G.number_of_nodes()}")

# ===================== 2. Compute structural features =====================
print("\n" + "=" * 60)
print("Step 2: Compute structural features")
print("=" * 60)

feature_names = [
    "in_degree",
    "out_degree",
    "total_degree",
    "degree_centrality",
    "pagerank",
    "betweenness_centrality",
    "clustering_coefficient",
    "local_reaching_centrality",
]

num_nodes = 878  # index 0 placeholder + nodes 1-877
num_features = len(feature_names)
features = np.zeros((num_nodes, num_features), dtype=np.float64)

# --- 2.1 Degree features ---
t0 = time.time()
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
total_deg = dict(G.degree())
for n in range(1, 878):
    features[n, 0] = in_deg.get(n, 0)
    features[n, 1] = out_deg.get(n, 0)
    features[n, 2] = total_deg.get(n, 0)
print(f"  [Degree features] done, time {time.time()-t0:.2f}s")

# --- 2.2 Degree centrality ---
t0 = time.time()
deg_cent = nx.degree_centrality(G)
for n in range(1, 878):
    features[n, 3] = deg_cent.get(n, 0.0)
print(f"  [Degree centrality] done, time {time.time()-t0:.2f}s")

# --- 2.3 PageRank ---
t0 = time.time()
pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-06)
for n in range(1, 878):
    features[n, 4] = pr.get(n, 0.0)
print(f"  [PageRank] done, time {time.time()-t0:.2f}s")

# --- 2.4 Betweenness centrality (approximate, k=200) ---
t0 = time.time()
bw = nx.betweenness_centrality(G, k=200, normalized=True)
for n in range(1, 878):
    features[n, 5] = bw.get(n, 0.0)
print(f"  [Betweenness centrality] done (k=200 approx), time {time.time()-t0:.2f}s")

# --- 2.5 Clustering coefficient (on undirected version) ---
t0 = time.time()
G_undirected = G.to_undirected()
clust = nx.clustering(G_undirected)
for n in range(1, 878):
    features[n, 6] = clust.get(n, 0.0)
print(f"  [Clustering coefficient] done (undirected), time {time.time()-t0:.2f}s")

# --- 2.6 Local reaching centrality ---
t0 = time.time()
count = 0
for n in range(1, 878):
    features[n, 7] = nx.local_reaching_centrality(G, n)
    count += 1
    if count % 100 == 0:
        print(f"    local_reaching_centrality: {count}/877 done, time {time.time()-t0:.1f}s")
print(f"  [Local reaching centrality] done, total time {time.time()-t0:.2f}s")

# ===================== 3. Statistics summary =====================
print("\n" + "=" * 60)
print("Step 3: Feature statistics summary (nodes 1-877 only)")
print("=" * 60)

valid_features = features[1:]  # exclude index 0 placeholder
for i, name in enumerate(feature_names):
    col = valid_features[:, i]
    print(f"  {name:30s}  mean={col.mean():.6f}  std={col.std():.6f}  "
          f"min={col.min():.6f}  max={col.max():.6f}")

# ===================== 4. Save results =====================
print("\n" + "=" * 60)
print("Step 4: Save results")
print("=" * 60)

np.save(OUTPUT_NPY, features)
print(f"  Feature matrix saved: {OUTPUT_NPY}")
print(f"  Matrix shape: {features.shape}, dtype: {features.dtype}")

with open(OUTPUT_NAMES, 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"  Feature names saved: {OUTPUT_NAMES}")
print(f"  Feature list: {feature_names}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
