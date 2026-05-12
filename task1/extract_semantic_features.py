#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract individual semantic features from BERT-encoded tweet embeddings.
Aggregate per source node: mean, variance, max embeddings + semantic consistency.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ============ Path Config ============
BASE_DIR = "/root/CORDGT/CorDGT"
EDGE_EMB_PATH = os.path.join(BASE_DIR, "processed/ml_twitter.npy")
EDGE_CSV_PATH = os.path.join(BASE_DIR, "processed/ml_twitter.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "lab3/task1/features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 1. Load Data ============
print("=" * 60)
print("Step 1: Load Data")
print("=" * 60)

print(f"  Loading edge embeddings: {EDGE_EMB_PATH}")
edge_emb = np.load(EDGE_EMB_PATH)
print(f"  Edge embeddings shape: {edge_emb.shape}, dtype: {edge_emb.dtype}")

print(f"  Loading edge table: {EDGE_CSV_PATH}")
edge_df = pd.read_csv(EDGE_CSV_PATH)
print(f"  Edge table shape: {edge_df.shape}")
print(f"  Columns: {list(edge_df.columns)}")

NUM_NODES = 878
EMB_DIM = 768

# ============ 2. Group by source node u ============
print("\n" + "=" * 60)
print("Step 2: Group by source node u")
print("=" * 60)

node_to_indices = defaultdict(list)
for _, row in edge_df.iterrows():
    u = int(row["u"])
    idx = int(row["idx"])
    node_to_indices[u].append(idx)

print(f"  Nodes with outgoing edges: {len(node_to_indices)}")
print(f"  Nodes without outgoing edges: {NUM_NODES - 1 - len(node_to_indices)} (excl. index 0)")

# ============ 3. Compute aggregated features ============
print("\n" + "=" * 60)
print("Step 3: Compute aggregated features (mean, var, max)")
print("=" * 60)

mean_embedding = np.zeros((NUM_NODES, EMB_DIM), dtype=np.float32)
var_embedding = np.zeros((NUM_NODES, EMB_DIM), dtype=np.float32)
max_embedding = np.zeros((NUM_NODES, EMB_DIM), dtype=np.float32)
tweet_counts = np.zeros(NUM_NODES, dtype=np.int32)

for u, indices in node_to_indices.items():
    embs = edge_emb[indices]  # (num_tweets, 768)
    tweet_counts[u] = len(indices)
    mean_embedding[u] = embs.mean(axis=0)
    var_embedding[u] = embs.var(axis=0)
    max_embedding[u] = embs.max(axis=0)

print(f"  mean_embedding shape: {mean_embedding.shape}")
print(f"  var_embedding shape: {var_embedding.shape}")
print(f"  max_embedding shape: {max_embedding.shape}")

# Free large array to save memory
del edge_emb

# ============ 4. Compute semantic consistency score ============
print("\n" + "=" * 60)
print("Step 4: Compute semantic consistency score")
print("=" * 60)

semantic_consistency = np.zeros(NUM_NODES, dtype=np.float32)

# Reload edge embeddings for consistency computation
edge_emb = np.load(EDGE_EMB_PATH)

for u, indices in node_to_indices.items():
    if len(indices) < 2:
        # Single tweet user: consistency = 1.0 (perfectly consistent with self)
        semantic_consistency[u] = 1.0
        continue

    embs = edge_emb[indices]  # (num_tweets, 768)
    mean_vec = mean_embedding[u].reshape(1, -1)  # (1, 768)

    # Cosine similarity: cos(a, b) = (a . b) / (||a|| * ||b||)
    embs_norm = np.linalg.norm(embs, axis=1, keepdims=True)  # (N, 1)
    mean_norm = np.linalg.norm(mean_vec, axis=1, keepdims=True)  # (1, 1)

    # Avoid division by zero
    embs_norm = np.maximum(embs_norm, 1e-8)
    mean_norm = np.maximum(mean_norm, 1e-8)

    cos_sims = (embs @ mean_vec.T) / (embs_norm * mean_norm)  # (N, 1)
    semantic_consistency[u] = float(cos_sims.mean())

del edge_emb

print(f"  semantic_consistency shape: {semantic_consistency.shape}")
print(f"  Active nodes (consistency > 0): {(semantic_consistency > 0).sum()}")
print(f"  Inactive nodes (consistency == 0): {(semantic_consistency == 0).sum()}")

# ============ 5. Combine features ============
print("\n" + "=" * 60)
print("Step 5: Combine and save features")
print("=" * 60)

# Scheme A (full): mean + var + max + consistency = 768*3 + 1 = 2305 dim
consistency_col = semantic_consistency.reshape(NUM_NODES, 1)  # (878, 1)
semantic_features_full = np.concatenate(
    [mean_embedding, var_embedding, max_embedding, consistency_col],
    axis=1
).astype(np.float32)

print(f"  Full version shape: {semantic_features_full.shape}")
print(f"  Row 0 all zeros: {np.all(semantic_features_full[0] == 0)}")

# Scheme B (compressed): mean + consistency = 768 + 1 = 769 dim
semantic_features = np.concatenate(
    [mean_embedding, consistency_col],
    axis=1
).astype(np.float32)

print(f"  Compressed version shape: {semantic_features.shape}")
print(f"  Row 0 all zeros: {np.all(semantic_features[0] == 0)}")

# ============ 6. Save ============
full_path = os.path.join(OUTPUT_DIR, "semantic_features_full.npy")
comp_path = os.path.join(OUTPUT_DIR, "semantic_features.npy")
cons_path = os.path.join(OUTPUT_DIR, "semantic_consistency.npy")

np.save(full_path, semantic_features_full)
np.save(comp_path, semantic_features)
np.save(cons_path, semantic_consistency)

print(f"\n  Saved: {full_path} shape={semantic_features_full.shape}")
print(f"  Saved: {comp_path} shape={semantic_features.shape}")
print(f"  Saved: {cons_path} shape={semantic_consistency.shape}")

# ============ 7. Statistics Summary ============
print("\n" + "=" * 60)
print("Statistics Summary")
print("=" * 60)

# Tweet count distribution
active_counts = tweet_counts[tweet_counts > 0]
print(f"\n  [Tweet Count Stats] (active nodes only, N={len(active_counts)})")
print(f"    Mean: {active_counts.mean():.1f}")
print(f"    Median: {np.median(active_counts):.1f}")
print(f"    Min: {active_counts.min()}")
print(f"    Max: {active_counts.max()}")
print(f"    Std: {active_counts.std():.1f}")
for q in [25, 50, 75, 90, 95, 99]:
    print(f"    P{q}: {np.percentile(active_counts, q):.0f}")

# Semantic consistency distribution
active_consistency = semantic_consistency[semantic_consistency > 0]
print(f"\n  [Semantic Consistency Stats] (active nodes, N={len(active_consistency)})")
print(f"    Mean: {active_consistency.mean():.4f}")
print(f"    Median: {np.median(active_consistency):.4f}")
print(f"    Min: {active_consistency.min():.4f}")
print(f"    Max: {active_consistency.max():.4f}")
print(f"    Std: {active_consistency.std():.4f}")
for q in [25, 50, 75, 90, 95, 99]:
    print(f"    P{q}: {np.percentile(active_consistency, q):.4f}")

# Consistency bins
bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0"]
hist, _ = np.histogram(active_consistency, bins=bins)
print(f"\n  [Semantic Consistency Distribution]")
for label, count in zip(labels, hist):
    pct = count / len(active_consistency) * 100
    print(f"    {label:>10s}: {count:4d} ({pct:5.1f}%)")

# Node tweet count breakdown
single_tweet = (tweet_counts == 1).sum()
multi_tweet = (tweet_counts > 1).sum()
print(f"\n  [Node Tweet Count Breakdown]")
print(f"    0 tweets: {(tweet_counts == 0).sum()} nodes")
print(f"    1 tweet:  {single_tweet} nodes (consistency fixed to 1.0)")
print(f"    2+ tweets: {multi_tweet} nodes")

# Sample nodes
print(f"\n  [Sample Nodes]")
for u in range(1, 10):
    cnt = tweet_counts[u]
    cons = semantic_consistency[u]
    if cnt > 0:
        norm = np.linalg.norm(mean_embedding[u])
        print(f"    Node {u}: {cnt} tweets, consistency={cons:.4f}, mean_norm={norm:.4f}")
    else:
        print(f"    Node {u}: no tweets")

print("\n" + "=" * 60)
print("Feature extraction complete!")
print("=" * 60)
