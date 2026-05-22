# -*- coding: utf-8 -*-
"""
Task 3 - Script 2: Group Feature Integration
Build group-level feature sets (super-individuals) for downstream stance detection.
Memory-optimized for 2GB container limit.
"""

import numpy as np
import pandas as pd
import torch
import os
import gc

# GPU/CPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path settings
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
FEATURE_DIR = os.path.join(BASE_DIR, 'task1/features')
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'task3')

os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_idx = np.arange(1, 878)

# ============================================================
# Phase 1: Group feature aggregation (low memory)
# ============================================================
print("=== Phase 1: Group Feature Aggregation ===")

# Load small arrays
structural = np.load(os.path.join(FEATURE_DIR, 'structural_features.npy'))
behavioral = np.load(os.path.join(FEATURE_DIR, 'behavioral_features.npy'))
tgan = np.load(os.path.join(FEATURE_DIR, 'tgan_embeddings.npy'))
individual_scores = np.load(os.path.join(OUTPUT_DIR, 'individual_scores.npy'))
filtered_mask = np.load(os.path.join(OUTPUT_DIR, 'filtered_mask.npy'))
group_labels = np.load(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'))
party_labels = np.load(os.path.join(PROCESSED_DIR, 'ml_twitter_node_labels.npy'))

print(f"  structural: {structural.shape}")
print(f"  behavioral: {behavioral.shape}")
print(f"  tgan: {tgan.shape}")
print(f"  filtered_mask kept: {filtered_mask.sum()}")

# Aggregate structural, behavioral, tgan per group
group_structural = np.zeros((7, 8))
group_behavioral = np.zeros((7, 8))
group_tgan = np.zeros((7, 64))

for g in range(7):
    members = np.where((group_labels[valid_idx] == g) & filtered_mask[valid_idx])[0] + 1
    n_members = len(members)
    if n_members == 0:
        print(f"  WARNING: Group {g} has no filtered members!")
        continue
    weights = individual_scores[members]
    weight_sum = weights.sum()
    if weight_sum < 1e-10:
        weights = np.ones(n_members) / n_members
    else:
        weights = weights / weight_sum
    group_structural[g] = np.average(structural[members], axis=0, weights=weights)
    group_behavioral[g] = np.average(behavioral[members], axis=0, weights=weights)
    group_tgan[g] = np.average(tgan[members], axis=0, weights=weights)
    print(f"  Group {g}: {n_members} members aggregated")

# Free large arrays not needed further
del structural, behavioral, tgan
gc.collect()

# Now load semantic (larger ~5MB) and aggregate
print("  Loading semantic features...")
semantic = np.load(os.path.join(FEATURE_DIR, 'semantic_features.npy'))
print(f"  semantic: {semantic.shape}")

group_semantic_mean = np.zeros((7, 768))
group_semantic_consistency = np.zeros((7, 1))

for g in range(7):
    members = np.where((group_labels[valid_idx] == g) & filtered_mask[valid_idx])[0] + 1
    n_members = len(members)
    if n_members == 0:
        continue
    weights = individual_scores[members]
    weight_sum = weights.sum()
    if weight_sum < 1e-10:
        weights = np.ones(n_members) / n_members
    else:
        weights = weights / weight_sum
    group_semantic_mean[g] = np.average(semantic[members, :768], axis=0, weights=weights)
    group_semantic_consistency[g, 0] = np.average(semantic[members, -1], weights=weights)

del semantic
gc.collect()

# Build and save group feature matrices
print("\n=== Building Group Feature Matrices ===")

# group_features_compact.npy: shape=(8, 81)
group_features_compact = np.zeros((8, 81))
for g in range(7):
    group_features_compact[g + 1] = np.concatenate([
        group_structural[g],
        group_behavioral[g],
        group_tgan[g],
        group_semantic_consistency[g]
    ])
print(f"  group_features_compact: {group_features_compact.shape}")
np.save(os.path.join(OUTPUT_DIR, 'group_features_compact.npy'), group_features_compact)

# group_features_full.npy: shape=(8, 849)
group_features_full = np.zeros((8, 849))
for g in range(7):
    semantic_full = np.concatenate([group_semantic_mean[g], group_semantic_consistency[g]])
    group_features_full[g + 1] = np.concatenate([
        group_structural[g],
        group_behavioral[g],
        semantic_full,
        group_tgan[g]
    ])
print(f"  group_features_full: {group_features_full.shape}")
np.save(os.path.join(OUTPUT_DIR, 'group_features_full.npy'), group_features_full)

# Group stance labels by majority vote
print("\n=== Group Stance Labels (Majority Vote) ===")
group_stance_labels = np.full(8, -1, dtype=int)
for g in range(7):
    members = np.where(group_labels[valid_idx] == g)[0] + 1
    member_parties = party_labels[members]
    n_dem = int(np.sum(member_parties == 0))
    n_rep = int(np.sum(member_parties == 1))
    if n_dem >= n_rep:
        group_stance_labels[g + 1] = 0
    else:
        group_stance_labels[g + 1] = 1
    stance_str = "Democrat" if group_stance_labels[g + 1] == 0 else "Republican"
    print(f"  Group {g}: {stance_str} (Dem={n_dem}, Rep={n_rep})")
np.save(os.path.join(OUTPUT_DIR, 'group_labels.npy'), group_stance_labels)

# Free phase 1 memory
del group_features_compact, group_features_full
del group_structural, group_behavioral, group_tgan
del group_semantic_mean, group_semantic_consistency
gc.collect()

# ============================================================
# Phase 2: Edge mapping (memory-sensitive)
# ============================================================
print("\n=== Phase 2: Building Group Interactions ===")

edges_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'ml_twitter.csv'))
print(f"  Original edges: {len(edges_df)}")

# Vectorized filtering
u_arr = edges_df['u'].values.astype(int)
i_arr = edges_df['i'].values.astype(int)

valid_mask = (u_arr >= 1) & (u_arr <= 877) & (i_arr >= 1) & (i_arr <= 877)
g_u_all = np.where(valid_mask, group_labels[np.clip(u_arr, 0, 877)], -1)
g_i_all = np.where(valid_mask, group_labels[np.clip(i_arr, 0, 877)], -1)
group_valid_mask = valid_mask & (g_u_all >= 0) & (g_i_all >= 0)

valid_indices = np.where(group_valid_mask)[0]
n_group_edges = len(valid_indices)
print(f"  Valid group edges: {n_group_edges}")

# Build interaction matrix (7x7)
group_u_arr = g_u_all[valid_indices]
group_i_arr = g_i_all[valid_indices]
interaction_matrix = np.zeros((7, 7), dtype=int)
for src in range(7):
    for dst in range(7):
        interaction_matrix[src, dst] = int(np.sum((group_u_arr == src) & (group_i_arr == dst)))
np.save(os.path.join(OUTPUT_DIR, 'interaction_matrix.npy'), interaction_matrix)
print(f"  interaction_matrix: {interaction_matrix.shape}, total={interaction_matrix.sum()}")

# Save group edge list
group_edge_df = pd.DataFrame({
    'u': group_u_arr + 1,
    'i': group_i_arr + 1,
    'ts': edges_df['ts'].values[valid_indices],
    'label': edges_df['label'].values[valid_indices],
    'idx': np.arange(1, n_group_edges + 1)
})
group_edge_df.to_csv(os.path.join(OUTPUT_DIR, 'group_edge_list.csv'), index=False)
print(f"  group_edge_list.csv: {n_group_edges} rows saved")

# Get original edge indices for feature mapping
orig_edge_indices = edges_df['idx'].values[valid_indices].astype(int)

# Free dataframe
del edges_df, u_arr, i_arr, g_u_all, g_i_all, group_u_arr, group_i_arr
gc.collect()

# Build group edge features using memmap to avoid RAM spike
print("  Building group edge features (disk-backed)...")
edge_feat_path = os.path.join(OUTPUT_DIR, 'group_edge_features.npy')

# Use memory-mapped source
src_features = np.load(os.path.join(PROCESSED_DIR, 'ml_twitter.npy'), mmap_mode='r')
print(f"  Source edge features: {src_features.shape}, dtype={src_features.dtype}")

# Create output on disk directly via open_memmap (no RAM allocation)
from numpy.lib.format import open_memmap
out_shape = (n_group_edges + 1, 768)
out_mmap = open_memmap(edge_feat_path, mode='w+', dtype=np.float32, shape=out_shape)
# Index 0 is already zeros

# Write in chunks
CHUNK_SIZE = 5000
for start in range(0, n_group_edges, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, n_group_edges)
    chunk_idx = orig_edge_indices[start:end]
    valid_chunk = chunk_idx < src_features.shape[0]
    if valid_chunk.any():
        chunk_data = np.array(src_features[chunk_idx[valid_chunk]], dtype=np.float32)
        # Place into output at correct positions
        positions = np.arange(start + 1, end + 1)[valid_chunk]
        out_mmap[positions[0]:positions[-1]+1] = chunk_data
        del chunk_data
    if (start // CHUNK_SIZE) % 5 == 0:
        print(f"    Chunk {start//CHUNK_SIZE}: processed {end}/{n_group_edges}")

out_mmap.flush()
del out_mmap, src_features
gc.collect()

print(f"  group_edge_features.npy: shape={out_shape}")

# ============================================================
# Final verification
# ============================================================
print("\n=== Data Quality Verification ===")

compact = np.load(os.path.join(OUTPUT_DIR, 'group_features_compact.npy'))
full = np.load(os.path.join(OUTPUT_DIR, 'group_features_full.npy'))
labels = np.load(os.path.join(OUTPUT_DIR, 'group_labels.npy'))

print(f"  group_features_compact: {compact.shape}, NaN={np.isnan(compact).any()}")
print(f"  group_features_full: {full.shape}, NaN={np.isnan(full).any()}")
print(f"  group_labels: {labels}")
print(f"  interaction_matrix total: {interaction_matrix.sum()}")

print("\n=== Output Files Summary ===")
output_files = [
    'group_features_compact.npy',
    'group_features_full.npy',
    'group_edge_list.csv',
    'group_edge_features.npy',
    'interaction_matrix.npy',
    'group_labels.npy',
]
for fname in output_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {fname}: {size_mb:.1f} MB")

print("\nGroup feature integration complete!")
