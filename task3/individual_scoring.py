# -*- coding: utf-8 -*-
"""
Task 3 - Script 1: Individual Information Strength Scoring and Filtering
"""

import numpy as np
import torch
import os

# GPU/CPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path settings
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
FEATURE_DIR = os.path.join(BASE_DIR, 'task1/features')
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
OUTPUT_DIR = os.path.join(BASE_DIR, 'task3')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
behavioral_raw = np.load(os.path.join(FEATURE_DIR, 'behavioral_features_raw.npy'))
structural = np.load(os.path.join(FEATURE_DIR, 'structural_features.npy'))
semantic = np.load(os.path.join(FEATURE_DIR, 'semantic_features.npy'))
tgan = np.load(os.path.join(FEATURE_DIR, 'tgan_embeddings.npy'))
group_labels = np.load(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'))

print("Data loaded:")
print(f"  behavioral_raw: {behavioral_raw.shape}")
print(f"  structural: {structural.shape}")
print(f"  semantic: {semantic.shape}")
print(f"  tgan: {tgan.shape}")
print(f"  group_labels: {group_labels.shape}, unique: {np.unique(group_labels)}")

# Valid node indices (1-877)
valid_idx = np.arange(1, 878)

# ============================================================
# 1. Compute composite information strength score
# ============================================================

def standardize(x):
    """Standardize to [0,1] range"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-10:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

# Activity score = standardize(tweet_count * active_time_span)
# behavioral_raw: col0=tweet_count, col5=active_time_span
tweet_count = behavioral_raw[valid_idx, 0]
active_time_span = behavioral_raw[valid_idx, 5]
activity_raw = tweet_count * active_time_span
activity_score = standardize(activity_raw)

# Influence score = standardize(in_degree + pagerank*1000)
# structural: col0=in_degree, col4=pagerank
in_degree = structural[valid_idx, 0]
pagerank = structural[valid_idx, 4]
influence_raw = in_degree + pagerank * 1000
influence_score = standardize(influence_raw)

# Information score = standardize(semantic diversity)
# semantic diversity = 1 - semantic_consistency (last column)
semantic_consistency = semantic[valid_idx, -1]
semantic_diversity = 1 - semantic_consistency
info_score = standardize(semantic_diversity)

# Composite score = 0.4 * activity + 0.35 * influence + 0.25 * information
composite_score = 0.4 * activity_score + 0.35 * influence_score + 0.25 * info_score

print(f"\n=== Score Statistics ===")
print(f"Activity score: mean={activity_score.mean():.4f}, std={activity_score.std():.4f}")
print(f"Influence score: mean={influence_score.mean():.4f}, std={influence_score.std():.4f}")
print(f"Information score: mean={info_score.mean():.4f}, std={info_score.std():.4f}")
print(f"Composite score: mean={composite_score.mean():.4f}, std={composite_score.std():.4f}")

# Build full score array (878,), index 0 = 0
individual_scores = np.zeros(878)
individual_scores[valid_idx] = composite_score

# ============================================================
# 2. Filter core individuals per group (keep Top-60%, min 15 per group)
# ============================================================

filtered_mask = np.zeros(878, dtype=bool)
TOP_RATIO = 0.6
MIN_KEEP = 15

print(f"\n=== Group Filtering Results ===")
print(f"{'Group':<8}{'Before':<8}{'Kept':<8}{'Ratio':<10}")
print("-" * 36)

for g in range(7):
    # Find members of this group
    members = np.where(group_labels[valid_idx] == g)[0] + 1  # convert to global index
    n_members = len(members)
    
    # Sort by score
    scores = individual_scores[members]
    sorted_indices = np.argsort(-scores)  # descending
    
    # Keep Top-60%, but at least 15
    n_keep = max(int(n_members * TOP_RATIO), min(MIN_KEEP, n_members))
    kept_indices = members[sorted_indices[:n_keep]]
    
    filtered_mask[kept_indices] = True
    
    print(f"  {g:<8}{n_members:<8}{n_keep:<8}{n_keep/n_members:.2%}")

total_kept = filtered_mask.sum()
print(f"\nTotal kept: {total_kept}/877 ({total_kept/877:.2%})")

# ============================================================
# 3. Save outputs
# ============================================================

np.save(os.path.join(OUTPUT_DIR, 'individual_scores.npy'), individual_scores)
np.save(os.path.join(OUTPUT_DIR, 'filtered_mask.npy'), filtered_mask)

print(f"\n=== Output Files ===")
print(f"  individual_scores.npy: shape={individual_scores.shape}")
print(f"  filtered_mask.npy: shape={filtered_mask.shape}, True count={filtered_mask.sum()}")
print("\nIndividual scoring and filtering complete!")
