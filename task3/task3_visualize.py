# -*- coding: utf-8 -*-
"""
Task 3 - Script 3: Visualization
Generate score distribution, group feature heatmap, interaction matrix, filtering comparison.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import gc

print("Using device: cpu (visualization only)")

# Chinese font config
for fname in fm.findSystemFonts():
    try:
        name = fm.FontProperties(fname=fname).get_name()
        if any(x in name for x in ['Noto Sans CJK', 'WenQuanYi', 'SimHei']):
            plt.rcParams['font.sans-serif'] = [name]
            break
    except:
        continue
plt.rcParams['axes.unicode_minus'] = False

# Path settings
BASE_DIR = '/root/CORDGT/CorDGT/lab3/GroupStanceAnalysis'
TASK2_DIR = os.path.join(BASE_DIR, 'task2')
OUTPUT_DIR = os.path.join(BASE_DIR, 'task3')
VIS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

# Load data
individual_scores = np.load(os.path.join(OUTPUT_DIR, 'individual_scores.npy'))
filtered_mask = np.load(os.path.join(OUTPUT_DIR, 'filtered_mask.npy'))
group_features_compact = np.load(os.path.join(OUTPUT_DIR, 'group_features_compact.npy'))
interaction_matrix = np.load(os.path.join(OUTPUT_DIR, 'interaction_matrix.npy'))
group_labels = np.load(os.path.join(TASK2_DIR, 'group_assignments_balanced.npy'))
group_stance_labels = np.load(os.path.join(OUTPUT_DIR, 'group_labels.npy'))

valid_idx = np.arange(1, 878)
valid_scores = individual_scores[valid_idx]

print("Data loaded, generating visualizations...")

# ============================================================
# 1. Individual score distribution
# ============================================================
print("  Generating: individual_scores_distribution.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram
axes[0].hist(valid_scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
axes[0].axvline(valid_scores.mean(), color='red', linestyle='--', label=f'Mean={valid_scores.mean():.3f}')
axes[0].set_xlabel('Composite Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Individual Information Strength Score Distribution')
axes[0].legend()

# Right: boxplot per group
group_score_data = []
group_tick_labels = []
for g in range(7):
    members = np.where(group_labels[valid_idx] == g)[0]
    scores = valid_scores[members]
    group_score_data.append(scores)
    stance = 'D' if group_stance_labels[g + 1] == 0 else 'R'
    group_tick_labels.append(f'G{g}\n({stance})')

bp = axes[1].boxplot(group_score_data, labels=group_tick_labels, patch_artist=True)
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47', '#264478']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_xlabel('Group')
axes[1].set_ylabel('Composite Score')
axes[1].set_title('Group Information Strength Score Boxplot')

plt.suptitle('Individual Information Strength Score Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'individual_scores_distribution.png'), dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

# ============================================================
# 2. Group features heatmap
# ============================================================
print("  Generating: group_features_heatmap.png")

fig, ax = plt.subplots(figsize=(14, 5))

# Select key feature dimensions (structural 8 + behavioral 8 + tgan first 10 + consistency 1 = 27)
feature_labels = (
    [f'Struct_{i}' for i in range(8)] +
    [f'Behav_{i}' for i in range(8)] +
    [f'TGAN_{i}' for i in range(10)] +
    ['Sem_Consist']
)
display_cols = list(range(16)) + list(range(16, 26)) + [80]
display_data = group_features_compact[1:8, :][:, display_cols]

# Normalize columns for display (manual MinMax to avoid sklearn memory overhead)
col_min = display_data.min(axis=0)
col_max = display_data.max(axis=0)
col_range = col_max - col_min
col_range[col_range < 1e-10] = 1.0
display_data_norm = (display_data - col_min) / col_range

im = ax.imshow(display_data_norm, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(7))
ax.set_yticklabels([f'Group{g} ({"D" if group_stance_labels[g+1]==0 else "R"})' for g in range(7)])
ax.set_xticks(range(len(feature_labels)))
ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=8)
ax.set_title('Group Feature Profile Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Value')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'group_features_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

# ============================================================
# 3. Group interaction matrix
# ============================================================
print("  Generating: group_interaction_matrix.png")

fig, ax = plt.subplots(figsize=(8, 7))

log_matrix = np.log1p(interaction_matrix)
im = ax.imshow(log_matrix, cmap='Blues', aspect='equal')

ax.set_xticks(range(7))
ax.set_yticks(range(7))
ax.set_xticklabels([f'G{g}' for g in range(7)])
ax.set_yticklabels([f'G{g}' for g in range(7)])

for i in range(7):
    for j in range(7):
        val = int(interaction_matrix[i, j])
        color = 'white' if log_matrix[i, j] > log_matrix.max() * 0.6 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=8)

ax.set_xlabel('Target Group')
ax.set_ylabel('Source Group')
ax.set_title('Group Interaction Strength Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Interactions (log1p)')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'group_interaction_matrix.png'), dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

# ============================================================
# 4. Filtering comparison
# ============================================================
print("  Generating: filtering_comparison.png")

fig, ax = plt.subplots(figsize=(10, 6))

before_counts = []
after_counts = []
for g in range(7):
    members_all = np.sum(group_labels[valid_idx] == g)
    members_kept = np.sum(filtered_mask & (group_labels == g))
    before_counts.append(members_all)
    after_counts.append(members_kept)

x = np.arange(7)
width = 0.35

bars1 = ax.bar(x - width/2, before_counts, width, label='Before Filter', color='#4472C4', alpha=0.8)
bars2 = ax.bar(x + width/2, after_counts, width, label='After Filter', color='#ED7D31', alpha=0.8)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Group ID')
ax.set_ylabel('Count')
ax.set_title('Group Size Before/After Filtering', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'G{g}\n({"D" if group_stance_labels[g+1]==0 else "R"})' for g in range(7)])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'filtering_comparison.png'), dpi=300, bbox_inches='tight')
plt.close('all')
gc.collect()

print(f"\n=== Visualization Complete ===")
print(f"Output dir: {VIS_DIR}")
print(f"Generated files:")
print(f"  - individual_scores_distribution.png")
print(f"  - group_features_heatmap.png")
print(f"  - group_interaction_matrix.png")
print(f"  - filtering_comparison.png")
