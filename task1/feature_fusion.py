#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Fusion & Analysis (Task 6)
"""

import os
import json
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# Config
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(BASE_DIR, 'features')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

# 配置中文字体
from matplotlib import font_manager
found = False
for font_name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']:
    fonts = font_manager.findSystemFonts()
    for f in fonts:
        try:
            if font_name.lower().replace(' ', '') in f.lower().replace(' ', ''):
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
                found = True
                break
        except:
            continue
    if found:
        break
# 若路径匹配失败，尝试通过字体名称匹配
if not found:
    for f in font_manager.fontManager.ttflist:
        if 'Noto Sans CJK' in f.name or 'WenQuanYi' in f.name or 'SimHei' in f.name:
            plt.rcParams['font.family'] = f.name
            break
plt.rcParams['axes.unicode_minus'] = False
DPI = 300

STANCE_COLORS = {0: '#2166AC', 1: '#B2182B', -1: '#BEBEBE'}
STANCE_NAMES = {0: '民主党', 1: '共和党', -1: '未标记'}

# ============================================================
# 1. Load feature matrices
# ============================================================
print("=" * 60)
print("1. Loading feature matrices")
print("=" * 60)

structural = np.load(os.path.join(FEAT_DIR, 'structural_features.npy'))
behavioral = np.load(os.path.join(FEAT_DIR, 'behavioral_features.npy'))
semantic = np.load(os.path.join(FEAT_DIR, 'semantic_features.npy'))
tgan = np.load(os.path.join(FEAT_DIR, 'tgan_embeddings.npy'))
consistency = np.load(os.path.join(FEAT_DIR, 'semantic_consistency.npy'))

with open(os.path.join(FEAT_DIR, 'structural_feature_names.json'), 'r') as f:
    struct_names = json.load(f)
with open(os.path.join(FEAT_DIR, 'behavioral_feature_names.json'), 'r') as f:
    behav_names = json.load(f)

labels = np.load(os.path.join(BASE_DIR, '..', '..', 'processed', 'ml_twitter_node_labels.npy'))

print(f"  Structural:    {structural.shape}")
print(f"  Behavioral:    {behavioral.shape}")
print(f"  Semantic:      {semantic.shape} (768-dim mean embedding + 1 consistency)")
print(f"  TGAN:          {tgan.shape}")
print(f"  Consistency:   {consistency.shape}")
label_counts = dict(zip(*np.unique(labels, return_counts=True)))
print(f"  Labels:        {labels.shape}, distribution: {label_counts}")

# ============================================================
# 2. Feature fusion
# ============================================================
print("\n" + "=" * 60)
print("2. Feature fusion")
print("=" * 60)

# Standardize structural features (behavioral already standardized)
scaler = StandardScaler()
structural_scaled = scaler.fit_transform(structural)
print("  Structural features: StandardScaler applied")

# Full fusion: structural(8) + behavioral(8) + semantic(769) + tgan(64) = 849
full_features = np.hstack([structural_scaled, behavioral, semantic, tgan])
print(f"  Full features:  {full_features.shape}")

# Compact: structural(8) + behavioral(8) + tgan(64) + consistency(1) = 81
compact_features = np.hstack([
    structural_scaled,
    behavioral,
    tgan,
    consistency.reshape(-1, 1)
])
print(f"  Compact features: {compact_features.shape}")

# Save
np.save(os.path.join(FEAT_DIR, 'individual_features.npy'), full_features)
np.save(os.path.join(FEAT_DIR, 'individual_features_compact.npy'), compact_features)
print("  Saved individual_features.npy and individual_features_compact.npy")

# Build feature name lists
compact_names = struct_names + behav_names + [f'tgan_{i}' for i in range(64)] + ['semantic_consistency']
low_dim_names = struct_names + behav_names

# ============================================================
# 3. PCA visualization
# ============================================================
print("\n" + "=" * 60)
print("3. PCA visualization (full features)")
print("=" * 60)

pca = PCA(n_components=10)
pca_result = pca.fit_transform(full_features)
explained = pca.explained_variance_ratio_
print(f"  Top-10 explained variance: {explained}")
print(f"  Top-2 cumulative: {explained[:2].sum():.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
for stance_val in [-1, 0, 1]:
    mask = labels == stance_val
    ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
               c=STANCE_COLORS[stance_val], label=STANCE_NAMES[stance_val],
               alpha=0.6 if stance_val != -1 else 0.3,
               s=20 if stance_val != -1 else 10,
               zorder=3 if stance_val != -1 else 1)

ax.set_xlabel(f'主成分1 ({explained[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'主成分2 ({explained[1]*100:.1f}%)', fontsize=12)
ax.set_title('PCA降维可视化（按政治立场着色）', fontsize=14)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, 'pca_stance_2d.png'), dpi=DPI)
plt.close(fig)
print("  Saved pca_stance_2d.png")

# ============================================================
# 4. t-SNE visualization
# ============================================================
print("\n" + "=" * 60)
print("4. t-SNE visualization (compact features)")
print("=" * 60)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_result = tsne.fit_transform(compact_features)
print(f"  t-SNE done: {tsne_result.shape}")

fig, ax = plt.subplots(figsize=(10, 8))
for stance_val in [-1, 0, 1]:
    mask = labels == stance_val
    ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
               c=STANCE_COLORS[stance_val], label=STANCE_NAMES[stance_val],
               alpha=0.6 if stance_val != -1 else 0.3,
               s=20 if stance_val != -1 else 10,
               zorder=3 if stance_val != -1 else 1)

ax.set_xlabel('t-SNE维度1', fontsize=12)
ax.set_ylabel('t-SNE维度2', fontsize=12)
ax.set_title('t-SNE降维可视化（按政治立场着色）', fontsize=14)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, 'tsne_stance_2d.png'), dpi=DPI)
plt.close(fig)
print("  Saved tsne_stance_2d.png")

# ============================================================
# 5. Feature-label correlation analysis
# ============================================================
print("\n" + "=" * 60)
print("5. Feature-label correlation (structural + behavioral)")
print("=" * 60)

labeled_mask = labels != -1
labeled_labels = labels[labeled_mask]

correlations = {}
p_values = {}

# Structural features (standardized)
for i, name in enumerate(struct_names):
    feat_vals = structural_scaled[labeled_mask, i]
    r, p = stats.pointbiserialr(labeled_labels, feat_vals)
    correlations[f'struct_{name}'] = (r, p)
    p_values[f'struct_{name}'] = p

# Behavioral features
for i, name in enumerate(behav_names):
    feat_vals = behavioral[labeled_mask, i]
    r, p = stats.pointbiserialr(labeled_labels, feat_vals)
    correlations[f'behav_{name}'] = (r, p)
    p_values[f'behav_{name}'] = p

# Semantic consistency
consist_vals = consistency[labeled_mask]
r, p = stats.pointbiserialr(labeled_labels, consist_vals)
correlations['semantic_consistency'] = (r, p)
p_values['semantic_consistency'] = p

# Sort by absolute correlation
sorted_feats = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

print("\n  Point-Biserial correlation (sorted by |r|):")
print(f"  {'Rank':<5} {'Feature':<35} {'r':>8} {'p-value':>12} {'Sig':>5}")
print("  " + "-" * 70)
for rank, (name, (r_val, p_val)) in enumerate(sorted_feats, 1):
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    print(f"  {rank:<5} {name:<35} {r_val:>+8.4f} {p_val:>12.6f} {sig:>5}")

# Top-10 bar chart
top10 = sorted_feats[:10]
top10_names = [item[0] for item in top10]
top10_r = [item[1][0] for item in top10]
top10_colors = ['#B2182B' if r > 0 else '#2166AC' for r in top10_r]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(top10_names)), top10_r, color=top10_colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(top10_names)))
ax.set_yticklabels(top10_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('相关系数', fontsize=12)
ax.set_ylabel('特征名称', fontsize=12)
ax.set_title('特征与立场标签相关性（Top-10）', fontsize=14)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, axis='x', alpha=0.3)

for i, (r_val, bar) in enumerate(zip(top10_r, bars)):
    ax.text(r_val + (0.01 if r_val >= 0 else -0.01), i, f'{r_val:.3f}',
            va='center', ha='left' if r_val >= 0 else 'right', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, 'feature_label_correlation.png'), dpi=DPI)
plt.close(fig)
print("\n  Saved feature_label_correlation.png")

# ============================================================
# 6. Generate analysis report
# ============================================================
print("\n" + "=" * 60)
print("6. Generating analysis report")
print("=" * 60)

# t-SNE cluster metrics
dem_mask = labels == 0
rep_mask = labels == 1
dem_center = tsne_result[dem_mask].mean(axis=0)
rep_center = tsne_result[rep_mask].mean(axis=0)
cluster_dist = np.linalg.norm(dem_center - rep_center)
dem_spread = np.mean(np.linalg.norm(tsne_result[dem_mask] - dem_center, axis=1))
rep_spread = np.mean(np.linalg.norm(tsne_result[rep_mask] - rep_center, axis=1))

# PCA cluster metrics
dem_pca_center = pca_result[dem_mask, :2].mean(axis=0)
rep_pca_center = pca_result[rep_mask, :2].mean(axis=0)
pca_cluster_dist = np.linalg.norm(dem_pca_center - rep_pca_center)

labeled_count = int(labeled_mask.sum())

report_lines = []
report_lines.append("# Feature Fusion & Analysis Report")
report_lines.append("")
report_lines.append("## 1. Feature Dimension Summary")
report_lines.append("")
report_lines.append("| Feature Type | Raw Dims | Preprocessing | Fused Dims |")
report_lines.append("|-------------|---------|--------------|-----------|")
report_lines.append("| Structural | 8 | StandardScaler | 8 |")
report_lines.append("| Behavioral | 8 | Already standardized | 8 |")
report_lines.append("| Semantic | 769 | - | 769 (768-dim mean embedding + 1 consistency) |")
report_lines.append("| TGAN | 64 | - | 64 |")
report_lines.append("| **Full fusion** | - | - | **849** |")
report_lines.append("| **Compact** | - | - | **81** (no 768-dim BERT embedding) |")
report_lines.append("")
report_lines.append("- Full: `individual_features.npy`, shape=(878, 849)")
report_lines.append("- Compact: `individual_features_compact.npy`, shape=(878, 81)")
report_lines.append("- Note: index 0 is a zero-vector placeholder; 877 effective nodes")
report_lines.append("")
report_lines.append("## 2. PCA Explained Variance Ratio")
report_lines.append("")
report_lines.append("| PC | Explained Var | Cumulative |")
report_lines.append("|----|-------------|-----------|")

cum_var = 0
for i in range(10):
    cum_var += explained[i]
    report_lines.append(f"| PC{i+1} | {explained[i]:.4f} ({explained[i]*100:.2f}%) | {cum_var:.4f} ({cum_var*100:.2f}%) |")

report_lines.append("")
report_lines.append(f"Top-2 cumulative: {explained[:2].sum()*100:.2f}%")
report_lines.append(f"Top-10 cumulative: {explained[:10].sum()*100:.2f}%")
report_lines.append("")

pca_separation = "moderate separation" if pca_cluster_dist > 5 else "low separation"
report_lines.append(f"**PCA cluster**: Inter-party center distance = {pca_cluster_dist:.2f}, showing {pca_separation}.")
report_lines.append("")
report_lines.append("## 3. Top Features by Stance Correlation")
report_lines.append("")
report_lines.append(f"Point-Biserial correlation on {labeled_count} labeled nodes (0=Democrat, 1=Republican):")
report_lines.append("")
report_lines.append("| Rank | Feature | r | p-value | Interpretation |")
report_lines.append("|------|---------|---|---------|---------------|")

for rank, (name, (r_val, p_val)) in enumerate(sorted_feats, 1):
    direction = "Positive -> GOP-leaning" if r_val > 0 else "Negative -> Dem-leaning"
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    report_lines.append(f"| {rank} | {name} | {r_val:+.4f} | {p_val:.6f}{sig} | {direction} |")

report_lines.append("")
report_lines.append("> Significance: *** p<0.001, ** p<0.01, * p<0.05")
report_lines.append("")

top_feat = sorted_feats[0]
report_lines.append(f"**Top feature**: {top_feat[0]} (r={top_feat[1][0]:+.4f})")
gop_feats = [n for n, (r, _) in sorted_feats[:5] if r > 0]
dem_feats = [n for n, (r, _) in sorted_feats[:5] if r < 0]
report_lines.append(f"- GOP-leaning: {', '.join(gop_feats)}")
report_lines.append(f"- Dem-leaning: {', '.join(dem_feats)}")
report_lines.append("")
report_lines.append("## 4. t-SNE Clustering Observation")
report_lines.append("")
report_lines.append("t-SNE on compact 81-d features (perplexity=30, random_state=42):")
report_lines.append("")
report_lines.append("| Metric | Value |")
report_lines.append("|--------|-------|")
report_lines.append(f"| Democrat center | ({dem_center[0]:.2f}, {dem_center[1]:.2f}) |")
report_lines.append(f"| Republican center | ({rep_center[0]:.2f}, {rep_center[1]:.2f}) |")
report_lines.append(f"| Inter-party distance | {cluster_dist:.2f} |")
report_lines.append(f"| Democrat spread | {dem_spread:.2f} |")
report_lines.append(f"| Republican spread | {rep_spread:.2f} |")
report_lines.append(f"| Distance/spread ratio (Dem) | {cluster_dist/dem_spread:.2f} |")
report_lines.append(f"| Distance/spread ratio (GOP) | {cluster_dist/rep_spread:.2f} |")
report_lines.append("")

ratio = cluster_dist / max(dem_spread, rep_spread)
if ratio > 1:
    tsne_eval = "clear cluster separation between the two parties in t-SNE space"
else:
    tsne_eval = "partial overlap between the two parties in t-SNE space"
tighter = "Republican" if rep_spread < dem_spread else "Democrat"
report_lines.append(f"**Evaluation**: Distance/spread ratio = {ratio:.2f}, indicating {tsne_eval}. The {tighter} group is more compact (spread: GOP={rep_spread:.2f} vs Dem={dem_spread:.2f}).")
report_lines.append("")
report_lines.append("## 5. Output Files")
report_lines.append("")
report_lines.append("| File | Description |")
report_lines.append("|------|-------------|")
report_lines.append("| features/individual_features.npy | Full fused features (878x849) |")
report_lines.append("| features/individual_features_compact.npy | Compact features (878x81) |")
report_lines.append("| visualizations/pca_stance_2d.png | PCA 2D scatter plot |")
report_lines.append("| visualizations/tsne_stance_2d.png | t-SNE 2D scatter plot |")
report_lines.append("| visualizations/feature_label_correlation.png | Feature-label correlation Top-10 bar chart |")
report_lines.append("| feature_analysis_report.md | This report |")

with open(os.path.join(BASE_DIR, 'feature_analysis_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines) + '\n')
print("  Saved feature_analysis_report.md")

print("\n" + "=" * 60)
print("Task 6 complete!")
print("=" * 60)
