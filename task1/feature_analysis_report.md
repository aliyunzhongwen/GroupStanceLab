# Feature Fusion & Analysis Report

## 1. Feature Dimension Summary

| Feature Type | Raw Dims | Preprocessing | Fused Dims |
|-------------|---------|--------------|-----------|
| Structural | 8 | StandardScaler | 8 |
| Behavioral | 8 | Already standardized | 8 |
| Semantic | 769 | - | 769 (768-dim mean embedding + 1 consistency) |
| TGAN | 64 | - | 64 |
| **Full fusion** | - | - | **849** |
| **Compact** | - | - | **81** (no 768-dim BERT embedding) |

- Full: `individual_features.npy`, shape=(878, 849)
- Compact: `individual_features_compact.npy`, shape=(878, 81)
- Note: index 0 is a zero-vector placeholder; 877 effective nodes

## 2. PCA Explained Variance Ratio

| PC | Explained Var | Cumulative |
|----|-------------|-----------|
| PC1 | 0.4368 (43.68%) | 0.4368 (43.68%) |
| PC2 | 0.1249 (12.49%) | 0.5616 (56.16%) |
| PC3 | 0.0756 (7.56%) | 0.6372 (63.72%) |
| PC4 | 0.0653 (6.53%) | 0.7025 (70.25%) |
| PC5 | 0.0406 (4.06%) | 0.7431 (74.31%) |
| PC6 | 0.0344 (3.44%) | 0.7775 (77.75%) |
| PC7 | 0.0264 (2.64%) | 0.8039 (80.39%) |
| PC8 | 0.0243 (2.43%) | 0.8283 (82.83%) |
| PC9 | 0.0207 (2.07%) | 0.8489 (84.89%) |
| PC10 | 0.0180 (1.80%) | 0.8670 (86.70%) |

Top-2 cumulative: 56.16%
Top-10 cumulative: 86.70%

**PCA cluster**: Inter-party center distance = 6.34, showing moderate separation.

## 3. Top Features by Stance Correlation

Point-Biserial correlation on 877 labeled nodes (0=Democrat, 1=Republican):

| Rank | Feature | r | p-value | Interpretation |
|------|---------|---|---------|---------------|
| 1 | struct_out_degree | -0.2138 | 0.000000*** | Negative -> Dem-leaning |
| 2 | struct_total_degree | -0.1556 | 0.000004*** | Negative -> Dem-leaning |
| 3 | struct_degree_centrality | -0.1556 | 0.000004*** | Negative -> Dem-leaning |
| 4 | struct_clustering_coefficient | -0.1378 | 0.000042*** | Negative -> Dem-leaning |
| 5 | behav_reply_ratio | -0.0876 | 0.009452** | Negative -> Dem-leaning |
| 6 | struct_in_degree | -0.0851 | 0.011732* | Negative -> Dem-leaning |
| 7 | struct_local_reaching_centrality | -0.0682 | 0.043565* | Negative -> Dem-leaning |
| 8 | struct_pagerank | +0.0594 | 0.078671 | Positive -> GOP-leaning |
| 9 | behav_mention_frequency | -0.0516 | 0.126710 | Negative -> Dem-leaning |
| 10 | behav_original_ratio | +0.0382 | 0.258409 | Positive -> GOP-leaning |
| 11 | behav_interaction_diversity | +0.0319 | 0.345266 | Positive -> GOP-leaning |
| 12 | behav_active_time_span | +0.0310 | 0.359868 | Positive -> GOP-leaning |
| 13 | behav_tweet_count | +0.0236 | 0.484345 | Positive -> GOP-leaning |
| 14 | behav_retweet_ratio | +0.0166 | 0.622444 | Positive -> GOP-leaning |
| 15 | semantic_consistency | -0.0129 | 0.703340 | Negative -> Dem-leaning |
| 16 | struct_betweenness_centrality | +0.0097 | 0.774363 | Positive -> GOP-leaning |
| 17 | behav_avg_tweet_length | -0.0077 | 0.818811 | Negative -> Dem-leaning |

> Significance: *** p<0.001, ** p<0.01, * p<0.05

**Top feature**: struct_out_degree (r=-0.2138)
- GOP-leaning: 
- Dem-leaning: struct_out_degree, struct_total_degree, struct_degree_centrality, struct_clustering_coefficient, behav_reply_ratio

## 4. t-SNE Clustering Observation

t-SNE on compact 81-d features (perplexity=30, random_state=42):

| Metric | Value |
|--------|-------|
| Democrat center | (13.97, -1.76) |
| Republican center | (-16.21, -0.75) |
| Inter-party distance | 30.20 |
| Democrat spread | 17.94 |
| Republican spread | 18.15 |
| Distance/spread ratio (Dem) | 1.68 |
| Distance/spread ratio (GOP) | 1.66 |

**Evaluation**: Distance/spread ratio = 1.66, indicating clear cluster separation between the two parties in t-SNE space. The Democrat group is more compact (spread: GOP=18.15 vs Dem=17.94).

## 5. Output Files

| File | Description |
|------|-------------|
| features/individual_features.npy | Full fused features (878x849) |
| features/individual_features_compact.npy | Compact features (878x81) |
| visualizations/pca_stance_2d.png | PCA 2D scatter plot |
| visualizations/tsne_stance_2d.png | t-SNE 2D scatter plot |
| visualizations/feature_label_correlation.png | Feature-label correlation Top-10 bar chart |
| feature_analysis_report.md | This report |
