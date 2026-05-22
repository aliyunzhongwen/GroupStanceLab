# Task 4 群体立场时序演化分析 — 任务规格说明

## 1. 任务目标

Task 4 承接 Task 3 构建的 **群体超个体** 数据集，以 7 个群体超个体作为节点，对其开展完整的**时序立场演化分析**。核心理念是：将 Task 3 输出的 7 个"超级个体"视作一个微型社会网络，利用 TGAN 在群体级交互数据上学习时序动态表示，从而追踪每个群体在 11 个时间窗口内的立场演变轨迹，并通过多维分析方法揭示群体间交互与立场极化的动力学机制。

具体目标：

1. **时序立场追踪**：在群体超个体上训练 TGAN，获取 (7, 11) 时序立场分数矩阵；
2. **交互-演化关联分析**：定量检验群体间交互频度与立场变化之间的因果-相关关系；
3. **演化模式分类**：将所有群体对的演化行为归入趋同/极化/传染/稳定四类；
4. **影响力传播建模**：对比 DeGroot、Friedkin-Johnsen、GNN 三种经典传播模型的预测精度；
5. **Granger 因果分析**：检验哪些群体间交互序列在时序层面 Granger 引起立场变化；
6. **动态网络演化分析**：计算每个时间窗口的网络拓扑指标，检测突变点。

---

## 2. 输入数据

所有输入数据均来自 Task 3 输出目录 `../task3/` 及原始数据。

| 文件 | Shape | 说明 |
|---|:---:|---|
| `../task3/group_features_compact.npy` | (8, 81) | 群体超个体紧凑特征（节点特征），索引 0 占位 |
| `../task3/group_edge_features.npy` | (114945, 768) | 群体级边的 768 维 BERT 特征，索引 0 占位 |
| `../task3/group_labels.npy` | (8,) | 群体立场标签，`[-1,1,0,0,1,1,0,0]`，0=民主党，1=共和党 |
| `../task3/group_edge_list.csv` | (114944, 5) | 群体级有向边表，列 `u,i,ts,label,idx` |
| `../task3/interaction_matrix.npy` | (7, 7) | 全局群体间交互强度矩阵（整型） |
| `../task3/group_features_full.npy` | (8, 849) | 完整版超个体特征（含 BERT 768 维） |

### 数据基本统计

- 时间步范围：ts ∈ [0, 54]，共 55 个时间步
- 时间窗口划分：11 个窗口，每窗口含 5 个 ts（窗口 w 对应 ts ∈ [w×5, w×5+4]）
- 群体构成：共和党 3 个（g0, g3, g4），民主党 4 个（g1, g2, g5, g6）
- 总边数：114,944 条（继承原始个体级交互全量）

---

## 3. 方法概述

### 3.1 时序立场追踪

**脚本**：`temporal_stance_tracking.py`

将 Task 3 输出的群体超个体数据集直接灌入 TGAN 框架进行立场分类训练。核心步骤：

1. **数据加载**：读取 `group_features_compact.npy`（节点特征）、`group_edge_features.npy`（边特征）、`group_edge_list.csv`（边序列），构建 `NeighborFinder` 双向邻接表；
2. **TGAN 初始化**：`d_model=64`，`n_layers=2`，`n_heads=2`，`num_neighbors=[20,1]`，配合轻量 3 层 `StanceClassifier`（64→32→2 维）；
3. **节点分类训练**：100 epoch，遍历所有 55 个时间步（ts 0-54），在每步的 7 个群体节点上做立场分类，类别加权（民主党 `w₀=0.875`，共和党 `w₁=1.167`），梯度裁剪 `max_norm=1.0`；
4. **时序提取**：训练完成后，对每个时间窗口 w 的截止时刻 `cut_time = w×5+4`，提取 7 个群体的 64 维嵌入向量，分类头输出 `P(共和党)` 即为立场分数；
5. **输出**：`temporal_stance_scores.npy`（shape=(7,11)）和 `group_temporal_embeddings.npy`（shape=(7,11,64)）。

### 3.2 交互-演化关联分析

**脚本**：`interaction_evolution_analysis.py`

- **时间窗口化**：将边表按窗口聚合为 `interaction_windows`，shape=(11,7,7)；
- **立场变化量**：`Δstance[g][t] = stance[g][t+1] - stance[g][t]`，shape=(7,10)；
- **逐对相关性**：对所有 42 个有向对 (A→B, A≠B)，计算 A→B 交互序列与 B 立场变化序列的 Pearson + Spearman 相关系数；
- **假设检验**：
  - **H1**（立场趋同假设）：跨立场交互是否导致目标群体立场趋同；
  - **H2**（回音室假设）：同立场交互是否强化立场极化。

### 3.3 演化模式分类

**脚本**：`evolution_pattern_classification.py`

对总交互量 ≥ 100 的群体对，按立场差距序列 `diff[t] = |stance_A[t] - stance_B[t]|` 的线性回归斜率与显著性分类：

| 模式 | 判定条件 |
|---|---|
| 趋同型（Convergence） | slope < 0 且 p < 0.05，差距显著减小 |
| 极化型（Polarization） | slope > 0 且 p < 0.05，差距显著增大 |
| 传染型（Contagion） | p ≥ 0.05 但交互相关性显著（p_corr < 0.05） |
| 稳定型（Stability） | 其他情况（无显著趋势变化） |

### 3.4 影响力传播建模

**脚本**：`influence_modeling.py`

对比三种经典传播模型，用前 t 个窗口预测第 t+1 个窗口的立场分数：

- **DeGroot 模型**：`x_i(t+1) = Σⱼ W[i,j] × x_j(t)`，权重矩阵由交互矩阵归一化得到，不保留初始锚点；
- **Friedkin-Johnsen (FJ) 模型**：`x_i(t+1) = (1-α_i) × x_i(0) + α_i × Σⱼ W[i,j] × x_j(t)`，`α_i`（固执度参数）通过 PyTorch 在训练集上学习；
- **GNN 演化预测模型**：三层图卷积 + LSTM 时序建模，在训练窗口上学习图动力学。

评价指标：MAE（平均绝对误差）、RMSE（均方根误差）。

### 3.5 Granger 因果分析

**脚本**：`granger_causality.py`

对所有 42 个有向群体对 (A→B)，使用 `statsmodels.grangercausalitytests`，以交互量序列 `iw[:, A, B]` 检验是否 Granger-cause 目标群体立场序列 `stance[B]`（maxlag=2，F-test，显著性阈值 p < 0.05）。汇总因果网络密度、同党/跨党因果对数。

### 3.6 动态网络演化分析

**脚本**：`dynamic_network_analysis.py`

对每个时间窗口构建 NetworkX 有向加权图，计算 8 项拓扑指标：

| 指标 | 说明 |
|---|---|
| 网络密度（density） | 实际边数 / 可能最大边数 |
| 总权重（total_weight） | 当前窗口总交互量 |
| 平均加权度（avg_weighted_degree） | 加权度之和 / 节点数 |
| 互惠性（reciprocity） | 双向边比例 |
| 聚类系数（clustering） | 局部传递性 |
| 强连通分量数（n_strongly_connected） | 网络连通性 |
| 度异配性（degree_assortativity） | 高度节点是否倾向于连接高度节点 |
| 模块度（modularity） | 社区结构强度 |

突变点检测：对每个指标序列，计算相邻窗口的差分，超过 1.5σ 阈值则标记为突变窗口。

### 3.7 评价指标体系

**脚本**：`evolution_metrics.py`

综合计算以下指标：

| 指标 | 计算方法 |
|---|---|
| E-I Index（极化指数） | `EI = (外部交互 - 内部交互) / 总交互`，范围 [-1, +1] |
| 极小化分数（merging_score） | 同立场对：`0.5 × cosine_sim + 0.5 × density_norm` |
| 极大化分数（splitting_score） | 跨立场对：`0.5 × sparsity + 0.5 × feature_diff` |
| 稳定性指数 | 各群体立场方差，越小越稳定 |

---

## 4. 输出文件清单

| 文件 | Shape / 大小 | 说明 |
|---|---|---|
| `results/temporal_stance_scores.npy` | (7, 11) float32 | 7群体×11窗口时序立场分数，P(共和党) |
| `results/group_temporal_embeddings.npy` | (7, 11, 64) float32 | 群体时序嵌入向量 |
| `results/interaction_windows.npy` | (11, 7, 7) float64 | 按窗口统计的群体间有向交互量 |
| `results/interaction_evolution_correlation.json` | ~18KB | 42对相关性分析结果 + 假设检验 |
| `results/evolution_patterns.json` | ~23KB | 38对演化模式分类结果及统计 |
| `results/influence_model_results.json` | ~13KB | DeGroot/FJ/GNN 三模型预测结果与误差 |
| `results/granger_results.json` | ~12KB | 42对 Granger 因果检验结果 |
| `results/dynamic_network_stats.json` | ~8KB | 11窗口拓扑指标 + 核-外围结构 |
| `results/change_points.json` | ~10KB | 各指标序列的突变点列表 |
| `results/evolution_metrics.json` | ~16KB | EI指数、极小化/极大化、稳定性综合指标 |
| `results/group_polarization_metrics.json` | ~2KB | EI极化指数时序详情 |
| `results/group_merging_metrics.json` | ~4KB | 同立场群体对合并分数 |
| `results/group_splitting_metrics.json` | ~5KB | 跨立场群体对分裂分数 |
| `visualizations/evolution_heatmap.png` | ~82KB | 群体立场时序热力图（7×11） |
| `visualizations/evolution_trajectory.png` | ~79KB | 群体立场轨迹折线图 |
| `visualizations/polarization_timeline.png` | ~89KB | EI极化指数时序图 |
| `visualizations/interaction_evolution_coupling.png` | ~294KB | 交互-演化耦合分析图 |
| `visualizations/stance_violin_plot.png` | ~52KB | 群体立场分布小提琴图 |
| `visualizations/causal_influence_network.png` | ~99KB | 因果影响力网络图 |
| `visualizations/firework_diagram.png` | ~192KB | 群体演化烟花图 |

---

## 5. 技术依赖

| 包 | 版本要求 | 用途 |
|---|---|---|
| `torch` | ≥ 1.9.1 | TGAN训练、GNN模型 |
| `numpy` | ≥ 1.21 | 数值计算 |
| `pandas` | ≥ 1.3 | 边表读取 |
| `scipy` | ≥ 1.7 | 相关性分析、线性回归 |
| `statsmodels` | ≥ 0.13 | Granger因果检验 |
| `networkx` | ≥ 2.6 | 动态图构建与分析 |
| `matplotlib` | ≥ 3.4 | 可视化 |
| `sklearn` | ≥ 0.24 | 归一化、余弦相似度 |

---

## 6. 运行指南

### 执行顺序

Task 4 各脚本存在依赖关系，须按以下顺序执行：

```bash
cd /root/CORDGT/CorDGT/lab3/GroupStanceAnalysis/task4

# 步骤1: 训练TGAN，生成时序立场分数和嵌入
python temporal_stance_tracking.py
# 输出: results/temporal_stance_scores.npy, results/group_temporal_embeddings.npy

# 步骤2: 统计时间窗口交互（interaction_tracking.py 内置，或由 dynamic_network_analysis.py 生成）
python interaction_evolution_analysis.py
# 输出: results/interaction_windows.npy, results/interaction_evolution_correlation.json

# 步骤3: 动态网络演化分析
python dynamic_network_analysis.py
# 输出: results/dynamic_network_stats.json, results/change_points.json

# 步骤4: 演化模式分类
python evolution_pattern_classification.py
# 输出: results/evolution_patterns.json

# 步骤5: 影响力传播建模
python influence_modeling.py
# 输出: results/influence_model_results.json

# 步骤6: Granger因果分析
python granger_causality.py
# 输出: results/granger_results.json

# 步骤7: 综合评价指标
python evolution_metrics.py
# 输出: results/evolution_metrics.json, results/group_polarization_metrics.json
#        results/group_merging_metrics.json, results/group_splitting_metrics.json

# 步骤8: 生成可视化图表
python task4_visualize.py
# 输出: visualizations/*.png
```

### 依赖关系说明

- `evolution_pattern_classification.py` 依赖 `temporal_stance_tracking.py` 和 `interaction_evolution_analysis.py` 的输出；
- `influence_modeling.py` 和 `granger_causality.py` 依赖 `temporal_stance_tracking.py` 和 `interaction_evolution_analysis.py` 的输出；
- `evolution_metrics.py` 依赖所有前序步骤的输出；
- `task4_visualize.py` 依赖所有结果文件。

### 预估运行时间

| 脚本 | 预估时长 | 备注 |
|---|---|---|
| `temporal_stance_tracking.py` | ~10-20 分钟 | 100 epoch × 55 ts，CPU 模式 |
| `interaction_evolution_analysis.py` | ~3-5 分钟 | 含边表聚合 |
| 其余脚本 | <1 分钟/个 | 快速统计计算 |
