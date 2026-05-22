# 群体立场分析实验计划

## 实验总目标
基于CorDGT个体立场数据集（877个Twitter政治用户，114,944条交互），实现从个体立场到群体立场的分析框架，包含群体划分、个体筛选、群体演化分析（Evolution Analysis）和群体立场分类四个核心阶段。

## Task 1：数据探索与个体特征工程
### 目标
深入分析数据集特性，构建多维度的个体特征表示
### 输入
- processed/ 目录下的所有数据文件
- 877个用户CSV文件
### 工作内容
1. 数据EDA：统计个体活跃度分布、交互网络拓扑特征、时间分布
   - 绘制小提琴图展示个体活跃度分布（比箱线图更直观地展示数据密度）
   - 生成词云图展示整体高频用词和议题分布
2. 构建个体结构特征：度中心性、PageRank、介数中心性、聚类系数
3. 构建个体行为特征：推文数、转发率、回复率、活跃时间段、交互多样性
4. 构建个体语义特征：利用已有BERT嵌入计算语义统计量
5. 利用已有TGAN模型提取个体嵌入表示（64维）
6. 特征融合：将结构+行为+语义+TGAN嵌入整合为统一的个体画像向量
### 输出
- data_exploration.ipynb / data_exploration.py：数据探索脚本和可视化
- individual_features.npy：个体多维度特征矩阵
- feature_analysis_report.md：特征分析报告
### 预计时间
3-4天

## Task 2：群体划分
### 目标
利用个体特征和交互信息将877个用户划分为若干有意义的群体
### 输入
- Task 1 输出的个体特征矩阵
- 交互网络（ml_twitter.csv）
### 工作内容
1. 构建静态交互图（忽略时间维度，聚合所有边）
2. 实现多种社区发现算法：
   - Louvain 社区检测
   - Label Propagation
   - Spectral Clustering
3. 实现基于特征的聚类：K-means, GMM
4. 实现混合方法：结合图结构和特征相似度
5. 群体质量评估：模块度、NMI（以政党标签为参考）、群体内立场纯度
6. 选择最优群体划分方案
### 输出
- community_detection.py：群体划分算法实现
- group_assignments.npy：每个用户的群体标签
- group_evaluation.md：各方法对比评估报告
### 预计时间
3-4天
### 依赖
Task 1

## Task 3：强信息个体筛选与群体特征整合
### 目标
从每个群体中筛选高信息量核心个体，并将群体整合为特征集合（"群体画像"），使其格式兼容个体立场检测方法，便于后续直接复用TGAN等模型做群体立场预测
### 输入
- Task 1 输出的个体特征矩阵（individual_features.npy, individual_features_compact.npy）
- Task 2 输出的群体划分结果（group_assignments_balanced.npy）
- 交互网络数据（ml_twitter.csv）
- 结构特征（structural_features.npy）、行为特征（behavioral_features.npy）、语义特征（semantic_features.npy）、TGAN嵌入（tgan_embeddings.npy）
### 工作内容
1. 个体信息强度评分：
   - 活跃度指标：推文数量、交互频率、活跃时间跨度
   - 影响力指标：入度（被转发/回复次数）、PageRank
   - 信息量指标：推文语义多样性、立场表达强度（BERT特征方差）
   - 综合评分：多指标标准化后加权求和
2. 强信息个体筛选：
   - 每群体保留Top-K核心成员（确保每群体≥10人）
   - 自适应阈值筛选（基于综合评分分布的肘部法则）
   - 验证筛选后群体完整性和立场分布平衡
3. **群体特征整合（核心步骤）**：
   - 均值聚合：群体内筛选后个体各维度特征的加权平均（按信息强度加权）
   - 构建群体级特征向量：结构特征聚合 + 行为特征聚合 + 语义特征聚合 + TGAN嵌入聚合
   - 构建群体间交互特征：统计群体间的交互边，生成群体级"伪边表"
   - 格式化为个体立场方法的输入格式：
     - 群体节点特征矩阵（类似ml_twitter_node.npy）
     - 群体间交互边表（类似ml_twitter.csv）
     - 群体间边特征（聚合的语义嵌入）
     - 群体立场标签（多数投票法）
4. 数据质量验证：确保群体级数据格式与原始个体数据兼容
### 输出
- individual_scoring.py：个体评分与筛选
- group_feature_integration.py：群体特征整合
- filtered_individuals.npy：筛选后的个体列表及评分
- group_features_matrix.npy：群体特征矩阵（7×D维）
- group_edge_list.csv：群体间交互边表
- group_edge_features.npy：群体间边的语义特征
- group_labels.npy：群体立场标签
- task3_analysis.md：分析报告
### 预计时间
3-4天
### 依赖
Task 1, Task 2

## Task 4：群体特征构建与演化分析
### 目标
构建群体级别的特征表示，分析群体立场在时间维度上因群体间交互影响而产生的动态演化过程
### 输入
- Task 2 输出的群体划分
- Task 3 输出的筛选后个体
- 原始交互数据和特征
### 工作内容
1. 群体特征聚合：
   - 均值池化：组内个体嵌入的简单平均
   - 注意力池化：学习个体权重的加权聚合
   - 图池化方法探索
2. 构建群体交互图：
   - 以群体为超节点
   - 计算群体间交互强度（边权重）
   - 提取群体间交互的语义特征
3. 时序立场追踪(Temporal Stance Tracking)：
   - 将55个时间切片划分为多个时间窗口（如每5个切片为一个窗口，共11个窗口）
   - 利用TGAN时态嵌入，通过分类器得到softmax立场概率作为连续立场强度分数
   - 按群体聚合，追踪群体立场强度随时间的变化轨迹
4. 交互-演化关联分析(Interaction-Evolution Correlation)：
   - 统计时间窗口t内群体A→B的交互量（转发+回复）
   - 计算时间窗口t+1内群体B的立场变化量
   - 分析交互量与立场变化的Pearson/Spearman相关性
   - 区分同立场交互 vs 跨立场交互的不同效果
   - 假设验证：跨立场交互→立场趋同？同立场交互→回音室强化？
5. 群体演化模式分类：
   - 趋同型(Convergence)、极化型(Polarization)、稳定型(Stability)、传染型(Contagion)
   - 对每对群体交互进行演化模式分类
6. 影响力传播建模：
   - DeGroot模型：x_i(t+1) = Σ_j w_ij * x_j(t)
   - Friedkin-Johnsen模型：x_i(t+1) = α_i * s_i + (1-α_i) * Σ_j w_ij * x_j(t)
   - 基于GNN的演化预测：输入t时刻特征+交互图，预测t+1立场（MSE损失）
7. Granger因果分析：
   - 检验群体间交互序列是否Granger-causes立场变化序列
   - 构建因果影响网络，识别核心影响群体
8. 动态网络演化分析：
   - 按时间窗口切片得到时序图序列
   - 分析网络拓扑变化（密度、路径长度、社区结构）
   - 检测交互模式突变点，分析突变前后立场变化
9. 可视化方案：
   - 演化轨迹图(Evolution Trajectory)、烟花图(Firework Diagram)、演化热力图(Evolution Heatmap)
   - 交互-演化联动图、因果影响网络图、小提琴图、词云图
10. 演化分析评价指标：
    - 群体极化指数（E-I Index时序）
    - 极小化指标（群体合并合理性）、极大化指标（群体分裂合理性）
    - 演化预测准确率（MAE/RMSE）
    - 因果显著性（p<0.05的因果对占比）
    - 立场稳定性指数（时序方差）
11. 关键发现总结：群体间的交互模式如何影响立场演化
### 输出
- group_features.py：群体特征构建
- evolution_analysis.py：演化分析实现（时序追踪、关联分析、模式分类）
- influence_modeling.py：影响力传播建模（DeGroot、Friedkin-Johnsen、GNN）
- granger_causality.py：Granger因果分析与因果网络构建
- group_interaction_graph.pkl：群体交互图数据
- visualizations/：可视化图表目录（演化轨迹图、烟花图、热力图、因果网络图等）
- evolution_metrics.json：演化分析评价指标结果
- group_polarization_metrics.json：群体极化指标结果
- group_merging_metrics.json：极小化指标结果
- group_splitting_metrics.json：极大化指标结果
- evolution_analysis_report.md：演化分析报告
### 预计时间
5-7天
### 依赖
Task 2, Task 3

## Task 5：群体立场分类网络
### 目标
设计并训练能够识别群体立场的神经网络
### 输入
- Task 4 输出的群体特征和群体交互图
- 群体立场标签（由组成个体的标签聚合得到）
### 工作内容
1. 群体立场标签生成：
   - 多数投票法
   - 加权投票法（按影响力加权）
   - 分析群体内立场纯度分布
2. 模型设计：
   - 基线模型：MLP分类器 + 群体特征
   - 图模型：GCN/GAT 作用在群体交互图上
   - 时态模型：结合时间动态的群体立场网络
   - 端到端模型：整合个体筛选+群体聚合+群体分类的联合训练
3. 训练与评估：
   - 数据集划分策略（群体级别）
   - 训练循环实现
   - Accuracy, F1, AUC 评估
   - 与个体立场分类的性能对比
4. 消融实验：
   - 不同群体划分方法的影响
   - 个体筛选的贡献
   - 演化分析特征的作用
   - 不同聚合方式的对比
### 输出
- group_stance_model.py：群体立场分类模型
- train_group_stance.py：训练脚本
- model_checkpoints/：模型权重
- experiment_results.md：实验结果报告
### 预计时间
5-7天
### 依赖
Task 4

## Task 6：实验总结与论文素材
### 目标
整理实验结果，生成论文级别的图表和分析
### 工作内容
1. 整理所有实验结果表格
2. 生成论文级别的可视化图表
3. 撰写实验分析和结论
4. 总结创新点和不足
### 输出
- final_report.md：完整实验报告
- paper_figures/：论文图表
### 预计时间
2-3天
### 依赖
Task 5

## 时间线总览

| Task | 内容 | 预计时间 | 依赖 |
|------|------|----------|------|
| Task 1 | 数据探索与个体特征工程 | 3-4天 | 无 |
| Task 2 | 群体划分 | 3-4天 | Task 1 |
| Task 3 | 强信息个体筛选与群体特征整合 | 3-4天 | Task 1, 2 |
| Task 4 | 群体特征与演化分析 | 5-7天 | Task 2, 3 |
| Task 5 | 群体立场分类网络 | 5-7天 | Task 4 |
| Task 6 | 实验总结与论文素材 | 2-3天 | Task 5 |
| **总计** | | **19-26天** | |

## 技术栈
- Python 3.8+
- PyTorch 1.x / 2.x
- PyTorch Geometric (图神经网络)
- NetworkX (图分析)
- scikit-learn (聚类、评估)
- community (Louvain)
- matplotlib / seaborn / plotly (可视化)
- transformers / sentence-transformers (BERT)
