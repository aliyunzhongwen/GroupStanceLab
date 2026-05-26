# Task 5: 群体立场分类实验规范

## 1. 实验目标

基于 Task 4 产出的时序嵌入、交互矩阵及立场分数，训练二分类模型预测各群体的政治立场（民主党 vs 共和党），并通过消融实验验证各特征块的贡献。

---

## 2. 数据来源

| 文件 | Shape | 说明 |
|------|-------|------|
| group_temporal_embeddings.npy | (7, 11, 64) | TGAN 群体嵌入，每群体每窗口 64 维 |
| interaction_windows.npy | (11, 7, 7) | 每窗口群体间交互强度矩阵 |
| temporal_stance_scores.npy | (7, 11) | 每群体每窗口立场分数 |
| influence_model_results.json | — | FJ 模型固执度 alpha (7,) |
| group_labels.npy | (8,) | 索引 0 = -1 占位，1-7 对应群体 0-6 |
| group_features_compact.npy | (8, 81) | 静态群体特征，索引 0 占位 |

---

## 3. 数据划分（30:12:12 窗口比）

| 集合 | 窗口索引 | 样本数 |
|------|---------|--------|
| 训练集 | W0~W5 (0-5) | 7 × 6 = 42 |
| 验证集 | W6~W7 (6-7) | 7 × 2 = 14 |
| 测试集 | W8~W10 (8-10) | 7 × 3 = 21 |

- 训练集数据增广：高斯噪声 ×5（noise_std=0.01），最终训练样本数 = 42 × 5 = 210
- 标签：group_labels[1:8] = [1, 0, 0, 1, 1, 0, 0]（0=民主党，1=共和党）
  - 民主党(0)：群体 1,2,5,6 → n_dem=4
  - 共和党(1)：群体 0,3,4 → n_rep=3

---

## 4. 特征维度表

### Block 1: 群体 TGAN 嵌入 (64d)
```
emb = group_temporal_embeddings[g, w, :]  # (64,)
```

### Block 2: 群体交互特征 (7d)
```
inter = interaction_windows[w, g, :]  # (7,) 与7个群体的交互强度
```

### Block 3: 时序变化特征 (3d)
```
vel   = stance_scores[g, w] - stance_scores[g, max(0, w-1)]  # 立场速度
vol   = std(stance_scores[g, 0:max(1, w)])                    # 历史波动性
alpha = fj_alpha[g]                                           # FJ 固执度
```

### Block 4: 静态群体特征 (81d)
```
static = group_features_compact[g+1, :]  # (81,)
```

### 特征组合

| 模型 | 特征块 | 维度 |
|------|--------|------|
| M0 | Block 4 (静态) | 81 |
| M1 | Block 1 (嵌入) | 64 |
| M2 | Block 1 + Block 2 + Block 3 | 64+7+3=74 |
| M3 | Block 1 + Block 2 + Block 3 + Block 4 | 74+81=155 |

---

## 5. 模型架构

```python
class StanceClassifier(nn.Module):
    fc_1: Linear(input_dim → 128)
    bn:   BatchNorm1d(128)
    act:  LeakyReLU()
    fc_2: Linear(128 → 64)
    fc_3: Linear(64 → 2)
    drop: Dropout(0.1)

    forward: x → drop(act(bn(fc_1(x)))) → drop(act(fc_2(x))) → fc_3(x)
```

---

## 6. 训练配置

| 超参数 | 值 |
|--------|-----|
| 优化器 | Adam, lr=1e-3, weight_decay=1e-5 |
| 调度器 | CosineAnnealingLR, T_max=100 |
| 损失函数 | CrossEntropyLoss (类别加权) |
| 早停 | patience=20，监控 val_F1（越大越好） |
| 随机种子 | 2023 |
| 最大轮数 | 300 |

类别权重：
- w_dem = 7 / (2 × 4) = 0.875
- w_rep = 7 / (2 × 3) = 1.167

---

## 7. 消融实验设计

| 实验 | 对比项 | 目的 |
|------|--------|------|
| E1 | M0(81d) vs M1(64d) | 静态特征 vs 动态嵌入 |
| E2 | M1(64d) vs M2(74d) | 嵌入 vs 嵌入+交互+时序 |
| E3 | M2(74d) vs M3(155d) | 核心特征 vs 完整特征 |
| E4 | M2无交互(67d) vs M2(74d) | 群体交互特征贡献 |
| E5 | M2无时序(71d) vs M2(74d) | 时序变化特征贡献 |
| E6 | 524过滤 vs 877全量 | 数据质量过滤效果 |

---

## 8. 评估指标

- **主要指标**：macro F1（类别不平衡下更公平）
- **辅助指标**：AUC-ROC，Accuracy
- **基准线**：随机分类 F1=0.50，多数类 F1≈0.571

---

## 9. 输出文件

```
task5/
├── results/
│   ├── model_checkpoints/     # 各模型最优权重 .pth
│   ├── visualizations/        # 可视化图表
│   ├── ablation_results.json  # 消融实验结果
│   └── evaluation_results.json# 主实验结果
├── models/
│   └── group_stance_classifier.py
├── data_builder.py
├── train_group_stance.py
├── ablation_study.py
├── task5_visualize.py
└── task5_analysis.md
```
