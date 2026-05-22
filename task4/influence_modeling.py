"""
影响力传播建模
实现三种经典模型，对比预测效果：
1. DeGroot模型
2. Friedkin-Johnsen模型
3. GNN演化预测模型
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(BASE_DIR)


def load_data():
    stance_scores = np.load(
        os.path.join(BASE_DIR, 'results', 'temporal_stance_scores.npy')
    )  # (7, 11)
    interaction_windows = np.load(
        os.path.join(BASE_DIR, 'results', 'interaction_windows.npy')
    )  # (11, 7, 7)
    interaction_matrix = np.load(
        os.path.join(LAB_DIR, 'task3', 'interaction_matrix.npy')
    )  # (7, 7)
    group_features = np.load(
        os.path.join(LAB_DIR, 'task3', 'group_features_compact.npy')
    )  # (8, 81), 索引1-7对应群体0-6
    group_labels = np.load(
        os.path.join(LAB_DIR, 'task3', 'group_labels.npy')
    )  # (8,)
    return stance_scores, interaction_windows, interaction_matrix, group_features, group_labels


def build_weight_matrix(interaction_matrix):
    """
    构建归一化权重矩阵W (7×7):
    W[i,j] = interaction_matrix[i,j] / sum_j(interaction_matrix[i,j])
    即：j对i的影响权重 = i从j收到的交互占i总收到交互的比例
    """
    W = interaction_matrix.astype(np.float64).copy()
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    W = W / row_sums
    # 对角线置零（不考虑自影响）
    np.fill_diagonal(W, 0)
    # 重新归一化
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


# ==================== DeGroot模型 ====================

def degroot_model(stance_scores, W):
    """
    DeGroot模型: x_i(t+1) = Σ_j W[i,j] * x_j(t)
    用stance[:,0:10]预测stance[:,1:11]
    """
    n_groups, n_windows = stance_scores.shape
    predictions = np.zeros_like(stance_scores[:, 1:])
    # W是(7,7), stance[:,t]是(7,), 所以 W @ stance[:,t] = (7,)
    for t in range(n_windows - 1):
        predictions[:, t] = W @ stance_scores[:, t]

    actual = stance_scores[:, 1:]
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))

    return predictions, actual, mae, rmse


# ==================== Friedkin-Johnsen模型 ====================

def friedkin_johnsen_model(stance_scores, W):
    """
    Friedkin-Johnsen模型: x_i(t+1) = α_i * s_i + (1-α_i) * Σ_j W[i,j] * x_j(t)
    α_i = 固执系数，反映群体立场的稳定性
    s_i = stance[i, 0]（初始立场作为固有立场）
    """
    n_groups, n_windows = stance_scores.shape

    # 估计α: 方差越小越固执
    var_stance = np.var(stance_scores, axis=1)  # (7,)
    max_var = var_stance.max()
    alpha = 1.0 - var_stance / max_var  # 方差越小，α越大（越固执）
    alpha = np.clip(alpha, 0.05, 0.95)  # 避免极端值

    s = stance_scores[:, 0]  # 初始立场 (7,)
    predictions = np.zeros_like(stance_scores[:, 1:])

    for t in range(n_windows - 1):
        social_influence = W @ stance_scores[:, t]  # (7,)
        predictions[:, t] = alpha * s + (1 - alpha) * social_influence

    actual = stance_scores[:, 1:]
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))

    return predictions, actual, mae, rmse, alpha


# ==================== GNN模型 ====================

class StanceGCN(nn.Module):
    """简单的2层GCN预测下一时刻立场"""

    def __init__(self, in_dim, hidden_dim=32, out_dim=1):
        super(StanceGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        """
        x: (n_groups, in_dim) 节点特征
        adj: (n_groups, n_groups) 归一化邻接矩阵
        """
        # 第一层图卷积
        h = torch.relu(adj @ x @ self.fc1.weight.T + self.fc1.bias)
        # 第二层图卷积
        out = adj @ h @ self.fc2.weight.T + self.fc2.bias
        return out.squeeze(-1)


def gnn_model(stance_scores, interaction_windows, group_features_compact, W):
    """
    GNN演化预测模型
    用窗口0-8预测1-9（训练），窗口9预测10（测试）
    """
    n_groups, n_windows = stance_scores.shape

    # 准备节点特征: 当前立场(1维) + 群体特征(81维) = 82维
    # group_features_compact索引1-7对应群体0-6
    g_feat = group_features_compact[1:]  # (7, 81)
    stance_feat = stance_scores.T  # (11, 7)

    # 构建特征矩阵: 每个时刻 (7, 82)
    # adj: 归一化交互矩阵
    adj = torch.FloatTensor(W)

    # 训练数据: 窗口0-8预测1-9
    train_x = []
    train_y = []
    for t in range(9):
        feat = np.column_stack([
            stance_scores[:, t].reshape(-1, 1),  # 当前立场 (7, 1)
            g_feat  # 群体特征 (7, 81)
        ])  # (7, 82)
        train_x.append(torch.FloatTensor(feat))
        train_y.append(torch.FloatTensor(stance_scores[:, t + 1]))

    # 测试数据: 窗口9预测10
    test_feat = np.column_stack([
        stance_scores[:, 9].reshape(-1, 1),
        g_feat
    ])
    test_x = torch.FloatTensor(test_feat)
    test_y = torch.FloatTensor(stance_scores[:, 10])

    # 训练
    model = StanceGCN(in_dim=82, hidden_dim=32, out_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(100):
        total_loss = 0
        for i in range(len(train_x)):
            optimizer.zero_grad()
            pred = model(train_x[i], adj)
            loss = criterion(pred, train_y[i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100, Loss: {total_loss/len(train_x):.6f}")

    # 评估（在所有窗口上）
    model.eval()
    all_predictions = []
    all_actual = []

    with torch.no_grad():
        for t in range(n_windows - 1):
            feat = np.column_stack([
                stance_scores[:, t].reshape(-1, 1),
                g_feat
            ])
            x = torch.FloatTensor(feat)
            pred = model(x, adj).numpy()
            all_predictions.append(pred)
            all_actual.append(stance_scores[:, t + 1])

    predictions = np.array(all_predictions).T  # (7, 10)
    actual = np.array(all_actual).T  # (7, 10)

    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))

    # 测试集单独评估
    with torch.no_grad():
        test_pred = model(test_x, adj).numpy()
    test_mae = np.mean(np.abs(test_pred - test_y.numpy()))
    test_rmse = np.sqrt(np.mean((test_pred - test_y.numpy()) ** 2))

    return predictions, actual, mae, rmse, test_mae, test_rmse


def print_comparison(degroot_mae, degroot_rmse,
                     fj_mae, fj_rmse, fj_alpha,
                     gnn_mae, gnn_rmse, gnn_test_mae, gnn_test_rmse):
    """打印模型对比表"""
    print("\n" + "=" * 70)
    print("影响力传播建模 — 模型对比")
    print("=" * 70)

    print("\n【全时序预测效果（窗口0-9预测1-10）】")
    print("-" * 50)
    print(f"{'模型':<25} {'MAE':>10} {'RMSE':>10}")
    print("-" * 50)
    print(f"{'DeGroot':<25} {degroot_mae:>10.4f} {degroot_rmse:>10.4f}")
    print(f"{'Friedkin-Johnsen':<25} {fj_mae:>10.4f} {fj_rmse:>10.4f}")
    print(f"{'GNN (StanceGCN)':<25} {gnn_mae:>10.4f} {gnn_rmse:>10.4f}")
    print("-" * 50)

    print(f"\n【GNN测试集效果（窗口9预测10）】")
    print(f"  MAE:  {gnn_test_mae:.4f}")
    print(f"  RMSE: {gnn_test_rmse:.4f}")

    print(f"\n【Friedkin-Johnsen固执系数α】")
    print(f"  α = {fj_alpha}")
    for i, a in enumerate(fj_alpha):
        party = "共和党" if a > 0.5 else "民主党(倾向)"
        print(f"  群体{i}: α={a:.4f} ({party})")


def main():
    stance_scores, interaction_windows, interaction_matrix, group_features, group_labels = load_data()
    print(f"数据加载完成:")
    print(f"  stance_scores: {stance_scores.shape}")
    print(f"  interaction_windows: {interaction_windows.shape}")
    print(f"  interaction_matrix: {interaction_matrix.shape}")
    print(f"  group_features: {group_features.shape}")

    # 构建权重矩阵
    W = build_weight_matrix(interaction_matrix)
    print(f"\n归一化权重矩阵W:\n{np.array2string(W, precision=3, suppress_small=True)}")

    # 1. DeGroot模型
    print("\n" + "-" * 40)
    print("1. DeGroot模型")
    print("-" * 40)
    deg_pred, deg_actual, deg_mae, deg_rmse = degroot_model(stance_scores, W)
    print(f"  MAE:  {deg_mae:.4f}")
    print(f"  RMSE: {deg_rmse:.4f}")
    print(f"  样本预测 vs 真实 (群体0, 窗口1-3): "
          f"pred={deg_pred[0, :3]}, actual={deg_actual[0, :3]}")

    # 2. Friedkin-Johnsen模型
    print("\n" + "-" * 40)
    print("2. Friedkin-Johnsen模型")
    print("-" * 40)
    fj_pred, fj_actual, fj_mae, fj_rmse, fj_alpha = friedkin_johnsen_model(stance_scores, W)
    print(f"  固执系数α: {fj_alpha}")
    print(f"  MAE:  {fj_mae:.4f}")
    print(f"  RMSE: {fj_rmse:.4f}")

    # 3. GNN模型
    print("\n" + "-" * 40)
    print("3. GNN演化预测模型")
    print("-" * 40)
    gnn_pred, gnn_actual, gnn_mae, gnn_rmse, gnn_test_mae, gnn_test_rmse = \
        gnn_model(stance_scores, interaction_windows, group_features, W)

    # 打印对比
    print_comparison(deg_mae, deg_rmse, fj_mae, fj_rmse, fj_alpha,
                     gnn_mae, gnn_rmse, gnn_test_mae, gnn_test_rmse)

    # 保存结果
    output = {
        'degroot': {
            'mae': float(deg_mae),
            'rmse': float(deg_rmse),
            'predictions': deg_pred.tolist(),
            'actual': deg_actual.tolist()
        },
        'friedkin_johnsen': {
            'mae': float(fj_mae),
            'rmse': float(fj_rmse),
            'alpha': fj_alpha.tolist(),
            'predictions': fj_pred.tolist(),
            'actual': fj_actual.tolist()
        },
        'gnn': {
            'mae': float(gnn_mae),
            'rmse': float(gnn_rmse),
            'test_mae': float(gnn_test_mae),
            'test_rmse': float(gnn_test_rmse),
            'predictions': gnn_pred.tolist(),
            'actual': gnn_actual.tolist()
        }
    }

    output_path = os.path.join(BASE_DIR, 'results', 'influence_model_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
