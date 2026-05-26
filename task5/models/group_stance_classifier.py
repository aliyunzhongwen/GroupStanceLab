# -*- coding: utf-8 -*-
"""
models/group_stance_classifier.py
群体立场分类器 — 与 learn_stance_node.py StanceClassifier 完全一致的结构
"""

import torch
import torch.nn as nn


class StanceClassifier(nn.Module):
    """
    二分类立场分类器（民主党 vs 共和党）。
    结构与 learn_stance_node.py 中 StanceClassifier 完全一致：
        Linear(in → 128) → BN → LeakyReLU → Dropout
        Linear(128 → 64) → LeakyReLU → Dropout
        Linear(64 → 2)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.bn   = nn.BatchNorm1d(128)
        self.act  = nn.LeakyReLU()
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.1)

        # 权重初始化（与原脚本一致）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, 2)
        """
        x = self.drop(self.act(self.bn(self.fc_1(x))))
        x = self.drop(self.act(self.fc_2(x)))
        return self.fc_3(x)


# ─────────────────────── 快速测试 ───────────────────────
if __name__ == '__main__':
    for dim in [64, 74, 81, 155]:
        model = StanceClassifier(dim)
        x = torch.randn(8, dim)
        out = model(x)
        print(f"input_dim={dim:3d}  output={out.shape}  params={sum(p.numel() for p in model.parameters())}")
