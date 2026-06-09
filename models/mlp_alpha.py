# -*- coding: utf-8 -*-
# @File    : mlp_alpha.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了MLP非线性Alpha模型的类定义。

网络结构: Linear(P, 64) → BN → Sigmoid → Linear(64, 64) → BN → Sigmoid → Linear(64, 1) → ReLU

关键设计：
- 前两层用Sigmoid（非ReLU）：因子zscore后≈正态分布均值0，BN后分布压在0附近，
  ReLU在x<0时梯度=0导致约50%神经元"死亡"，Sigmoid在0附近有非零梯度。
- 末层用ReLU：输出zscore，只关心相对排序，ReLU确保非负单调性。
- 必须返回第二层隐藏层输出（用于正交惩罚）。
"""

import torch
import torch.nn as nn

from config import N_HIDDEN


class MLPAlphaModel(nn.Module):
    """MLP非线性Alpha模型。

    三层MLP，前两层Sigmoid+BN激活，末层ReLU。
    forward返回(prediction, hidden_layer2_output)用于正交惩罚计算。

    Attributes:
        n_factors: 输入因子数量。
    """

    def __init__(self, n_factors: int, n_hidden: int = N_HIDDEN) -> None:
        """初始化MLP模型。

        Args:
            n_factors: 输入因子数量P。
            n_hidden: 隐藏层维度，默认64。
        """
        super().__init__()
        self.n_factors = n_hidden  # 用于正交惩罚维度校验

        # 第一层: P → 64
        self.fc1 = nn.Linear(n_factors, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)

        # 第二层: 64 → 64
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)

        # 输出层: 64 → 1
        self.fc3 = nn.Linear(n_hidden, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回预测值和第二层隐藏层输出。

        Args:
            x: 输入因子张量，shape=(N, P)。

        Returns:
            (prediction, hidden_layer2_output) 元组:
            - prediction: 预测zscore，shape=(N,)。
            - hidden_layer2_output: 第二隐藏层输出，shape=(N, 64)，
              用于计算正交惩罚。
        """
        # Layer 1: Linear → BN → Sigmoid
        h1 = self.sigmoid(self.bn1(self.fc1(x)))

        # Layer 2: Linear → BN → Sigmoid
        h2 = self.sigmoid(self.bn2(self.fc2(h1)))

        # Output: Linear → ReLU
        pred = self.relu(self.fc3(h2).squeeze(-1))

        return pred, h2
