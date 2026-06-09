# -*- coding: utf-8 -*-
# @File    : linear_alpha.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了线性基准Alpha模型的类定义。

网络结构: Linear(P, 64) → BN → Linear(64, 6) → BN → Linear(6, 1)
损失: MSE + L2正则

经济含义：
- 第一层(64维): 学习到64个"细分因子"
- 第二层(6维): 学习到6个"大类因子组合"
- 输出层: 预测zscore
"""

import torch
import torch.nn as nn

from config import N_HIDDEN


class LinearAlphaModel(nn.Module):
    """线性基准Alpha模型。

    使用BatchNorm但不使用非线性激活函数，保持线性结构。
    作为MLP非线性模型的对照组。

    Attributes:
        n_factors: 输入因子数量。
    """

    def __init__(self, n_factors: int, n_hidden: int = N_HIDDEN) -> None:
        """初始化线性基准模型。

        Args:
            n_factors: 输入因子数量P。
            n_hidden: 第一隐藏层维度，默认64。
        """
        super().__init__()
        self.n_factors = n_factors

        self.net = nn.Sequential(
            nn.Linear(n_factors, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(n_hidden, 6),
            nn.BatchNorm1d(6),
            nn.Linear(6, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入因子张量，shape=(N, P)。

        Returns:
            预测zscore，shape=(N,)。
        """
        return self.net(x).squeeze(-1)
