# -*- coding: utf-8 -*-
# @File    : mlp_alpha.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 2.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了MLP非线性Alpha模型的类定义。

网络结构（v2.5+ 可变）:
    Linear(P, d1) -> Norm -> Act -> Dropout -> Linear(d1, d2) -> Norm -> Act -> ...
    -> Linear(d_last, 1)  [末层无激活，标签zscore有正负]

关键设计：
- 激活函数可选 Sigmoid/GELU/LeakyReLU，默认Sigmoid。
- 因子zscore后approx正态分布均值0，BN后分布压在0附近，
  ReLU在x<0时梯度=0导致约50%神经元"死亡"，Sigmoid在0附近有非零梯度。
- 末层不加激活函数：标签是截面zscore（均值approx 0，有正有负），
  加ReLU会强制预测>=0，导致无法拟合负标签，IC为负。
- 必须返回最后一层隐藏层输出（用于正交惩罚）。
- v2.5: 支持Dropout、可变隐藏维度序列、可选激活函数、LayerNorm。
"""

import torch
import torch.nn as nn

from config import (
    ACTIVATION, DROPOUT_RATE, HIDDEN_DIMS, N_HIDDEN, USE_LAYER_NORM,
)


def _get_activation(name: str) -> nn.Module:
    """根据名称获取激活函数模块。

    Args:
        name: 激活函数名称，"sigmoid"/"gelu"/"leaky_relu"。

    Returns:
        PyTorch激活函数模块。
    """
    if name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    else:
        return nn.Sigmoid()


def _get_norm(dim: int, use_ln: bool) -> nn.Module:
    """根据配置获取归一化层。

    Args:
        dim: 特征维度。
        use_ln: True=LayerNorm, False=BatchNorm1d。

    Returns:
        归一化模块。
    """
    if use_ln:
        return nn.LayerNorm(dim)
    return nn.BatchNorm1d(dim)


class MLPAlphaModel(nn.Module):
    """MLP非线性Alpha模型（v2.5 可变架构）。

    隐藏层数量和维度由 hidden_dims 控制。
    forward返回(prediction, last_hidden_output)用于正交惩罚计算。

    Attributes:
        n_factors: 输入因子数量。
        hidden_dims: 隐藏层维度元组，如 (64, 64) 或 (128, 64, 32)。
        n_hidden_out: 最后一层隐藏层输出维度（正交惩罚用）。
    """

    def __init__(
        self,
        n_factors: int,
        hidden_dims: tuple = HIDDEN_DIMS,
        dropout: float = DROPOUT_RATE,
        activation: str = ACTIVATION,
        use_layer_norm: bool = USE_LAYER_NORM,
    ) -> None:
        """初始化可变架构MLP模型。

        Args:
            n_factors: 输入因子数量P。
            hidden_dims: 隐藏层维度序列，默认(64, 64)对应2隐藏层。
            dropout: Dropout比例，0=不启用。
            activation: 激活函数名称。
            use_layer_norm: True=LayerNorm替代BatchNorm。
        """
        super().__init__()
        self.n_factors = n_factors
        self.hidden_dims = hidden_dims
        self.n_hidden_out = hidden_dims[-1]  # 最后一层维度（正交惩罚用）

        # 构建隐藏层
        layers = []
        in_dim = n_factors
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(_get_norm(h_dim, use_layer_norm))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)

        # 输出层: 最后一层隐藏维度 → 1 (无激活函数)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回预测值和最后一层隐藏层输出。

        Args:
            x: 输入因子张量，shape=(N, P)。

        Returns:
            (prediction, last_hidden_output) 元组:
            - prediction: 预测zscore，shape=(N,)。
            - last_hidden_output: 最后一隐藏层输出（正交惩罚用），
              shape=(N, hidden_dims[-1])。
        """
        hidden = self.hidden_layers(x)
        pred = self.output_layer(hidden).squeeze(-1)
        return pred, hidden

    @classmethod
    def from_legacy(
        cls, n_factors: int, n_hidden: int = N_HIDDEN,
    ) -> "MLPAlphaModel":
        """向后兼容：使用旧版 n_hidden 参数创建模型。

        Args:
            n_factors: 输入因子数量。
            n_hidden: 每层隐藏维度（2层等宽）。

        Returns:
            MLPAlphaModel 实例，hidden_dims=(n_hidden, n_hidden)。
        """
        return cls(n_factors, hidden_dims=(n_hidden, n_hidden))
