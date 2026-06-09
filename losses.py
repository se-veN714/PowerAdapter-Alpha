# -*- coding: utf-8 -*-
# @File    : losses.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了三种损失函数和正交惩罚项的函数定义。

包含三种损失函数（MSE/IC/CCC）和正交惩罚项。
总损失组合: total_loss = loss_fn(pred, target) + lambda_orth * orthogonal_penalty(hidden)
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """均方误差损失。

    基准损失函数，稳定但仅优化绝对偏差。

    Args:
        pred: 模型预测值，shape=(N,)。
        target: 真实标签值，shape=(N,)。

    Returns:
        MSE损失标量。
    """
    return F.mse_loss(pred, target)


def ic_loss(pred: Tensor, target: Tensor) -> Tensor:
    """IC损失 = -Pearson相关系数。

    直接优化排序能力，多头选股效果最好。
    非凸目标函数，收敛可能不稳定。

    Args:
        pred: 模型预测值，shape=(N,)。
        target: 真实标签值，shape=(N,)。

    Returns:
        -Pearson相关系数标量。
    """
    # 中心化
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()

    # Pearson相关系数
    pred_std = pred_centered.std()
    target_std = target_centered.std()

    # 防止除零
    eps = 1e-8
    if pred_std < eps or target_std < eps:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    correlation = (pred_centered * target_centered).mean() / (pred_std * target_std + eps)
    return -correlation  # 取负，最大化相关系数=最小化负相关系数


def ccc_loss(pred: Tensor, target: Tensor) -> Tensor:
    """CCC一致性相关系数损失。

    CCC = 2 * ρ * σ_pred * σ_target / (σ_pred² + σ_target² + (μ_pred - μ_target)²)
    结合MSE和IC的优点：同时优化相关性和绝对偏差。
    稳定性最好，论文推荐的首选损失。

    Args:
        pred: 模型预测值，shape=(N,)。
        target: 真实标签值，shape=(N,)。

    Returns:
        -CCC标量。
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()

    # Pearson相关系数
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    pred_std = torch.sqrt(pred_var + 1e-8)
    target_std = torch.sqrt(target_var + 1e-8)
    correlation = covariance / (pred_std * target_std + 1e-8)

    # CCC
    numerator = 2 * correlation * pred_std * target_std
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

    ccc = numerator / (denominator + 1e-8)
    return -ccc  # 取负，最大化CCC


def orthogonal_penalty(h: Tensor) -> Tensor:
    """正交惩罚项：惩罚隐藏层输出的非对角协方差。

    H: 隐藏层输出 (N, d)
    C = H^T @ H / N  # 协方差矩阵
    L_orth = ||C - diag(C)||_F  # 只惩罚非对角元素（协方差），保留对角（方差）

    设计原因：F范数||C||_F会同时惩罚方差和协方差，但方差是有用的信息量，
    所以只惩罚非对角元素，让学到的隐因子尽量不相关。

    Args:
        h: 第二层隐藏层输出，shape=(N, d)。

    Returns:
        正交惩罚标量。
    """
    n = h.shape[0]
    # 协方差矩阵
    cov_matrix = (h.T @ h) / n
    # 只保留非对角元素
    diagonal = torch.diag(torch.diag(cov_matrix))
    off_diagonal = cov_matrix - diagonal
    # F范数
    return torch.norm(off_diagonal, p="fro")


# 损失函数注册表
LOSS_FUNCTIONS: dict[str, type] = {
    "mse": type("MSELossFn", (), {"__call__": staticmethod(mse_loss)}),
    "ic": type("ICLossFn", (), {"__call__": staticmethod(ic_loss)}),
    "ccc": type("CCCLossFn", (), {"__call__": staticmethod(ccc_loss)}),
}


def get_loss_fn(name: str) -> type:
    """按名称获取损失函数。

    Args:
        name: 损失函数名称，"mse"/"ic"/"ccc"。

    Returns:
        对应的损失函数类。

    Raises:
        ValueError: 不支持的损失函数名称。
    """
    if name not in LOSS_FUNCTIONS:
        msg = f"不支持的损失函数: {name}，可选: {list(LOSS_FUNCTIONS.keys())}"
        raise ValueError(msg)
    return LOSS_FUNCTIONS[name]
