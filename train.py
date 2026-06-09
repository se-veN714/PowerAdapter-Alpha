# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了模型训练的完整流程，包括单次训练、滚动训练和训练曲线绘制。

支持单次训练和滚动训练两种模式。
- 早停：验证集loss连续patience个epoch不下降则停止。
- 保存最佳模型到checkpoints/。
- 每个epoch记录train_loss和val_loss。
- GPU训练：模型和数据均需.to(DEVICE)。
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR, DEVICE, FACTOR_COLS, LAMBDA_ORTH,
    LEARNING_RATE, LOG_DIR, MAX_EPOCHS, N_HIDDEN, OPTIMIZER, PATIENCE,
    PROCESSED_DATA_DIR, TRAIN_END, VAL_END,
)
from losses import ccc_loss, ic_loss, mse_loss, orthogonal_penalty
from models.linear_alpha import LinearAlphaModel
from models.mlp_alpha import MLPAlphaModel
from utils.dataset import FactorDataset, split_dataset
from utils.metrics import rank_ic


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: type,
    lambda_orth: float = LAMBDA_ORTH,
    is_mlp: bool = False,
) -> float:
    """训练一个epoch。

    Args:
        model: PyTorch模型。
        loader: 训练数据DataLoader。
        optimizer: 优化器。
        loss_fn: 损失函数。
        lambda_orth: 正交惩罚系数。
        is_mlp: 是否为MLP模型（需返回hidden层输出）。

    Returns:
        平均训练loss。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        if is_mlp:
            pred, hidden = model(x_batch)
            loss = loss_fn(pred, y_batch) + lambda_orth * orthogonal_penalty(hidden)
        else:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: type,
    lambda_orth: float = LAMBDA_ORTH,
    is_mlp: bool = False,
) -> tuple[float, float]:
    """在验证/测试集上评估。

    Args:
        model: PyTorch模型。
        loader: 数据DataLoader。
        loss_fn: 损失函数。
        lambda_orth: 正交惩罚系数。
        is_mlp: 是否为MLP模型。

    Returns:
        (平均loss, 平均RankIC) 元组。
    """
    model.eval()
    total_loss = 0.0
    ic_values: list[float] = []
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if is_mlp:
                pred, hidden = model(x_batch)
                loss = loss_fn(pred, y_batch) + lambda_orth * orthogonal_penalty(hidden)
            else:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

            # 计算RankIC（需转回CPU/NumPy）
            ic = rank_ic(pred.cpu().numpy(), y_batch.cpu().numpy())
            ic_values.append(ic)

    avg_loss = total_loss / max(n_batches, 1)
    avg_ic = np.mean(ic_values) if ic_values else 0.0
    return avg_loss, avg_ic


def train(
    model: nn.Module,
    train_ds: FactorDataset,
    val_ds: FactorDataset,
    loss_name: str = "mse",
    is_mlp: bool = False,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LEARNING_RATE,
    lambda_orth: float = LAMBDA_ORTH,
    save_name: str | None = None,
) -> dict[str, list[float]]:
    """完整训练流程，含早停和模型保存。

    Args:
        model: PyTorch模型。
        train_ds: 训练数据集。
        val_ds: 验证数据集。
        loss_name: 损失函数名称，"mse"/"ic"/"ccc"。
        is_mlp: 是否为MLP模型。
        max_epochs: 最大训练轮数。
        patience: 早停耐心值。
        lr: 学习率。
        lambda_orth: 正交惩罚系数。
        save_name: 模型保存名称。

    Returns:
        训练历史字典，含train_loss, val_loss, val_ic列表。
    """
    # 损失函数映射
    loss_fn_map = {"mse": mse_loss, "ic": ic_loss, "ccc": ccc_loss}
    loss_fn = loss_fn_map[loss_name]

    # 模型移至GPU
    model = model.to(DEVICE)

    # 优化器
    optimizer_map = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    optimizer = optimizer_map[OPTIMIZER](model.parameters(), lr=lr)

    # DataLoader（batch_size=1因为每个截面就是一个完整batch）
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 训练历史
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_ic": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"训练开始 | 设备: {DEVICE} | 模型: {model.__class__.__name__} | 损失: {loss_name}")
    print("-" * 70)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, lambda_orth, is_mlp)
        val_loss, val_ic = evaluate(model, val_loader, loss_fn, lambda_orth, is_mlp)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(val_ic)

        elapsed = time.time() - t0

        # 每10个epoch打印一次
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{max_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_ic={val_ic:.4f} | "
                f"time={elapsed:.1f}s"
            )

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            if save_name is None:
                save_name = f"{model.__class__.__name__}_{loss_name}"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = CHECKPOINT_DIR / f"{save_name}_best.pt"
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发: 验证loss连续{patience}个epoch未下降")
                break

    print("-" * 70)
    print(f"训练完成 | 最佳val_loss={best_val_loss:.6f}")

    return history


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: Path | None = None,
) -> None:
    """绘制训练曲线。

    Args:
        history: 训练历史字典。
        title: 图表标题。
        save_path: 图片保存路径，None则显示。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss曲线
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IC曲线
    ax2.plot(epochs, history["val_ic"], label="Val RankIC", color="green", linewidth=1.5)
    ax2.axhline(y=0.03, color="red", linestyle="--", alpha=0.5, label="IC=0.03 threshold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RankIC")
    ax2.set_title("Validation RankIC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存至 {save_path}")
    else:
        plt.show()
    plt.close()


def run_training(
    model_type: str = "mlp",
    loss_name: str = "mse",
) -> None:
    """运行单次训练的便捷入口。

    Args:
        model_type: 模型类型，"mlp"或"linear"。
        loss_name: 损失函数名称。
    """
    # 加载预处理数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("请先运行 preprocess.py 生成预处理数据")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])
    train_ds, val_ds, test_ds = split_dataset(df)

    # 获取因子数
    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    n_factors = len(available_factors)
    print(f"可用因子数: {n_factors}")

    # 创建模型
    is_mlp = model_type == "mlp"
    if is_mlp:
        model: nn.Module = MLPAlphaModel(n_factors)
    else:
        model = LinearAlphaModel(n_factors)

    # 训练
    history = train(
        model, train_ds, val_ds,
        loss_name=loss_name,
        is_mlp=is_mlp,
        save_name=f"{model_type}_{loss_name}",
    )

    # 保存训练历史
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOG_DIR / f"{model_type}_{loss_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 绘制曲线
    plot_path = LOG_DIR / f"{model_type}_{loss_name}_curve.png"
    plot_training_history(history, title=f"{model_type.upper()} + {loss_name.upper()}", save_path=plot_path)


def rolling_train(
    model_type: str = "mlp",
    loss_name: str = "mse",
    n_windows: int = 3,
) -> None:
    """滚动训练（扩展窗口）。

    训练集用所有历史数据，验证集取最后252个交易日，
    测试集为后续半年。每轮滚动向前推进一步。

    Args:
        model_type: 模型类型。
        loss_name: 损失函数名称。
        n_windows: 滚动窗口数量。
    """
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("请先运行 preprocess.py 生成预处理数据")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])
    all_dates = sorted(df["date"].unique())

    # 至少需要252+126个交易日
    min_dates = 252 + 126
    if len(all_dates) < min_dates:
        print(f"数据不足: 需要至少{min_dates}个交易日，当前{len(all_dates)}个")
        return

    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    n_factors = len(available_factors)
    is_mlp = model_type == "mlp"

    results: list[dict[str, float]] = []

    for w in range(n_windows):
        # 滚动窗口划分
        val_start_idx = len(all_dates) - (n_windows - w) * 126
        val_start = all_dates[max(val_start_idx, 252)]
        test_start_idx = val_start_idx + 126
        test_start = all_dates[min(test_start_idx, len(all_dates) - 1)]

        train_df = df[df["date"] < val_start].copy()
        val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
        test_df = df[df["date"] >= test_start].copy()

        if len(train_df) < 100 or len(val_df) < 20:
            print(f"窗口{w}: 数据不足，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"滚动窗口 {w+1}/{n_windows}")
        print(f"训练: ~{train_df['date'].min().date()} → {train_df['date'].max().date()} ({len(train_df)}行)")
        print(f"验证: {val_df['date'].min().date()} → {val_df['date'].max().date()} ({len(val_df)}行)")
        print(f"测试: {test_df['date'].min().date()} → {test_df['date'].max().date()} ({len(test_df)}行)")

        train_ds = FactorDataset(train_df, available_factors)
        val_ds = FactorDataset(val_df, available_factors)
        test_ds = FactorDataset(test_df, available_factors)

        # 创建新模型
        if is_mlp:
            model: nn.Module = MLPAlphaModel(n_factors)
        else:
            model = LinearAlphaModel(n_factors)

        history = train(
            model, train_ds, val_ds,
            loss_name=loss_name,
            is_mlp=is_mlp,
            save_name=f"{model_type}_{loss_name}_rolling{w}",
        )

        # 加载最佳模型评估测试集
        ckpt_path = CHECKPOINT_DIR / f"{model_type}_{loss_name}_rolling{w}_best.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        loss_fn_map = {"mse": mse_loss, "ic": ic_loss, "ccc": ccc_loss}
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        test_loss, test_ic = evaluate(model, test_loader, loss_fn_map[loss_name], is_mlp=is_mlp)

        results.append({"window": w, "test_loss": test_loss, "test_ic": test_ic})
        print(f"测试集: loss={test_loss:.6f}, RankIC={test_ic:.4f}")

    # 汇总
    if results:
        print(f"\n{'='*50}")
        print("滚动训练汇总:")
        for r in results:
            print(f"  窗口{r['window']}: loss={r['test_loss']:.6f}, RankIC={r['test_ic']:.4f}")
        avg_ic = np.mean([r["test_ic"] for r in results])
        print(f"  平均RankIC: {avg_ic:.4f}")


if __name__ == "__main__":
    # 单次训练
    run_training(model_type="mlp", loss_name="mse")
    # 滚动训练（取消注释启用）
    # rolling_train(model_type="mlp", loss_name="mse", n_windows=3)
