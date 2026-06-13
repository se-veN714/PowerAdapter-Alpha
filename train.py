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
from tqdm import tqdm, trange

from config import (
    ACTIVATION, CHECKPOINT_DIR, DEVICE, DROPOUT_RATE, FACTOR_COLS,
    HIDDEN_DIMS, LAMBDA_ORTH, LEARNING_RATE, LOG_DIR, MAX_EPOCHS,
    N_HIDDEN, OPTIMIZER, PATIENCE, PROCESSED_DATA_DIR, ROLLING_WINDOWS,
    TRAIN_END, USE_LAYER_NORM, VAL_END, WEIGHT_DECAY,
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
        x_batch = x_batch.squeeze(0).to(DEVICE)  # (1,N,P)→(N,P)
        y_batch = y_batch.squeeze(0).to(DEVICE)  # (1,N)→(N,)

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
            x_batch = x_batch.squeeze(0).to(DEVICE)  # (1,N,P)→(N,P)
            y_batch = y_batch.squeeze(0).to(DEVICE)  # (1,N)→(N,)

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
    weight_decay: float = WEIGHT_DECAY,
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
        weight_decay: Adam权重衰减(L2正则)。
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
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

    # DataLoader（batch_size=1因为每个截面就是一个完整batch）
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 训练历史
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_ic": [],
    }

    best_val_ic = -float("inf")
    patience_counter = 0

    print(f"Training | device: {DEVICE} | model: {model.__class__.__name__} | loss: {loss_name}")
    print(f"Config  | lr={lr} | lambda_orth={lambda_orth} | wd={weight_decay}")
    print("-" * 70)

    pbar = trange(1, max_epochs + 1, desc="Epoch", ncols=100,
                  unit="ep", leave=True)
    for epoch in pbar:
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, lambda_orth, is_mlp)
        val_loss, val_ic = evaluate(model, val_loader, loss_fn, lambda_orth, is_mlp)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(val_ic)

        elapsed = time.time() - t0

        # Update tqdm postfix with live metrics
        pbar.set_postfix(
            t_loss=f"{train_loss:.4f}",
            v_loss=f"{val_loss:.4f}",
            v_ic=f"{val_ic:.4f}",
            best_ic=f"{best_val_ic:.4f}",
            pat=f"{patience_counter}/{patience}",
            time=f"{elapsed:.1f}s",
        )

        # Early stopping based on val_ic (higher is better)
        if val_ic > best_val_ic:
            best_val_ic = val_ic
            patience_counter = 0
            # Save best model
            if save_name is None:
                save_name = f"{model.__class__.__name__}_{loss_name}"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = CHECKPOINT_DIR / f"{save_name}_best.pt"
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.close()
                print(f"Early stop: val IC no improvement for {patience} epochs")
                break

    print("-" * 70)
    print(f"Training done | best val_ic={best_val_ic:.4f}")

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
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()
    plt.close()


def run_training(
    model_type: str = "mlp",
    loss_name: str = "mse",
    hidden_dims: tuple = HIDDEN_DIMS,
    dropout: float = DROPOUT_RATE,
    activation: str = ACTIVATION,
    use_layer_norm: bool = USE_LAYER_NORM,
    lr: float = LEARNING_RATE,
    lambda_orth: float = LAMBDA_ORTH,
    weight_decay: float = WEIGHT_DECAY,
) -> None:
    """运行单次训练的便捷入口。

    Args:
        model_type: 模型类型，"mlp"或"linear"。
        loss_name: 损失函数名称。
        hidden_dims: MLP隐藏层维度序列。
        dropout: Dropout比例。
        activation: 激活函数名称。
        use_layer_norm: True=LayerNorm。
        lr: 学习率。
        lambda_orth: 正交惩罚系数。
        weight_decay: 权重衰减。
    """
    # 加载预处理数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("Please run preprocess.py first")
        return

    df = pd.read_csv(processed_path, parse_dates=["date"])
    train_ds, val_ds, test_ds = split_dataset(df)

    # get factor count
    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    n_factors = len(available_factors)
    print(f"Available factors: {n_factors}")

    # 创建模型
    is_mlp = model_type == "mlp"
    if is_mlp:
        model: nn.Module = MLPAlphaModel(
            n_factors, hidden_dims=hidden_dims, dropout=dropout,
            activation=activation, use_layer_norm=use_layer_norm,
        )
    else:
        model = LinearAlphaModel(n_factors)

    # 训练
    history = train(
        model, train_ds, val_ds,
        loss_name=loss_name,
        is_mlp=is_mlp,
        lr=lr,
        lambda_orth=lambda_orth,
        weight_decay=weight_decay,
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
    windows: list[tuple[str, str, str]] | None = None,
    hidden_dims: tuple = HIDDEN_DIMS,
    dropout: float = DROPOUT_RATE,
    activation: str = ACTIVATION,
    use_layer_norm: bool = USE_LAYER_NORM,
    lr: float = LEARNING_RATE,
    lambda_orth: float = LAMBDA_ORTH,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
) -> list[dict]:
    """滚动训练（扩展窗口）。

    训练集用所有历史数据（不断扩大），验证集和测试集各约半年。
    每轮滚动向前推进一步，训练集不断增大（扩展窗口）。

    Args:
        model_type: 模型类型，"mlp"或"linear"。
        loss_name: 损失函数名称，"mse"/"ic"/"ccc"。
        windows: 滚动窗口列表，每项为(train_end, val_end, test_end)。
        hidden_dims: MLP隐藏层维度序列。
        dropout: Dropout比例。
        activation: 激活函数名称。
        use_layer_norm: True=LayerNorm。
        lr: 学习率。
        lambda_orth: 正交惩罚系数。
        weight_decay: 权重衰减(L2正则)。
        patience: 早停耐心值。

    Returns:
        每个窗口的评估结果列表。
    """
    if windows is None:
        windows = list(ROLLING_WINDOWS)

    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("Please run preprocess.py first")
        return []

    df = pd.read_csv(processed_path, parse_dates=["date"])
    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    n_factors = len(available_factors)
    is_mlp = model_type == "mlp"

    # Print tuning config once
    if is_mlp:
        print(f"\nMLP Config: hidden_dims={hidden_dims} dropout={dropout} "
              f"act={activation} ln={use_layer_norm}")
    print(f"Train Config: lr={lr} lambda_orth={lambda_orth} "
          f"wd={weight_decay} patience={patience}")

    results: list[dict] = []

    for w, (train_end, val_end, test_end) in enumerate(tqdm(
        windows, desc=f"Rolling {model_type.upper()}+{loss_name.upper()}",
        ncols=100, leave=True,
    )):
        train_end_dt = pd.Timestamp(train_end)
        val_end_dt = pd.Timestamp(val_end)
        test_end_dt = pd.Timestamp(test_end)

        train_df = df[df["date"] < train_end_dt].copy()
        val_df = df[(df["date"] >= train_end_dt) & (df["date"] < val_end_dt)].copy()
        test_df = df[(df["date"] >= val_end_dt) & (df["date"] < test_end_dt)].copy()

        if len(train_df) < 100 or len(val_df) < 20:
            print(f"Window {w}: not enough data, skipped")
            continue

        print(f"\n{'='*60}")
        print(f"Rolling window {w+1}/{len(windows)} | "
              f"{model_type.upper()} + {loss_name.upper()}")
        print(f"Train: {train_df['date'].min().date()} -> "
              f"{train_df['date'].max().date()} ({len(train_df)} rows, "
              f"{train_df['date'].nunique()} dates)")
        print(f"Val:   {val_df['date'].min().date()} -> "
              f"{val_df['date'].max().date()} ({len(val_df)} rows, "
              f"{val_df['date'].nunique()} dates)")
        print(f"Test:  {test_df['date'].min().date()} -> "
              f"{test_df['date'].max().date()} ({len(test_df)} rows, "
              f"{test_df['date'].nunique()} dates)")

        train_ds = FactorDataset(train_df, available_factors)
        val_ds = FactorDataset(val_df, available_factors)
        test_ds = FactorDataset(test_df, available_factors)

        # Create new model for each window
        if is_mlp:
            model: nn.Module = MLPAlphaModel(
                n_factors, hidden_dims=hidden_dims, dropout=dropout,
                activation=activation, use_layer_norm=use_layer_norm,
            )
        else:
            model = LinearAlphaModel(n_factors)

        save_name = f"{model_type}_{loss_name}_rolling{w}"
        history = train(
            model, train_ds, val_ds,
            loss_name=loss_name,
            is_mlp=is_mlp,
            lr=lr,
            lambda_orth=lambda_orth,
            weight_decay=weight_decay,
            patience=patience,
            save_name=save_name,
        )

        # Load best checkpoint and evaluate on test set
        ckpt_path = CHECKPOINT_DIR / f"{save_name}_best.pt"
        if ckpt_path.exists():
            model.load_state_dict(
                torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
            )
        model = model.to(DEVICE)

        # Full evaluation on test set
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        loss_fn_map = {"mse": mse_loss, "ic": ic_loss, "ccc": ccc_loss}
        test_loss, test_ic = evaluate(
            model, test_loader, loss_fn_map[loss_name], is_mlp=is_mlp
        )

        # Compute detailed IC metrics
        from utils.metrics import calc_ic_series, ic_summary, group_return
        from evaluate import predict

        predictions, actuals, pred_dates = predict(model, test_ds, is_mlp=is_mlp)
        pred_series = pd.Series(predictions, name="prediction")
        actual_series = pd.Series(actuals, name="actual")
        date_series = pd.Series(pred_dates, name="date")

        ic_series = calc_ic_series(pred_series, actual_series, date_series)
        summary = ic_summary(ic_series)

        # Group return for long-short
        group_df = pd.DataFrame({
            "date": date_series,
            "prediction": pred_series,
            "label": actual_series,
        })
        groups = group_return(group_df, n_groups=10)
        ls_row = groups[groups["group"] == "long_short"]
        ls_return = float(ls_row["mean_return"].values[0]) if len(ls_row) > 0 else 0.0

        result = {
            "window": w,
            "model": f"{model_type}_{loss_name}",
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "rank_ic_mean": summary["rank_ic_mean"],
            "icir": summary["icir"],
            "ic_win_rate": summary["ic_win_rate"],
            "long_short_return": ls_return,
            "best_val_ic": max(history["val_ic"]),
        }
        results.append(result)

        print(f"\nTest Results | {model_type.upper()} + {loss_name.upper()}")
        print(f"  Rank IC: {summary['rank_ic_mean']:.4f}")
        print(f"  ICIR:    {summary['icir']:.4f}")
        print(f"  IC Win:  {summary['ic_win_rate']:.2%}")
        print(f"  L-S Ret: {ls_return:.4f}")

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"Rolling Summary | {model_type.upper()} + {loss_name.upper()}")
        print("-" * 60)
        print(f"{'Window':<8}{'IC Mean':>10}{'ICIR':>10}{'IC Win':>10}{'L-S Ret':>10}")
        print("-" * 60)
        for r in results:
            print(f"  W{r['window']:<6}{r['rank_ic_mean']:>10.4f}{r['icir']:>10.4f}"
                  f"{r['ic_win_rate']:>10.2%}{r['long_short_return']:>10.4f}")
        avg_ic = np.mean([r["rank_ic_mean"] for r in results])
        avg_icir = np.mean([r["icir"] for r in results])
        avg_win = np.mean([r["ic_win_rate"] for r in results])
        avg_ls = np.mean([r["long_short_return"] for r in results])
        print("-" * 60)
        print(f"  {'Avg':<8}{avg_ic:>10.4f}{avg_icir:>10.4f}"
              f"{avg_win:>10.2%}{avg_ls:>10.4f}")

    return results


def run_all_rolling(
    hidden_dims: tuple = HIDDEN_DIMS,
    dropout: float = DROPOUT_RATE,
    activation: str = ACTIVATION,
    use_layer_norm: bool = USE_LAYER_NORM,
    lr: float = LEARNING_RATE,
    lambda_orth: float = LAMBDA_ORTH,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
) -> None:
    """运行所有模型+损失函数组合的滚动训练，并生成对比表。

    Args:
        hidden_dims: MLP隐藏层维度序列。
        dropout: Dropout比例。
        activation: 激活函数名称。
        use_layer_norm: True=LayerNorm。
        lr: 学习率。
        lambda_orth: 正交惩罚系数。
        weight_decay: 权重衰减。
        patience: 早停耐心值。
    """
    from evaluate import evaluate_rolling_results

    all_results: list[dict] = []
    model_configs = [
        ("linear", "mse"),
        ("mlp", "mse"),
        ("mlp", "ic"),
        ("mlp", "ccc"),
    ]

    for model_type, loss_name in tqdm(
        model_configs, desc="All Rolling", ncols=100, leave=True,
    ):
        print(f"\n{'#'*60}")
        print(f"# Rolling Training: {model_type.upper()} + {loss_name.upper()}")
        print(f"{'#'*60}")
        results = rolling_train(
            model_type=model_type, loss_name=loss_name,
            hidden_dims=hidden_dims, dropout=dropout,
            activation=activation, use_layer_norm=use_layer_norm,
            lr=lr, lambda_orth=lambda_orth, weight_decay=weight_decay,
            patience=patience,
        )
        all_results.extend(results)

    # Save and visualize results
    if all_results:
        evaluate_rolling_results(all_results)


if __name__ == "__main__":
    import argparse
    import ast

    parser = argparse.ArgumentParser(description="Project-Alpha Training")
    parser.add_argument("--mode", choices=["single", "rolling", "tuning"],
                        default="rolling", help="Training mode")
    parser.add_argument("--model", default="mlp", help="Model type: mlp/linear")
    parser.add_argument("--loss", default="ic", help="Loss function: mse/ic/ccc")
    # Tuning params
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE,
                        help="Dropout rate")
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY,
                        help="Weight decay (L2 reg)")
    parser.add_argument("--lambda_orth", type=float, default=LAMBDA_ORTH,
                        help="Orthogonal penalty coefficient")
    parser.add_argument("--hidden_dims", type=str, default=str(HIDDEN_DIMS),
                        help="Hidden dims tuple, e.g. '(64,64)' or '(32,)'")
    parser.add_argument("--activation", default=ACTIVATION,
                        choices=["sigmoid", "gelu", "leaky_relu"],
                        help="Activation function")
    parser.add_argument("--layer_norm", action="store_true",
                        default=USE_LAYER_NORM, help="Use LayerNorm")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early stopping patience")
    args = parser.parse_args()

    hidden_dims_tuple = ast.literal_eval(args.hidden_dims)

    if args.mode == "tuning":
        # Quick test: MLP+IC only, logging key params
        print("=" * 60)
        print("TUNING MODE: MLP+IC (single model, 6 windows)")
        print(f"hidden_dims={hidden_dims_tuple} dropout={args.dropout} "
              f"act={args.activation} ln={args.layer_norm}")
        print(f"lr={args.lr} lambda_orth={args.lambda_orth} "
              f"wd={args.wd} patience={args.patience}")
        print("=" * 60)
        from evaluate import evaluate_rolling_results
        results = rolling_train(
            model_type="mlp", loss_name="ic",
            hidden_dims=hidden_dims_tuple, dropout=args.dropout,
            activation=args.activation, use_layer_norm=args.layer_norm,
            lr=args.lr, lambda_orth=args.lambda_orth,
            weight_decay=args.wd, patience=args.patience,
        )
        if results:
            evaluate_rolling_results(results)
    elif args.mode == "rolling":
        run_all_rolling(
            hidden_dims=hidden_dims_tuple, dropout=args.dropout,
            activation=args.activation, use_layer_norm=args.layer_norm,
            lr=args.lr, lambda_orth=args.lambda_orth,
            weight_decay=args.wd, patience=args.patience,
        )
    elif args.mode == "single":
        run_training(
            model_type=args.model, loss_name=args.loss,
            hidden_dims=hidden_dims_tuple, dropout=args.dropout,
            activation=args.activation, use_layer_norm=args.layer_norm,
            lr=args.lr, lambda_orth=args.lambda_orth,
            weight_decay=args.wd,
        )
