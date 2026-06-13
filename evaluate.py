# -*- coding: utf-8 -*-
# @File    : evaluate.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了模型评估的完整流程，包括RankIC计算、分组回测和多模型对比。

加载模型，在测试集上生成预测，计算RankIC/ICIR/IC胜率，
生成分组收益柱状图，不同模型对比表。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, DEVICE, FACTOR_COLS, N_HIDDEN,
    PROCESSED_DATA_DIR, TRAIN_END, VAL_END, N_GROUPS,
)
from models.linear_alpha import LinearAlphaModel
from models.mlp_alpha import MLPAlphaModel
from utils.dataset import FactorDataset, split_dataset
from utils.metrics import calc_ic_series, group_return, ic_summary


def predict(
    model: nn.Module,
    dataset: FactorDataset,
    is_mlp: bool = False,
) -> tuple[np.ndarray, np.ndarray, list]:
    """在数据集上生成预测。

    Args:
        model: 训练好的PyTorch模型。
        dataset: 因子数据集。
        is_mlp: 是否为MLP模型。

    Returns:
        (predictions, actuals, dates) 元组，均为NumPy数组/列表。
    """
    model.eval()
    model = model.to(DEVICE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    all_dates: list = []

    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(tqdm(
            loader, desc="Predicting", ncols=80, leave=False,
        )):
            x_batch = x_batch.to(DEVICE)
            x_batch = x_batch.squeeze(0)
            y_batch = y_batch.squeeze(0)

            if is_mlp:
                pred, _ = model(x_batch)
            else:
                pred = model(x_batch)

            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y_batch.numpy())
            # Expand date per-section to per-sample
            section_date = dataset.get_date(idx)
            all_dates.extend([section_date] * len(y_batch))

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    return predictions, actuals, all_dates


def evaluate_model(
    model_name: str,
    model: nn.Module,
    test_ds: FactorDataset,
    is_mlp: bool = False,
) -> dict:
    """评估单个模型并返回结果。

    Args:
        model_name: 模型名称。
        model: PyTorch模型。
        test_ds: 测试数据集。
        is_mlp: 是否为MLP模型。

    Returns:
        评估结果字典，含ic_summary和分组收益。
    """
    model.eval()
    model = model.to(DEVICE)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_preds: list[float] = []
    all_actuals: list[float] = []
    all_dates: list = []

    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(tqdm(
            loader, desc=f"Eval {model_name}", ncols=80, leave=False,
        )):
            x_batch = x_batch.squeeze(0).to(DEVICE)
            y_batch = y_batch.squeeze(0)

            if is_mlp:
                pred, _ = model(x_batch)
            else:
                pred = model(x_batch)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_actuals.extend(y_batch.numpy().tolist())
            all_dates.extend([test_ds.get_date(idx)] * len(y_batch))

    pred_series = pd.Series(all_preds, name="prediction")
    actual_series = pd.Series(all_actuals, name="actual")
    date_series = pd.Series(all_dates, name="date")

    # IC指标
    ic_series = calc_ic_series(pred_series, actual_series, date_series)
    summary = ic_summary(ic_series)

    # 分组收益
    group_df = pd.DataFrame({
        "date": date_series,
        "prediction": pred_series,
        "label": actual_series,
    })
    groups = group_return(group_df, n_groups=N_GROUPS)

    return {
        "model_name": model_name,
        "ic_summary": summary,
        "ic_series": ic_series,
        "group_return": groups,
    }


def plot_group_return(
    group_df: pd.DataFrame,
    title: str = "Group Return",
    save_path: Path | None = None,
) -> None:
    """绘制分组收益柱状图。

    Args:
        group_df: 分组回测结果DataFrame。
        title: 图表标题。
        save_path: 图片保存路径。
    """
    # 过滤多空行
    plot_df = group_df[group_df["group"] != "long_short"].copy()
    plot_df["group"] = plot_df["group"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(plot_df)))
    ax.bar(plot_df["group"], plot_df["mean_return"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Group (1=Lowest, 10=Highest)")
    ax.set_ylabel("Mean Return (zscore)")
    ax.set_title(title)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 标注多空对冲收益
    ls_row = group_df[group_df["group"] == "long_short"]
    if not ls_row.empty:
        ls_return = ls_row.iloc[0]["mean_return"]
        ax.text(0.98, 0.98, f"Long-Short: {ls_return:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Group return chart saved to {save_path}")
    else:
        plt.show()
    plt.close()


def compare_models(
    model_configs: list[dict],
    test_ds: FactorDataset,
) -> pd.DataFrame:
    """多模型对比评估。

    Args:
        model_configs: 模型配置列表，每项含name, path, is_mlp, n_factors。
        test_ds: 测试数据集。

    Returns:
        对比结果DataFrame。
    """
    results: list[dict] = []

    for config in model_configs:
        name = config["name"]
        ckpt_path = config["path"]
        is_mlp = config["is_mlp"]
        n_factors = config["n_factors"]

        if not Path(ckpt_path).exists():
            print(f"[WARN] model not found: {ckpt_path}, skipping")
            continue

        # 加载模型
        if is_mlp:
            model: nn.Module = MLPAlphaModel(n_factors)
        else:
            model = LinearAlphaModel(n_factors)

        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEVICE))

        # 评估
        eval_result = evaluate_model(name, model, test_ds, is_mlp)

        results.append({
            "model": name,
            "rank_ic_mean": eval_result["ic_summary"]["rank_ic_mean"],
            "icir": eval_result["ic_summary"]["icir"],
            "ic_win_rate": eval_result["ic_summary"]["ic_win_rate"],
        })

        # 绘制分组收益图
        from config import LOG_DIR
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        plot_group_return(
            eval_result["group_return"],
            title=f"Group Return - {name}",
            save_path=LOG_DIR / f"{name}_group_return.png",
        )

    if not results:
        print("No models to evaluate")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(results)
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))
    return comparison_df


if __name__ == "__main__":
    # 加载预处理数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("Please run preprocess.py first")
    else:
        df = pd.read_csv(processed_path, parse_dates=["date"])
        _, _, test_ds = split_dataset(df)

        available_factors = [c for c in FACTOR_COLS if c in df.columns]
        n_factors = len(available_factors)

        # 定义要对比的模型
        model_configs = [
            {
                "name": "Linear_MSE",
                "path": str(CHECKPOINT_DIR / "linear_mse_best.pt"),
                "is_mlp": False,
                "n_factors": n_factors,
            },
            {
                "name": "MLP_MSE",
                "path": str(CHECKPOINT_DIR / "mlp_mse_best.pt"),
                "is_mlp": True,
                "n_factors": n_factors,
            },
            {
                "name": "MLP_IC",
                "path": str(CHECKPOINT_DIR / "mlp_ic_best.pt"),
                "is_mlp": True,
                "n_factors": n_factors,
            },
            {
                "name": "MLP_CCC",
                "path": str(CHECKPOINT_DIR / "mlp_ccc_best.pt"),
                "is_mlp": True,
                "n_factors": n_factors,
            },
        ]

        compare_models(model_configs, test_ds)


def evaluate_rolling_results(
    all_results: list[dict],
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """汇总滚动训练结果，生成对比表和可视化。

    Args:
        all_results: 各窗口的评估结果列表，来自rolling_train()。
        save_dir: 保存目录，默认LOG_DIR。

    Returns:
        汇总DataFrame。
    """
    from config import LOG_DIR

    if save_dir is None:
        save_dir = LOG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)

    # ---- 1. Per-window comparison table ----
    print(f"\n{'='*70}")
    print("Rolling Training - Full Comparison Table")
    print("=" * 70)
    pivot_cols = ["model", "window", "rank_ic_mean", "icir",
                  "ic_win_rate", "long_short_return"]
    print(results_df[pivot_cols].to_string(index=False))

    # ---- 2. Average across windows per model ----
    avg_df = (
        results_df.groupby("model")
        .agg({
            "rank_ic_mean": "mean",
            "icir": "mean",
            "ic_win_rate": "mean",
            "long_short_return": "mean",
        })
        .reset_index()
        .sort_values("rank_ic_mean", ascending=False)
    )
    avg_df.columns = ["model", "avg_ic", "avg_icir", "avg_ic_win",
                       "avg_ls_return"]

    print(f"\n{'='*70}")
    print("Average Across Windows (sorted by avg IC)")
    print("=" * 70)
    print(avg_df.to_string(index=False))

    # Save to CSV
    csv_path = save_dir / "rolling_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ---- 3. Visualization: grouped bar chart ----
    models = avg_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Avg Rank IC
    bars = axes[0].bar(x, avg_df["avg_ic"], width, color="steelblue",
                       edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Avg Rank IC")
    axes[0].set_title("Average Rank IC (Rolling)")
    axes[0].axhline(y=0.03, color="red", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, avg_df["avg_ic"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Avg ICIR
    bars = axes[1].bar(x, avg_df["avg_icir"], width, color="darkorange",
                       edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Avg ICIR")
    axes[1].set_title("Average ICIR (Rolling)")
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, avg_df["avg_icir"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Avg L-S Return
    bars = axes[2].bar(x, avg_df["avg_ls_return"], width, color="green",
                       edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("Avg Long-Short Return")
    axes[2].set_title("Average L-S Return (Rolling)")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, avg_df["avg_ls_return"]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Rolling Training Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = save_dir / "rolling_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {chart_path}")
    plt.close()

    # ---- 4. Per-window IC line chart ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for model_name in results_df["model"].unique():
        model_data = results_df[results_df["model"] == model_name]
        ax2.plot(model_data["window"], model_data["rank_ic_mean"],
                 marker="o", label=model_name, linewidth=2)
    ax2.set_xlabel("Window Index")
    ax2.set_ylabel("Rank IC Mean")
    ax2.set_title("Rolling Rank IC per Window")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.03, color="red", linestyle="--", alpha=0.5,
                label="IC=0.03 threshold")
    plt.tight_layout()
    line_path = save_dir / "rolling_ic_trend.png"
    plt.savefig(line_path, dpi=150, bbox_inches="tight")
    print(f"IC trend chart saved to {line_path}")
    plt.close()

    return avg_df
