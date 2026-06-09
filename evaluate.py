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
        for idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(DEVICE)
            # 去掉DataLoader添加的batch维度
            x_batch = x_batch.squeeze(0)
            y_batch = y_batch.squeeze(0)

            if is_mlp:
                pred, _ = model(x_batch)
            else:
                pred = model(x_batch)

            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y_batch.numpy())
            all_dates.append(dataset.get_date(idx))

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_preds)  # Fix: should be actuals
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
        for idx, (x_batch, y_batch) in enumerate(loader):
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
        print(f"分组收益图已保存至 {save_path}")
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
            print(f"⚠ 模型文件不存在: {ckpt_path}，跳过")
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
        print("没有可评估的模型")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(results)
    print("\n模型对比表:")
    print(comparison_df.to_string(index=False))
    return comparison_df


if __name__ == "__main__":
    # 加载预处理数据
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("请先运行 preprocess.py 生成预处理数据")
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
