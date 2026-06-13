# -*- coding: utf-8 -*-
# @File    : run_tuning.py
# @Time    : 2026/06/12
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""批量超参数调优实验脚本。

按 TUNING.md 中的实验矩阵依次运行实验，并记录结果。
每个实验运行 MLP+IC 的 6 窗口滚动训练。
支持断点续跑：检测已完成的实验自动跳过。
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "logs" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_experiments() -> list[dict[str, Any]]:
    """构建实验矩阵。

    Returns:
        实验配置列表，每项含 name 和 kwargs。
    """
    experiments: list[dict[str, Any]] = []

    # ===== Round 1: 正则化 =====
    experiments.extend([
        {
            "name": "R1_baseline",
            "round": 1,
            "desc": "Baseline (no dropout, no WD)",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R1.1_dropout02",
            "round": 1,
            "desc": "Dropout=0.2",
            "kwargs": {"dropout": 0.2, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R1.2_dropout03",
            "round": 1,
            "desc": "Dropout=0.3",
            "kwargs": {"dropout": 0.3, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R1.3_wd1e4",
            "round": 1,
            "desc": "Weight Decay=1e-4",
            "kwargs": {"dropout": 0.0, "weight_decay": 1e-4,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R1.4_d02_wd1e4",
            "round": 1,
            "desc": "Dropout=0.2 + WD=1e-4",
            "kwargs": {"dropout": 0.2, "weight_decay": 1e-4,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
    ])

    # ===== Round 2: 学习率 =====
    experiments.extend([
        {
            "name": "R2.1_lr1e4",
            "round": 2,
            "desc": "LR=0.0001",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0001,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R2.2_lr1e3",
            "round": 2,
            "desc": "LR=0.001",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.001,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R2.3_lr3e3",
            "round": 2,
            "desc": "LR=0.003",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.003,
                       "lambda_orth": 0.01, "patience": 50},
        },
    ])

    # ===== Round 3: 网络架构 =====
    experiments.extend([
        {
            "name": "R3.1_h32",
            "round": 3,
            "desc": "2 layers, hidden=32",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (32, 32), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R3.2_h128",
            "round": 3,
            "desc": "2 layers, hidden=128",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (128, 128), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R3.3_deep",
            "round": 3,
            "desc": "3 layers [64,32,16]",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 32, 16), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
    ])

    # ===== Round 4: 正交惩罚λ =====
    experiments.extend([
        {
            "name": "R4.1_orth0",
            "round": 4,
            "desc": "lambda_orth=0 (no orth penalty)",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.0, "patience": 50},
        },
        {
            "name": "R4.2_orth1e3",
            "round": 4,
            "desc": "lambda_orth=0.001",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.001, "patience": 50},
        },
        {
            "name": "R4.3_orth01",
            "round": 4,
            "desc": "lambda_orth=0.1",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.1, "patience": 50},
        },
    ])

    # ===== Round 5: 激活函数 =====
    experiments.extend([
        {
            "name": "R5.1_gelu",
            "round": 5,
            "desc": "GELU activation",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "gelu",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R5.2_leaky_relu",
            "round": 5,
            "desc": "LeakyReLU activation",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "leaky_relu",
                       "use_layer_norm": False, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
        {
            "name": "R5.3_layernorm",
            "round": 5,
            "desc": "LayerNorm instead of BatchNorm",
            "kwargs": {"dropout": 0.0, "weight_decay": 0.0,
                       "hidden_dims": (64, 64), "activation": "sigmoid",
                       "use_layer_norm": True, "lr": 0.0005,
                       "lambda_orth": 0.01, "patience": 50},
        },
    ])

    return experiments


def is_completed(exp_name: str) -> bool:
    """检查实验是否已完成（结果文件存在）。

    Args:
        exp_name: 实验名称。

    Returns:
        True if result file exists and is valid JSON.
    """
    result_path = RESULTS_DIR / f"{exp_name}.json"
    if not result_path.exists():
        return False
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "results" in data
    except (json.JSONDecodeError, KeyError):
        return False


def save_results(exp_name: str, exp_desc: str, exp_round: int,
                 exp_kwargs: dict, results: list[dict]) -> None:
    """保存实验结果。

    Args:
        exp_name: 实验名称。
        exp_desc: 实验描述。
        exp_round: 实验轮次。
        exp_kwargs: 实验参数。
        results: 滚动训练结果列表。
    """
    output = {
        "name": exp_name,
        "description": exp_desc,
        "round": exp_round,
        "timestamp": datetime.now().isoformat(),
        "kwargs": {k: str(v) if isinstance(v, tuple) else v
                    for k, v in exp_kwargs.items()},
        "results": results,
        "summary": _compute_summary(results) if results else {},
    }
    result_path = RESULTS_DIR / f"{exp_name}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {result_path}")


def _compute_summary(results: list[dict]) -> dict:
    """计算实验汇总统计。

    Args:
        results: 滚动训练结果列表。

    Returns:
        汇总统计字典。
    """
    if not results:
        return {}
    ic_values = [r["rank_ic_mean"] for r in results]
    icir_values = [r["icir"] for r in results]
    win_rates = [r["ic_win_rate"] for r in results]
    n_positive = sum(1 for ic in ic_values if ic > 0)

    import numpy as np
    return {
        "mean_ic": float(np.mean(ic_values)),
        "std_ic": float(np.std(ic_values, ddof=1)),
        "min_ic": float(np.min(ic_values)),
        "max_ic": float(np.max(ic_values)),
        "mean_icir": float(np.mean(icir_values)),
        "mean_win_rate": float(np.mean(win_rates)),
        "positive_windows": f"{n_positive}/{len(ic_values)}",
    }


def run_experiment(exp: dict[str, Any]) -> list[dict]:
    """运行单个实验。

    Args:
        exp: 实验配置字典，含 name, kwargs 等。

    Returns:
        滚动训练结果列表。
    """
    from train import rolling_train

    kwargs = exp["kwargs"]
    print(f"\n{'#' * 70}")
    print(f"# Experiment: {exp['name']} (Round {exp['round']})")
    print(f"# {exp['desc']}")
    print(f"# Config: {kwargs}")
    print(f"{'#' * 70}")

    results = rolling_train(
        model_type="mlp",
        loss_name="ic",
        hidden_dims=kwargs["hidden_dims"],
        dropout=kwargs["dropout"],
        activation=kwargs["activation"],
        use_layer_norm=kwargs["use_layer_norm"],
        lr=kwargs["lr"],
        lambda_orth=kwargs["lambda_orth"],
        weight_decay=kwargs["weight_decay"],
        patience=kwargs["patience"],
    )
    return results


def print_all_summaries(experiments: list[dict]) -> None:
    """打印所有已完成实验的汇总对比表。

    Args:
        experiments: 实验配置列表。
    """
    print(f"\n{'=' * 100}")
    print("TUNING RESULTS SUMMARY")
    print(f"{'=' * 100}")
    header = (f"{'Round':<6}{'Experiment':<20}{'Mean IC':>10}"
              f"{'Std IC':>10}{'Min IC':>10}{'Max IC':>10}"
              f"{'Mean ICIR':>10}{'+Win':>8}{'Desc'}")
    print(header)
    print("-" * 100)

    for exp in experiments:
        if not is_completed(exp["name"]):
            continue
        result_path = RESULTS_DIR / f"{exp['name']}.json"
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary", {})
        if not s:
            continue
        line = (f"R{exp['round']:<5} {exp['name']:<20}"
                f"{s.get('mean_ic', 0):>10.4f}"
                f"{s.get('std_ic', 0):>10.4f}"
                f"{s.get('min_ic', 0):>10.4f}"
                f"{s.get('max_ic', 0):>10.4f}"
                f"{s.get('mean_icir', 0):>10.4f}"
                f"{s.get('positive_windows', 'N/A'):>8}"
                f"  {exp['desc']}")
        print(line)
    print("-" * 100)


def main() -> None:
    """主入口：运行所有未完成的实验。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch hyperparameter tuning for Project-Alpha")
    parser.add_argument("--round", type=int, default=0,
                        help="Only run experiments in this round (0=all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if already completed")
    parser.add_argument("--summary", action="store_true",
                        help="Only print summary of completed experiments")
    args = parser.parse_args()

    experiments = build_experiments()

    if args.summary:
        print_all_summaries(experiments)
        return

    # Filter by round if specified
    if args.round > 0:
        experiments = [e for e in experiments if e["round"] == args.round]
        print(f"Running Round {args.round}: {len(experiments)} experiments")
    else:
        print(f"Running All Rounds: {len(experiments)} experiments")

    completed_count = 0
    skipped_count = 0

    for i, exp in enumerate(experiments):
        exp_name = exp["name"]

        # Skip completed experiments unless --force
        if is_completed(exp_name) and not args.force:
            print(f"\n[{i+1}/{len(experiments)}] {exp_name}: SKIPPED "
                  f"(already completed, use --force to re-run)")
            skipped_count += 1
            continue

        print(f"\n[{i+1}/{len(experiments)}] {exp_name}: RUNNING")
        try:
            results = run_experiment(exp)
            save_results(exp_name, exp["desc"], exp["round"],
                         exp["kwargs"], results)
            completed_count += 1
        except Exception as e:
            print(f"ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Tuning complete: {completed_count} run, {skipped_count} skipped")
    print(f"{'=' * 70}")

    # Print summary
    print_all_summaries(experiments)


if __name__ == "__main__":
    main()
