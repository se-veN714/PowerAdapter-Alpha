# -*- coding: utf-8 -*-
# @File    : preprocess.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了因子预处理的类和函数，包括MAD去极值、zscore标准化和收益率标签构建。

核心功能：MAD去极值、zscore标准化、收益率标签构建、完整预处理流水线。
所有标准化/去极值操作必须在groupby('date')截面内执行，严禁全局操作。
"""

import pandas as pd
import numpy as np

from config import MAD_MULTIPLIER, FACTOR_COLS, FACTOR_DIRECTION, LABEL_PERIOD


def mad_clip_section(df: pd.DataFrame, col: str, n: float = MAD_MULTIPLIER) -> pd.Series:
    """对指定列做截面MAD去极值。

    MAD = Median Absolute Deviation = median(|x_i - median(x)|)
    截断范围: [median - n*MAD, median + n*MAD]

    Args:
        df: 包含date列和因子列的DataFrame。
        col: 需要处理的列名。
        n: MAD倍数，默认3。

    Returns:
        处理后的Series，极值被截断到[median - n*MAD, median + n*MAD]。
    """
    def _clip_group(group: pd.Series) -> pd.Series:
        median = group.median()
        mad = (group - median).abs().median()
        lower = median - n * mad
        upper = median + n * mad
        return group.clip(lower=lower, upper=upper)

    return df.groupby("date")[col].transform(_clip_group)


def zscore_section(df: pd.DataFrame, col: str) -> pd.Series:
    """截面zscore标准化。

    在每个交易日截面内: (x - mean) / std
    截面均值≈0，标准差≈1。

    Args:
        df: 包含date列和因子列的DataFrame。
        col: 需要处理的列名。

    Returns:
        截面zscore化后的Series。
    """
    def _zscore_group(group: pd.Series) -> pd.Series:
        std = group.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean()) / std

    return df.groupby("date")[col].transform(_zscore_group)


def build_return_label(df: pd.DataFrame, period: int = LABEL_PERIOD) -> pd.Series:
    """构建T+1到T+period收益率标签，再做截面zscore。

    标签 = T日买入持有period天的收益率
    使用shift(-period)向前偏移，确保标签不含未来信息泄露。

    Args:
        df: 包含date, stock_code, close列的DataFrame。
        period: 持有期天数，默认20。

    Returns:
        截面zscore化后的收益率标签Series。
    """
    # 按股票计算period日收益率
    future_return = df.groupby("stock_code")["close"].pct_change(period).shift(-period)
    # 截面zscore
    temp_df = df.copy()
    temp_df["_raw_return"] = future_return
    return zscore_section(temp_df, "_raw_return")


def apply_factor_direction(df: pd.DataFrame, factor_cols: list[str] | None = None) -> pd.DataFrame:
    """处理因子方向：负方向因子取负值。

    Args:
        df: 包含因子列的DataFrame。
        factor_cols: 因子列名列表，默认使用config.FACTOR_COLS。

    Returns:
        方向调整后的DataFrame。
    """
    if factor_cols is None:
        factor_cols = FACTOR_COLS

    df = df.copy()
    for col in factor_cols:
        if col in df.columns and col in FACTOR_DIRECTION:
            df[col] = df[col] * FACTOR_DIRECTION[col]
    return df


def preprocess_pipeline(
    df: pd.DataFrame,
    factor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """完整预处理流水线：方向调整 → MAD → zscore → 缺失填0 → 标签构建。

    Args:
        df: 原始数据DataFrame，需包含date, stock_code, close列及因子列。
        factor_cols: 因子列名列表，默认使用config.FACTOR_COLS。

    Returns:
        预处理后的DataFrame，包含label列和所有处理后的因子列。
    """
    if factor_cols is None:
        factor_cols = FACTOR_COLS

    df = df.copy()
    print(f"预处理前: {len(df)} 行，{df['stock_code'].nunique()} 只股票")

    # Step 1: 因子方向调整
    df = apply_factor_direction(df, factor_cols)
    print("[1/5] 因子方向调整完成")

    # Step 2: 逐因子 MAD去极值 + zscore标准化
    for col in factor_cols:
        if col not in df.columns:
            print(f"  ⚠ 因子列 '{col}' 不存在，跳过")
            continue
        # 跳过全NaN列
        if df[col].notna().sum() == 0:
            print(f"  ⚠ 因子列 '{col}' 全为NaN，跳过")
            continue
        df[col] = mad_clip_section(df, col)
        df[col] = zscore_section(df, col)
    print("[2/5] MAD去极值 + zscore标准化完成")

    # Step 3: 缺失值填充为0
    for col in factor_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    print("[3/5] 缺失值填充完成")

    # Step 4: 构建收益率标签
    df["label"] = build_return_label(df)
    print("[4/5] 收益率标签构建完成")

    # Step 5: 删除标签为NaN的行（未来period天的数据不可用）
    n_before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].fillna(0)
    print(f"[5/5] 清理完成，删除 {n_before - len(df)} 行（标签缺失）")

    print(f"预处理后: {len(df)} 行")
    return df


if __name__ == "__main__":
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from utils.data_loader import fetch_stock_data, compute_derived_factors

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 检查是否已有原始数据
    raw_path = RAW_DATA_DIR / "stock_data_with_factors.csv"
    if raw_path.exists():
        print(f"加载已有数据: {raw_path}")
        raw_df = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        raw_df = fetch_stock_data()
        raw_df = compute_derived_factors(raw_df)

    processed_df = preprocess_pipeline(raw_df)
    processed_df.to_csv(PROCESSED_DATA_DIR / "processed_data.csv", index=False, encoding="utf-8-sig")
    print(f"预处理数据已保存")
