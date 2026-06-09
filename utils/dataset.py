# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了按交易日截面加载的PyTorch数据集类和时间划分函数。

按交易日截面加载数据，每个截面包含当日所有股票的因子和标签。
不使用PyTorch默认的batch_size，需按日期分组。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FACTOR_COLS, PROCESSED_DATA_DIR, TRAIN_END, VAL_END


class FactorDataset(Dataset):
    """按交易日截面加载的因子数据集。

    每个样本 = 一个交易日截面的所有股票。
    __getitem__返回该截面的(factor_tensor, label_tensor)。

    Attributes:
        dates: 排序后的日期列表。
        factor_cols: 使用的因子列名。
        data: 完整DataFrame。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factor_cols: list[str] | None = None,
    ) -> None:
        """初始化数据集。

        Args:
            df: 预处理后的DataFrame，需包含date, stock_code列及因子列和label列。
            factor_cols: 因子列名列表，默认使用config.FACTOR_COLS。
        """
        if factor_cols is None:
            factor_cols = FACTOR_COLS

        # 确保date列为datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        self.data = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        self.factor_cols = [c for c in factor_cols if c in self.data.columns]
        self.dates = sorted(self.data["date"].unique())

        # 按日期预分组索引，加速__getitem__
        self._group_indices: dict[pd.Timestamp, pd.Index] = {}
        for date_val, group in self.data.groupby("date"):
            self._group_indices[pd.Timestamp(date_val)] = group.index

    def __len__(self) -> int:
        """返回截面数量（交易日数）。"""
        return len(self.dates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取第idx个截面的因子和标签。

        Args:
            idx: 截面索引。

        Returns:
            (factor_tensor, label_tensor) 元组，dtype均为float32。
            factor_tensor: shape=(N, P)，N为当日股票数，P为因子数。
            label_tensor: shape=(N,)。
        """
        date_val = self.dates[idx]
        indices = self._group_indices[date_val]
        section = self.data.loc[indices]

        factor_array = section[self.factor_cols].values.astype(np.float32)
        label_array = section["label"].values.astype(np.float32)

        return torch.from_numpy(factor_array), torch.from_numpy(label_array)

    def get_date(self, idx: int) -> pd.Timestamp:
        """获取第idx个截面对应的日期。

        Args:
            idx: 截面索引。

        Returns:
            对应的日期Timestamp。
        """
        return self.dates[idx]


def split_dataset(
    df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> tuple[FactorDataset, FactorDataset, FactorDataset]:
    """按时间切分训练/验证/测试数据集。

    严禁随机split，必须按时间划分以避免未来信息泄露。

    Args:
        df: 预处理后的DataFrame。
        factor_cols: 因子列名列表。
        train_end: 训练集截止日期（不含）。
        val_end: 验证集截止日期（不含）。

    Returns:
        (train_dataset, val_dataset, test_dataset) 元组。
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_df = df[df["date"] < train_end_dt].copy()
    val_df = df[(df["date"] >= train_end_dt) & (df["date"] < val_end_dt)].copy()
    test_df = df[df["date"] >= val_end_dt].copy()

    print(f"训练集: {len(train_df)} 行, {train_df['date'].nunique()} 个交易日")
    print(f"验证集: {len(val_df)} 行, {val_df['date'].nunique()} 个交易日")
    print(f"测试集: {len(test_df)} 行, {test_df['date'].nunique()} 个交易日")

    return (
        FactorDataset(train_df, factor_cols),
        FactorDataset(val_df, factor_cols),
        FactorDataset(test_df, factor_cols),
    )


if __name__ == "__main__":
    processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
    if not processed_path.exists():
        print("请先运行 preprocess.py 生成预处理数据")
    else:
        df = pd.read_csv(processed_path, parse_dates=["date"])
        train_ds, val_ds, test_ds = split_dataset(df)
        # 验证单个截面
        x, y = train_ds[0]
        print(f"截面0: factors shape={x.shape}, labels shape={y.shape}")
        print(f"截面0日期: {train_ds.get_date(0)}")
