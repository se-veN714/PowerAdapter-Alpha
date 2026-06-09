# -*- coding: utf-8 -*-
# @File    : __init__.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了工具包的公共接口和统一导出。"""

from utils.data_loader import fetch_stock_data
from utils.preprocess import preprocess_pipeline
from utils.dataset import FactorDataset
from utils.metrics import rank_ic, calc_ic_series, ic_summary, group_return

__all__ = [
    "fetch_stock_data",
    "preprocess_pipeline",
    "FactorDataset",
    "rank_ic",
    "calc_ic_series",
    "ic_summary",
    "group_return",
]
