# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了项目全局参数配置的常量和变量。

所有可调参数集中管理，禁止硬编码散落各文件。
"""

from pathlib import Path
from typing import Final

try:
    import torch
    _CUDA_AVAILABLE: Final = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE: Final = False

# ===== 路径配置 =====
PROJECT_ROOT: Final = Path(__file__).resolve().parent
RAW_DATA_DIR: Final = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Final = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR: Final = PROJECT_ROOT / "checkpoints"
LOG_DIR: Final = PROJECT_ROOT / "logs"

# ===== 股票池（20-30只，覆盖主要行业） =====
STOCK_POOL: Final = [
    # 银行
    "600036",  # 招商银行
    "601398",  # 工商银行
    "601288",  # 农业银行
    # 保险/券商
    "601318",  # 中国平安
    "600030",  # 中信证券
    # 食品饮料
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "000568",  # 泸州老窖
    # 家电
    "000651",  # 格力电器
    "000333",  # 美的集团
    # 科技/制造
    "002415",  # 海康威视
    "300750",  # 宁德时代
    # 医药
    "600276",  # 恒瑞医药
    "000538",  # 云南白药
    # 地产
    "000002",  # 万科A
    # 钢铁
    "600019",  # 宝钢股份
    # 化工
    "600309",  # 万华化学
    # 电力
    "600900",  # 长江电力
    # 建筑
    "601668",  # 中国建筑
    # 通信
    "600050",  # 中国联通
    # 石油
    "601857",  # 中国石油
    # 煤炭
    "601088",  # 中国神华
    # 新能源
    "601012",  # 隆基绿能
    # 汽车
    "600104",  # 上汽集团
    # 电子
    "002475",  # 立讯精密
]

# ===== 时间范围 =====
DATE_RANGE: Final = ("2020-01-01", "2023-12-31")

# ===== 因子配置 =====
FACTOR_COLS: Final = [
    # 估值类
    "ep",    # Earnings-to-Price (盈利收益率)
    "bp",    # Book-to-Price (账面市值比)
    "dp",    # Dividend-to-Price (股息率)
    # 成长类
    "roe_growth",       # ROE增长率
    "profit_growth",    # 净利润增长率
    "revenue_growth",   # 营收增长率
    # 经营类
    "roe",              # 净资产收益率
    "asset_turnover",   # 总资产周转率
    # 流动性
    "turnover_rate",    # 换手率（负方向）
    "amplitude",        # 振幅（负方向）
    # 技术
    "momentum_20",      # 20日动量
    "reversal_5",       # 5日反转（负方向）
]

# 因子方向映射：1=正方向，-1=负方向
FACTOR_DIRECTION: Final = {
    "ep": 1,
    "bp": 1,
    "dp": 1,
    "roe_growth": 1,
    "profit_growth": 1,
    "revenue_growth": 1,
    "roe": 1,
    "asset_turnover": 1,
    "turnover_rate": -1,
    "amplitude": -1,
    "momentum_20": 1,
    "reversal_5": -1,
}

# ===== 预处理参数 =====
MAD_MULTIPLIER: Final = 3

# ===== 模型参数 =====
N_HIDDEN: Final = 64
LEARNING_RATE: Final = 0.0005
OPTIMIZER: Final = "Adam"
MAX_EPOCHS: Final = 1000
PATIENCE: Final = 5
LAMBDA_ORTH: Final = 0.01

# ===== 训练/验证/测试时间划分 =====
TRAIN_END: Final = "2022-06-30"
VAL_END: Final = "2023-06-30"

# ===== 标签参数 =====
LABEL_PERIOD: Final = 20  # T+1到T+20收益率

# ===== GPU =====
DEVICE: Final = "cuda" if _CUDA_AVAILABLE else "cpu"

# ===== 分组回测 =====
N_GROUPS: Final = 10
