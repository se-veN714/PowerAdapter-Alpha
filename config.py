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

# ===== 股票池（50只，覆盖主要行业，i5-12450H + RTX 4060 可承载） =====
STOCK_POOL: Final = [
    # ---- 银行 (6) ----
    "600036",  # 招商银行
    "601398",  # 工商银行
    "601288",  # 农业银行
    "601166",  # 兴业银行
    "600000",  # 浦发银行
    "601328",  # 交通银行
    # ---- 保险/券商 (4) ----
    "601318",  # 中国平安
    "600030",  # 中信证券
    "601601",  # 中国太保
    "601688",  # 华泰证券
    # ---- 食品饮料 (5) ----
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "000568",  # 泸州老窖
    "000333",  # 美的集团
    "600887",  # 伊利股份
    # ---- 家电 (1) ----
    "000651",  # 格力电器
    # ---- 科技/制造/电子 (6) ----
    "002415",  # 海康威视
    "300750",  # 宁德时代
    "002475",  # 立讯精密
    "002230",  # 科大讯飞
    "600588",  # 用友网络
    "002049",  # 紫光国微
    # ---- 医药 (5) ----
    "600276",  # 恒瑞医药
    "000538",  # 云南白药
    "300760",  # 迈瑞医疗
    "600196",  # 复星医药
    "002007",  # 华兰生物
    # ---- 地产/建材 (3) ----
    "000002",  # 万科A
    "600048",  # 保利发展
    "600585",  # 海螺水泥
    # ---- 钢铁/有色 (3) ----
    "600019",  # 宝钢股份
    "601899",  # 紫金矿业
    "600362",  # 江西铜业
    # ---- 化工 (3) ----
    "600309",  # 万华化学
    "600426",  # 华鲁恒升
    "002493",  # 荣盛石化
    # ---- 电力/公用 (3) ----
    "600900",  # 长江电力
    "601985",  # 中国核电
    "600025",  # 华能水电
    # ---- 建筑 (2) ----
    "601668",  # 中国建筑
    "601186",  # 中国铁建
    # ---- 通信/传媒 (2) ----
    "600050",  # 中国联通
    "000063",  # 中兴通讯
    # ---- 石油/煤炭 (3) ----
    "601857",  # 中国石油
    "601088",  # 中国神华
    "601225",  # 陕西煤业
    # ---- 新能源 (2) ----
    "601012",  # 隆基绿能
    "002129",  # 中环股份
    # ---- 汽车 (2) ----
    "600104",  # 上汽集团
    "002594",  # 比亚迪
]

# ===== 时间范围 =====
DATE_RANGE: Final = ("2018-01-01", "2023-12-31")

# ===== 因子配置 =====
FACTOR_COLS: Final = [
    # 估值类
    "ep",    # Earnings-to-Price (盈利收益率)
    "bp",    # Book-to-Price (账面市值比)
    # "dp",  # Dividend-to-Price (股息率) - 无股息数据，暂移除
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
    # "dp": 1,  # 无股息数据，暂移除
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
PATIENCE: Final = 50
LAMBDA_ORTH: Final = 0.01

# ===== 调优参数（v2.5+） =====
DROPOUT_RATE: Final = 0.0        # Dropout比例，0=不启用
WEIGHT_DECAY: Final = 0.0        # Adam权重衰减(L2正则)
ACTIVATION: Final = "sigmoid"    # 激活函数: "sigmoid"/"gelu"/"leaky_relu"
HIDDEN_DIMS: tuple = (64, 64)    # 隐藏层维度序列，默认2层各64
USE_LAYER_NORM: Final = False    # True=LayerNorm替代BatchNorm

# ===== 训练/验证/测试时间划分 =====
TRAIN_END: Final = "2022-06-30"
VAL_END: Final = "2023-06-30"

# ===== 标签参数 =====
LABEL_PERIOD: Final = 20  # T+1到T+20收益率

# ===== GPU =====
DEVICE: Final = "cuda" if _CUDA_AVAILABLE else "cpu"

# ===== 分组回测 =====
N_GROUPS: Final = 10

# ===== 滚动训练窗口（扩展窗口） =====
# 每个元组: (train_end, val_end, test_end)
# 训练集: date < train_end
# 验证集: train_end <= date < val_end
# 测试集: val_end <= date < test_end  (与验证集等长，避免测试集跨度过长)
ROLLING_WINDOWS: Final = [
    ("2020-07-01", "2021-01-01", "2021-07-01"),  # W0: train 2018-2020H1, val 2020H2, test 2021H1
    ("2021-01-01", "2021-07-01", "2022-01-01"),  # W1: train 2018-2020, val 2021H1, test 2021H2
    ("2021-07-01", "2022-01-01", "2022-07-01"),  # W2: train 2018-2021H1, val 2021H2, test 2022H1
    ("2022-01-01", "2022-07-01", "2023-01-01"),  # W3: train 2018-2021, val 2022H1, test 2022H2
    ("2022-07-01", "2023-01-01", "2023-07-01"),  # W4: train 2018-2022H1, val 2022H2, test 2023H1
    ("2023-01-01", "2023-07-01", "2024-01-01"),  # W5: train 2018-2022, val 2023H1, test 2023H2
]
