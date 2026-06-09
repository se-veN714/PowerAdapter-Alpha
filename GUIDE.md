# Project-Alpha Guide

> CB（Codebuddy）编程上下文参考。本文档提供项目全貌，使你能独立高效执行开发任务。

---

## 1. 项目概述与目标

**复现论文**：招商证券《端到端的动态Alpha模型》（2023），基于MLP的端到端因子权重训练与收益率预测。

**核心思路**：将传统"分步因子合成"改造为一个神经网络端到端训练：
- 输入：10-15个预处理后的因子（估值/成长/经营/流动性/技术）
- 网络：3层MLP（前两层Sigmoid+BN，末层ReLU）
- 输出：每只股票的预测收益排名zscore
- 亮点：正交惩罚使学到的隐因子不相关 + 三种loss函数对比

**场景**：面试驱动，10天冲刺。目标是跑通核心链路 + 能讲清每步逻辑。简化版：20-30只股票、3年数据、10-15个因子。

---

## 2. 技术栈

| 类别 | 技术 | 版本/说明 |
|------|------|-----------|
| 语言 | Python | 3.14（项目venv已配） |
| 深度学习 | PyTorch | **CUDA版（GPU: RTX 4060）**，见§7安装命令 |
| GPU加速 | CUDA | 12.x，PyTorch自动检测，config.py统一管理device |
| 数据处理 | Pandas, NumPy | 截面操作核心 |
| 评估 | SciPy (spearmanr), Scikit-learn | RankIC计算 |
| 可视化 | Matplotlib | loss曲线/分组收益图 |
| 数据源 | AKShare / Tushare / JoinQuant | 三选一，见下文 |
| 开发环境 | PyCharm + CB | 项目已初始化 |

---

## 3. 目录结构

```
Project-Alpha/
├── GUIDE.md                 # 本文档
├── config.py                # 全局参数配置（路径、超参、因子列表）
├── data/
│   ├── raw/                 # 原始CSV数据
│   └── processed/           # 预处理后CSV
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # 数据获取脚本（AKShare/Tushare）
│   ├── preprocess.py        # 因子预处理（MAD/zscore/标签构建）
│   ├── dataset.py           # PyTorch Dataset（按截面加载）
│   └── metrics.py           # RankIC/ICIR/IC胜率/分组收益
├── models/
│   ├── __init__.py
│   ├── linear_alpha.py      # 线性基准模型（L2正则）
│   └── mlp_alpha.py         # MLP非线性模型（Sigmoid+ReLU）
├── losses.py                # 三种损失函数（MSE/IC/CCC）+ 正交惩罚
├── train.py                 # 训练入口（含早停/滚动训练）
├── evaluate.py              # 评估入口（RankIC/分组/对比表）
└── notebooks/
    └── reproduction.ipynb   # 最终交付notebook
```

---

## 4. 模块划分与职责

### config.py — 全局参数

所有可调参数集中管理，禁止硬编码散落各文件：

```python
# 数据
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
STOCK_POOL = [...]           # 股票代码列表，20-30只
DATE_RANGE = ("2020-01-01", "2023-12-31")

# 因子
FACTOR_COLS = [...]          # 因子列名列表
FACTOR_DIRECTION = {...}     # 因子方向映射：1=正，-1=负

# 预处理
MAD_MULTIPLIER = 3           # MAD截断倍数

# 模型
N_HIDDEN = 64                # 隐藏层神经元数
LEARNING_RATE = 0.0005
OPTIMIZER = "Adam"
MAX_EPOCHS = 1000
PATIENCE = 5                 # 早停耐心值
LAMBDA_ORTH = 0.01           # 正交惩罚系数

# 训练/验证/测试划分
TRAIN_END = "2022-06-30"
VAL_END = "2023-06-30"

# GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # RTX 4060，务必用CUDA
```

### utils/data_loader.py — 数据获取

- 从数据源拉取行情和因子数据
- 输出统一格式CSV到 `data/raw/`
- DataFrame必须包含列：`date, stock_code, close, <各因子列>`
- 处理方向：负方向因子取负值（`factor * direction`）

### utils/preprocess.py — 因子预处理

核心函数签名：

```python
def mad_clip_section(df: pd.DataFrame, col: str, n: float = 3) -> pd.Series:
    """截面MAD去极值：groupby('date')后clip"""

def zscore_section(df: pd.DataFrame, col: str) -> pd.Series:
    """截面zscore标准化"""

def build_return_label(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """构建T+1到T+period收益率标签，再做截面zscore"""

def preprocess_pipeline(df: pd.DataFrame, factor_cols: list, label_col: str) -> pd.DataFrame:
    """完整预处理流水线：MAD → zscore → 缺失填0 → 标签构建"""
```

**关键原则**：所有标准化/去极值操作必须在 `groupby('date')` 截面内执行，严禁全局操作。

### utils/dataset.py — PyTorch数据集

```python
class FactorDataset(Dataset):
    """按交易日截面加载，每个__getitem__返回一个截面的(factor_tensor, label_tensor)"""
```

- 一个batch = 一个交易日截面的所有股票
- 不使用PyTorch默认的batch_size，需自定义Dataset按日期分组
- 返回tensor的dtype为float32
- **数据送入模型前需 `.to(DEVICE)`，DEVICE从config.py读取**

### models/linear_alpha.py — 线性基准

```
Linear(P, 64) → BN → Linear(64, 6) → BN → Linear(6, 1)
损失：MSE + L2正则
```

### models/mlp_alpha.py — MLP模型

```
Linear(P, 64) → BN → Sigmoid → Linear(64, 64) → BN → Sigmoid → Linear(64, 1) → ReLU
```

**必须返回第二层隐藏层输出**（用于正交惩罚），forward签名：

```python
def forward(self, x) -> tuple[Tensor, Tensor]:
    """返回 (prediction, hidden_layer2_output)"""
```

### losses.py — 损失函数

```python
def mse_loss(pred, target) -> Tensor
def ic_loss(pred, target) -> Tensor      # -Pearson相关系数
def ccc_loss(pred, target) -> Tensor     # -CCC一致性相关系数
def orthogonal_penalty(h: Tensor) -> Tensor  # 非对角协方差F范数
```

总损失组合：`total_loss = loss_fn(pred, target) + lambda_orth * orthogonal_penalty(hidden)`

### utils/metrics.py — 评估指标

```python
def rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman秩相关系数"""

def calc_ic_series(predictions: pd.Series, actuals: pd.Series, dates: pd.Series) -> pd.Series:
    """每个截面的RankIC序列"""

def ic_summary(ic_series: pd.Series) -> dict:
    """返回 {rank_ic_mean, icir, ic_win_rate}"""

def group_return(df: pd.DataFrame, pred_col: str, return_col: str, n_groups: int = 10) -> pd.DataFrame:
    """分组回测：按预测zscore分n组，计算各组平均收益"""
```

### train.py — 训练入口

- 支持单次训练和滚动训练两种模式
- 早停：验证集loss连续patience个epoch不下降则停止
- 保存最佳模型到 `checkpoints/`
- 每个epoch记录train_loss和val_loss，用于画曲线
- **GPU训练**：模型和数据均需 `.to(DEVICE)`，训练循环模板：

```python
model = model.to(DEVICE)
for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
    pred, hidden = model(x_batch)
    loss = loss_fn(pred, y_batch) + LAMBDA_ORTH * orthogonal_penalty(hidden)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- **注意**：评估指标（spearmanr等）需先将tensor `.cpu().numpy()` 转回NumPy再计算

### evaluate.py — 评估入口

- 加载模型，在测试集上生成预测
- 计算RankIC/ICIR/IC胜率
- 生成分组收益柱状图
- 不同模型对比表

---

## 5. 编码规范与约定

遵循 **Google Python Style Guide**，要点：

### 命名

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块/包 | lowercase | `preprocess.py`, `data_loader.py` |
| 类 | CapWords | `MLPAlphaModel`, `FactorDataset` |
| 函数/方法 | lowercase_underscore | `mad_clip_section`, `build_return_label` |
| 常量 | UPPER_SNAKE | `MAD_MULTIPLIER`, `LEARNING_RATE` |
| 变量 | lowercase_underscore | `train_loss`, `hidden_output` |

### Docstring

所有公开函数和类必须写docstring，格式：

```python
def mad_clip_section(df: pd.DataFrame, col: str, n: float = 3) -> pd.Series:
    """对指定列做截面MAD去极值。

    Args:
        df: 包含date列和因子列的DataFrame。
        col: 需要处理的列名。
        n: MAD倍数，默认3。

    Returns:
        处理后的Series，极值被截断到[n*MAD, median+n*MAD]。
    """
```

### 类型注解

所有函数签名必须加类型注解：

```python
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer, loss_fn: Callable) -> float:
```

### 其他约定

- 每个文件顶部：`"""模块级docstring，说明该模块职责。"""`
- import顺序：标准库 → 第三方库 → 本项目模块，各组间空一行
- 行宽上限120字符
- 使用f-string而非format/%
- 禁止裸except，必须指定异常类型
- 配置参数一律从config.py读取，禁止硬编码

### 文件头部注释规范

所有Python文件必须包含以下标准化头部注释，紧跟编码声明之后：

```python
# -*- coding: utf-8 -*-
# @File    : [当前文件名]
# @Time    : [创建日期时间，格式 YYYY/MM/DD HH:MM]
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了XXX功能的类和函数。

[模块详细说明]
"""
```

要求：
1. `@File` 必须与实际文件名一致
2. `@Time` 使用创建/修改时的日期时间
3. 模块docstring中"XXX"需根据文件实际功能生成简述

### Google Python Style Guide 详细规范

以下为本项目强制执行的Google风格注释规范完整说明：

#### 模块级Docstring

每个模块文件必须在头部注释后紧跟模块级docstring，说明模块的整体职责：

```python
"""本模块提供了[功能简述]的类和函数。

[可选的详细说明、设计要点、使用注意事项等]
"""
```

#### 函数/方法Docstring

所有公开函数和类方法必须使用Google风格docstring，包含以下段落（按顺序）：

1. **摘要行**：一句话描述函数功能，以句号结尾。
2. **详细描述**（可选）：多行详细说明。
3. **Args**：逐个列出参数，格式 `参数名: 描述。`
4. **Returns**：描述返回值。
5. **Raises**（可选）：列出可能抛出的异常。

```python
def build_return_label(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """构建T+1到T+period收益率标签，再做截面zscore。

    标签 = T日买入持有period天的收益率。
    使用shift(-period)向前偏移，确保标签不含未来信息泄露。

    Args:
        df: 包含date, stock_code, close列的DataFrame。
        period: 持有期天数，默认20。

    Returns:
        截面zscore化后的收益率标签Series。

    Raises:
        ValueError: 当DataFrame缺少必要列时。
    """
```

#### 类Docstring

类docstring应描述类的用途，并使用Attributes段落列出公开属性：

```python
class FactorDataset(Dataset):
    """按交易日截面加载的因子数据集。

    每个样本 = 一个交易日截面的所有股票。
    __getitem__返回该截面的(factor_tensor, label_tensor)。

    Attributes:
        dates: 排序后的日期列表。
        factor_cols: 使用的因子列名。
        data: 完整DataFrame。
    """
```

#### Docstring格式要点

| 规则 | 说明 | 示例 |
|------|------|------|
| 摘要行 | 第三人称动词开头，一句完整陈述 | `计算Spearman秩相关系数。` |
| Args段 | 每个参数占一行，名称后冒号+空格+描述 | `n: MAD倍数，默认3。` |
| Returns段 | 描述返回值类型和含义 | `处理后的Series，极值被截断。` |
| Raises段 | 仅列出公开抛出的异常 | `ValueError: 股票代码格式错误。` |
| 空行分隔 | 摘要行与详细描述之间空一行 | 见上方示例 |
| 缩进 | Docstring内内容与起始引号对齐 | 缩进4空格 |

#### 类型注解规范

所有函数签名必须加类型注解，遵循PEP 526和PEP 484：

```python
# 基本类型
def rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:

# 可选参数
def fetch_stock_data(stock_pool: list[str] | None = None) -> pd.DataFrame:

# 复杂返回类型
def split_dataset(df: pd.DataFrame) -> tuple[FactorDataset, FactorDataset, FactorDataset]:

# Callable类型
from typing import Callable
def train_one_epoch(model: nn.Module, loss_fn: Callable) -> float:
```

#### import顺序

按以下三组排列，各组之间空一行，组内按字母序排列：

```python
# 1. 标准库
import json
import time
from pathlib import Path

# 2. 第三方库
import numpy as np
import pandas as pd
import torch

# 3. 本项目模块
from config import DEVICE, LEARNING_RATE
from models.mlp_alpha import MLPAlphaModel
```

---

## 6. 开发流程与任务规划

### 执行顺序（严格按天推进）

| 天 | 任务 | 对应模块 | 交付物 |
|----|------|----------|--------|
| D1 | 搭建目录结构，写config.py，数据获取脚本 | config.py, utils/data_loader.py | CSV到data/raw/ |
| D2 | MAD去极值 + zscore标准化 | utils/preprocess.py | 预处理后CSV到data/processed/ |
| D3 | 收益率标签构建 + 完整预处理pipeline | utils/preprocess.py | label列生成，验证截面统计 |
| D4 | PyTorch Dataset + 线性基准模型 | utils/dataset.py, models/linear_alpha.py | 线性模型跑通1-2 epoch |
| D5 | MLP模型搭建 | models/mlp_alpha.py | forward输出shape正确 |
| D6 | 训练循环 + 早停 | train.py | loss曲线图 |
| D7 | 正交惩罚 + 三种损失函数 | losses.py | 三种loss分别跑出结果 |
| D8 | RankIC评估 + 分组回测 | utils/metrics.py, evaluate.py | RankIC数值 + 分组收益图 |
| D9 | 滚动训练 | train.py | 滚动窗口结果对比 |
| D10 | 整理notebook | notebooks/ | 可运行reproduction.ipynb |

### 每步完成后的验证

- D1：`pd.read_csv()` 能打印shape和head
- D2：截面内均值≈0、标准差≈1
- D3：label无NaN，shift方向正确
- D4：模型输出shape=(N,)，loss可计算
- D5：`model(x)` 返回(pred, hidden)，pred.shape=(N,)，hidden.shape=(N,64)
- D6：train_loss单调下降，val_loss有早停
- D7：正交惩罚项>0，三种loss数值合理
- D8：RankIC均值>0.03，分组收益大致单调
- D9：滚动预测可拼接为完整序列
- D10：notebook从头运行无报错

---

## 7. 关键依赖与配置说明

### 依赖清单

```
pandas>=2.0
numpy>=1.24
torch>=2.0        # CUDA版，RTX 4060专用
scipy>=1.10       # spearmanr
scikit-learn>=1.2
matplotlib>=3.7
akshare>=1.10     # 数据源，可选tushare替代
jupyter           # notebook环境
```

安装命令（**必须用CUDA版PyTorch**）：
```bash
# 先装PyTorch CUDA版（RTX 4060支持CUDA 12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# 再装其余依赖
pip install pandas numpy scipy scikit-learn matplotlib akshare jupyter
```

验证GPU可用：
```python
import torch
print(torch.cuda.is_available())   # 应输出 True
print(torch.cuda.get_device_name(0))  # 应输出 NVIDIA GeForce RTX 4060
```

### 数据源选择

| 方案 | 安装 | 优点 | 缺点 |
|------|------|------|------|
| **BaoStock** | `pip install baostock` | 免费无需注册、TCP连接不受代理影响、含PE/PB/PS | 数据更新有延迟 |
| AKShare | `pip install akshare` | 数据全面 | 需VPN才能访问东方财富API（见§9.5） |
| Tushare | `pip install tushare` | 数据质量好 | 需注册+积分 |
| JoinQuant | 网页端操作 | 因子数据最全 | 免费版有限制 |

**推荐使用 BaoStock**（默认），无需 VPN，数据源切换在 data_loader.py 中配置。
详见 §9.5 网络代理与数据源。

### 环境信息

- Python：3.14.0（项目.venv已配置）
- 项目路径：`D:/.shigodo/shigodo/Quantification/Project-Alpha/`
- Git已初始化

---

## 8. 关键量化概念速查（CB必读）

以下是本项目涉及的核心量化概念，CB在编码时需严格遵守：

### 截面操作（Cross-Section）

**定义**：截面 = 某一个交易日所有股票的数据快照。

**铁律**：所有标准化、去极值操作必须在 `groupby('date')` 截面内执行。全局操作会导致数据泄露（未来信息混入），RankIC虚高但实际无效。

```python
# 正确 ✓
df[factor] = df.groupby('date')[factor].transform(lambda x: (x - x.mean()) / x.std())
# 错误 ✗
df[factor] = (df[factor] - df[factor].mean()) / df[factor].std()
```

### 时序划分（Temporal Split）

**铁律**：训练/验证/测试集必须按时间切分，严禁随机split。随机划分会导致未来信息泄露。

```python
# 正确 ✓
train = df[df['date'] < '2022-06-30']
val = df[(df['date'] >= '2022-06-30') & (df['date'] < '2023-06-30')]
test = df[df['date'] >= '2023-06-30']
# 错误 ✗
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)  # 严禁！
```

### 标签构建

```python
# T日的标签 = T+1到T+20的收益率
df['label'] = df.groupby('stock_code')['close'].pct_change(20).shift(-20)
# shift(-20)是关键：用未来20天数据，所以必须往前移20行
```

### RankIC

Spearman秩相关系数，衡量预测排序能力。每个截面算一次，再汇总：
- RankIC均值 > 0.03 算有效
- ICIR = RankIC均值 / RankIC标准差 > 0.5 算稳定
- IC胜率 = IC>0的截面占比 > 50% 算有效

### 正交惩罚

```
H: 隐藏层输出 (N, 64)
C = H^T @ H           # 协方差矩阵
L_orth = ||C - diag(C)||_F  # 只惩罚非对角（协方差），保留对角（方差）
```

### 三种损失函数

| 损失 | 公式要点 | 特点 |
|------|----------|------|
| MSE | (y - y_hat)^2 均值 | 基准，稳定 |
| IC | -Pearson(pred, target) | 直接优化排序，多头最好 |
| CCC | -2*cov/(var_pred+var_target+(mean_pred-mean_target)^2) | 结合MSE+IC，最稳定 |

### 激活函数选择

前两层用Sigmoid（不用ReLU），原因：
1. 因子zscore后≈正态分布，均值0
2. BN层把分布压在0附近
3. ReLU在x<0时梯度=0，约50%神经元"死亡"
4. Sigmoid在0附近有非零梯度

最后一层用ReLU：输出zscore，只关心相对排序。

---

## 9. 常见坑点

| 坑 | 症状 | 解决 |
|----|------|------|
| 随机split | 训练loss低但测试极差 | 按时间切分 |
| 全局标准化 | RankIC虚高 | 截面内标准化 |
| 全用ReLU | loss不下降 | 前两层改Sigmoid |
| batch按固定size | shape不匹配 | 按截面group，每个截面一个batch |
| NaN传播 | loss变nan | MAD后检查NaN，fillna(0) |
| 标签泄露 | IC=100%+ | return列shift(-20) |
| 学习率过大 | loss震荡/nan | lr=0.0005 |
| 数据太少 | 不收敛 | 至少20只股票3年 |
| tensor没to(device) | RuntimeError: Expected all tensors on same device | 所有输入tensor和模型都要.to(DEVICE) |
| 评估时忘转cpu | spearmanr收到GPU tensor | 先.cpu().detach().numpy() |
| CUDA OOM | 隐藏层太大 | RTX 4060有8GB显存，64维隐藏层足够，减少epoch内打印频率 |
| VPN/代理导致数据获取失败 | ProxyError / RemoteDisconnected | 见下文"网络代理与数据源" |

---

## 9.5 网络代理与数据源

### 问题背景

开发机常驻 VPN/梯子（如 Clash、v2rayN 等），会设置 Windows 系统代理（`127.0.0.1:PORT`）。
这会导致基于 HTTP 的数据源（如 AKShare/东方财富 API）出现 `ProxyError` 或 `RemoteDisconnected`。

### 根因分析

| 场景 | push2his.eastmoney.com 状态 | 说明 |
|------|-----------------------------|------|
| VPN 开启 + 代理正常 | 可达（经VPN隧道） | 流量经代理→VPN服务器→东方财富 |
| VPN 开启 + 代理端口不通 | ProxyError | Windows 指向代理端口但服务未监听 |
| VPN 关闭 | RemoteDisconnected | 直连东方财富 API 服务器断开 TLS 连接 |

**关键结论**：`push2his.eastmoney.com`（东方财富历史行情 API）从本机直连网络**不可达**，
TLS 握手后服务器主动断开连接（疑似 CDN/ISP 层面限制）。只有通过 VPN 隧道才能访问。

### 代理绕行规则说明

VPN 软件内置的系统代理绕行规则：
```
localhost, 127.*, 10.*, 172.16.*-172.20.*, 192.168.*, *.fine-smart.com, <local>
```

**这些绕行规则无法解决数据获取问题**，原因：
1. 绕行 = 直连，但 `push2his.eastmoney.com` 从直连网络不可达
2. 添加 `*.eastmoney.com` 到绕行列表会使情况**更糟**（绕过 VPN 隧道后直连仍失败）
3. 该 API **需要** VPN 隧道才能访问

### 解决方案：BaoStock 主数据源

项目已切换到 **BaoStock** 作为主数据源（v3.0 data_loader.py）：

| 特性 | BaoStock | AKShare |
|------|----------|---------|
| 协议 | TCP 长连接 | HTTP/HTTPS |
| 代理影响 | **不受影响** | 受系统代理影响 |
| 数据覆盖 | OHLCV + PE/PB/PS + 财务 | OHLCV + 更多衍生指标 |
| 注册要求 | 免费，无需注册 | 免费，无需注册 |
| VPN 要求 | **不需要** | 需要 VPN 开启 |

### 操作指引

- **BaoStock（默认）**：直接运行 `python -m utils.data_loader`，无需 VPN
- **AKShare（备选）**：需确保 VPN 开启且代理端口正常监听
- 若遇到 ProxyError：检查 VPN 是否开启，或切换到 BaoStock 数据源

---

## 10. 研究提示（论文章节映射）

> 每个开发阶段对应论文的具体章节，编码前应阅读对应章节。以下是每个阶段你需要研究的论文内容以及相关延伸知识点，帮助你理解"为什么"而不仅是"怎么做"。

### Phase 1 — 数据获取与预处理（对应论文 §1.1 + 附录）

**必读章节**：
- §1.1 多因子Alpha模型构建流程（P4）：理解传统因子模型的5步流程（数据清洗→因子构建→分类合成→权重组合→收益预测），端到端网络就是用MLP替代后3步
- 附录 因子说明表（P21-22）：所有因子的定义、方向、计算方法

**延伸研究**：
- **因子方向**：为什么lncap是负方向（小市值溢价）、换手率是负方向（低流动性溢价）？这是Fama-French三因子和后续学术研究的经典结论
- **MAD vs 3σ截断**：研究稳健统计量（robust statistics），理解为什么金融数据需要用基于中位数的截断而非基于均值的截断
- **市值+行业中性化**：论文表1提到"市值、行业中性"，这通过WLS回归取残差实现（Barra模型用1/√(市值)做权重）。简化版可跳过此步，但面试需要知道

### Phase 2 — 线性基准模型（对应论文 §1.2 + §2.1）

**必读章节**：
- §1.2 当前因子模型遇到的问题（P5-7）：理解线性模型的4个缺陷——APT线性假设不成立、多头拥挤、量价因子负Alpha属性、策略趋同
- §2.1 线性基准网络（P8-10）：网络结构 `Linear(P,64)→BN→Linear(64,6)→BN→Linear(6,1)`，第一层=细分因子合成大类因子，第二层=大类因子组合预测

**延伸研究**：
- **APT定价模型**：了解套利定价理论，理解为什么线性因子模型理论上需要一个"随机折现因子线性结构"的假设，以及实际数据如何违反这些假设
- **Fama-Macbeth回归**：传统方法中处理截面相关性的标准工具，理解为什么神经网络不需要这个（端到端训练隐式处理）
- **BP因子拥挤现象**：论文图4-6展示BP因子在2019-2020几乎失效，理解因子IC≠因子可用性（空头贡献占主导时，实际多头选股效果差）
- **线性网络结构的经济含义**：64维隐藏层=64个"学习到的大类因子"，6维=6个"大类因子组合"，这与传统因子模型的结构一一对应

### Phase 3 — MLP非线性模型（对应论文 §2.2）

**必读章节**：
- §2.2 非线性Alpha网络（P10-12）：完整网络结构、表2的参数设置、表3-4的对比结果

**延伸研究**：
- **激活函数选择**：论文图16专门对比了ReLU vs Sigmoid。深入理解：为什么BN+ReLU组合在CV中没问题（输入是像素值0-255非负），但在因子模型中是致命的（输入是zscore≈0对称分布）
- **隐藏层神经元个数公式**：论文给出 `N_h = √(α*(N_i+N_o))`，α∈[2,10]。研究这个经验法则的来源（参考论文引用3: Jason Brownlee, How to Configure the Number of Layers and Nodes in a Neural Network）
- **2层隐藏层的理论意义**：0层=线性可分，2层=任意有限空间连续映射，3+层=自动特征工程。论文选3层隐藏层，思考为什么金融因子需要这种深度
- **论文表3的关键发现**：MLP的RankIC(9.27%)比线性基准(10.43%)略低，但ICIR更高、多空夏普更好——非线性换的不是排序能力，而是稳定性。理解这个trade-off

### Phase 4 — 正交惩罚（对应论文 §2.3）

**必读章节**：
- §2.3 因子正交化正则（P13-14）：完整推导过程，从传统共线性处理到正交惩罚的数学过渡

**延伸研究**：
- **传统共线性处理**：了解LASSO、岭回归如何处理多重共线性，以及它们的局限性（只能处理线性相关性，无法处理非线性交互）
- **为什么惩罚F范数不够**：论文明确指出 `||X^TX||_F` 会同时惩罚方差和协方差，但方差是有用的（信息量），所以只惩罚非对角元素。理解这个设计选择
- **正交惩罚与PCA的关系**：PCA做的是对X^TX做特征分解取主成分，正交惩罚是直接压制非对角元。两者目的类似但手段不同——PCA是变换，正交惩罚是正则
- **λ的选择**：论文没有给出λ的具体值，从0.01开始。研究正则化系数的一般调参策略（grid search / 验证集loss曲线观察）
- **论文表5关键发现**：正交惩罚后RankIC从9.27%→10.07%仅小幅提升，但多头对冲年化从13.25%→15.91%提升显著——正交惩罚的主要收益不在IC而在多头端

### Phase 5 — 三种损失函数（对应论文 §2.4）

**必读章节**：
- §2.4 优化目标的改进（P14-16）：MSE/IC/CCC三种loss的数学定义和实验对比

**延伸研究**：
- **MSE与IC的等价条件**：论文指出线性模型中优化MSE≈优化IC（因为R²=ρ²），但在深度网络中这个等价关系被打破。研究为什么（随机初始化、非凸优化、batch训练等扰动因素）
- **Pearson相关系数的非凸性**：理解为什么直接用IC做loss会收敛困难（目标函数非凸，存在多个局部最优），这解释了为什么论文推荐CCC作为折中
- **CCC损失函数**：论文引用5（Liao & Lewis, 2000）是CCC的原始论文。CCC = ρ*MSE项/方差项，同时优化相关性和绝对偏差
- **论文表6的核心洞察**：IC loss多头最好(11.02% RankIC)，CCC稳定性最好(5.26夏普, -6.60%最大回撤)，**但CCC的RankIC最低(9.11%)**——这是论文最重要的结论之一：IC单一指标无法正确评估多头选股能力
- **参考论文**：Gu, Kelly & Xiu (2021) "Autoencoder Asset Pricing Models"——同类工作中对损失函数的讨论

### Phase 6 — 模型分析与归因（对应论文 §2.5）

**必读章节**：
- §2.5 模型分析（P16-17）：SHAP归因方法论、表7风格因子相关性矩阵
- §2.5续 不同成分股中的选股效果（P18）：中证500/800/1000的表现差异，表8

**延伸研究**：
- **SHAP（Shapley Additive Explanations）**：了解合作博弈论中Shapley值的概念，以及SHAP如何将其应用于神经网络的可解释性。论文图26-27展示了特征重要性排序
- **论文关键发现**：流动性因子>量价指标>基本面因子。理解为什么（量价因子信息更新频率更高，更容易捕获短期Alpha；基本面因子更新慢，更适合长期选股）
- **表7的相关性矩阵**：模型输出zscore与流动性(-0.46)和残差波动率(-0.43)的负相关性最高，与成长因子(0.01)几乎无关——说明MLP自动学会了与传统因子模型不同的因子组合
- **不同股票池的差异**：中证1000>中证500>中证800，因为小盘股定价效率更低，非线性模型的增量信息更多。理解市场微观结构与Alpha的关系

### Phase 7 — 滚动训练（对应论文表1 + 图10）

**必读章节**：
- 表1 固定参数说明（P8）：滚动训练模式的具体设置
- 图10 数据集划分说明（P8）：训练集/验证集/测试集的时间窗口定义

**延伸研究**：
- **论文的滚动方式**：训练集用所有历史数据，验证集取最后252个交易日，测试集为最近半年。这不是固定窗口而是扩展窗口（expanding window）
- **为什么不用滑动窗口**：金融数据样本珍贵，扩展窗口利用了所有历史信息；滑动窗口会丢弃早期数据
- **过拟合风险**：滚动训练的目的是验证模型的样本外表现，不是调参。避免在测试集上反复调参后再测试（这等于在测试集上训练）
- **参考论文**：Kelly, Pruitt & Su (2019) "Characteristics are Covariances"——同类工作中的滚动训练方法论
