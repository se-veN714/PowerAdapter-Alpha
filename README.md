# Project-Alpha

复现招商证券《端到端的动态Alpha模型》(2023)，基于MLP的端到端因子权重训练与收益率预测。

## 环境要求

- Python 3.12.7
- NVIDIA RTX 4060 (8GB) + CUDA 13.0 驱动 (581.15+)
- Windows 64-bit

## 快速开始

### 1. 创建虚拟环境

```powershell
cd D:\.shigodo\shigodo\Quantification\Project-Alpha
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装PyTorch（CUDA版）

驱动版本 581.15 支持 CUDA 13.0，向下兼容 cu124：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证GPU：

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 4060
```

### 3. 安装其余依赖

```powershell
pip install -r requirements.txt
```

### 4. 初始化项目目录

```powershell
python init_project.py
```

### 5. 运行流程

```powershell
# D1: 数据获取
python -m utils.data_loader

# D2-D3: 预处理
python -m utils.preprocess

# D4-D7: 训练（线性基准 + MLP + 三种损失）
python train.py

# D8: 评估
python evaluate.py
```

## 项目结构

```
Project-Alpha/
├── GUIDE.md                 # 项目完整指南
├── README.md                # 本文档
├── init_project.py          # 项目初始化脚本
├── config.py                # 全局参数配置
├── requirements.txt         # 依赖清单
├── data/
│   ├── raw/                 # 原始CSV数据
│   └── processed/           # 预处理后CSV
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # 数据获取（AKShare）
│   ├── preprocess.py        # 因子预处理（MAD/zscore/标签）
│   ├── dataset.py           # PyTorch Dataset（按截面加载）
│   └── metrics.py           # RankIC/ICIR/分组收益
├── models/
│   ├── __init__.py
│   ├── linear_alpha.py      # 线性基准模型
│   └── mlp_alpha.py         # MLP非线性模型
├── losses.py                # 三种损失 + 正交惩罚
├── train.py                 # 训练入口（早停/滚动训练）
├── evaluate.py              # 评估入口（对比表/分组图）
├── checkpoints/             # 模型保存目录
├── logs/                    # 训练日志和曲线
└── notebooks/
    └── reproduction.ipynb   # 最终交付notebook
```

## RTX 4060 显存优化策略

| 策略 | 说明 |
|------|------|
| 隐藏层64维 | 论文设定，8GB显存完全够用 |
| 按截面batch | 每个截面约25只股票，单batch极小 |
| 截面内shuffle | 训练时shuffle截面顺序，不改变截面内容 |
| 早停 | 防止过拟合，同时减少不必要的epoch |
| 避免频繁打印 | 减少CPU-GPU同步点 |
