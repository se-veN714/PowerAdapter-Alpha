# -*- coding: utf-8 -*-
# @File    : init_project.py
# @Time    : 2026/06/09 17:51
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了项目初始化的功能，包括目录创建和环境验证。

创建所有缺失的目录结构，验证环境和依赖。
"""

import sys
from pathlib import Path

# 修复Windows GBK终端编码问题
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 需要创建的目录
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "models",
    "utils",
    "checkpoints",
    "logs",
    "notebooks",
]

# 需要验证的Python包
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "torch": "torch",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "akshare": "akshare",
}


def create_directories() -> None:
    """创建所有缺失的项目目录。"""
    print("=" * 50)
    print("创建项目目录结构")
    print("=" * 50)

    for dir_path in REQUIRED_DIRS:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}/ (已存在)")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  [++] {dir_path}/ (已创建)")

    # 创建 __init__.py 占位（如果不存在）
    init_dirs = ["models", "utils"]
    for dir_name in init_dirs:
        init_file = PROJECT_ROOT / dir_name / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"  [++] {dir_name}/__init__.py (已创建)")


def verify_environment() -> None:
    """验证Python版本和虚拟环境。"""
    print("\n" + "=" * 50)
    print("环境验证")
    print("=" * 50)

    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor != 12:
        print(f"  [!] 建议使用Python 3.12.x，当前为 {version.major}.{version.minor}")

    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    venv_status = "[OK] 已激活" if in_venv else "[!] 未检测到虚拟环境"
    print(f"  虚拟环境: {venv_status}")
    print(f"  Python路径: {sys.executable}")


def verify_gpu() -> None:
    """验证CUDA/GPU可用性。"""
    print("\n" + "=" * 50)
    print("GPU验证")
    print("=" * 50)

    try:
        import torch
    except ImportError:
        print("  [X] PyTorch未安装，无法验证GPU")
        return

    if not torch.cuda.is_available():
        print("  [!] CUDA不可用，将使用CPU训练")
        print("  请检查PyTorch是否为CUDA版本")
        return

    print(f"  [OK] CUDA可用")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA版本: {torch.version.cuda}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  显存: {vram_gb:.1f} GB")
    print(f"  PyTorch版本: {torch.__version__}")


def verify_packages() -> None:
    """验证必需的Python包。"""
    print("\n" + "=" * 50)
    print("依赖验证")
    print("=" * 50)

    missing: list[str] = []

    for import_name, display_name in REQUIRED_PACKAGES.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {display_name}: {version}")
        except ImportError:
            print(f"  [X] {display_name}: 未安装")
            missing.append(display_name)

    if missing:
        print(f"\n  缺失包: {', '.join(missing)}")
        print("  运行: pip install -r requirements.txt")


def main() -> None:
    """执行完整的项目初始化流程。"""
    print("Project-Alpha 初始化")
    print()

    create_directories()
    verify_environment()
    verify_gpu()
    verify_packages()

    print("\n" + "=" * 50)
    print("初始化完成！")
    print("=" * 50)
    print("\n下一步:")
    print("  1. python -m utils.data_loader    # 获取数据")
    print("  2. python -m utils.preprocess     # 预处理")
    print("  3. python train.py                # 训练模型")
    print("  4. python evaluate.py             # 评估模型")


if __name__ == "__main__":
    main()
