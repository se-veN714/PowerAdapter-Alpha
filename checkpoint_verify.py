# -*- coding: utf-8 -*-
# @File    : checkpoint_verify.py
# @Time    : 2026/06/09 17:55
# @Author  : seveN1foR
# @Version : 1.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了项目检查点验证的函数，用于逐项确认各阶段交付物的完整性。

每个检查点对应GUIDE.md中的一天任务，包含明确的通过/失败判定标准。
"""

import sys
import importlib
from pathlib import Path

# 修复Windows GBK终端编码问题
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 检查点1待办任务清单
CHECKPOINT1_TASKS = {
    "dir_structure": {
        "description": "项目目录结构完整",
        "required_dirs": [
            "data/raw",
            "data/processed",
            "models",
            "utils",
            "checkpoints",
            "logs",
            "notebooks",
        ],
    },
    "dir_init_files": {
        "description": "包__init__.py文件存在",
        "required_files": [
            "models/__init__.py",
            "utils/__init__.py",
        ],
    },
    "pip_mirror": {
        "description": "pip镜像源已配置为国内源",
    },
    "python_version": {
        "description": "Python版本为3.12.x",
        "expected_major": 3,
        "expected_minor": 12,
    },
    "venv_active": {
        "description": "虚拟环境已激活",
    },
    "pytorch_cuda": {
        "description": "PyTorch CUDA版安装成功",
        "package": "torch",
        "check_cuda": True,
    },
    "core_packages": {
        "description": "核心依赖包已安装",
        "packages": [
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("scipy", "scipy"),
            ("sklearn", "scikit-learn"),
            ("matplotlib", "matplotlib"),
            ("akshare", "akshare"),
            ("jupyter", "jupyter"),
        ],
    },
    "source_files": {
        "description": "核心源码文件存在",
        "required_files": [
            "config.py",
            "train.py",
            "evaluate.py",
            "losses.py",
            "utils/data_loader.py",
            "utils/preprocess.py",
            "utils/dataset.py",
            "utils/metrics.py",
            "models/linear_alpha.py",
            "models/mlp_alpha.py",
        ],
    },
    "file_headers": {
        "description": "Python文件包含标准化头部注释",
        "header_marker": "# -*- coding: utf-8 -*-",
        "author_marker": "@Author  : seveN1foR",
    },
    "gpu_available": {
        "description": "GPU可被PyTorch识别",
    },
}


def _pass(msg: str) -> str:
    """生成通过标记。

    Args:
        msg: 通过信息。

    Returns:
        带PASS前缀的字符串。
    """
    return f"  [PASS] {msg}"


def _fail(msg: str) -> str:
    """生成失败标记。

    Args:
        msg: 失败信息。

    Returns:
        带FAIL前缀的字符串。
    """
    return f"  [FAIL] {msg}"


def check_dir_structure() -> bool:
    """验证项目目录结构完整性。

    Returns:
        所有目录均存在则返回True。
    """
    print("\n--- 1. 目录结构 ---")
    task = CHECKPOINT1_TASKS["dir_structure"]
    all_ok = True
    for dir_path in task["required_dirs"]:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists() and full_path.is_dir():
            print(_pass(f"{dir_path}/"))
        else:
            print(_fail(f"{dir_path}/ 不存在"))
            all_ok = False
    return all_ok


def check_init_files() -> bool:
    """验证包__init__.py文件。

    Returns:
        所有__init__.py均存在则返回True。
    """
    print("\n--- 2. 包初始化文件 ---")
    task = CHECKPOINT1_TASKS["dir_init_files"]
    all_ok = True
    for file_path in task["required_files"]:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists() and full_path.is_file():
            print(_pass(file_path))
        else:
            print(_fail(f"{file_path} 不存在"))
            all_ok = False
    return all_ok


def check_pip_mirror() -> bool:
    """验证pip镜像源配置。

    Returns:
        镜像源已配置为国内源则返回True。
    """
    print("\n--- 3. pip镜像源 ---")
    try:
        import pip._internal.configuration as cfg
        from pip._internal.commands import create_command

        config = create_command("config")._get_configuration()
        index_url = config.get_value("global.index-url")
        if index_url and "tsinghua" in str(index_url):
            print(_pass(f"镜像源: {index_url}"))
            return True
        elif index_url:
            print(_fail(f"镜像源非清华: {index_url}"))
            return False
    except Exception:
        pass

    # 备选：直接读pip.ini
    pip_ini = Path.home() / "AppData" / "Roaming" / "pip" / "pip.ini"
    if pip_ini.exists():
        content = pip_ini.read_text(encoding="utf-8", errors="replace")
        if "tsinghua" in content:
            for line in content.splitlines():
                if "index-url" in line:
                    print(_pass(f"镜像源: {line.strip()}"))
                    return True
        print(_fail("pip.ini中未找到清华镜像配置"))
        return False

    print(_fail("未找到pip配置文件"))
    return False


def check_python_version() -> bool:
    """验证Python版本。

    Returns:
        Python版本为3.12.x则返回True。
    """
    print("\n--- 4. Python版本 ---")
    task = CHECKPOINT1_TASKS["python_version"]
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == task["expected_major"] and v.minor == task["expected_minor"]:
        print(_pass(f"Python {version_str}"))
        return True
    print(_fail(f"Python {version_str}，期望 3.12.x"))
    return False


def check_venv() -> bool:
    """验证虚拟环境已激活。

    Returns:
        虚拟环境已激活则返回True。
    """
    print("\n--- 5. 虚拟环境 ---")
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(_pass(f"已激活: {sys.prefix}"))
        return True
    print(_fail("虚拟环境未激活"))
    return False


def check_pytorch_cuda() -> bool:
    """验证PyTorch CUDA安装。

    Returns:
        PyTorch CUDA版可用则返回True。
    """
    print("\n--- 6. PyTorch CUDA ---")
    try:
        import torch
        print(_pass(f"PyTorch {torch.__version__}"))
        if torch.cuda.is_available():
            print(_pass(f"CUDA {torch.version.cuda}"))
            print(_pass(f"GPU: {torch.cuda.get_device_name(0)}"))
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(_pass(f"显存: {vram:.1f} GB"))
            return True
        print(_fail("CUDA不可用，请安装CUDA版PyTorch"))
        return False
    except ImportError:
        print(_fail("PyTorch未安装"))
        print("  安装命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return False


def check_core_packages() -> bool:
    """验证核心依赖包。

    Returns:
        所有核心包均已安装则返回True。
    """
    print("\n--- 7. 核心依赖包 ---")
    task = CHECKPOINT1_TASKS["core_packages"]
    all_ok = True
    for import_name, display_name in task["packages"]:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(_pass(f"{display_name}: {version}"))
        except ImportError:
            print(_fail(f"{display_name}: 未安装"))
            all_ok = False
    return all_ok


def check_source_files() -> bool:
    """验证核心源码文件存在。

    Returns:
        所有源码文件均存在则返回True。
    """
    print("\n--- 8. 核心源码文件 ---")
    task = CHECKPOINT1_TASKS["source_files"]
    all_ok = True
    for file_path in task["required_files"]:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists() and full_path.is_file():
            print(_pass(file_path))
        else:
            print(_fail(f"{file_path} 不存在"))
            all_ok = False
    return all_ok


def check_file_headers() -> bool:
    """验证Python文件包含标准化头部注释。

    Returns:
        所有Python文件头部注释合规则返回True。
    """
    print("\n--- 9. 文件头部注释规范 ---")
    task = CHECKPOINT1_TASKS["file_headers"]
    py_files = list(PROJECT_ROOT.glob("*.py")) + list(
        PROJECT_ROOT.glob("utils/*.py")
    ) + list(PROJECT_ROOT.glob("models/*.py"))

    all_ok = True
    for py_file in sorted(py_files):
        if py_file.name == "checkpoint_verify.py":
            continue
        rel_path = py_file.relative_to(PROJECT_ROOT)
        try:
            content = py_file.read_text(encoding="utf-8")
            has_encoding = task["header_marker"] in content
            has_author = task["author_marker"] in content
            if has_encoding and has_author:
                print(_pass(f"{rel_path}"))
            else:
                missing = []
                if not has_encoding:
                    missing.append("coding声明")
                if not has_author:
                    missing.append("@Author")
                print(_fail(f"{rel_path} 缺少: {', '.join(missing)}"))
                all_ok = False
        except Exception as e:
            print(_fail(f"{rel_path} 读取失败: {e}"))
            all_ok = False
    return all_ok


def check_gpu() -> bool:
    """验证GPU可用性。

    Returns:
        GPU可被PyTorch识别则返回True。
    """
    print("\n--- 10. GPU验证 ---")
    try:
        import torch
        if not torch.cuda.is_available():
            print(_fail("GPU不可用"))
            return False
        # 简单CUDA运算测试
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        print(_pass("CUDA计算测试通过"))
        return True
    except ImportError:
        print(_fail("PyTorch未安装，跳过GPU测试"))
        return False
    except Exception as e:
        print(_fail(f"GPU测试失败: {e}"))
        return False


def run_checkpoint1() -> dict[str, bool]:
    """执行检查点1全部验证项。

    Returns:
        各验证项的通过/失败结果字典。
    """
    print("=" * 60)
    print("  检查点1 - 环境搭建验证")
    print("  对应 GUIDE.md D1阶段")
    print("=" * 60)

    checks = {
        "目录结构": check_dir_structure(),
        "包初始化文件": check_init_files(),
        "pip镜像源": check_pip_mirror(),
        "Python版本": check_python_version(),
        "虚拟环境": check_venv(),
        "PyTorch CUDA": check_pytorch_cuda(),
        "核心依赖包": check_core_packages(),
        "源码文件": check_source_files(),
        "文件头部注释": check_file_headers(),
        "GPU可用性": check_gpu(),
    }

    print("\n" + "=" * 60)
    print("  检查点1 汇总")
    print("=" * 60)

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  结果: {passed}/{total} 项通过")

    if passed == total:
        print("\n  检查点1 全部通过! 可进入下一阶段开发。")
    else:
        print("\n  存在未通过项，请修复后重新验证。")
        failed = [k for k, v in checks.items() if not v]
        print(f"  未通过项: {', '.join(failed)}")

    return checks


if __name__ == "__main__":
    run_checkpoint1()
