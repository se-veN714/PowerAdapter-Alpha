# -*- coding: utf-8 -*-
# @File    : data_loader.py
# @Time    : 2026/06/09 18:30
# @Author  : seveN1foR
# @Version : 3.0
# @Software: PyCharm
# @Contact : qingyudong942@gmail.com

"""本模块提供了从数据源拉取行情和因子数据的类和函数。

主数据源为 BaoStock（免费、无需代理），备选为 AKShare（需网络畅通）。
输出统一格式 CSV 到 data/raw/，包含行情、估值与衍生因子。
支持代理检测、自动重试和断点续传机制。
"""

import json
import os
import time
from pathlib import Path
from typing import Final

import baostock as bs
import pandas as pd

from config import RAW_DATA_DIR, STOCK_POOL, DATE_RANGE

# ===== 代理配置 =====
# 本项目默认使用 BaoStock，走 TCP 长连接，不受 HTTP 代理影响。
# 若需切换到 AKShare 数据源，可能需要 VPN/代理：
#   - VPN 开启时：AKShare 可通过代理隧道访问东方财富 API
#   - VPN 关闭时：push2his.eastmoney.com 不可达，AKShare 会失败
# 若需使用代理，设置环境变量或修改下方配置
CUSTOM_PROXY: str | None = None

# ===== 请求配置 =====
MAX_RETRIES: int = 5
RETRY_BACKOFF: float = 2.0
DEFAULT_DELAY: float = 0.3       # BaoStock 请求间隔（秒）
BATCH_PAUSE: float = 5.0         # 每批暂停（秒）
BATCH_SIZE: int = 10             # 每批股票数量

# ===== 断点续传进度文件 =====
PROGRESS_FILE: Final = RAW_DATA_DIR / "_fetch_progress.json"

# ===== BaoStock 字段映射 =====
# BaoStock query_history_k_data_plus 可用字段
KLINE_FIELDS: str = (
    "date,open,high,low,close,volume,amount,turn,pctChg,"
    "peTTM,pbMRQ,psTTM,pcfNcfTTM"
)

# BaoStock 代码前缀映射
_BS_PREFIX: dict[str, str] = {
    "6": "sh",   # 上海主板
    "0": "sz",   # 深圳主板
    "3": "sz",   # 创业板
    "4": "sz",   # 深市B股
    "8": "bj",   # 北交所（BaoStock 可能不支持）
}

# BaoStock 季度财务数据字段
FINANCE_FIELDS: str = (
    "statDate,roeAvg,npParentMineYoY,revenueYoY,roeAvg,"
    "npParentMineYoY,revenueYoY,totalAssetTurnover"
)


def _to_bs_code(stock_code: str) -> str:
    """将6位股票代码转换为BaoStock格式。

    Args:
        stock_code: 6位股票代码，如 "600036"。

    Returns:
        BaoStock格式代码，如 "sh.600036"。

    Raises:
        ValueError: 股票代码格式无法识别。
    """
    if not (stock_code.isdigit() and len(stock_code) == 6):
        msg = f"股票代码格式错误: {stock_code}，需为6位数字"
        raise ValueError(msg)
    prefix = _BS_PREFIX.get(stock_code[0])
    if prefix is None:
        msg = f"无法识别的股票代码前缀: {stock_code}"
        raise ValueError(msg)
    return f"{prefix}.{stock_code}"


def _from_bs_code(bs_code: str) -> str:
    """将BaoStock格式代码转换回6位股票代码。

    Args:
        bs_code: BaoStock格式代码，如 "sh.600036"。

    Returns:
        6位股票代码，如 "600036"。
    """
    return bs_code.split(".")[1]


def _load_progress() -> set[str]:
    """加载断点续传进度，返回已成功获取的股票代码集合。

    Returns:
        已完成的股票代码集合。
    """
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            completed = set(data.get("completed", []))
            if completed:
                print(f"[续传] 发现进度文件，已完成 {len(completed)} 只股票")
            return completed
        except (json.JSONDecodeError, KeyError):
            print("[续传] 进度文件损坏，将从头开始")
    return set()


def _save_progress(completed: set[str]) -> None:
    """保存当前获取进度到文件。

    Args:
        completed: 已完成的股票代码集合。
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "completed": sorted(completed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    PROGRESS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _exponential_backoff(attempt: int, base: float = RETRY_BACKOFF) -> float:
    """计算指数退避等待时间。

    Args:
        attempt: 当前重试次数（从0开始）。
        base: 退避基数秒数。

    Returns:
        等待秒数，上限60秒。
    """
    return min(base * (2 ** attempt), 60.0)


def _check_network_proxy() -> None:
    """检测并提示系统代理配置问题。

    当系统存在 HTTP 代理设置且代理端口不可达时，发出警告。
    BaoStock 使用 TCP 长连接，不受 HTTP 代理影响，但如果将来
    需要使用 AKShare（基于 HTTP 请求东方财富 API），代理问题
    会导致 ProxyError / RemoteDisconnected 错误。
    """
    try:
        import urllib.request
        sys_proxies = urllib.request.getproxies()
        if sys_proxies:
            proxy_addr = sys_proxies.get("https") or sys_proxies.get("http", "")
            print(f"[代理] 检测到系统代理: {sys_proxies}")
            print(f"[代理] 当前数据源 BaoStock 使用 TCP 连接，不受 HTTP 代理影响")
            print(f"[代理] 若需使用 AKShare，请确保代理可用或关闭 VPN/梯子")
    except Exception:
        pass


def fetch_single_stock_kline(
    stock_code: str,
    start_date: str,
    end_date: str,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """通过BaoStock获取单只股票的日K线及估值数据，带自动重试。

    Args:
        stock_code: 6位股票代码，如 "600036"。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 结束日期，格式 "YYYY-MM-DD"。
        max_retries: 最大重试次数。

    Returns:
        包含行情和估值数据的DataFrame，列包括：
        date, stock_code, open, high, low, close, volume, amount,
        turnover_rate, pct_change, pe_ttm, pb_mrq, ps_ttm, pcf_ncf_ttm。

    Raises:
        ValueError: 股票代码格式错误。
        RuntimeError: 超过最大重试次数仍失败。
    """
    bs_code = _to_bs_code(stock_code)
    # BaoStock 日期格式: "YYYY-MM-DD"
    start_fmt = start_date if "-" in start_date else f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end_fmt = end_date if "-" in end_date else f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                KLINE_FIELDS,
                start_date=start_fmt,
                end_date=end_fmt,
                frequency="d",
                adjustflag="2",  # 前复权
            )

            if rs.error_code != "0":
                msg = f"BaoStock 错误 [{rs.error_code}]: {rs.error_msg}"
                raise RuntimeError(msg)

            rows: list[list[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                msg = f"获取 {stock_code} 返回空数据（可能已退市或停牌）"
                raise RuntimeError(msg)

            df = pd.DataFrame(rows, columns=rs.fields)

            # 类型转换
            numeric_cols = [
                "open", "high", "low", "close", "volume", "amount",
                "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 统一列名
            column_map = {
                "turn": "turnover_rate",
                "pctChg": "pct_change",
                "peTTM": "pe_ttm",
                "pbMRQ": "pb_mrq",
                "psTTM": "ps_ttm",
                "pcfNcfTTM": "pcf_ncf_ttm",
            }
            df = df.rename(columns=column_map)
            df["date"] = pd.to_datetime(df["date"])
            df["stock_code"] = stock_code

            return df[[
                "date", "stock_code", "open", "high", "low", "close",
                "volume", "amount", "turnover_rate", "pct_change",
                "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm",
            ]]

        except (ConnectionError, OSError, RuntimeError) as e:
            last_error = e
            wait = _exponential_backoff(attempt)
            error_type = type(e).__name__
            print(f"\n    [{error_type}] {stock_code} 第{attempt + 1}/{max_retries}次重试，"
                  f"等待{wait:.1f}s... ({e})")
            time.sleep(wait)

        except Exception as e:
            error_type = type(e).__name__
            if attempt == 0:
                last_error = e
                print(f"\n    [{error_type}] {stock_code} 意外错误，1次重试... ({e})")
                time.sleep(1.0)
            else:
                msg = f"获取 {stock_code} 数据失败: [{error_type}] {e}"
                raise RuntimeError(msg) from e

    msg = f"获取 {stock_code} 数据失败（已重试{max_retries}次）: {last_error}"
    raise RuntimeError(msg)


def fetch_single_stock_finance(
    stock_code: str,
    start_date: str,
    end_date: str,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """通过BaoStock获取单只股票的季度财务数据。

    分别调用三个API并按季度合并：
    - query_profit_data → roe_avg（盈利能力）
    - query_growth_data → yoy_pni（归属净利润同比）、yoy_equity（净资产同比）
    - query_operation_data → asset_turn_ratio（总资产周转率）

    Args:
        stock_code: 6位股票代码，如 "600036"。
        start_date: 起始日期，格式 "YYYY-MM-DD"。
        end_date: 结束日期，格式 "YYYY-MM-DD"。
        max_retries: 最大重试次数。

    Returns:
        包含财务数据的DataFrame，列：stat_date, stock_code,
        roe_avg, profit_growth_yoy, revenue_growth_yoy, asset_turnover。
    """
    bs_code = _to_bs_code(stock_code)
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            records: dict[str, dict[str, str]] = {}  # statDate -> {field: value}

            for year in range(start_year, end_year + 1):
                for quarter in (1, 2, 3, 4):
                    # 盈利能力：roeAvg
                    rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        sd = row.get("statDate", "")
                        if sd and sd not in records:
                            records[sd] = {}
                        if sd:
                            records[sd]["roe_avg"] = row.get("roeAvg", "")

                    # 成长能力：YOYPNI, YOYEquity
                    rs = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        sd = row.get("statDate", "")
                        if sd and sd not in records:
                            records[sd] = {}
                        if sd:
                            records[sd]["profit_growth_yoy"] = row.get("YOYPNI", "")
                            records[sd]["revenue_growth_yoy"] = row.get("YOYEquity", "")

                    # 经营效率：AssetTurnRatio
                    rs = bs.query_operation_data(code=bs_code, year=year, quarter=quarter)
                    while rs.next():
                        row = dict(zip(rs.fields, rs.get_row_data()))
                        sd = row.get("statDate", "")
                        if sd and sd not in records:
                            records[sd] = {}
                        if sd:
                            records[sd]["asset_turnover"] = row.get("AssetTurnRatio", "")

            if not records:
                return pd.DataFrame(columns=[
                    "stat_date", "stock_code", "roe_avg",
                    "profit_growth_yoy", "revenue_growth_yoy", "asset_turnover",
                ])

            df = pd.DataFrame.from_dict(records, orient="index")
            df.index.name = "stat_date"
            df = df.reset_index()

            # 类型转换
            for col in df.columns:
                if col not in ("stat_date", "stock_code"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["stock_code"] = stock_code
            df["stat_date"] = pd.to_datetime(df["stat_date"])

            # 去重
            df = df.drop_duplicates(subset=["stat_date"], keep="last")

            return df

        except (ConnectionError, OSError, RuntimeError) as e:
            last_error = e
            wait = _exponential_backoff(attempt)
            print(f"\n    [财务数据重试] {stock_code} 第{attempt + 1}次，等待{wait:.1f}s...")
            time.sleep(wait)

    print(f"[!] {stock_code} 财务数据获取失败: {last_error}")
    return pd.DataFrame(columns=[
        "stat_date", "stock_code", "roe_avg",
        "profit_growth_yoy", "revenue_growth_yoy", "asset_turnover",
    ])


def fetch_stock_data(
    stock_pool: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    save_dir: Path | None = None,
    delay: float = DEFAULT_DELAY,
    resume: bool = True,
) -> pd.DataFrame:
    """批量获取股票池行情和因子数据并保存为CSV，支持断点续传。

    使用 BaoStock 作为主数据源，通过 TCP 长连接获取数据，
    不受 HTTP 代理/VPN 影响。断点续传：每次成功获取一只股票后
    更新进度文件，中断后再次运行自动跳过已完成的股票。

    Args:
        stock_pool: 股票代码列表，默认使用config.STOCK_POOL。
        date_range: (起始日期, 结束日期)元组，默认使用config.DATE_RANGE。
        save_dir: CSV保存目录，默认使用config.RAW_DATA_DIR。
        delay: 请求间隔秒数，避免触发频率限制。
        resume: 是否启用断点续传，默认True。

    Returns:
        合并后的完整DataFrame，包含行情、估值和衍生因子。

    Raises:
        RuntimeError: 所有股票数据获取失败。
    """
    if stock_pool is None:
        stock_pool = STOCK_POOL
    if date_range is None:
        date_range = DATE_RANGE
    if save_dir is None:
        save_dir = RAW_DATA_DIR

    save_dir.mkdir(parents=True, exist_ok=True)

    start_date = date_range[0]  # "YYYY-MM-DD" 格式
    end_date = date_range[1]

    # 加载已完成进度
    completed: set[str] = _load_progress() if resume else set()

    # 检查是否已有完整CSV
    csv_path = save_dir / "stock_data.csv"
    existing_frames: list[pd.DataFrame] = []
    if resume and csv_path.exists():
        try:
            existing_df = pd.read_csv(
                csv_path, encoding="utf-8-sig", parse_dates=["date"]
            )
            existing_codes = set(existing_df["stock_code"].unique())
            for code in completed - existing_codes:
                completed.discard(code)
            if existing_codes.issubset(completed):
                existing_frames.append(existing_df)
                print(f"[续传] 加载已有CSV: {len(existing_df)} 行, "
                      f"{len(existing_codes)} 只股票")
        except Exception as e:
            print(f"[续传] 读取已有CSV失败: {e}，将从头获取")

    # 先测试单只股票以验证 BaoStock 可用
    print("[BaoStock] 连接测试...", end=" ", flush=True)
    lg = bs.login()
    if lg.error_code != "0":
        msg = f"BaoStock 登录失败: [{lg.error_code}] {lg.error_msg}"
        raise RuntimeError(msg)
    print(f"[BaoStock] 登录成功")

    all_frames: list[pd.DataFrame] = list(existing_frames)
    failed_codes: list[str] = []
    total = len(stock_pool)
    batch_count = 0

    try:
        for i, code in enumerate(stock_pool, 1):
            if code in completed:
                print(f"[{i}/{total}] {code} 已完成，跳过")
                continue

            print(f"[{i}/{total}] 获取 {code} ...", end=" ", flush=True)
            try:
                df = fetch_single_stock_kline(code, start_date, end_date)
                all_frames.append(df)
                completed.add(code)
                _save_progress(completed)
                print(f"成功 ({len(df)} 行)")
            except RuntimeError as e:
                failed_codes.append(code)
                print(f"失败: {e}")

            time.sleep(delay)

            batch_count += 1
            if batch_count >= BATCH_SIZE:
                batch_count = 0
                print(f"  -- 批次暂停 {BATCH_PAUSE:.0f}s --")
                time.sleep(BATCH_PAUSE)

    finally:
        bs.logout()

    if failed_codes:
        print(f"\n[!] 以下股票获取失败: {failed_codes}")
        # 第二轮重试
        print(f"\n[重试] 对 {len(failed_codes)} 只失败股票进行第二轮重试...")
        lg = bs.login()
        retry_failed: list[str] = []
        try:
            for code in failed_codes:
                print(f"  [重试] {code} ...", end=" ", flush=True)
                try:
                    df = fetch_single_stock_kline(code, start_date, end_date)
                    all_frames.append(df)
                    completed.add(code)
                    _save_progress(completed)
                    print(f"成功 ({len(df)} 行)")
                except RuntimeError as e:
                    retry_failed.append(code)
                    print(f"仍失败: {e}")
                time.sleep(delay * 3)
        finally:
            bs.logout()
        failed_codes = retry_failed

    if not all_frames:
        msg = "所有股票数据获取失败"
        raise RuntimeError(msg)

    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["date", "stock_code"], keep="last")

    # 确保 stock_code 为6位零填充字符串
    result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)

    # 计算衍生因子
    result = compute_derived_factors(result)

    # 保存
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n数据已保存至 {csv_path}，共 {len(result)} 行，"
          f"{result['stock_code'].nunique()} 只股票")

    # 清理进度文件
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("[续传] 进度文件已清理")

    if failed_codes:
        print(f"[!] 最终失败 {len(failed_codes)} 只: {failed_codes}")

    return result


def compute_derived_factors(df: pd.DataFrame) -> pd.DataFrame:
    """基于行情和估值数据计算衍生因子。

    从BaoStock获取的数据可以直接计算：
    - ep: Earnings-to-Price = 1/PE(TTM)
    - bp: Book-to-Price = 1/PB(MRQ)
    - momentum_20: 20日动量
    - reversal_5: 5日反转
    - amplitude: 5日平均振幅

    以下因子需从季度财务数据补充（通过 fetch_finance_data 方法）：
    - roe_growth, profit_growth, revenue_growth, roe, asset_turnover, dp

    Args:
        df: fetch_stock_data返回的原始DataFrame。

    Returns:
        添加了衍生因子列的DataFrame。
    """
    df = df.sort_values(["stock_code", "date"]).copy()

    # 估值因子：从PE/PB反推
    if "pe_ttm" in df.columns:
        df["ep"] = 1.0 / df["pe_ttm"].replace(0, float("nan"))
    else:
        df["ep"] = float("nan")

    if "pb_mrq" in df.columns:
        df["bp"] = 1.0 / df["pb_mrq"].replace(0, float("nan"))
    else:
        df["bp"] = float("nan")

    # 动量因子：20日收益率
    df["momentum_20"] = df.groupby("stock_code")["close"].pct_change(20)

    # 反转因子：5日收益率
    df["reversal_5"] = df.groupby("stock_code")["close"].pct_change(5)

    # 振幅：5日平均振幅
    df["amplitude"] = (df["high"] - df["low"]) / df["close"]
    df["amplitude"] = (
        df.groupby("stock_code")["amplitude"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 以下因子需从财务数据补充，暂用 NaN 占位
    for col in ("dp", "roe_growth", "profit_growth",
                "revenue_growth", "roe", "asset_turnover"):
        if col not in df.columns:
            df[col] = float("nan")

    return df


def fetch_finance_data(
    stock_pool: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """批量获取季度财务数据，用于补充估值和成长因子。

    Args:
        stock_pool: 股票代码列表，默认使用config.STOCK_POOL。
        date_range: 日期范围元组，默认使用config.DATE_RANGE。
        save_dir: CSV保存目录，默认使用config.RAW_DATA_DIR。

    Returns:
        包含季度财务数据的DataFrame。
    """
    if stock_pool is None:
        stock_pool = STOCK_POOL
    if date_range is None:
        date_range = DATE_RANGE
    if save_dir is None:
        save_dir = RAW_DATA_DIR

    save_dir.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    if lg.error_code != "0":
        msg = f"BaoStock 登录失败: [{lg.error_code}] {lg.error_msg}"
        raise RuntimeError(msg)

    all_frames: list[pd.DataFrame] = []
    try:
        for i, code in enumerate(stock_pool, 1):
            print(f"[{i}/{len(stock_pool)}] 获取 {code} 财务数据...", end=" ", flush=True)
            df = fetch_single_stock_finance(code, date_range[0], date_range[1])
            if not df.empty:
                all_frames.append(df)
                print(f"成功 ({len(df)} 条)")
            else:
                print("无数据")
            time.sleep(0.1)
    finally:
        bs.logout()

    if not all_frames:
        print("[!] 未获取到任何财务数据")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    csv_path = save_dir / "finance_data.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n财务数据已保存至 {csv_path}，共 {len(result)} 行")
    return result


def merge_finance_to_kline(
    kline_df: pd.DataFrame,
    finance_df: pd.DataFrame,
) -> pd.DataFrame:
    """将季度财务数据合并到日K线数据中（向前填充）。

    财务数据为季度频率，需按 stock_code 匹配后向前填充到每个交易日。

    Args:
        kline_df: 日K线数据DataFrame。
        finance_df: 季度财务数据DataFrame。

    Returns:
        合并后的DataFrame，财务因子已向前填充。
    """
    if finance_df.empty:
        return kline_df

    # 确保日期格式一致
    finance_df = finance_df.copy()
    if "stat_date" in finance_df.columns:
        finance_df["stat_date"] = pd.to_datetime(finance_df["stat_date"])

    # 为财务数据添加有效日期范围（财报发布日 = 报告期后45天）
    finance_df["effective_date"] = finance_df["stat_date"] + pd.Timedelta(days=45)

    # 排序
    finance_df = finance_df.sort_values(["stock_code", "effective_date"])

    # 确定可合并的财务列
    finance_value_cols = [c for c in finance_df.columns
                          if c not in ("stat_date", "stock_code", "effective_date")]

    if not finance_value_cols:
        return kline_df

    # 因子列名映射：财务API字段 → 项目因子名
    col_mapping = {
        "roe_avg": "roe",
        "profit_growth_yoy": "profit_growth",
        "revenue_growth_yoy": "revenue_growth",
        "asset_turnover": "asset_turnover",
    }

    # 在合并前，删除 kline_df 中可能被财务数据覆盖的 NaN 占位列
    kline_df_clean = kline_df.copy()
    finance_factor_cols = list(col_mapping.values()) + ["roe_growth"]
    for col in finance_factor_cols:
        if col in kline_df_clean.columns:
            kline_df_clean = kline_df_clean.drop(columns=[col])

    # 按股票代码逐个合并
    result_frames: list[pd.DataFrame] = []
    for code, group in kline_df_clean.groupby("stock_code"):
        fin = finance_df[finance_df["stock_code"] == code].copy()
        if fin.empty:
            result_frames.append(group)
            continue

        # 用 merge_asof 向前填充
        fin_sorted = fin.sort_values("effective_date")
        group_sorted = group.sort_values("date")

        merged = pd.merge_asof(
            group_sorted,
            fin_sorted[["stock_code", "effective_date"] + finance_value_cols],
            left_on="date",
            right_on="effective_date",
            by="stock_code",
            direction="backward",
        )

        # 映射到因子列名
        for src_col, dst_col in col_mapping.items():
            if src_col in merged.columns:
                merged[dst_col] = merged[src_col]

        # ROE growth 近似用 ROE 本身（简化处理）
        if "roe" in merged.columns and "roe_growth" not in merged.columns:
            merged["roe_growth"] = merged["roe"]

        result_frames.append(merged)

    result = pd.concat(result_frames, ignore_index=True)

    # 清理临时列（只删除源财务列中与因子名不同的列，保留因子列）
    factor_names = {"roe", "roe_growth", "profit_growth", "revenue_growth", "asset_turnover"}
    cleanup_cols = {"effective_date"} | {c for c in finance_value_cols if c not in factor_names}
    result = result.drop(columns=[c for c in cleanup_cols if c in result.columns])

    # 确保所有因子列存在（缺失的补NaN）
    for col in factor_names:
        if col not in result.columns:
            result[col] = float("nan")

    return result


# ===== 模块初始化：检测代理 =====
_check_network_proxy()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("Project-Alpha 数据获取（BaoStock）")
    print("=" * 60)

    # 第一步：获取行情+估值数据
    data = fetch_stock_data()

    # 第二步：获取财务数据
    print("\n" + "=" * 60)
    print("获取季度财务数据...")
    print("=" * 60)
    finance = fetch_finance_data()

    # 第三步：合并
    if not finance.empty:
        print("\n合并财务数据到行情数据...")
        data = merge_finance_to_kline(data, finance)

    # 保存最终结果
    output_path = RAW_DATA_DIR / "stock_data_with_factors.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n最终数据已保存至 {output_path}，共 {len(data)} 行")
    print(f"因子列: {[c for c in data.columns if c not in ('date', 'stock_code')]}")
    print(data.head(10))
