"""
题目一：欧式看涨期权定价
========================
参数: S=100, K=105, r=0.05, σ=0.2, T=1

要求:
1. 输出 Black-Scholes 理论价格
2. Monte Carlo 模拟 100000 次，输出 payoff 和价格
3. 输出 MC 的标准差
4. 对比两种价格

背景知识（Black-Scholes 公式）:
    C = S₀·N(d₁) - K·e^(-rT)·N(d₂)

    d₁ = [ln(S₀/K) + (r + σ²/2)·T] / (σ·√T)
    d₂ = d₁ - σ·√T

    其中 N(x) 是标准正态分布的累积分布函数(CDF)。
    在 Python 中可以用 math.erf 实现: N(x) = 0.5 * (1 + erf(x / √2))

背景知识（Monte Carlo 模拟）:
    在风险中性测度下，股票价格的几何布朗运动:
        S_T = S₀ · exp((r - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)

    payoff = max(S_T - K, 0)，折现后价格 = e^(-rT) · mean(payoff)
    MC 标准差 = std(payoff) / √N （样本均值的标准误）
"""

import math
import random
import statistics


def normal_cdf(x: float) -> float:
    """标准正态分布累积分布函数 N(x)。

    使用 math.erf 实现，避免引入 scipy 依赖。

    Args:
        x: 自变量。

    Returns:
        N(x) 的值，范围 [0, 1]。
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes 欧式看涨期权定价公式。

    Args:
        S: 标的资产当前价格。
        K: 行权价。
        r: 无风险利率（连续复利）。
        sigma: 年化波动率。
        T: 到期时间（年）。

    Returns:
        期权的理论价格。
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    call = (
            S * normal_cdf(d1)
            - K * math.exp(-r * T) * normal_cdf(d2)
    )

    return call


def monte_carlo_call(
        S: float, K: float, r: float, sigma: float, T: float, N: int = 100000,
) -> tuple[float, float]:
    """Monte Carlo 模拟欧式看涨期权定价。

    在风险中性测度下模拟 N 条股价路径，计算 payoff 的折现均值。
    使用 random.gauss() 生成正态随机数（或 math.normalvariate）。

    Args:
        S: 标的资产当前价格。
        K: 行权价。
        r: 无风险利率（连续复利）。
        sigma: 年化波动率。
        T: 到期时间（年）。
        N: 模拟路径数，默认 100000。

    Returns:
        (mc_price, mc_std): 元组
            mc_price: MC 估计的期权价格 (= discount * mean(payoff))。
            mc_std: 样本均值的标准误 (= std(payoff) / sqrt(N))。
    """
    payoffs = []

    for _ in range(N):
        Z = random.gauss(0, 1)
        S_T = S * math.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)
        payoff = max(S_T - K, 0)
        payoffs.append(payoff)

    mean_payoff = statistics.mean(payoffs)
    std_payoff = statistics.stdev(payoffs)

    mc_price = math.exp(-r * T) * mean_payoff
    mc_std = std_payoff / math.sqrt(N)

    return mc_price, mc_std


if __name__ == "__main__":
    S, K, r, sigma, T = 100, 105, 0.05, 0.2, 1

    bs_price = black_scholes_call(S, K, r, sigma, T)
    mc_price, mc_std = monte_carlo_call(S, K, r, sigma, T)

    print("=== Black-Scholes ===")
    # TODO: 输出 d1, d2, BS Price，保留 6 位小数
    print(bs_price)

    print("\n=== Monte Carlo (N=100000) ===")
    # TODO: 输出 MC Price, MC Std，保留 6 位小数
    print(mc_price)
    print(mc_std)

    print("\n=== Comparison ===")
    # TODO: 输出 absolute difference 和 relative error (%)
    # relative error = abs(bs_price - mc_price) / bs_price * 100
