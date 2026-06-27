"""
题目二：投资组合优化
===================
给定:
  μ = [0.12, 0.08, 0.15]     (三只资产的预期收益率)
  Σ = [[0.10, 0.02, 0.03],    (3×3 协方差矩阵)
       [0.02, 0.08, 0.01],
       [0.03, 0.01, 0.12]]

要求:
1. 最大化 Sharpe Ratio，约束 w_i ≥ 0，Σw_i = 1
2. 输出权重、预期收益、波动率、Sharpe Ratio
3. 画出有效前沿曲线

关键公式:
    组合收益:   μ_p = wᵀμ
    组合波动:   σ_p = sqrt(wᵀ Σ w)
    Sharpe:     (μ_p - r_f) / σ_p

优化思路（scipy.optimize.minimize）:
    1. 约束条件:
       - 等式约束: sum(w) - 1 = 0  →  LinearConstraint 或 constraints=[{'type': 'eq', ...}]
       - 不等式约束: w_i >= 0      →  bounds=[(0, None), ...]  N 个资产 N 个 (0, None)
    2. 初始值: 等权重 w0 = [1/n, 1/n, ...]。
    3. minimize(neg_sharpe, ...) 即可得到最大 Sharpe 组合。

有效前沿计算:
    对一系列目标收益率 target_ret，求解最小方差组合:
        minimize  σ_p
        s.t.     μ_p >= target_ret,  sum(w)=1,  w_i >= 0
    遍历 target_ret 从 MVP 收益率到 max(μ)，得到 (σ, μ) 点集连线。
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """计算投资组合的预期收益率。

    Args:
        weights: 权重向量，shape (n_assets,)。
        mu: 预期收益率向量，shape (n_assets,)。

    Returns:
        组合预期收益 μ_p = wᵀμ。
    """
    mu_p = np.dot(weights, mu)
    return mu_p


def portfolio_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    """计算投资组合的波动率（标准差）。

    Args:
        weights: 权重向量，shape (n_assets,)。
        sigma: 协方差矩阵，shape (n_assets, n_assets)。

    Returns:
        组合波动率 σ_p = sqrt(wᵀ Σ w)。
    """
    sigma_p = np.sqrt(weights @ sigma @ weights)
    return sigma_p


def neg_sharpe(weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    """负 Sharpe Ratio，用于 scipy.optimize.minimize 最小化。

    scipy 的 minimize 只能最小化目标函数，因此返回 -Sharpe，
    最小化 -Sharpe 等价于最大化 Sharpe。

    Args:
        weights: 权重向量，shape (n_assets,)。
        mu: 预期收益率向量，shape (n_assets,)。
        sigma: 协方差矩阵，shape (n_assets, n_assets)。
        rf: 无风险利率。

    Returns:
        -(μ_p - r_f) / σ_p。
    """
    mu_p = portfolio_return(weights, mu)
    sigma_p = portfolio_volatility(weights, sigma)

    return -(mu_p - rf) / sigma_p


def min_variance_portfolio(
        mu: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """求解最小方差组合（MVP, Minimum Variance Portfolio）。

    不约束收益，只追求风险最小。MVP 是有效前沿的左端点。

    Args:
        mu: 预期收益率向量，shape (n_assets,)。
        sigma: 协方差矩阵，shape (n_assets, n_assets)。

    Returns:
        最小方差组合的权重向量，shape (n_assets,)。
    """
    n_assets = len(mu)
    w0 = np.ones(n_assets) / n_assets
    bounds = [(0, None)] * n_assets
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    res = minimize(
        portfolio_volatility, w0, args=(sigma,),
        method='SLSQP', bounds=bounds, constraints=constraints,
    )
    return res.x


def optimal_portfolio(
        mu: np.ndarray, sigma: np.ndarray, rf: float,
) -> tuple[np.ndarray, float, float, float]:
    """求解最大 Sharpe Ratio 的最优投资组合。

    使用 scipy.optimize.minimize 最小化 neg_sharpe，
    约束: sum(w) = 1, w_i >= 0。

    Args:
        mu: 预期收益率向量，shape (n_assets,)。
        sigma: 协方差矩阵，shape (n_assets, n_assets)。
        rf: 无风险利率。

    Returns:
        (weights, ret, vol, sharpe): 元组
            weights: 最优权重向量。
            ret: 组合预期收益率。
            vol: 组合波动率。
            sharpe: 最大 Sharpe Ratio。
    """
    n_assets = len(mu)
    w0 = np.ones(n_assets) / n_assets
    bounds = [(0, None)] * n_assets
    constraints = [
        {'type': 'eq',
         'fun': lambda w: np.sum(w) - 1}
    ]
    res = minimize(
        neg_sharpe, w0, args=(mu, sigma, rf),
        method='SLSQP', bounds=bounds, constraints=constraints,
    )
    weights = res.x
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, sigma)
    sharpe = (ret - rf) / vol
    return weights, ret, vol, sharpe


def efficient_frontier(
        mu: np.ndarray, sigma: np.ndarray, rf: float, num_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """计算有效前沿上的点集。

    对每个目标收益率 target_ret，求解最小方差组合（二次规划）。
    目标: minimize σ_p, s.t. μ_p >= target_ret, sum(w)=1, w_i >= 0。

    有效前沿的起点是最小方差组合（MVP）的收益率，而非 min(mu)。

    Args:
        mu: 预期收益率向量。
        sigma: 协方差矩阵。
        rf: 无风险利率。
        num_points: 前沿上的点数。

    Returns:
        (vols, rets): 元组
            vols: 有效前沿上各点的波动率，shape (num_points,)。
            rets: 有效前沿上各点的收益率，shape (num_points,)。
    """
    n_assets = len(mu)

    # 有效前沿起点：最小方差组合（MVP）的收益率
    w_mvp = min_variance_portfolio(mu, sigma)
    ret_mvp = portfolio_return(w_mvp, mu)

    # 收益率区间：从 MVP 收益率到最高资产收益率
    target_returns = np.linspace(ret_mvp, np.max(mu), num_points)

    vols = []
    rets = []

    for target_ret in target_returns:

        # 等权重初始猜测
        w0 = np.ones(n_assets) / n_assets

        # long-only
        bounds = [(0.0, None)] * n_assets

        constraints = [
            {
                'type': 'eq',
                'fun': lambda w: float(np.sum(w) - 1)
            },
            {
                'type': 'ineq',
                'fun': lambda w, tr=target_ret:
                float(portfolio_return(w, mu) - tr)
            }
        ]

        res = minimize(
            portfolio_volatility,
            w0,
            args=(sigma,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        if res.success:
            w_opt = res.x

            ret = portfolio_return(
                w_opt,
                mu
            )

            vol = portfolio_volatility(
                w_opt,
                sigma
            )

            rets.append(ret)
            vols.append(vol)

    return (
        np.array(vols),
        np.array(rets)
    )


if __name__ == "__main__":
    mu = np.array([0.12, 0.08, 0.15])
    sigma = np.array([
        [0.10, 0.02, 0.03],
        [0.02, 0.08, 0.01],
        [0.03, 0.01, 0.12],
    ])
    rf = 0.0  # 题目未给出无风险利率，假设为 0
            # 若实际 rf > 0，Sharpe Ratio 会相应降低

    w_opt, ret_opt, vol_opt, sharpe_opt = optimal_portfolio(mu, sigma, rf)

    print("=== Optimal Portfolio ===")
    for i, w in enumerate(w_opt):
        print(f"Asset {i + 1}: {w:.4f}")

    print(f"\nExpected Return : {ret_opt:.4f}")
    print(f"Volatility      : {vol_opt:.4f}")
    print(f"Sharpe Ratio    : {sharpe_opt:.4f}")

    print("\n=== Efficient Frontier ===")
    vols, rets = efficient_frontier(mu, sigma, rf)
    plt.figure(figsize=(8, 6))
    plt.plot(vols, rets, label="Efficient Frontier")
    plt.scatter(vol_opt, ret_opt, marker="*", s=200, label="Max Sharpe Ratio")
    asset_vols = np.sqrt(np.diag(sigma))

    for i in range(len(mu)):
        plt.scatter(asset_vols[i], mu[i], label=f"Asset {i + 1}")

    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.show()

    print("\n=== Summary ===")
    print(
        f"Best Portfolio: "
        f"Return={ret_opt:.4f}, "
        f"Volatility={vol_opt:.4f}, "
        f"Sharpe={sharpe_opt:.4f}"
    )
