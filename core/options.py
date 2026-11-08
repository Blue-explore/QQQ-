"""Black-Scholes 欧式期权定价 + 按目标 Delta 反解行权价。

无 scipy 依赖：
  * 正态 CDF/PDF 用 math.erf / exp 实现
  * 反解 K 用二分法（call delta 关于 K 单调递减，保证收敛）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """欧式看涨期权理论价（每股）。"""
    if T <= 0:
        return max(0.0, S - K)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """看涨期权 Delta（含股息率 q）。"""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * norm_cdf(d1)


def call_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    v = sigma * math.sqrt(T)
    return math.exp(-q * T) * norm_pdf(d1) / (S * v)


def strike_for_delta(
    target_delta: float,
    S: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """二分法反解使 call_delta(K) == target_delta 的 K。

    Delta 随 K 增大而单调递减（K->0 时 delta->exp(-qT)，K->inf 时->0）。
    """
    target_delta = min(max(target_delta, 1e-6), 0.999999)
    lo, hi = S * 1e-4, S * 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d = call_delta(S, mid, T, r, q, sigma)
        if d > target_delta:      # K 太小，delta 太大 -> 抬高 K
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * S:
            break
    K = 0.5 * (lo + hi)
    # 行权价按 1 美元整数档取整（美股 ETF 常见 1 美元间隔），更贴近真实成交
    return float(round(K))


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    premium: float          # 每股权利金（含滑点后实际可成交价）
    theo: float             # 理论价（每股，未含滑点）
    delta: float
    iv: float               # 实际用于定价的年化 IV（小数）
    dte: int                # 剩余自然日
    T: float                # 年化期限

    @property
    def contract_value(self) -> float:
        return self.premium * 100


def quote_call(
    S: float,
    target_delta: float,
    dte: int,
    vxn: float,
    r: float,
    cfg=None,
) -> OptionQuote:
    """按目标 Delta 生成一张 call 的报价。

    VXN 是指数隐含波动率（百分数），乘以经验系数 iv_scale 后作为 QQQ 期权 IV。
    卖 call 得到 premium*(1-slippage)，买 call 支付 premium*(1+slippage) 在
    引擎层面处理；这里返回的 premium 是「理论价」，滑点由引擎按方向施加。
    """
    from .. import config as C
    cfg = cfg or C.PR
    sigma = min(max(vxn / 100.0 * cfg.iv_scale, cfg.iv_floor), cfg.iv_cap)
    T = max(dte, 1) / 365.0
    q = cfg.dividend_yield
    K = strike_for_delta(target_delta, S, T, r, q, sigma)
    theo = call_price(S, K, T, r, q, sigma)
    delta = call_delta(S, K, T, r, q, sigma)
    return OptionQuote(
        strike=K,
        premium=theo,
        theo=theo,
        delta=delta,
        iv=sigma,
        dte=dte,
        T=T,
    )


def mark_call(
    S: float, K: float, dte: int, vxn: float, r: float, cfg=None
) -> float:
    """给已持有的合约做逐日盯市：返回每股理论价。"""
    from .. import config as C
    cfg = cfg or C.PR
    sigma = min(max(vxn / 100.0 * cfg.iv_scale, cfg.iv_floor), cfg.iv_cap)
    T = max(dte, 0) / 365.0
    if T <= 0:
        return max(0.0, S - K)
    return call_price(S, K, T, r, cfg.dividend_yield, sigma)


def mark_delta(
    S: float, K: float, dte: int, vxn: float, r: float, cfg=None
) -> float:
    """给已持有的合约算当前 Delta（Roll 判定用）。"""
    from .. import config as C
    cfg = cfg or C.PR
    sigma = min(max(vxn / 100.0 * cfg.iv_scale, cfg.iv_floor), cfg.iv_cap)
    T = max(dte, 0) / 365.0
    return call_delta(S, K, T, r, cfg.dividend_yield, sigma)


if __name__ == "__main__":
    S, T, r, q, sig = 500.0, 7 / 365, 0.04, 0.006, 0.25
    for td in (0.2, 0.6):
        K = strike_for_delta(td, S, T, r, q, sig)
        print(f"target delta {td}: K={K:.2f}, "
              f"delta={call_delta(S,K,T,r,q,sig):.6f}, "
              f"price={call_price(S,K,T,r,q,sig):.4f}")
    print("ATM delta", call_delta(S, S, T, r, q, sig))
