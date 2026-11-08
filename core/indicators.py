"""技术指标：滚动 5 年 5 日涨幅分位门槛、最大回撤、单日涨跌。

严格按策略定义实现：
  * 每个交易日 t 计算 r5(t) = close[t] / close[t-5] - 1
  * 过去 5 年（约 1260 个交易日）的 r5 历史从大到小排序，
    第 126 个（= 10% 位置）就是门槛；当日 r5 >= 门槛 => 「刚冲完一波」
  * 分位门槛用「回看窗口」计算，绝不使用未来数据（无前视偏差）
"""

from __future__ import annotations

import numpy as np

from .. import config as C


def rolling_5d_returns(close: np.ndarray, window: int = C.CC.rally_window) -> np.ndarray:
    """r[t] = close[t]/close[t-window] - 1；前 window 个为 nan。"""
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan)
    if len(close) > window:
        out[window:] = close[window:] / close[:-window] - 1.0
    return out


def pct_threshold(history: np.ndarray, pct: float = C.CC.pct_threshold) -> float:
    """从大到小排第 int(N*pct) 个（0-based）作为门槛。

    例：1260 个数、pct=0.10 -> int(126)=126 -> 第 127 大的数。
    为避免 off-by-one 歧义，这里采用「>= 门槛才算前 10%」的等价定义：
    threshold = 降序数组的第 k 个值，k = max(1, floor(N*pct)) - 1 (0-based)
    即 N=1260, pct=0.10 => k=125 => 第 126 大的数。
    """
    h = np.asarray(history, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        return np.inf
    k = max(1, int(np.floor(h.size * pct)))
    arr = np.sort(h)[::-1]            # 降序
    return float(arr[min(k, arr.size) - 1])


def rolling_threshold_series(
    r5: np.ndarray,
    lookback: int = C.CC.lookback_days,
    pct: float = C.CC.pct_threshold,
) -> np.ndarray:
    """thr[t]：用 t 之前（含 t-lookback .. t-1，不含 t）的 lookback 个 r5 计算门槛。

    注意：不含当日 r5，避免「当日自己抬高门槛」的前视偏差。
    需要至少 lookback 个历史样本才有有效值，前期为 nan。
    """
    n = len(r5)
    thr = np.full(n, np.inf)
    for t in range(lookback, n):
        hist = r5[t - lookback:t]
        thr[t] = pct_threshold(hist, pct)
    return thr


def max_drawdown(prices: np.ndarray) -> float:
    """最大回撤（正数，0.10 = 10%）。"""
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return 0.0
    peak = np.maximum.accumulate(p)
    dd = (peak - p) / peak
    return float(np.max(dd))


def rolling_max_drawdown_series(close: np.ndarray, window: int) -> np.ndarray:
    """dd[t]：截至 t 的过去 window 根收盘价内的最大回撤（含 t）。"""
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = close[lo:t + 1]
        if len(seg) >= 2:
            out[t] = max_drawdown(seg)
        else:
            out[t] = 0.0
    return out


def daily_returns(close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) > 1:
        out[1:] = close[1:] / close[:-1] - 1.0
    return out


if __name__ == "__main__":
    # 小自测
    rng = np.random.default_rng(0)
    x = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    r = rolling_5d_returns(x)
    print("r5 sample", r[5:8])
    hist = np.linspace(-0.1, 0.1, 1260)
    print("threshold(1260,10%) =", pct_threshold(hist, 0.10))
    print("expect ~ 0.1 - 0.1*(126/1260) =", 0.1 - 0.1 * 126 / 1260)
    print("maxdd", max_drawdown(np.array([100, 120, 90, 130])))
