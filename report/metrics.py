"""绩效指标计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import config as C


@dataclass
class Metrics:
    total_return: float
    cagr: float
    vol: float
    sharpe: float
    max_drawdown: float
    max_dd_start: int
    max_dd_end: int
    calmar: float
    win_rate_daily: float
    n_days: int
    final_equity: float

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items()}


def compute_metrics(equity: np.ndarray, rf_annual: float = 0.02) -> Metrics:
    eq = np.asarray(equity, dtype=float)
    n = len(eq)
    if n < 2:
        raise ValueError("equity too short")
    rets = eq[1:] / eq[:-1] - 1.0
    total = eq[-1] / eq[0] - 1.0
    years = n / C.CFG.trading_days_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = float(np.std(rets, ddof=1) * math.sqrt(252)) if n > 2 else 0.0
    sharpe = (float(np.mean(rets)) * 252 - rf_annual) / vol if vol > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    i_end = int(np.argmax(dd))
    i_start = int(np.argmax(eq[:i_end + 1])) if i_end > 0 else 0
    mdd = float(dd[i_end])
    calmar = cagr / mdd if mdd > 0 else 0.0
    win = float(np.mean(rets > 0))

    return Metrics(
        total_return=float(total), cagr=float(cagr), vol=vol,
        sharpe=float(sharpe), max_drawdown=mdd,
        max_dd_start=i_start, max_dd_end=i_end,
        calmar=float(calmar), win_rate_daily=win,
        n_days=n, final_equity=float(eq[-1]),
    )


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"
