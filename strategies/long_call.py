"""策略二：买入 Call（趋势回踩）信号判定。

两个条件必须同时满足：
    B1 牛市趋势：满足任一即可
       a) 策略一在近3个月（63个交易日）内触发 >= 3 次
       b) 近一年（252个交易日）最大回撤 <= 10%
    B2 单日跌幅 > 1%（close[t]/close[t-1]-1 < -0.01）

满足后买入一张 Delta=0.6、约一年的 call。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .. import config as C
from ..core.indicators import max_drawdown, daily_returns


@dataclass
class LongCallSignal:
    as_of: date
    idx: int
    cond1_bull: bool
    cond1_via_triggers: bool
    cond1_via_drawdown: bool
    triggers_3m: int
    max_dd_1y: float
    cond2_drop: bool
    daily_ret: float
    triggered: bool
    reason: str


def evaluate_long_call(
    md, idx: int,
    cc_trigger_flags: np.ndarray,
) -> LongCallSignal:
    d = md.dates[idx]

    # B1a: 近3个月策略一触发次数（用 idx 之前的窗口，含前一日，避免前视）
    lo = max(0, idx - C.LC.bull_lookback_days)
    n_trig = int(np.sum(cc_trigger_flags[lo:idx]))
    via_triggers = n_trig >= C.LC.bull_trigger_min

    # B1b: 近一年最大回撤
    lo2 = max(0, idx - C.LC.max_dd_lookback_days + 1)
    dd = max_drawdown(md.close[lo2:idx + 1])
    via_dd = dd <= C.LC.max_dd_limit

    c1 = bool(via_triggers or via_dd)

    # B2: 单日跌幅
    r = daily_returns(md.close)[idx]
    r = float(r) if np.isfinite(r) else 0.0
    c2 = r < C.LC.drop_threshold

    triggered = bool(c1 and c2)
    return LongCallSignal(
        as_of=d, idx=idx,
        cond1_bull=c1,
        cond1_via_triggers=bool(via_triggers),
        cond1_via_drawdown=bool(via_dd),
        triggers_3m=n_trig,
        max_dd_1y=dd,
        cond2_drop=bool(c2),
        daily_ret=r,
        triggered=triggered,
        reason=(f"B1牛市={'✓' if c1 else '✗'}"
                f"(3月内策略一触发={n_trig}次[{'✓' if via_triggers else '✗'}], "
                f"近1年最大回撤={dd*100:.1f}%[{'✓' if via_dd else '✗'}])；"
                f"B2单日跌={'✓' if c2 else '✗'}({r*100:.2f}%)"),
    )
