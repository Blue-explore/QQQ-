"""策略一：Covered Call（备兑看涨）信号判定。

三个条件必须同时满足才允许卖出 call：
    C1 股价刚冲完一波：近5日涨幅 >= 过去5年所有5日涨幅的前10%门槛
    C2 有短期技术阻力：NewsAgent.has_resistance == True
                       且 human_caused == False（人为动作不算）
    C3 IV 够低：VXN <= 过去 252 个交易日 VXN 的 10 分位
        （用户最终定稿：只保留相对口径，固定绝对阈值已移除）

「从不裸卖」的实现不在信号层，而在引擎层做**硬约束**：
    开仓前必须校验 shares >= contracts*100；不足则先买入股票，
    校验失败直接拒绝开仓（raise / 返回 blocked）。见 engine/backtest.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .. import config as C
from ..agent.news_agent import NewsAgent, NewsVerdict


@dataclass
class CoveredCallSignal:
    as_of: date
    idx: int
    cond1_rally: bool
    cond2_resistance: bool
    cond3_iv_ok: bool            # C3：IV 条件是否满足（相对分位）
    triggered: bool
    r5: float
    threshold: float
    vxn: float
    iv_percentile_threshold: float     # 当日滚动分位门槛
    news: NewsVerdict | None
    reason: str


def iv_condition_ok(md, idx: int) -> tuple[bool, float, float]:
    """返回 (C3 是否满足, 当日 VXN, 当日滚动分位门槛)。

    唯一口径（用户最终定稿）：
        取 idx 之前 iv_lookback_days 个交易日的 VXN 历史（**不含当日**，
        避免前视偏差），算第 iv_percentile 百分位数作为门槛；
        当日 VXN <= 门槛 → C3 成立（IV 处于近一年相对低位）。
    """
    v = float(md.vxn[idx])
    lo = max(0, idx - C.CC.iv_lookback_days)
    hist = md.vxn[lo:idx]
    th = float(np.percentile(hist, C.CC.iv_percentile)) if len(hist) >= 60 else -np.inf
    return (np.isfinite(th) and v <= th), v, th


def evaluate_covered_call(
    md, idx: int, r5: np.ndarray, thr: np.ndarray, agent: NewsAgent,
) -> CoveredCallSignal:
    """评估第 idx 个交易日是否满足策略一开仓条件。"""
    d = md.dates[idx]
    r = float(r5[idx]) if np.isfinite(r5[idx]) else float("nan")
    t = float(thr[idx]) if np.isfinite(thr[idx]) else float("inf")

    c1 = np.isfinite(r) and np.isfinite(t) and r >= t
    c3, vxn, used_th = iv_condition_ok(md, idx)

    # 性能：仅当 C1 与 C3 都成立才调用新闻 Agent（Agent 是有成本/有 I/O 的）
    news = None
    c2 = False
    if c1 and c3:
        news = agent.analyze(as_of=d, md=md, idx=idx, r5=r, vxn=vxn)
        c2 = news.has_resistance and not news.human_caused

    triggered = bool(c1 and c2 and c3)
    parts = [
        f"C1涨幅={'✓' if c1 else '✗'}(r5={r:.4f},门槛={t:.4f})",
        f"C2技术阻力={'✓' if c2 else '✗'}"
        + (f"[{news.verdict}]" if news else "[未调用]"),
        f"C3IV满足条件={'✓' if c3 else '✗'}"
        f"(VXN={vxn:.2f} <= {used_th:.2f} 近1年{C.CC.iv_percentile:g}分位)",
    ]
    return CoveredCallSignal(
        as_of=d, idx=idx,
        cond1_rally=bool(c1), cond2_resistance=bool(c2),
        cond3_iv_ok=bool(c3), triggered=triggered,
        r5=r, threshold=t, vxn=vxn, iv_percentile_threshold=used_th,
        news=news,
        reason="；".join(parts) + (f" → {news.reasoning}" if news else ""),
    )
