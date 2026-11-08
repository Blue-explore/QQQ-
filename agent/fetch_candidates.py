"""只为「策略一条件1+条件3（或相对IV口径）」的候选交易日抓新闻。

比起全量 5 年（260 次请求 / ~24 分钟），只抓候选日附近窗口可把请求数
压到几十次，且完全覆盖 Agent 真正会被调用的日子。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .. import config as C
from ..core.market_data import build_market_data
from ..core.indicators import rolling_5d_returns, rolling_threshold_series
from .fetch_news import _fetch_window


def candidate_dates() -> list[date]:
    """策略一候选日 = C1（5日涨幅前10%）∩ C3（相对 IV 低分位）。"""
    md = build_market_data()
    start = md.index_of(date.fromisoformat(C.CFG.start))
    r5 = rolling_5d_returns(md.close, C.CC.rally_window)
    thr = rolling_threshold_series(r5, C.CC.lookback_days, C.CC.pct_threshold)

    out = []
    for t in range(start, len(md.dates)):
        if not (np.isfinite(r5[t]) and r5[t] >= thr[t]):
            continue
        lo = max(0, t - C.CC.iv_lookback_days)
        q = np.percentile(md.vxn[lo:t], C.CC.iv_percentile) \
            if t - lo >= 60 else -np.inf
        if np.isfinite(q) and md.vxn[t] <= q:
            out.append(md.dates[t])
    return out


def build_candidate_cache(days: list[date], out_path: Path | None = None,
                          pad: int = 4, sleep: float = 12.0,
                          retries: int = 3) -> dict:
    """按候选日聚合出需要抓取的周窗口，去重后逐窗口请求。

    GDELT 实测限流比官方 5s 更严，默认 12s 间隔 + 429 指数退避重试。
    已抓过的窗口（以窗口内任一天已有缓存为标志）会跳过，便于断点续跑。
    """
    import time
    out_path = out_path or C.NEWS_CACHE
    cache = {}
    if out_path.exists():
        try:
            cache = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    weeks = sorted({(d - timedelta(days=pad)).strftime("%G-W%V") for d in days})
    print(f"candidate days={len(days)}, unique week-windows={len(weeks)}")

    for i, wk in enumerate(weeks):
        y, w = int(wk[:4]), int(wk[-2:])
        lo = date.fromisocalendar(y, w, 1)
        hi = lo + timedelta(days=6)
        # 断点续跑：该窗口覆盖的日期里若已有 >=2 天缓存，视为已抓过
        covered = sum(1 for j in range(7)
                      if (lo + timedelta(days=j)).isoformat() in cache)
        if covered >= 2:
            print(f"  [{i+1}/{len(weeks)}] {lo}..{hi}: SKIP (cached {covered} days)")
            continue

        arts = []
        for attempt in range(retries):
            try:
                arts = _fetch_window(lo, hi, max_records=250)
                break
            except Exception as e:                  # noqa: BLE001
                wait = sleep * (2 ** attempt)
                print(f"  [{i+1}/{len(weeks)}] {lo}..{hi}: attempt {attempt+1} "
                      f"failed ({e}), backoff {wait:.0f}s")
                time.sleep(wait)
        if arts:
            for a in arts:
                cache.setdefault(a["date"], [])
                if a["url"] not in {x.get("url") for x in cache[a["date"]]}:
                    cache[a["date"]].append(a)
            print(f"  [{i+1}/{len(weeks)}] {lo}..{hi}: +{len(arts)}  cache_days={len(cache)}")
        out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        if i < len(weeks) - 1:
            time.sleep(sleep)
    return cache


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sleep", type=float, default=12.0)
    a = p.parse_args()
    days = candidate_dates()
    print("candidate dates:", [str(d) for d in days])
    build_candidate_cache(days, sleep=a.sleep)


if __name__ == "__main__":
    main()
