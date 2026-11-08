"""市场数据加载与清洗：QQQ 日线 / VXN 隐含波动率 / 无风险利率。

数据来源（均为公开官方源，已缓存到 data/ 便于离线复现）：
  * QQQ 日线   : Nasdaq 官方历史行情 API（2016-11-07 ~ 2026-11-06，2514 根）
  * VXN        : CBOE 官方 CSV（纳指100隐含波动率指数，2009 至今）
  * 无风险利率 : FRED DGS3MO（3个月国债收益率）
  * NASDAQ100  : FRED（仅作校验/兜底，不参与主回测）

所有价格统一为「收盘价」，日期统一为 datetime.date，按交易日升序。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C


def _parse_date(s: str) -> date:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {s!r}")


def _num(s: str) -> float:
    """Nasdaq 的价格带 $ 和千分位逗号；FRED 用 '.' 表示缺失。"""
    if s is None:
        return np.nan
    s = str(s).strip().replace("$", "").replace(",", "")
    if s in ("", ".", "N/A", "null", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- QQQ
def load_qqq(path: Path | None = None) -> pd.DataFrame:
    """返回按日期升序的 QQQ OHLCV，index 为 date。"""
    path = path or C.QQQ_RAW
    raw = json.loads(Path(path).read_text())
    rows = raw["data"]["tradesTable"]["rows"]
    recs = []
    for r in rows:
        recs.append({
            "date": _parse_date(r["date"]),
            "open": _num(r["open"]),
            "high": _num(r["high"]),
            "low": _num(r["low"]),
            "close": _num(r["close"]),
            "volume": _num(r["volume"]),
        })
    df = pd.DataFrame(recs).sort_values("date").drop_duplicates("date")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- VXN
def load_vxn(path: Path | None = None) -> pd.Series:
    """CBOE VXN 收盘价 Series，index 为 date。"""
    path = path or C.VXN_RAW
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = _parse_date(r["DATE"])
            c = _num(r["CLOSE"])
            if np.isfinite(c):
                out[d] = c
    s = pd.Series(out, name="vxn").sort_index()
    return s


# ---------------------------------------------------------------- 利率
def load_risk_free(path: Path | None = None) -> pd.Series:
    """FRED DGS3MO，百分数 -> 小数。缺失日向前填充。"""
    path = path or C.RATE_RAW
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                d = _parse_date(row[0])
            except ValueError:
                continue
            v = _num(row[1])
            if np.isfinite(v):
                out[d] = v / 100.0
    s = pd.Series(out, name="rf").sort_index()
    return s


@dataclass
class MarketData:
    """对齐到 QQQ 交易日历的合并数据。"""

    df: pd.DataFrame           # date, open, high, low, close, volume, vxn, rf
    dates: list[date]
    close: np.ndarray
    vxn: np.ndarray            # 已前向填充
    rf: np.ndarray             # 已前向填充

    def index_of(self, d: date) -> int:
        """返回 <= d 的最近交易日下标（用于历史窗口查询）。"""
        # dates 已升序；用 bisect
        import bisect
        i = bisect.bisect_right(self.dates, d) - 1
        return max(i, 0)


def build_market_data() -> MarketData:
    qqq = load_qqq()
    vxn = load_vxn()
    rf = load_risk_free()

    dates = [d for d in qqq["date"]]
    # 用 reindex + ffill 把外部序列对齐到交易日
    vxn_aligned = vxn.reindex(dates).ffill().bfill().to_numpy(dtype=float)
    rf_aligned = rf.reindex(dates).ffill().bfill().to_numpy(dtype=float)
    rf_aligned = np.where(np.isfinite(rf_aligned), rf_aligned, C.PR.default_risk_free)

    df = qqq.copy()
    df["vxn"] = vxn_aligned
    df["rf"] = rf_aligned
    return MarketData(
        df=df,
        dates=dates,
        close=df["close"].to_numpy(dtype=float),
        vxn=vxn_aligned,
        rf=rf_aligned,
    )


if __name__ == "__main__":
    md = build_market_data()
    print(f"QQQ bars: {len(md.dates)}  {md.dates[0]} -> {md.dates[-1]}")
    print(f"close range: {md.close.min():.2f} .. {md.close.max():.2f}")
    print(f"VXN range : {np.nanmin(md.vxn):.2f} .. {np.nanmax(md.vxn):.2f}")
    from .. import config as _C
    lo = _C.CC.iv_lookback_days
    p = _C.CC.iv_percentile
    n = sum(1 for t in range(lo, len(md.vxn))
            if md.vxn[t] <= np.percentile(md.vxn[t-lo:t], p))
    print(f"相对 IV 口径 (VXN<=近{lo}日{p:g}分位) 满足天数: {n}")
    print(md.df.tail(3).to_string(index=False))
