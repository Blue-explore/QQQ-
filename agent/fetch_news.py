"""按需抓取 GDELT DOC 2.0 新闻并写入本地缓存，供 NewsAgent 离线回测使用。

GDELT 限制：查询间隔 >= 5 秒。本脚本按「周」批量查询以控制请求数，
并把结果按发布日期归档到 { "YYYY-MM-DD": [articles...] }。

用法：
    python -m qqq_options.agent.fetch_news --start 2021-11-01 --end 2026-11-06
    python -m qqq_options.agent.fetch_news --dates 2024-01-05,2024-03-08   # 只补指定日
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from urllib.parse import quote
from urllib.request import Request, urlopen

from .. import config as C

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = "(QQQ OR \"Nasdaq 100\" OR \"Nasdaq-100\" OR \"tech stocks\")"


def _fetch_window(lo: date, hi: date, max_records: int = 250) -> list[dict]:
    sd = lo.strftime("%Y%m%d") + "000000"
    ed = hi.strftime("%Y%m%d") + "235959"
    url = (f"{GDELT}?query={quote(QUERY)}&mode=artlist&maxrecords={max_records}"
           f"&startdatetime={sd}&enddatetime={ed}&format=json&sort=datedesc")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (quant-research)"})
    with urlopen(req, timeout=40) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    out = []
    for a in data.get("articles", []):
        try:
            d = date(int(a["seendate"][:4]), int(a["seendate"][4:6]), int(a["seendate"][6:8]))
            out.append({
                "date": d.isoformat(),
                "title": a.get("title", ""),
                "domain": a.get("domain", ""),
                "url": a.get("url", ""),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def fetch_range(start: date, end: date, out_path=None, sleep: float = 5.5) -> dict:
    """按周抓取，合并已有缓存。"""
    out_path = out_path or C.NEWS_CACHE
    cache = {}
    if out_path.exists():
        try:
            cache = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    cur = start
    n = 0
    while cur <= end:
        hi = min(cur + timedelta(days=6), end)
        try:
            arts = _fetch_window(cur, hi)
            for a in arts:
                cache.setdefault(a["date"], [])
                if a["url"] not in {x.get("url") for x in cache[a["date"]]}:
                    cache[a["date"]].append(a)
            print(f"  {cur} .. {hi}: {len(arts)} articles -> cache days={len(cache)}")
        except Exception as e:                       # noqa: BLE001
            print(f"  {cur} .. {hi}: FAILED {e}")
        n += 1
        cur = hi + timedelta(days=1)
        out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        if cur <= end:
            time.sleep(sleep)
    print(f"done. {n} requests, {sum(len(v) for v in cache.values())} articles, "
          f"{len(cache)} days -> {out_path}")
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default=C.CFG.end)
    p.add_argument("--dates", default=None, help="comma separated YYYY-MM-DD")
    p.add_argument("--sleep", type=float, default=5.5)
    a = p.parse_args()
    if a.dates:
        ds = [date.fromisoformat(x.strip()) for x in a.dates.split(",")]
        for d in ds:
            fetch_range(d - timedelta(days=3), d, sleep=a.sleep)
    else:
        fetch_range(date.fromisoformat(a.start), date.fromisoformat(a.end), sleep=a.sleep)


if __name__ == "__main__":
    main()
