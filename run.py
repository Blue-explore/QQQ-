#!/usr/bin/env python3
"""主入口：跑五年回测，输出 CSV / JSON / HTML 报告。

IV 口径只有一种（用户最终定稿）：VXN ≤ 过去 252 个交易日 VXN 的 10 分位。
固定绝对阈值口径已移除。

用法：
    python -m qqq_options.run                          # 默认：两策略一起
    python -m qqq_options.run --strategies cc_only    # 只跑策略一
    python -m qqq_options.run --start 2020-11-08      # 换区间（含 2020 疫情尾部）
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace

from . import config as C
from .engine.backtest import Backtester
from .report.build_report import build_html
from .report.metrics import compute_metrics, fmt_pct


def dump_csv(res, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "equity", "bench_equity", "qqq_close", "cash", "shares"])
        for i, d in enumerate(res.dates):
            w.writerow([d.isoformat(), f"{res.equity[i]:.2f}",
                        f"{res.bench_equity[i]:.2f}", f"{res.qqq_close[i]:.2f}",
                        f"{res.cash[i]:.2f}", int(res.shares[i])])


def dump_trades(res, path):
    with open(path, "w", newline="") as f:
        if not res.trades:
            f.write("no trades\n"); return
        w = csv.DictWriter(f, fieldnames=list(res.trades[0].keys()))
        w.writeheader(); w.writerows(res.trades)


def run_one(strategies: str, start: str | None = None) -> dict:
    cfg = C.CFG
    if start:
        cfg = replace(cfg, start=start)
    res = Backtester(cfg=cfg, strategies=strategies).run()
    m = compute_metrics(res.equity)
    bm = compute_metrics(res.bench_equity)

    tag = strategies
    html_path = build_html(res, out=C.REPORT_DIR / f"backtest_{tag}.html")
    dump_csv(res, C.REPORT_DIR / f"equity_{tag}.csv")
    dump_trades(res, C.REPORT_DIR / f"trades_{tag}.csv")

    cc_open = sum(1 for t in res.trades if t["action"] == "SELL_CALL")
    lc_open = sum(1 for t in res.trades if t["action"] == "BUY_CALL")
    summary = {
        "strategies": strategies,
        "iv_rule": (f"VXN <= 过去 {C.CC.iv_lookback_days} 个交易日 VXN 的 "
                    f"{C.CC.iv_percentile:g} 分位"),
        "start": str(res.start), "end": str(res.end),
        "config": res.config_note,
        "strategy_metrics": m.to_dict(),
        "benchmark_metrics": bm.to_dict(),
        "cc": {
            "cond1_days": sum(1 for s in res.cc_signals if s.cond1_rally),
            "cond3_days": sum(1 for s in res.cc_signals if s.cond3_iv_ok),
            "candidate_days": sum(1 for s in res.cc_signals
                                  if s.cond1_rally and s.cond3_iv_ok),
            "triggered_days": sum(1 for s in res.cc_signals if s.triggered),
            "triggered_dates": [str(s.as_of) for s in res.cc_signals if s.triggered],
            "opened": cc_open,
        },
        "lc": {
            "triggered_dates": [str(s.as_of) for s in res.lc_signals if s.triggered],
            "opened": lc_open,
            "contracts_per_entry": C.LC.contracts,
        },
        "roll": {
            "enabled": C.LC.roll_enabled and strategies in ("both", "lc_only"),
            "delta_trigger": C.LC.roll_delta_trigger,
            "target_delta": C.LC.roll_target_delta,
            "n_rolls": len(res.roll_events),
            "events": res.roll_events,
            "n_fundamental_checks": len(res.fundamental_checks),
            "fundamental_decisions": {
                k: sum(1 for x in res.fundamental_checks if x["decision"] == k)
                for k in sorted({x["decision"] for x in res.fundamental_checks})
            },
        },
        "agent_calls": res.agent_calls,
        "agent_news_days": sum(1 for v in res.agent_verdicts
                               if v["source"] == "gdelt-rule"),
        "agent_proxy_days": sum(1 for v in res.agent_verdicts
                                if v["source"] == "price-structure"),
        "agent_verdicts": res.agent_verdicts,
        "n_trades": len(res.trades),
        "n_events": len(res.events),
        "report": str(html_path),
    }
    (C.REPORT_DIR / f"summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    label = {"both": "策略一+策略二", "cc_only": "仅策略一",
             "lc_only": "仅策略二"}[strategies]
    print(f"\n{'='*70}\n  {label}  |  IV 口径：VXN≤近{C.CC.iv_lookback_days}日"
          f"{C.CC.iv_percentile:g}分位  |  {res.start} ~ {res.end}\n{'='*70}")
    print(f"  期末权益   ${m.final_equity:,.2f}   (初始 ${cfg.initial_cash:,.0f})")
    print(f"  总收益     {fmt_pct(m.total_return)}    基准(买入持有) {fmt_pct(bm.total_return)}")
    print(f"  年化 CAGR  {fmt_pct(m.cagr)}     基准 {fmt_pct(bm.cagr)}")
    print(f"  年化波动   {fmt_pct(m.vol)}      夏普 {m.sharpe:.2f} (基准 {bm.sharpe:.2f})")
    print(f"  最大回撤   {fmt_pct(m.max_drawdown)}   基准 {fmt_pct(bm.max_drawdown)}")
    print(f"  策略一: C1={summary['cc']['cond1_days']}天, C3={summary['cc']['cond3_days']}天, "
          f"候选={summary['cc']['candidate_days']}天, 触发={summary['cc']['triggered_days']}天, "
          f"实卖={cc_open}张")
    lc_dates = summary["lc"]["triggered_dates"]
    print(f"  策略二: 条件触发 {len(lc_dates)} 次，实际开仓 {lc_open} 批"
          f"（每批 {C.LC.contracts} 张）"
          + ("（本模式未启用策略二）" if strategies == "cc_only" else
             f"（受 max_open_long_calls={C.LC.max_open_long_calls} 限制）"))
    if summary["roll"]["enabled"]:
        print(f"  Roll: Delta<{C.LC.roll_delta_trigger} → Δ{C.LC.roll_target_delta}/"
              f"{C.LC.roll_dte_days}天，实际滚动 {summary['roll']['n_rolls']} 次，"
              f"基本面检查 {summary['roll']['n_fundamental_checks']} 次 "
              f"{summary['roll']['fundamental_decisions']}")
    print(f"  Agent 调用: {res.agent_calls} 次   交易: {len(res.trades)} 笔")
    print(f"  报告: {html_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", default=None,
                   choices=["both", "cc_only", "lc_only"],
                   help="默认跟随 config.py 的 strategies（当前为 both）")
    p.add_argument("--start", default=None)
    a = p.parse_args()
    run_one(a.strategies or C.CFG.strategies, a.start)


if __name__ == "__main__":
    main()
