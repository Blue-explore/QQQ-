"""用生成的 summary_<strategies>.json 把真实回测数字填入 README 的结果表格占位符。

用法： python -m qqq_options.report.fill_readme [--strategies both]
"""

from __future__ import annotations

import argparse
import csv
import json

from .. import config as C


def _m(s, key, pct=False):
    d = s.get("strategy_metrics") or s.get("benchmark_metrics")
    v = d[key]
    return f"{v*100:.2f}%" if pct else f"{v:,.2f}"


def fill(strategies: str = "both"):
    tag = strategies
    s = json.loads((C.REPORT_DIR / f"summary_{tag}.json").read_text())
    p = C.REPORT_DIR / f"trades_{tag}.csv"
    trades = (list(csv.DictReader(p.open()))
              if p.exists() and p.stat().st_size > 20 else [])
    b = s["benchmark_metrics"]

    def metrics_row(label, key, pct=False):
        return f"| {label} | {_m(s, key, pct)} | {_m({'benchmark_metrics': b}, key, pct)} |"

    lines = [
        "| 指标 | 策略组合 | 买入持有 QQQ |",
        "|---|---|---|",
        metrics_row("期末权益 ($)", "final_equity"),
        metrics_row("总收益", "total_return", pct=True),
        metrics_row("年化 CAGR", "cagr", pct=True),
        metrics_row("年化波动率", "vol", pct=True),
        metrics_row("夏普比率", "sharpe"),
        metrics_row("最大回撤", "max_drawdown", pct=True),
        f"| 策略一触发 | {s['cc']['triggered_days']} 天 | — |",
        f"| 策略一实际卖出 | {s['cc']['opened']} 张 | — |",
        f"| 策略二触发 | {len(s['lc']['triggered_dates'])} 次 | — |",
        f"| 策略二实际开仓 | {s['lc']['opened']} 批 × "
        f"{s['lc']['contracts_per_entry']} 张 | — |",
        f"| Delta Roll 次数 | {s['roll']['n_rolls']} 次 | — |",
        f"| 基本面检查 | {s['roll']['n_fundamental_checks']} 次 | — |",
        f"| Agent 调用 | {s['agent_calls']} 次 | — |",
        f"| 交易笔数 | {s['n_trades']} 笔 | — |",
    ]
    table = "\n".join(lines)

    readme = C.PKG / "README.md"
    txt = readme.read_text()
    marker = "<!--RESULT_TABLE-->"
    start, end = "<!--RESULT_START-->", "<!--RESULT_END-->"
    block = start + "\n" + table + "\n" + end
    if marker in txt:
        txt = txt.replace(marker, block)
    elif start in txt and end in txt:
        pre, rest = txt.split(start, 1)
        _, post = rest.split(end, 1)
        txt = pre + block + post
    else:
        raise RuntimeError("README 缺少结果表格占位符，请检查")
    readme.write_text(txt)
    print(table)
    return readme


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", default=C.CFG.strategies,
                   choices=["both", "cc_only", "lc_only"])
    fill(p.parse_args().strategies)
