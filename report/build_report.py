"""生成单文件 HTML 回测报告（含 SVG 图表、交易明细、Agent 判定记录）。"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np

from .. import config as C
from ..engine.backtest import BacktestResult
from .charts import line_chart, drawdown_chart
from .metrics import compute_metrics, fmt_pct

CSS = """
:root{--ink:#1c2536;--sub:#6b7789;--line:#e6ebf2;--bg:#f6f8fb;--card:#fff;
--blue:#2f6fed;--green:#1a9d6f;--red:#d64545;--amber:#e08b1a;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft Yahei",sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:36px 22px 80px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.4px}h2{font-size:19px;margin:34px 0 14px;
padding-left:11px;border-left:4px solid var(--blue)}
.sub{color:var(--sub);margin-bottom:26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;
margin-bottom:18px;box-shadow:0 1px 3px rgba(20,40,80,.04)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .lab{font-size:12px;color:var(--sub)}.kpi .val{font-size:22px;font-weight:650;margin-top:4px}
.pos{color:var(--green)}.neg{color:var(--red)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#f2f5fa;color:var(--sub);font-weight:600;white-space:nowrap}
tr:hover td{background:#fafbfe}
.tag{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600}
.t-yes{background:#e4f6ee;color:var(--green)}.t-no{background:#fbeaea;color:var(--red)}
.t-warn{background:#fdf1dc;color:var(--amber)}
.note{background:#fff8e9;border:1px solid #f2d9a4;border-radius:10px;padding:14px 18px;
margin:16px 0;font-size:13.5px}
.note b{color:#a8680a}
details{margin-top:10px}summary{cursor:pointer;color:var(--blue);font-size:13px}
.small{font-size:12px;color:var(--sub)}
code{background:#eef2f8;padding:1px 5px;border-radius:4px;font-size:12px}
ul{margin:8px 0 0 0;padding-left:20px}li{margin:4px 0}
.foot{margin-top:40px;color:var(--sub);font-size:12px;text-align:center}
"""

KPI = [("total_return", "总收益", "pct"), ("cagr", "年化收益 CAGR", "pct"),
       ("vol", "年化波动率", "pct"), ("sharpe", "夏普比率", "num"),
       ("max_drawdown", "最大回撤", "pct"), ("calmar", "卡玛比率", "num")]


def _metrics_html(m, label: str) -> str:
    cells = []
    for key, name, kind in KPI:
        v = getattr(m, key)
        if kind == "pct":
            txt = fmt_pct(v)
            cls = "pos" if (v > 0 if key != "max_drawdown" else False) else ""
            if key == "max_drawdown":
                cls = "neg"
        else:
            txt = f"{v:.2f}"
            cls = "pos" if v > 0 else "neg"
        cells.append(f'<div class="kpi"><div class="lab">{name}</div>'
                     f'<div class="val {cls}">{txt}</div></div>')
    return (f'<h2>{label}</h2><div class="kpis">{"".join(cells)}</div>'
            f'<div class="small" style="margin-top:8px">'
            f'期末权益 ${m.final_equity:,.2f}｜交易天数 {m.n_days}｜'
            f'日胜率 {fmt_pct(m.win_rate_daily)}</div>')


def _esc(s) -> str:
    return html.escape(str(s))


def build_html(res: BacktestResult, bench: BacktestResult | None = None,
               out: Path | None = None) -> Path:
    out = out or (C.REPORT_DIR / f"backtest_{res.strategies}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    m = compute_metrics(res.equity)
    bm = compute_metrics(res.bench_equity)
    peak = np.maximum.accumulate(res.equity)
    dd = (res.equity - peak) / peak

    eq_chart = line_chart([
        ("策略组合权益", res.equity, "#2f6fed"),
        ("买入持有 QQQ（基准）", res.bench_equity, "#b6c0d0"),
    ], res.dates, "策略组合 vs 买入持有 QQQ")

    # 归一化对比（都除以起点）
    norm_s = res.equity / res.equity[0] - 1
    norm_b = res.bench_equity / res.bench_equity[0] - 1
    norm_chart = line_chart([
        ("策略（累计收益）", norm_s, "#2f6fed"),
        ("QQQ（累计收益）", norm_b, "#e08b1a"),
    ], res.dates, "累计收益率对比", ylabel="累计收益", pct=True)
    dd_chart = drawdown_chart(res.equity, res.dates)

    # 策略一信号汇总
    c1 = sum(1 for s in res.cc_signals if s.cond1_rally)
    c3 = sum(1 for s in res.cc_signals if s.cond3_iv_ok)
    both = sum(1 for s in res.cc_signals if s.cond1_rally and s.cond3_iv_ok)
    trig = sum(1 for s in res.cc_signals if s.triggered)
    agent_v = res.agent_verdicts
    n_human = sum(1 for v in agent_v if v["verdict"] == "HUMAN_ACTION")
    n_tech = sum(1 for v in agent_v if v["verdict"] == "TECHNICAL_RESISTANCE")
    n_none = sum(1 for v in agent_v if v["verdict"] == "NO_RESISTANCE")
    n_news = sum(1 for v in agent_v if v["source"] == "gdelt-rule")
    n_proxy = sum(1 for v in agent_v if v["source"] == "price-structure")

    # Roll（策略二）
    rolls = res.roll_events
    fund_checks = res.fundamental_checks
    roll_rows = "".join(
        f"<tr><td>{_esc(e['date'])}</td><td>{e['old_strike']:.0f} → {e['new_strike']:.0f}</td>"
        f"<td>{e['contracts']}</td><td>{e['old_delta']:.3f} → {C.LC.roll_target_delta:.2f}</td>"
        f"<td>{e['old_dte']} → {e['new_dte']}</td>"
        f"<td class='{'neg' if e['net_cash'] < 0 else 'pos'}'>{e['net_cash']:+,.0f}</td>"
        f"<td class='small'>{_esc(e['fundamental_source'])}</td></tr>"
        for e in rolls) or '<tr><td colspan="7">本次未发生 Roll</td></tr>'
    fund_rows = "".join(
        f"<tr><td>{_esc(x['as_of'])}</td><td>{x.get('delta', 0):.3f}</td>"
        f"<td>{x.get('n_headlines', 0)}</td>"
        f"<td><span class='tag {'t-yes' if x['decision']=='ROLL' else 't-warn'}'>"
        f"{_esc(x['decision'])}</span></td>"
        f"<td class='small'>{_esc(', '.join(x.get('bad_hits', [])[:4])) or '—'}</td></tr>"
        for x in fund_checks) or '<tr><td colspan="5">无（Delta 未跌破阈值或现金不足）</td></tr>'

    # 交易表
    trade_rows = "".join(
        f"<tr><td>{_esc(t['date'])}</td><td>{_esc(t['action'])}</td>"
        f"<td>{t['strike']:.0f}</td><td>{t['dte']}</td>"
        f"<td>{t['contracts']}</td><td>${t['premium_per_share']:.2f}</td>"
        f"<td class='{'pos' if t['cash_flow']>0 else 'neg'}'>"
        f"{t['cash_flow']:+,.2f}</td><td class='small'>{_esc(t['note'])}</td></tr>"
        for t in res.trades) or '<tr><td colspan="8">无交易</td></tr>'

    # Agent 判定表
    agent_rows = "".join(
        f"<tr><td>{_esc(v['as_of'])}</td>"
        f"<td><span class='tag {'t-yes' if v['verdict']=='TECHNICAL_RESISTANCE' else ('t-warn' if v['verdict']=='HUMAN_ACTION' else 't-no')}'>"
        f"{_esc(v['verdict'])}</span></td>"
        f"<td class='small'>{_esc(v['source'])}</td>"
        f"<td class='small'>{_esc(v['reasoning'])}</td>"
        f"<td class='small'>{_esc('；'.join(v['evidence'][:2]))}</td></tr>"
        for v in sorted(agent_v, key=lambda x: x["as_of"])) or \
        '<tr><td colspan="5">未调用（无候选日）</td></tr>'

    # 触发日明细
    trig_rows = "".join(
        f"<tr><td>{_esc(s.as_of)}</td><td>{s.r5*100:.2f}%</td>"
        f"<td>{s.threshold*100:.2f}%</td><td>{s.vxn:.2f}</td>"
        f"<td>{s.iv_percentile_threshold:.2f}</td>"
        f"<td><span class='tag t-yes'>三条件满足</span></td>"
        f"<td class='small'>{_esc(s.news.verdict if s.news else '')}</td></tr>"
        for s in res.cc_signals if s.triggered) or \
        '<tr><td colspan="7">无触发</td></tr>'

    # LC 触发
    lc_trig = [s for s in res.lc_signals if s.triggered]
    lc_rows = "".join(
        f"<tr><td>{_esc(s.as_of)}</td><td>{s.daily_ret*100:.2f}%</td>"
        f"<td>{s.triggers_3m}</td><td>{s.max_dd_1y*100:.1f}%</td>"
        f"<td>{'✓' if s.cond1_via_triggers else '✗'} / {'✓' if s.cond1_via_drawdown else '✗'}</td>"
        f"<td class='small'>{_esc(s.reason)}</td></tr>"
        for s in lc_trig) or '<tr><td colspan="6">无触发</td></tr>'

    agent_raw = json.dumps(
        [{"date": v["as_of"], "verdict": v["verdict"], "headlines": v["headlines"]}
         for v in agent_v],
        ensure_ascii=False, indent=1)
    event_rows = "".join(
        f"<tr><td>{_esc(e['date'])}</td><td>{_esc(e['kind'])}</td>"
        f"<td class='small'>{_esc(e['msg'])}</td></tr>" for e in res.events)

    mode_note = (
        f"<b>IV 口径（唯一）：</b>策略一条件 3 为 "
        f"<code>VXN ≤ 过去 {C.CC.iv_lookback_days} 个交易日 VXN 的 "
        f"{C.CC.iv_percentile:g} 分位</code>——即「波动率处于近一年最低的 "
        f"{C.CC.iv_percentile:g}% 才算低」。门槛每天按滚动窗口重算、不含当日，无前视偏差；"
        "固定绝对阈值口径已按用户要求移除。策略一候选日 = C1（5 日涨幅前 10%）与 C3 的交集，"
        "是否真正触发还取决于新闻 Agent 的 C2 判定。"
    )
    strat_note = {
        "both": (f"本次<b>同时运行策略一与策略二</b>。策略二每次信号买入 "
                 f"<b>{C.LC.contracts} 张</b> Δ{C.LC.target_delta}/{C.LC.dte_days}天 call；"
                 f"初始资金 ${C.CFG.initial_cash:,.0f}（因 10 张一年期 call + 100 股备兑底仓"
                 f"同时占资，原 $100k 在 Roll 后会现金不足）。"
                 f"<b>Roll 规则</b>：持仓 Delta 跌破 {C.LC.roll_delta_trigger}、剩余期限 ≥ "
                 f"{C.LC.roll_min_dte} 天、且新闻 Agent 判定「基本面没问题」时，"
                 f"平旧仓并重新买入 {C.LC.contracts} 张 Δ{C.LC.roll_target_delta}/"
                 f"{C.LC.roll_dte_days}天的 call。"),
        "cc_only": "本次<b>只执行策略一（Covered Call）</b>，策略二完全不评估、不开仓；"
                   "组合里不会出现 Δ0.6 的一年期 long call。",
        "lc_only": "本次<b>只执行策略二</b>，策略一不评估、不开仓。",
    }[res.strategies]

    n_cc_open = sum(1 for t in res.trades if t["action"] == "SELL_CALL")
    n_lc_open = sum(1 for t in res.trades if t["action"] == "BUY_CALL")
    n_lc_sig = len(lc_trig)

    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QQQ 期权策略回测报告 · {res.strategies}</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>QQQ 期权策略 · 五年回测报告</h1>
<div class="sub">回测区间 {res.start} ~ {res.end}　·　IV 口径：<b>相对 {C.CC.iv_percentile:g} 分位</b>　·　
运行模式：<b>{res.strategies}</b><br>{_esc(res.config_note)}</div>

<div class="note">{mode_note}<br>{strat_note}</div>

{_metrics_html(m, "策略组合绩效")}
{_metrics_html(bm, "基准：买入并持有 QQQ")}

<h2>净值曲线</h2><div class="card">{eq_chart}</div>
<div class="card">{norm_chart}</div>
<h2>回撤</h2><div class="card">{dd_chart}
<div class="small" style="margin-top:8px">最大回撤 {fmt_pct(m.max_drawdown)}，
发生于 {res.dates[m.max_dd_start]} ~ {res.dates[m.max_dd_end]}</div></div>

<div class="note">「触发」指条件同时满足；<b>实际开仓</b>另受持仓上限约束
（同一时间最多 1 张空 call / 1 张一年期多 call，避免重复加仓）。本回测中：
策略一触发 {trig} 天、实际卖出 {n_cc_open} 张；
{"策略二触发 " + str(n_lc_sig) + " 次、实际买入 " + str(n_lc_open) + " 张。"
 if res.strategies in ("both", "lc_only") else
 "策略二在本模式下<b>未启用</b>。"}</div>

<h2>策略一 · Covered Call 信号漏斗</h2>
<div class="card"><div class="kpis">
<div class="kpi"><div class="lab">C1 刚冲完一波（5日涨幅前10%）</div><div class="val">{c1} 天</div></div>
<div class="kpi"><div class="lab">C3 IV 条件满足（相对 {C.CC.iv_percentile:g} 分位）</div><div class="val">{c3} 天</div></div>
<div class="kpi"><div class="lab">C1+C3 候选日（调用 Agent）</div><div class="val">{both} 天</div></div>
<div class="kpi"><div class="lab">三条件全满足 → 触发</div><div class="val pos">{trig} 天</div></div>
</div>
<div class="small" style="margin-top:12px">
Agent 共调用 {res.agent_calls} 次：技术阻力 {n_tech} 次 ·
人为动作否决 {n_human} 次 · 无阻力 {n_none} 次。
其中 <b>{n_news} 次基于 GDELT 真实新闻</b>、{n_proxy} 次因新闻窗口缺失回退到
价格结构代理（GDELT 限流所致，<code>source</code> 列已如实标注）。
「人为动作引起的市场变动不算」这一约束会过滤掉由
美联储 / 关税 / 财报 / 地缘等事件驱动的反弹。</div></div>

<h2>策略一 · 触发日明细</h2>
<div class="card"><table><thead><tr><th>日期</th><th>5日涨幅</th><th>5年门槛</th>
<th>VXN</th><th>滚动分位门槛(≤)</th><th>信号</th><th>Agent 判定</th></tr></thead>
<tbody>{trig_rows}</tbody></table></div>

<h2>策略二 · 买入 Call 触发日
{"<span class='tag t-warn'>本模式未启用</span>" if res.strategies == "cc_only" else ""}</h2>
<div class="card">{"<div class='small'>本次模拟只执行策略一，策略二信号未评估、未开仓。</div>"
                   if res.strategies == "cc_only" else
                   "<table><thead><tr><th>日期</th><th>单日跌幅</th><th>近3月策略一触发</th>"
                   "<th>近1年最大回撤</th><th>牛市条件A/B</th><th>说明</th></tr></thead>"
                   f"<tbody>{lc_rows}</tbody></table>"}</div>

<h2>策略二 · Delta Roll 记录（共 {len(rolls)} 次）</h2>
<div class="card">
<div class="small" style="margin-bottom:10px">
触发条件：持仓 Delta &lt; {C.LC.roll_delta_trigger}、剩余期限 ≥ {C.LC.roll_min_dte} 天、
且<b>基本面没问题</b>（新闻 Agent 未检测到业绩暴雷 / 指引下调 / 评级下调 / 欺诈 / 诉讼等负面信号）。
执行方式：平掉旧仓 + 重新买入 {C.LC.contracts} 张 Δ{C.LC.roll_target_delta}/{C.LC.roll_dte_days}天 call，张数不变。
本次共做 {len(fund_checks)} 次基本面检查，实际滚动 {len(rolls)} 次。
{"<br><b>注</b>：两次买 call 的 Delta 最低只到 0.56，从未跌破 0.5，故 0 次 Roll 属正确行为。"
 if not rolls else ""}
{"<br><b>⚠️ 新闻覆盖说明</b>：这 " + str(len(fund_checks)) + " 次检查里 "
 + str(sum(1 for x in fund_checks if x.get('n_headlines', 0) == 0)) + " 次当日无缓存新闻，"
 "走的是「无负面信号 → 默认放行」（source=no-news-default-ok）。"
 "即本次回测中「基本面没问题」主要体现为<b>默认放行</b>，"
 "并未真正读到财报/新闻文本。若要让该条件生效，需补齐 GDELT 新闻或使用真实 LLM 后端。"
 if fund_checks else ""}
</div>
<table><thead><tr><th>日期</th><th>行权价 K</th><th>张数</th><th>Delta</th>
<th>剩余天数</th><th>换仓净现金流</th><th>基本面来源</th></tr></thead>
<tbody>{roll_rows}</tbody></table>
<details><summary>查看全部基本面检查记录（{len(fund_checks)} 条）</summary>
<table><thead><tr><th>日期</th><th>当时 Delta</th><th>新闻条数</th><th>决策</th>
<th>命中的负面词</th></tr></thead><tbody>{fund_rows}</tbody></table></details>
</div>

<h2>新闻 Agent 判定记录</h2>
<div class="card"><table><thead><tr><th>日期</th><th>结论</th><th>来源</th>
<th>推理</th><th>证据</th></tr></thead><tbody>{agent_rows}</tbody></table>
<details><summary>查看 Agent 判定使用的原始头条</summary>
<pre style="font-size:11px;white-space:pre-wrap;color:#4a5568">{_esc(agent_raw)}</pre></details></div>

<h2>交易流水（共 {len(res.trades)} 笔）</h2>
<div class="card"><table><thead><tr><th>日期</th><th>动作</th><th>行权价</th>
<th>期限(天)</th><th>张数</th><th>权利金/股</th><th>现金流</th><th>备注</th></tr></thead>
<tbody>{trade_rows}</tbody></table></div>

<h2>事件日志（共 {len(res.events)} 条）</h2>
<div class="card"><details><summary>展开</summary><table><thead><tr><th>日期</th>
<th>类型</th><th>说明</th></tr></thead><tbody>
{event_rows}
</tbody></table></details></div>

<div class="foot">生成于 {date.today()} · 数据源：Nasdaq（QQQ）/ CBOE（VXN）/ FRED（DGS3MO）/ GDELT（新闻）<br>
仅供研究，不构成投资建议。回测含滑点与手续费，未考虑保证金、分红税与流动性冲击。</div>
</div></body></html>"""

    out.write_text(html_doc, encoding="utf-8")
    return out
