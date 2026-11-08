"""逐日回测引擎。

时间推进方式
------------
对每个交易日 t（按真实交易日历，跳过周末/节假日）：
  1. 先处理「到期日 <= 今日」的持仓（按自然日计，7天/365天期权按时到期）
  2. 用截至 t 收盘的数据计算策略一/策略二信号（无前视）
  3. 按信号开仓（策略一有备兑硬约束）
  4. 按收盘价盯市，记录权益

策略二依赖「策略一近3个月触发次数」，因此先跑策略一再跑策略二，
且触发标记只在当日收盘后写入，供后续交易日使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from .. import config as C
from ..core.market_data import build_market_data, MarketData
from ..core.indicators import rolling_5d_returns, rolling_threshold_series
from ..core.options import quote_call, mark_delta
from ..agent.news_agent import NewsAgent
from ..strategies.covered_call import evaluate_covered_call, CoveredCallSignal
from ..strategies.long_call import evaluate_long_call, LongCallSignal
from .portfolio import Portfolio, Position


@dataclass
class BacktestResult:
    config_note: str
    strategies: str          # "both" | "cc_only" | "lc_only"
    start: date
    end: date
    dates: list[date] = field(default_factory=list)
    equity: np.ndarray = field(default_factory=lambda: np.array([]))
    cash: np.ndarray = field(default_factory=lambda: np.array([]))
    shares: np.ndarray = field(default_factory=lambda: np.array([]))
    qqq_close: np.ndarray = field(default_factory=lambda: np.array([]))
    cc_signals: list[CoveredCallSignal] = field(default_factory=list)
    lc_signals: list[LongCallSignal] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    agent_calls: int = 0
    agent_verdicts: list[dict] = field(default_factory=list)
    roll_events: list[dict] = field(default_factory=list)
    fundamental_checks: list[dict] = field(default_factory=list)
    bench_equity: np.ndarray = field(default_factory=lambda: np.array([]))


VALID_STRATEGIES = ("both", "cc_only", "lc_only")


class Backtester:
    def __init__(self, md: MarketData | None = None,
                 agent: NewsAgent | None = None, cfg=None, cc=None, lc=None,
                 strategies: str | None = None):
        self.md = md or build_market_data()
        self.agent = agent or NewsAgent()
        self.cfg = cfg or C.CFG
        self.cc = cc or C.CC
        self.lc = lc or C.LC
        # CLI --strategies 优先，其次 cfg.strategies，默认 "both" 兜底
        self.strategies = strategies or getattr(self.cfg, "strategies", "both")
        if self.strategies not in VALID_STRATEGIES:
            raise ValueError(f"strategies 必须是 {VALID_STRATEGIES} 之一，收到 {self.strategies!r}")
        self.run_cc = self.strategies in ("both", "cc_only")
        self.run_lc = self.strategies in ("both", "lc_only")
        self.r5 = rolling_5d_returns(self.md.close, self.cc.rally_window)
        self.thr = rolling_threshold_series(self.r5, self.cc.lookback_days,
                                            self.cc.pct_threshold)

    # ------------------------------------------------------------ 到期处理
    @staticmethod
    def _process_expiries(pf: Portfolio, md: MarketData, idx: int) -> None:
        """按自然日处理到期；在开新仓之前调用。"""
        d = md.dates[idx]
        S = float(md.close[idx])
        r = float(md.rf[idx]); vxn = float(md.vxn[idx])

        for p in list(pf.short_calls):
            if p.expiry_date <= d:
                pf.close_short_call(p, S, d, reason="到期")
        for p in list(pf.long_calls):
            if p.expiry_date <= d:
                dte = (p.expiry_date - d).days
                pf.close_long_call(p, S, vxn, r, dte, d, reason="到期")

    # ------------------------------------------------------------ Roll
    def _maybe_roll(self, pf: Portfolio, md: MarketData, idx: int,
                    res: BacktestResult) -> None:
        """策略二持仓 Roll 检查：Delta 掉到阈值以下 + 基本面没问题 → 滚到 Δ0.6 / 365 天。

        由用户在 2026-11 提出：「如果策略二基本面没问题，但 Delta 掉到了 0.5 以下，
        Roll 到 Delta 为 0.6」。
        """
        cfg = self.lc
        if not cfg.roll_enabled or not self.run_lc or not pf.long_calls:
            return
        d = md.dates[idx]
        S = float(md.close[idx]); r = float(md.rf[idx]); vxn = float(md.vxn[idx])

        for pos in list(pf.long_calls):
            dte = (pos.expiry_date - d).days
            if dte < cfg.roll_min_dte:
                continue                       # 临近到期不滚
            delta = mark_delta(S, pos.strike, dte, vxn, r)
            if delta >= cfg.roll_delta_trigger:
                continue                       # Delta 还没掉到 0.5 以下

            # 冷却：同一批次两次 Roll 至少间隔 roll_cooldown_days 个自然日
            if (d - pos.open_date).days < cfg.roll_cooldown_days:
                continue

            # 1) 先做资金可行性检查（Roll 是净额换仓：平旧仓所得可抵扣新仓成本）
            #    这步不涉及基本面，也不产生逐日重复日志。
            q = quote_call(S, cfg.roll_target_delta, cfg.roll_dte_days, vxn, r)
            new_cost = (q.premium * pos.multiplier * pos.contracts
                        * (1 + C.PR.slippage)
                        + C.PR.commission_per_contract * pos.contracts)
            old_proceeds = pos.mark(S, dte, vxn, r) * pos.multiplier * pos.contracts
            if new_cost > pf.cash + old_proceeds:
                # 每个持仓只记录一次，避免 100+ 条重复噪音
                if not pos.roll_cash_blocked:
                    pos.roll_cash_blocked = True
                    pf._log(d, "ROLL_BLOCKED_CASH", 0.0,
                            f"Delta={delta:.3f}<{cfg.roll_delta_trigger}，但换仓净支出 "
                            f"{new_cost - old_proceeds:,.0f}（新仓 {new_cost:,.0f} / "
                            f"平旧仓 {old_proceeds:,.0f}）超出可用现金 "
                            f"{pf.cash:,.0f}，Roll 暂时不可行")
                continue

            # 2) 现金够了才调用 Agent 做「基本面没问题」检查
            fv = self.agent.fundamentals_ok(d)
            decision = "ROLL" if fv.ok else "SKIP_BAD_FUNDAMENTALS"
            res.fundamental_checks.append({
                **fv.to_dict(),
                "delta": delta, "dte": dte, "strike": pos.strike,
                "decision": decision,
            })
            if not fv.ok:
                pf._log(d, "ROLL_SKIPPED", 0.0,
                        f"Delta={delta:.3f}<{cfg.roll_delta_trigger} 但基本面有问题"
                        f"（{', '.join(fv.bad_hits[:4])}），不 Roll")
                continue

            # 3) 执行 Roll：平旧仓 + 开新的 365 天 Δ0.6 call，张数保持 10
            new_pos = Position(
                kind="long_call", strike=q.strike,
                entry_premium=q.premium, dte=cfg.roll_dte_days,
                open_date=d, expiry_date=d + timedelta(days=cfg.roll_dte_days),
                contracts=pos.contracts,
                strategy=f"{cfg.roll_target_delta}",
            )
            note = (f"｜基本面检查 OK（{fv.n_headlines} 条新闻，无负面信号，"
                    f"source={fv.source}）")
            try:
                pf.roll_long_call(pos, new_pos, S, vxn, r, dte, d,
                                  old_delta=delta, fundamental_note=note)
                res.roll_events.append({
                    "date": d.isoformat(),
                    "old_strike": pos.strike, "new_strike": new_pos.strike,
                    "contracts": new_pos.contracts,
                    "old_delta": delta, "old_dte": dte,
                    "new_dte": cfg.roll_dte_days,
                    "net_cash": old_proceeds - new_cost,
                    "fundamental_ok": True,
                    "fundamental_source": fv.source,
                    "n_headlines": fv.n_headlines,
                })
            except RuntimeError as e:
                pf._log(d, "ROLL_FAILED", 0.0, str(e))

    # ------------------------------------------------------------ 主循环
    def run(self) -> BacktestResult:
        md = self.md
        start_i = md.index_of(date.fromisoformat(self.cfg.start))
        end_i = md.index_of(date.fromisoformat(self.cfg.end))
        if end_i <= start_i:
            raise ValueError("回测区间无效")

        pf = Portfolio(cash=self.cfg.initial_cash)
        n = end_i - start_i + 1
        cc_flags = np.zeros(len(md.dates), dtype=bool)   # 策略一每日触发标记

        mode_label = {"both": "策略一 + 策略二",
                      "cc_only": "仅策略一（Covered Call）",
                      "lc_only": "仅策略二（买入 Call）"}[self.strategies]
        roll_note = ("，Roll: Delta<{:.2f}→{:.2f}/{}天(基本面OK)".format(
            self.lc.roll_delta_trigger, self.lc.roll_target_delta,
            self.lc.roll_dte_days) if (self.run_lc and self.lc.roll_enabled) else "")
        res = BacktestResult(
            config_note=(f"{mode_label}｜初始资金 ${self.cfg.initial_cash:,.0f}，"
                         f"策略一 Δ={self.cc.target_delta}/{self.cc.dte_days}天 "
                         f"IV条件 VXN≤近{self.cc.iv_lookback_days}日"
                         f"{self.cc.iv_percentile:g}分位，"
                         f"策略二 {self.lc.contracts}张 Δ={self.lc.target_delta}/"
                         f"{self.lc.dte_days}天 跌幅>{abs(self.lc.drop_threshold)*100:.0f}%"
                         f"{roll_note}"),
            strategies=self.strategies,
            start=md.dates[start_i], end=md.dates[end_i],
        )

        eq = np.zeros(n); cash_a = np.zeros(n); sh_a = np.zeros(n)
        close_a = np.zeros(n); bench = np.zeros(n)
        S0 = float(md.close[start_i])
        bench_shares = self.cfg.initial_cash / S0   # 基准：买入并持有 QQQ

        for k, t in enumerate(range(start_i, end_i + 1)):
            d = md.dates[t]
            S = float(md.close[t]); r = float(md.rf[t]); vxn = float(md.vxn[t])

            # 0) 到期结算（先于任何新信号）
            self._process_expiries(pf, md, t)

            # 0.5) 策略二持仓 Roll 检查（Delta<0.5 + 基本面 OK → 滚到 Δ0.6/365天）
            self._maybe_roll(pf, md, t, res)

            # 1) 策略一信号
            if self.run_cc:
                sig1 = evaluate_covered_call(md, t, self.r5, self.thr,
                                             self.agent)
                if t >= start_i:
                    res.cc_signals.append(sig1)

                # 2) 策略一开仓（含备兑硬约束）
                #    「触发」= 三条件满足，即计入策略二的 3 个月触发计数；
                #    实际是否开仓另受 max_open_short_calls 限制（避免重复加仓）。
                if sig1.triggered:
                    cc_flags[t] = True
                if sig1.triggered and len(pf.short_calls) < self.cc.max_open_short_calls:
                    q = quote_call(S, self.cc.target_delta, self.cc.dte_days,
                                   vxn, r)
                    expiry = d + timedelta(days=self.cc.dte_days)
                    pos = Position(
                        kind="short_call", strike=q.strike,
                        entry_premium=q.premium, dte=self.cc.dte_days,
                        open_date=d, expiry_date=expiry,
                        contracts=self.cc.contracts,
                        strategy=f"{self.cc.target_delta}",
                    )
                    try:
                        pf.ensure_covered(self.cc.contracts, S, d)   # 绝不裸卖
                        # 二次断言：引擎级硬约束
                        assert pf.shares >= self.cc.contracts * self.cc.shares_per_contract, \
                            "裸卖被阻止：持股不足"
                        pf.open_short_call(pos, q.premium, d)
                    except (RuntimeError, AssertionError) as e:
                        pf._log(d, "BLOCKED_NAKED", 0.0, str(e))

            # 3) 策略二信号（用「截至昨日」的触发计数，避免当日自我引用）
            if self.run_lc:
                sig2 = evaluate_long_call(md, t, cc_flags)
                res.lc_signals.append(sig2)

                # 4) 策略二开仓
                if sig2.triggered and len(pf.long_calls) < self.lc.max_open_long_calls:
                    q = quote_call(S, self.lc.target_delta, self.lc.dte_days,
                                   vxn, r)
                    expiry = d + timedelta(days=self.lc.dte_days)
                    pos = Position(
                        kind="long_call", strike=q.strike,
                        entry_premium=q.premium, dte=self.lc.dte_days,
                        open_date=d, expiry_date=expiry,
                        contracts=self.lc.contracts,
                        strategy=f"{self.lc.target_delta}",
                    )
                    try:
                        pf.open_long_call(pos, d)
                    except RuntimeError as e:
                        pf._log(d, "SKIP_LONG", 0.0, str(e))

            # 5) 盯市
            eq[k] = pf.equity(S, md, t)
            cash_a[k] = pf.cash
            sh_a[k] = pf.shares
            close_a[k] = S
            bench[k] = bench_shares * S

        # 6) 收尾：区间末按盯市价平掉所有未到期持仓（便于统计已实现+浮动）
        d = md.dates[end_i]; S = float(md.close[end_i])
        r = float(md.rf[end_i]); vxn = float(md.vxn[end_i])
        for p in list(pf.short_calls):
            dte = (p.expiry_date - d).days
            px = p.mark(S, dte, vxn, r)
            cost = px * p.multiplier * p.contracts
            pf.cash -= cost
            pf.short_calls.remove(p)
            pf._log(d, "MARK_SHORT_CLOSE", -cost,
                    f"回测结束按盯市价 {px:.2f} 买回 short call K={p.strike:.0f}")
        for p in list(pf.long_calls):
            dte = (p.expiry_date - d).days
            pf.close_long_call(p, S, vxn, r, dte, d, reason="回测结束盯市平仓")
        eq[-1] = pf.cash + pf.shares * S

        res.dates = md.dates[start_i:end_i + 1]
        res.equity = eq; res.cash = cash_a; res.shares = sh_a
        res.qqq_close = close_a; res.bench_equity = bench
        res.events = list(pf.events); res.trades = list(pf.trades)
        res.agent_calls = self.agent.calls
        res.agent_verdicts = [v.to_dict() for v in self.agent._cache.values()]
        return res


def run_backtest(strategies: str | None = None, **kwargs) -> BacktestResult:
    return Backtester(strategies=strategies, **kwargs).run()


if __name__ == "__main__":
    for s in ("cc_only", "both"):
        r = run_backtest(strategies=s)
        print(f"{s:8s} final=${r.equity[-1]:,.2f} "
              f"trades={len(r.trades)} agent={r.agent_calls} "
              f"cc_trig={sum(1 for x in r.cc_signals if x.triggered)} "
              f"lc_trig={sum(1 for x in r.lc_signals if x.triggered)}")
