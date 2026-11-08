"""单元测试：核心数学、指标、Agent 判定、备兑硬约束、回测一致性。

运行： python -m unittest qqq_options.tests.test_all -v
"""

from __future__ import annotations

import math
import unittest
from datetime import date, timedelta

import numpy as np

from .. import config as C
from ..core.options import (call_price, call_delta, strike_for_delta,
                            norm_cdf, quote_call, mark_call, mark_delta)
from ..core.indicators import (rolling_5d_returns, pct_threshold,
                               rolling_threshold_series, max_drawdown,
                               daily_returns)
from ..core.market_data import build_market_data
from ..agent.news_agent import (RuleClassifier, NewsItem, NewsAgent,
                                CachedGDELTProvider, NewsVerdict)
from ..engine.portfolio import Portfolio, Position
from ..engine.backtest import Backtester
from ..strategies.covered_call import iv_condition_ok


class TestMath(unittest.TestCase):
    def test_norm_cdf(self):
        self.assertAlmostEqual(norm_cdf(0), 0.5, places=10)
        self.assertAlmostEqual(norm_cdf(1.96), 0.975, places=3)
        self.assertAlmostEqual(norm_cdf(-1.96), 0.025, places=3)

    def test_call_price_put_call_parity(self):
        S, K, T, r, q, sig = 500, 520, 0.5, 0.04, 0.006, 0.25
        c = call_price(S, K, T, r, q, sig)
        # put = c - S*e^{-qT} + K*e^{-rT}
        put = c - S * math.exp(-q * T) + K * math.exp(-r * T)
        self.assertGreater(put, 0)
        self.assertAlmostEqual(c - put, S * math.exp(-q * T) - K * math.exp(-r * T), places=8)
        self.assertGreater(c, max(0, S - K * math.exp(-r * T)))

    def test_delta_monotonic_in_strike(self):
        S, T, r, q, sig = 500, 7 / 365, 0.04, 0.006, 0.25
        ds = [call_delta(S, K, T, r, q, sig) for K in (450, 500, 550, 600)]
        self.assertTrue(all(ds[i] > ds[i + 1] for i in range(3)))

    def test_strike_for_delta(self):
        S, T, r, q, sig = 500, 7 / 365, 0.04, 0.006, 0.25
        for target in (0.2, 0.6):
            K = strike_for_delta(target, S, T, r, q, sig)
            d = call_delta(S, K, T, r, q, sig)
            # 行权价按 1 美元整数档取整，允许 0.02 偏差
            self.assertLess(abs(d - target), 0.02, f"target={target} K={K} delta={d}")
        self.assertGreater(strike_for_delta(0.2, S, T, r, q, sig),
                           strike_for_delta(0.6, S, T, r, q, sig))  # 越低 delta 越虚值、K 越高

    def test_quote_and_mark(self):
        q = quote_call(500, 0.2, 7, 30.0, 0.04)
        self.assertGreater(q.premium, 0)
        self.assertAlmostEqual(q.delta, 0.2, delta=0.02)
        self.assertAlmostEqual(mark_call(500, q.strike, 7, 30.0, 0.04),
                               q.theo, places=8)


class TestIndicators(unittest.TestCase):
    def test_rolling_5d(self):
        c = np.array([100, 101, 102, 103, 104, 110.0])
        r = rolling_5d_returns(c, 5)
        self.assertTrue(np.isnan(r[:5]).all())
        self.assertAlmostEqual(r[5], 110 / 100 - 1)

    def test_pct_threshold_1260(self):
        # 1260 个等差数，10% 门槛应为第 126 大的值
        h = np.linspace(-0.1, 0.1, 1260)
        t = pct_threshold(h, 0.10)
        expected = np.sort(h)[::-1][125]
        self.assertAlmostEqual(t, expected, places=12)
        self.assertAlmostEqual(t, 0.1 - 0.2 * 125 / 1259, places=6)

    def test_no_lookahead_in_threshold(self):
        r = np.arange(1300, dtype=float) / 1000.0
        thr = rolling_threshold_series(r, lookback=1260)
        self.assertTrue(np.isinf(thr[:1260]).all())
        # thr[1260] 只由 r[0:1260] 决定
        self.assertAlmostEqual(thr[1260], pct_threshold(r[:1260]), places=12)

    def test_max_drawdown(self):
        self.assertAlmostEqual(max_drawdown(np.array([100, 120, 90, 130])), 0.25)
        self.assertAlmostEqual(max_drawdown(np.array([100, 110, 120])), 0.0)

    def test_daily_returns(self):
        r = daily_returns(np.array([100, 99, 100.0]))
        self.assertAlmostEqual(r[1], -0.01)
        self.assertAlmostEqual(r[2], 1 / 99)


class TestNewsAgent(unittest.TestCase):
    def setUp(self):
        self.rc = RuleClassifier()
        self.d = date(2024, 8, 13)

    def test_human_action_veto(self):
        items = [NewsItem(self.d, "Nasdaq jumps after Fed Chair Powell signals rate cut"),
                 NewsItem(self.d, "QQQ hits resistance at 200-day moving average")]
        has, human, ev, why = self.rc.classify(items, self.d, {"recent_closes": [100.0] * 60})
        self.assertTrue(human)
        self.assertFalse(has)
        self.assertIn("人为", why)

    def test_technical_resistance(self):
        items = [NewsItem(self.d, "QQQ overbought, RSI stalls at key resistance level"),
                 NewsItem(self.d, "Tech stocks struggle at trendline resistance")]
        has, human, ev, why = self.rc.classify(items, self.d, {})
        self.assertTrue(has)
        self.assertFalse(human)

    def test_price_structure_fallback(self):
        # 现价低于 MA50 且距前高 3~20% -> 判定有阻力
        closes = [100.0] * 50 + [95.0] * 5 + [90.0, 91, 92, 93, 94]
        # 让 60 日高点约 100，现价 94 -> 距前高 6.4%
        closes = [100.0] * 55 + [90, 91, 92, 93, 94]
        has, human, ev, why = self.rc.classify([], self.d, {"recent_closes": closes})
        self.assertTrue(has)
        self.assertFalse(human)
        self.assertTrue(any("均线" in x or "套牢" in x for x in ev))

    def test_no_future_news(self):
        future = NewsItem(self.d + timedelta(days=3), "Fed cuts rates, QQQ surges")
        has, human, ev, why = self.rc.classify([future], self.d, {})
        self.assertFalse(human)     # 未来新闻不得参与判定

    def test_agent_verdict_dataclass(self):
        a = NewsAgent()
        v = a.analyze(self.d, md=build_market_data(), idx=100, r5=0.05, vxn=25)
        self.assertIn(v.verdict, ("TECHNICAL_RESISTANCE", "HUMAN_ACTION", "NO_RESISTANCE"))
        self.assertEqual(v.has_resistance, v.verdict == "TECHNICAL_RESISTANCE")
        self.assertEqual(a.analyze(self.d).as_of, self.d)   # 缓存命中


class TestCoveredConstraint(unittest.TestCase):
    def test_ensure_covered_buys_stock(self):
        pf = Portfolio(cash=100_000)
        pf.ensure_covered(1, S=500.0, d=date(2024, 1, 2))
        self.assertGreaterEqual(pf.shares, 100)
        self.assertLess(pf.cash, 100_000)

    def test_insufficient_cash_blocks_naked(self):
        pf = Portfolio(cash=100.0)      # 买不起 100 股
        with self.assertRaises(RuntimeError):
            pf.ensure_covered(1, S=500.0, d=date(2024, 1, 2))
        self.assertEqual(pf.shares, 0)

    def test_existing_shares_keeps_covered(self):
        pf = Portfolio(cash=100_000, shares=100)
        pf.ensure_covered(1, S=500.0, d=date(2024, 1, 2))
        self.assertEqual(pf.shares, 100)   # 不重复买入


class TestBacktest(unittest.TestCase):
    def test_only_relative_iv_rule_exists(self):
        """固定绝对阈值已从配置与代码中移除（用户最终定稿）。"""
        self.assertFalse(hasattr(C.CC, "iv_threshold"))
        self.assertFalse(hasattr(C.CC, "iv_relative_percentile"))
        self.assertEqual(C.CC.iv_lookback_days, 252)
        self.assertEqual(C.CC.iv_percentile, 10.0)
        # iv_condition_ok 不再接受 mode 参数
        import inspect
        sig = inspect.signature(iv_condition_ok)
        self.assertNotIn("mode", sig.parameters)
        self.assertNotIn("fixed", sig.parameters)

    def test_relative_threshold_varies_over_time(self):
        """相对口径的门槛随滚动窗口变化，不是常数（区别于固定阈值的核心特征）。"""
        md = build_market_data()
        s = md.index_of(date.fromisoformat(C.CFG.start))
        ths = [iv_condition_ok(md, t)[2]
               for t in range(s + 60, len(md.dates), 20)]
        ths = [x for x in ths if np.isfinite(x)]
        self.assertGreater(len(set(round(x, 2) for x in ths)), 3)
        self.assertLess(min(ths), max(ths))

    def test_relative_c3_is_a_real_filter(self):
        """C3 在近5年成立但并非恒真。"""
        md = build_market_data()
        s = md.index_of(date.fromisoformat(C.CFG.start))
        n_ok = sum(1 for t in range(s, len(md.dates))
                   if iv_condition_ok(md, t)[0])
        self.assertGreater(n_ok, 0)
        self.assertLess(n_ok, len(md.dates) - s)

    def test_cc_mode_has_covered_calls(self):
        """相对口径下策略一仍应产生实际开仓。"""
        r = Backtester(strategies="cc_only").run()
        c1 = sum(1 for s in r.cc_signals if s.cond1_rally)
        c3 = sum(1 for s in r.cc_signals if s.cond3_iv_ok)
        self.assertGreater(c1, 0)
        self.assertGreater(c3, 0)
        self.assertLess(c3, len(r.cc_signals))
        self.assertGreater(sum(1 for s in r.cc_signals if s.triggered), 0)
        self.assertTrue(any(t["action"] == "SELL_CALL" for t in r.trades))

    def test_all_short_calls_covered(self):
        """每一笔 SELL_CALL 发生时，持股必须 >=100（绝不裸卖）。"""
        for st in ("cc_only", "both"):
            r = Backtester(strategies=st).run()
            sh_by_date = {d.isoformat(): int(s) for d, s in zip(r.dates, r.shares)}
            for t in r.trades:
                if t["action"] == "SELL_CALL":
                    self.assertGreaterEqual(sh_by_date[t["date"]], 100,
                                            f"{st} {t['date']} 裸卖!")

    def test_cc_only_never_trades_strategy_two(self):
        """仅策略一模式：不评估策略二、不买 call、无 long_call 持仓。"""
        r = Backtester(strategies="cc_only").run()
        self.assertEqual(r.strategies, "cc_only")
        self.assertEqual(len(r.lc_signals), 0)
        self.assertFalse(any(t["action"] == "BUY_CALL" for t in r.trades))
        self.assertFalse(any("LONG_CALL" in t["action"] for t in r.trades))

    def test_both_runs_both_strategies(self):
        r = Backtester(strategies="both").run()
        self.assertGreater(len(r.lc_signals), 0)
        self.assertTrue(any(t["action"] == "SELL_CALL" for t in r.trades))

    def test_invalid_strategies_rejected(self):
        with self.assertRaises(ValueError):
            Backtester(strategies="nonsense")

    def test_no_lookahead_candidate_count(self):
        """候选日数量与引擎实际调用 Agent 次数一致。"""
        r = Backtester(strategies="cc_only").run()
        cand = sum(1 for s in r.cc_signals if s.cond1_rally and s.cond3_iv_ok)
        self.assertEqual(cand, r.agent_calls)

    def test_equity_positive_and_consistent(self):
        for st in ("cc_only", "both"):
            r = Backtester(strategies=st).run()
            self.assertTrue(np.all(np.isfinite(r.equity)))
            self.assertTrue(np.all(r.equity > 0))
            self.assertEqual(len(r.equity), len(r.dates))

    def test_iv_condition_direction(self):
        """VXN 低于滚动分位门槛 → C3 成立；高于则不成立。"""
        md = build_market_data()
        t = md.index_of(date.fromisoformat(C.CFG.start)) + 300
        ok, v, th = iv_condition_ok(md, t)
        self.assertEqual(ok, v <= th)
        self.assertAlmostEqual(v, float(md.vxn[t]))
        self.assertTrue(np.isfinite(th))
        # 门槛是「低分位」，应低于该窗口的中位数
        hist = md.vxn[max(0, t - C.CC.iv_lookback_days):t]
        self.assertLessEqual(th, float(np.median(hist)))


class TestRollAndFundamentals(unittest.TestCase):
    """策略二：10 张 + Delta Roll + 基本面前置检查。"""

    def test_config_values(self):
        self.assertEqual(C.LC.contracts, 10)
        self.assertTrue(C.LC.roll_enabled)
        self.assertEqual(C.LC.roll_delta_trigger, 0.50)
        self.assertEqual(C.LC.roll_target_delta, 0.60)
        self.assertEqual(C.CFG.initial_cash, 200_000.0)
        self.assertEqual(C.CFG.strategies, "both")

    def test_fundamentals_bad_news_blocks(self):
        d = date(2024, 1, 5)

        class P:
            def fetch(self, dd, cfg):
                return [NewsItem(dd, "Megacap issues profit warning, cuts guidance")]
        v = NewsAgent(provider=P()).fundamentals_ok(d)
        self.assertFalse(v.ok)
        self.assertTrue(any("guidance" in x for x in v.bad_hits))

    def test_fundamentals_clean_and_no_news_ok(self):
        d = date(2024, 1, 5)

        class Clean:
            def fetch(self, dd, cfg):
                return [NewsItem(dd, "Tech shares edge higher in quiet session")]
        self.assertTrue(NewsAgent(provider=Clean()).fundamentals_ok(d).ok)

        class Empty:
            def fetch(self, dd, cfg):
                return []
        v = NewsAgent(provider=Empty()).fundamentals_ok(d)
        self.assertTrue(v.ok)
        self.assertEqual(v.source, "no-news-default-ok")

    def test_fundamentals_ignores_macro_human_action(self):
        """宏观人为事件（Fed/关税）不等于基本面恶化，不应拦截 Roll。"""
        d = date(2024, 1, 5)

        class P:
            def fetch(self, dd, cfg):
                return [NewsItem(dd, "Fed Chair Powell signals higher rates; tariff fears")]
        self.assertTrue(NewsAgent(provider=P()).fundamentals_ok(d).ok)

    def test_mark_delta_consistency(self):
        S, K, dte, vxn, r = 500.0, 520.0, 200, 25.0, 0.04
        from ..core.options import call_delta
        from .. import config as C
        sig = min(max(vxn / 100 * C.PR.iv_scale, C.PR.iv_floor), C.PR.iv_cap)
        expect = call_delta(S, K, dte / 365, r, C.PR.dividend_yield, sig)
        self.assertAlmostEqual(mark_delta(S, K, dte, vxn, r), expect, places=10)

    def test_both_mode_buys_10_contracts(self):
        r = Backtester(strategies="both").run()
        buys = [t for t in r.trades if t["action"] == "BUY_CALL"]
        self.assertGreater(len(buys), 0)
        self.assertTrue(all(t["contracts"] == 10 for t in buys))

    def test_roll_zero_in_relative_only_regime(self):
        """只保留相对口径后，回测期内 Delta 从未跌破 0.5 → 0 次 Roll 是正确行为。

        （历史对照：固定 VXN<30 口径在 2022 熊市会触发 4 次 Roll，
        但那套口径已按用户要求移除。）
        """
        r = Backtester(strategies="both").run()
        self.assertEqual(len(r.roll_events), 0)
        self.assertEqual(len(r.fundamental_checks), 0)
        self.assertFalse(any(t["action"] in ("ROLL_CLOSE", "ROLL_OPEN")
                             for t in r.trades))

    def test_roll_logic_still_works_on_synthetic_position(self):
        """用合成持仓直接验证 Roll 逻辑本身未被移除/改坏：
        Delta<0.5 + 现金充足 + 基本面 OK → 平旧开新，张数与期限保持不变。
        """
        from datetime import timedelta
        from ..engine.portfolio import Portfolio
        from ..agent.news_agent import NewsItem

        class CleanNews:
            def fetch(self, d, cfg):
                return [NewsItem(d, "Tech shares edge higher in quiet session")]

        bt = Backtester(strategies="both")
        bt.agent = NewsAgent(provider=CleanNews())
        md = bt.md
        t = md.index_of(date(2022, 8, 1))
        d = md.dates[t]
        S, vxn, r = float(md.close[t]), float(md.vxn[t]), float(md.rf[t])

        # 构造一个深虚值（Delta 远低于 0.5）的 10 张 long call
        old = Position(kind="long_call", strike=round(S * 1.4),
                       entry_premium=5.0, dte=200, open_date=d - timedelta(days=100),
                       expiry_date=d + timedelta(days=200), contracts=10,
                       strategy="0.6")
        pf = Portfolio(cash=500_000.0, long_calls=[old])
        res = Backtester.__new__(Backtester)      # 只借 _maybe_roll 方法
        # 手动补齐 _maybe_roll 需要的属性
        md2 = md
        res.lc = C.LC; res.run_lc = True; res.agent = bt.agent
        from ..engine.backtest import BacktestResult
        r_dummy = BacktestResult(config_note="", strategies="both",
                                 start=d, end=d)
        res._maybe_roll(pf, md2, t, r_dummy)

        self.assertEqual(len(r_dummy.roll_events), 1, "应触发一次 Roll")
        e = r_dummy.roll_events[0]
        self.assertEqual(e["contracts"], 10)
        self.assertEqual(e["new_dte"], 365)
        self.assertLess(e["old_delta"], C.LC.roll_delta_trigger)
        self.assertTrue(e["fundamental_ok"])
        opens = [x for x in pf.long_calls]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0].contracts, 10)
        self.assertEqual(opens[0].dte, 365)
        self.assertEqual(len([x for x in pf.trades if x["action"] == "ROLL_CLOSE"]), 1)
        self.assertEqual(len([x for x in pf.trades if x["action"] == "ROLL_OPEN"]), 1)

    def test_roll_blocked_by_bad_fundamentals(self):
        """合成场景：基本面负面新闻 → 即使 Delta<0.5 也不 Roll。"""
        from datetime import timedelta
        from ..engine.portfolio import Portfolio
        from ..agent.news_agent import NewsItem

        class BadNews:
            def fetch(self, d, cfg):
                return [NewsItem(d, "Megacap issues profit warning, cuts guidance")]

        bt = Backtester(strategies="both")
        md = bt.md
        t = md.index_of(date(2022, 8, 1))
        d = md.dates[t]
        S = float(md.close[t])
        old = Position(kind="long_call", strike=round(S * 1.4),
                       entry_premium=5.0, dte=200, open_date=d - timedelta(days=100),
                       expiry_date=d + timedelta(days=200), contracts=10,
                       strategy="0.6")
        pf = Portfolio(cash=500_000.0, long_calls=[old])
        res = Backtester.__new__(Backtester)
        res.lc = C.LC; res.run_lc = True; res.agent = NewsAgent(provider=BadNews())
        from ..engine.backtest import BacktestResult
        r_dummy = BacktestResult(config_note="", strategies="both", start=d, end=d)
        res._maybe_roll(pf, md, t, r_dummy)
        self.assertEqual(len(r_dummy.roll_events), 0)
        self.assertEqual(r_dummy.fundamental_checks[0]["decision"],
                         "SKIP_BAD_FUNDAMENTALS")
        self.assertFalse(r_dummy.fundamental_checks[0]["ok"])

    def test_roll_trades_are_paired_if_any(self):
        """若发生 Roll，ROLL_CLOSE 与 ROLL_OPEN 必须成对。"""
        r = Backtester(strategies="both").run()
        n_close = sum(1 for t in r.trades if t["action"] == "ROLL_CLOSE")
        n_open = sum(1 for t in r.trades if t["action"] == "ROLL_OPEN")
        self.assertEqual(n_close, n_open)
        self.assertEqual(n_open, len(r.roll_events))

    def test_cc_only_still_has_no_rolls(self):
        r = Backtester(strategies="cc_only").run()
        self.assertEqual(len(r.roll_events), 0)
        self.assertEqual(len(r.fundamental_checks), 0)
        self.assertFalse(any(t["action"] == "ROLL_OPEN" for t in r.trades))


if __name__ == "__main__":
    unittest.main(verbosity=2)
