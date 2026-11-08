"""投资组合：股票、期权空头/多头持仓、现金、逐日盯市。

所有金额单位：美元。期权价格按「每股」存储，合约价值 = price*100*contracts。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from .. import config as C
from ..core.options import mark_call


@dataclass
class Position:
    """一张（或多张）同规格期权持仓。"""
    kind: Literal["short_call", "long_call"]
    strike: float
    entry_premium: float          # 每股，开仓时的理论价（未含滑点）
    dte: int                      # 开仓时期限（自然日）
    open_date: date
    expiry_date: date
    contracts: int = 1
    multiplier: int = 100
    entry_price_paid: float = 0.0  # 实际现金流（每股，含滑点/手续费后）
    strategy: str = ""
    roll_cash_blocked: bool = False   # 已记过「现金不足」日志，避免重复噪音

    def intrinsic(self, S: float) -> float:
        return max(0.0, S - self.strike)

    def mark(self, S: float, dte: int, vxn: float, r: float) -> float:
        """返回每股盯市价。"""
        if dte <= 0:
            return self.intrinsic(S)
        return mark_call(S, self.strike, dte, vxn, r)

    def value(self, S: float, dte: int, vxn: float, r: float) -> float:
        """合约市值（多头为正、空头为负债取负）。"""
        px = self.mark(S, dte, vxn, r)
        v = px * self.multiplier * self.contracts
        return v if self.kind == "long_call" else -v


@dataclass
class Portfolio:
    cash: float
    shares: int = 0
    short_calls: list[Position] = field(default_factory=list)
    long_calls: list[Position] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------ 股票
    def buy_shares(self, n: int, price: float, d: date, tag: str = "") -> float:
        cost = n * price
        fee = cost * C.PR.stock_fee_rate
        self.cash -= cost + fee
        self.shares += n
        self._log(d, "BUY_STOCK", -cost - fee,
                  f"买入 {n} 股 QQQ @ {price:.2f} 手续费 {fee:.2f} {tag}")
        return cost + fee

    def sell_shares(self, n: int, price: float, d: date, tag: str = "") -> float:
        proceeds = n * price
        fee = proceeds * C.PR.stock_fee_rate
        self.cash += proceeds - fee
        self.shares -= n
        self._log(d, "SELL_STOCK", proceeds - fee,
                  f"卖出 {n} 股 QQQ @ {price:.2f} 手续费 {fee:.2f} {tag}")
        return proceeds - fee

    # ------------------------------------------------------------ 备兑硬约束
    def ensure_covered(self, contracts: int, S: float, d: date) -> None:
        """【绝不裸卖】开空 call 前必须持有足额股票，不足先补。

        若现金不足以补足股票，则直接拒绝（由调用方捕获）。
        """
        need = contracts * C.CC.shares_per_contract
        if self.shares >= need:
            return
        missing = need - self.shares
        cost = missing * S * (1 + C.PR.stock_fee_rate)
        if cost > self.cash:
            raise RuntimeError(
                f"备兑约束不满足且现金不足：需补 {missing} 股约 {cost:.2f}，"
                f"仅有现金 {self.cash:.2f}。拒绝裸卖。"
            )
        self.buy_shares(missing, S, d, tag="[为备兑补仓]")
        assert self.shares >= need, "备兑约束未满足"

    # ------------------------------------------------------------ 期权开仓
    def open_short_call(self, pos: Position, premium_net: float, d: date) -> None:
        """卖出 call：收到权利金（扣滑点、手续费）。"""
        gross = pos.entry_premium * pos.multiplier * pos.contracts
        slip = gross * C.PR.slippage
        fee = C.PR.commission_per_contract * pos.contracts
        net = gross - slip - fee
        pos.entry_price_paid = net / (pos.multiplier * pos.contracts)
        self.cash += net
        self.short_calls.append(pos)
        self._log(d, "SELL_CALL", net,
                  f"卖出 {pos.contracts} 张 Δ≈{pos.strategy} call K={pos.strike:.0f} "
                  f"{pos.dte}天到期 权利金 {gross:.2f}-滑点{slip:.2f}-手续费{fee:.2f}"
                  f"=净收 {net:.2f}｜持股 {self.shares} 股(备兑)")
        self._trade(d, "SELL_CALL", pos, net, "open")

    def open_long_call(self, pos: Position, d: date, action: str = "BUY_CALL") -> None:
        """买入 call：支付权利金（加滑点、手续费）。action 可覆盖用于 Roll 开仓。"""
        gross = pos.entry_premium * pos.multiplier * pos.contracts
        slip = gross * C.PR.slippage
        fee = C.PR.commission_per_contract * pos.contracts
        cost = gross + slip + fee
        if cost > self.cash:
            raise RuntimeError(f"现金不足，无法买入 call：需 {cost:.2f}，有 {self.cash:.2f}")
        pos.entry_price_paid = cost / (pos.multiplier * pos.contracts)
        self.cash -= cost
        self.long_calls.append(pos)
        self._log(d, action, -cost,
                  f"买入 {pos.contracts} 张 Δ≈{pos.strategy} call K={pos.strike:.0f} "
                  f"{pos.dte}天到期 权利金 {gross:.2f}+滑点{slip:.2f}+手续费{fee:.2f}"
                  f"=净付 {cost:.2f}")
        self._trade(d, action, pos, -cost, "open")

    # ------------------------------------------------------------ 收盘/到期
    def close_short_call(self, pos: Position, S: float, d: date,
                         reason: str = "EXPIRY") -> None:
        """空 call 了结：到期若价内则被行权（按 K 卖出 100 股），否则归零。"""
        itm = S > pos.strike
        if itm and reason == "EXPIRY":
            # 被行权：以 K 卖出 100 股/张
            n = pos.contracts * C.CC.shares_per_contract
            n = min(n, self.shares)
            proceeds = n * pos.strike
            self.cash += proceeds
            self.shares -= n
            pnl = pos.entry_price_paid * pos.multiplier * pos.contracts
            self._log(d, "ASSIGNED", proceeds,
                      f"call 被行权：以 K={pos.strike:.0f} 卖出 {n} 股，"
                      f"收到 {proceeds:.2f}；权利金已入账 {pnl:.2f} {reason}")
            self._trade(d, "ASSIGNED", pos, proceeds, "close", note="被行权交割股票")
        else:
            self._log(d, "SHORT_CALL_EXPIRE", 0.0,
                      f"short call K={pos.strike:.0f} 价外到期归零，"
                      f"保留全部权利金 {reason}")
            self._trade(d, "SHORT_CALL_EXPIRE", pos, 0.0, "close", note="价外到期")
        self.short_calls.remove(pos)

    def close_long_call(self, pos: Position, S: float, vxn: float,
                        r: float, dte: int, d: date, reason: str = "EXPIRY",
                        action: str = "LONG_CALL_CLOSE") -> None:
        """多 call 了结：按内在价值（到期）或盯市价（提前平仓）结算。"""
        px = pos.mark(S, dte, vxn, r)
        proceeds = px * pos.multiplier * pos.contracts
        fee = C.PR.commission_per_contract * pos.contracts
        self.cash += proceeds - fee
        self._log(d, action, proceeds - fee,
                  f"long call K={pos.strike:.0f} 平仓 @ {px:.2f} "
                  f"收到 {proceeds-fee:.2f} {reason}")
        self._trade(d, action, pos, proceeds - fee, "close", note=reason)
        self.long_calls.remove(pos)

    # ------------------------------------------------------------ Roll
    def roll_long_call(self, old: Position, new: Position, S: float,
                       vxn: float, r: float, dte: int, d: date,
                       old_delta: float, fundamental_note: str = "") -> None:
        """把 long call 批次 Roll 到新的 Delta=0.6 / 365 天合约。

        顺序：先平旧仓（收到盯市价）→ 再开新仓（支付新权利金）。
        张数保持 new.contracts == old.contracts，由调用方保证。
        """
        self.close_long_call(old, S, vxn, r, dte, d,
                             reason=f"Roll 平旧仓（Delta={old_delta:.3f}）",
                             action="ROLL_CLOSE")
        self.open_long_call(new, d, action="ROLL_OPEN")
        new_delta = self._new_delta(new, S, vxn, r)
        self._log(d, "ROLL", 0.0,
                  f"Roll 完成：K {old.strike:.0f} → {new.strike:.0f}，"
                  f"{new.contracts} 张，Delta {old_delta:.3f} → {new_delta:.3f}，"
                  f"期限重置为 {new.dte} 天 {fundamental_note}")

    @staticmethod
    def _new_delta(pos: Position, S: float, vxn: float, r: float) -> float:
        from ..core.options import mark_delta
        return mark_delta(S, pos.strike, pos.dte, vxn, r)

    # ------------------------------------------------------------ 权益
    def equity(self, S: float, md, idx: int) -> float:
        """现金 + 股票市值 + 期权盯市市值（空头为负）。"""
        v = self.cash + self.shares * S
        r = float(md.rf[idx])
        vxn = float(md.vxn[idx])
        for p in self.short_calls + self.long_calls:
            dte = (p.expiry_date - md.dates[idx]).days
            v += p.value(S, dte, vxn, r)
        return v

    # ------------------------------------------------------------ 日志
    def _log(self, d: date, kind: str, cash_flow: float, msg: str) -> None:
        self.events.append({"date": d.isoformat(), "kind": kind,
                            "cash_flow": cash_flow, "msg": msg})

    def _trade(self, d: date, action: str, pos: Position, cash_flow: float,
               side: str, note: str = "") -> None:
        self.trades.append({
            "date": d.isoformat(), "action": action, "side": side,
            "kind": pos.kind, "strike": pos.strike, "dte": pos.dte,
            "contracts": pos.contracts, "premium_per_share": pos.entry_premium,
            "cash_flow": cash_flow, "strategy": pos.strategy, "note": note,
        })
