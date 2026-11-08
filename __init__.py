"""
QQQ 期权双策略系统 + 新闻 Agent

策略一：Covered Call（备兑看涨，绝不裸卖）
    同时满足三条件才卖出一张约一周、Delta=0.2 的 call：
      1) 股价刚冲完一波：近5日涨幅进入「过去5年所有5日涨幅」的前10%
      2) 有短期技术阻力：由新闻 Agent 判定（人为动作引起的波动不算）
      3) IV 够低：VXN <= 过去 252 个交易日 VXN 的 10 分位
         ← 用户最终定稿：只保留相对口径，固定绝对阈值已移除

策略二：买入 Call（趋势回踩）
    同时满足两条件才买入 10 张 Delta=0.6、约一年的 call（用户三次修订，原 1 张）：
      1) 牛市趋势：策略一近3个月触发>=3次 或 近一年最大回撤<=10%
      2) 单日跌幅 > 1%
    Roll：持仓 Delta < 0.5、剩余期限 >= 30 天且「基本面没问题」时，
          平旧仓并重新买入 10 张 Delta=0.6 / 365 天 call。

运行模式：config.BacktestConfig.strategies = "both"（默认，两策略一起）
          / "cc_only"（只跑策略一）/ "lc_only"
初始资金 $200,000（10 张一年期 call + 100 股备兑底仓同时占资）。

本包只依赖 Python 标准库 + numpy/pandas（Harness 自带），
Black-Scholes 用 math.erf 实现正态 CDF，Delta 反解用二分法，无需 scipy。
"""

__version__ = "1.4.0"
