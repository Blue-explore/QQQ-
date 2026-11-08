"""全局配置：所有策略参数集中在此，便于复现与调参。"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent           # 工作区根
PKG = Path(__file__).resolve().parent                  # qqq_options/
DATA_DIR = PKG / "data"
REPORT_DIR = PKG / "report" / "output"
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, REPORT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

QQQ_RAW = DATA_DIR / "qqq_nasdaq_raw.json"
VXN_RAW = DATA_DIR / "vxn_cboe_raw.csv"
NDX_RAW = DATA_DIR / "nasdaq100_fred_raw.csv"
RATE_RAW = DATA_DIR / "dgs3mo_fred_raw.csv"
NEWS_CACHE = CACHE_DIR / "gdelt_news.json"


# ---------------------------------------------------------------- 回测区间
@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2021-11-08"      # 回测起点（近5年）
    end: str = "2026-11-06"        # 回测终点（数据最后一天）
    # 【用户修订】策略二改为买 10 张一年期 call（约 $35k~$80k），
    # 加上 100 股备兑底仓，原 $100k 在 Roll 后会现金不足；
    # 提高到 $200k 以便 Roll 机制能充分运作。
    initial_cash: float = 200_000.0
    # 预热窗口：5年历史用于计算「5日涨幅分位门槛」，
    # 因此回测开始前必须已有 >= 5 年的 QQQ 数据。
    lookback_years: int = 5
    trading_days_per_year: int = 252
    # 运行哪些策略："both" | "cc_only"（只执行策略一）| "lc_only"
    # 【用户修订】本次模拟「策略一 + 策略二一起」，默认改为 both。
    strategies: str = "both"


# ---------------------------------------------------------------- 策略一：Covered Call
@dataclass(frozen=True)
class CoveredCallConfig:
    # 条件1：5日涨幅分位
    rally_window: int = 5                 # 「近5天」= 收盘价 / 5个交易日前 - 1
    pct_threshold: float = 0.10          # 前 10%（1260个数里第126个）
    lookback_days: int = 1260            # 5年 ≈ 1260 个交易日
    # 条件2：由新闻 Agent 给出 has_resistance 布尔值
    # 条件3：IV 指数 —— 【用户最终定稿】只保留「相对口径」：
    #   VXN <= 过去 iv_lookback_days 个交易日 VXN 的 iv_percentile 分位
    #   （即「波动率处于近一年最低的 10% 才算低」），固定绝对阈值已移除。
    iv_lookback_days: int = 252
    iv_percentile: float = 10.0
    # 合约
    target_delta: float = 0.20
    dte_days: int = 7                    # 约一周
    dte_tolerance: float = 2.0           # 允许 5~9 天
    contracts: int = 1
    multiplier: int = 100
    # 备兑：每卖出 1 张 call 必须对应持有 100 股现货
    shares_per_contract: int = 100
    # 同一时间最多允许持有的空 call 张数（防止重复加仓）
    max_open_short_calls: int = 1
    # 若被行权，重新补回股票的成本按当日收盘价计（现金扣减）


# ---------------------------------------------------------------- 策略二：Long Call
@dataclass(frozen=True)
class LongCallConfig:
    drop_threshold: float = -0.01        # 单日跌幅超过 1%
    target_delta: float = 0.60
    dte_days: int = 365                  # 约一年
    # 【用户修订】每次信号触发买入 10 张（原为 1 张）
    contracts: int = 10
    multiplier: int = 100
    max_open_long_calls: int = 1         # 同时最多 1 个「批次」（1 批 = 10 张）
    # 牛市条件A：策略一近3个月（63个交易日）触发次数
    bull_lookback_days: int = 63
    bull_trigger_min: int = 3            # >= 3 次
    # 牛市条件B：近一年最大回撤不超过 10%
    max_dd_lookback_days: int = 252
    max_dd_limit: float = 0.10

    # ---- Roll 规则（用户修订）----
    # 当持仓 call 的 Delta 掉到 roll_delta_trigger 以下、且「基本面没问题」时，
    # 平掉旧仓并重新买入 365 天、Delta=roll_target_delta 的 call，张数保持 contracts。
    roll_enabled: bool = True
    roll_delta_trigger: float = 0.50     # Delta < 0.5 触发
    roll_target_delta: float = 0.60      # 滚到 Delta 0.6
    roll_dte_days: int = 365             # 新合约重新给一年期
    # 临近到期不滚（避免反复换月产生过多滑点）
    roll_min_dte: int = 30
    # 每次 Roll 最多检查一次/日；同一天不重复滚
    roll_cooldown_days: int = 1


# ---------------------------------------------------------------- 期权定价
@dataclass(frozen=True)
class PricingConfig:
    # 定价使用的 IV 源：VXN 是「指数」的隐含波动率，
    # 对个股/ETF 期权需要乘一个经验系数（QQQ 通常低于 VXN 一点）。
    iv_scale: float = 0.85
    # 波动率下限/上限，避免极端数据导致价格失真
    iv_floor: float = 0.05
    iv_cap: float = 2.00
    # 买卖价差（单边，按权利金比例），让回测更贴近真实
    slippage: float = 0.02
    # 每张合约手续费（美元）
    commission_per_contract: float = 1.0
    # 股票买卖手续费（按金额比例）
    stock_fee_rate: float = 0.0005
    risk_free_source: str = "DGS3MO"     # FRED 3个月国债利率
    default_risk_free: float = 0.03
    dividend_yield: float = 0.006        # QQQ 股息率约 0.6%


# ---------------------------------------------------------------- 新闻 Agent
@dataclass(frozen=True)
class NewsAgentConfig:
    # GDELT DOC 2.0 查询窗口（交易日前后各 N 天）
    lookahead_days: int = 0
    lookback_days: int = 3
    max_records: int = 50
    # 离线模式：使用内置「事件日历 + 技术位」规则生成判定，
    # 不依赖网络/LLM，保证回测可复现。
    offline: bool = True
    # LLM 后端（可选）："openai" / "anthropic" / None
    llm_backend: str | None = None
    llm_model: str = "gpt-4o-mini"
    # 作为「人为动作」过滤的关键词（命中即视为人为，不算技术阻力）
    human_action_keywords: tuple = (
        "fed", "fomc", "powell", "rate cut", "rate hike", "interest rate",
        "tariff", "trade war", "sanction", "executive order", "trump",
        "election", "congress", "government shutdown", "stimulus",
        "geopolit", "war ", "invasion", "ceasefire", "opec",
        "earnings", "guidance", "downgrade", "upgrade", "analyst",
        "merger", "acquisition", "buyback", "sec ", "lawsuit", "antitrust",
        "jobs report", "cpi", "inflation report", "nonfarm", "ppi",
        "treasury", "yield", "bond", "dollar", "currency", "intervention",
        "crypto", "bitcoin", "hack", "fraud", "recall",
    )
    # 【Roll 用】「基本面有问题」的关键词。
    # 命中即视为基本面恶化 → 不 Roll。用现有 NewsAgent 兜底判定，无需额外数据源。
    fundamental_bad_keywords: tuple = (
        "earnings miss", "profit warning", "guidance cut", "cuts guidance",
        "lowers guidance", "downgrade", "analyst downgrade", "revenue miss",
        "loss warning", "bankruptcy", "insolvency", "fraud", "accounting",
        "sec investigation", "sec probe", "lawsuit", "antitrust",
        "recall", "data breach", "hack", "ceo resign", "cfo resign",
        "layoffs", "job cuts", "restructuring", "write-down", "writedown",
        "impairment", "default", "credit downgrade", "debt downgrade",
    )
    # 作为「技术阻力」证据的关键词
    resistance_keywords: tuple = (
        "resistance", "overbought", "rsi", "technical", "chart",
        "moving average", "50-day", "200-day", "trendline", "ceiling",
        "profit-taking", "profit taking", "pullback", "consolidat",
        "stall", "pause", "reversal", "double top", "head and shoulders",
        "fibonacci", "support level", "barrier", "capped", "struggle",
        "failed to break", "rejected at", "resistance level",
    )


CFG = BacktestConfig()
CC = CoveredCallConfig()
LC = LongCallConfig()
PR = PricingConfig()
NA = NewsAgentConfig()
