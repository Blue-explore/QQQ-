"""新闻分析 Agent：判定「短期技术阻力」并排除「人为动作引起的市场变动」。

职责（策略一条件2）：
    给定交易日 t，读取 t 前后若干天与 QQQ / Nasdaq-100 相关的新闻，
    判断市场上方是否存在「技术性阻力」（前期高点、均线压制、超买、
    获利了结、双顶等），且**该阻力不是由人为动作（美联储、关税、
    财报指引、地缘政治、监管、干预等）造成的**。

    输出 NewsVerdict：
        has_resistance: bool
        human_caused  : bool   # 若为 True，即使有压制也不算数
        verdict       : "TECHNICAL_RESISTANCE" | "HUMAN_ACTION" | "NO_RESISTANCE"
        evidence      : list[str]
        reasoning     : str

设计要点
--------
1. **可复现优先**：回测默认 offline=True。使用「新闻源(GDELT 缓存) + 规则化
   classifier」；对没有缓存的日期，使用基于价格结构的技术阻力代理
   （前期高点/密集成交区/超买），并显式标注 source。
2. **可接真实 LLM**：llm_backend 支持 openai / anthropic / http，传入
   api_key 或自定义 client 即可。LLM 的 prompt 固化在 SYSTEM_PROMPT 里，
   输出强制 JSON。
3. **人为动作过滤是硬约束**：只要新闻命中 human_action_keywords 且
   被判定为该轮波动主因，直接返回 HUMAN_ACTION（不算技术阻力）。
   这正是用户要求的「如果是人为的动作引起市场的变动不算」。
4. **绝不前视**：只使用 t 及 t 之前发布的新闻（seendate <= t 收盘）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

from .. import config as C


# ------------------------------------------------------------------ 数据结构
@dataclass
class NewsItem:
    date: date
    title: str
    domain: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(), "title": self.title,
                "domain": self.domain, "url": self.url}


@dataclass
class NewsVerdict:
    as_of: date
    has_resistance: bool
    human_caused: bool
    verdict: str
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""
    source: str = "offline-rule"      # gdelt-rule / gdelt-llm / price-structure
    headlines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


@dataclass
class FundamentalVerdict:
    """Roll 前置检查：「基本面没问题」的判定结果。"""
    as_of: date
    ok: bool                  # True = 基本面没问题，允许 Roll
    n_headlines: int          # 检查了多少条新闻
    bad_hits: list[str] = field(default_factory=list)   # 命中的负面词
    bad_titles: list[str] = field(default_factory=list)
    source: str = "gdelt-rule"        # 无新闻时为 no-news-default-ok

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


# ------------------------------------------------------------------ 新闻源
class NewsProvider(Protocol):
    def fetch(self, d: date, cfg: NewsAgentConfig) -> list[NewsItem]: ...


class CachedGDELTProvider:
    """优先读本地 GDELT 缓存；无缓存时返回空列表（由 Agent 回退到价格结构）。

    缓存格式：{ "YYYY-MM-DD": [ {date,title,domain,url}, ... ] }
    抓取脚本见 fetch_news.py，可离线预抓，也可在线按需抓（受 GDELT 5s 限流）。
    """

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or C.NEWS_CACHE
        self._data: dict[str, list[dict]] = {}
        if self.cache_path.exists():
            try:
                self._data = json.loads(self.cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def fetch(self, d: date, cfg: NewsAgentConfig) -> list[NewsItem]:
        items: list[NewsItem] = []
        lo = d - timedelta(days=cfg.lookback_days)
        hi = d + timedelta(days=cfg.lookahead_days)
        for k in self._data:
            try:
                kd = date.fromisoformat(k)
            except ValueError:
                continue
            if lo <= kd <= hi:
                for r in self._data[k]:
                    try:
                        items.append(NewsItem(
                            date=date.fromisoformat(r["date"]),
                            title=r.get("title", ""),
                            domain=r.get("domain", ""),
                            url=r.get("url", ""),
                        ))
                    except (KeyError, ValueError):
                        continue
        # 只用 as_of 当天及之前的新闻（杜绝前视）
        items = [x for x in items if x.date <= d]
        items.sort(key=lambda x: x.date, reverse=True)
        return items[: cfg.max_records]


# ------------------------------------------------------------------ 分类器后端
class Classifier(Protocol):
    def classify(self, headlines: list[NewsItem], as_of: date,
                 ctx: dict[str, Any]) -> tuple[bool, bool, list[str], str]:
        """返回 (has_technical_resistance, human_caused, evidence, reasoning)"""


SYSTEM_PROMPT = """你是量化交易团队的「市场结构分析 Agent」。
任务：判断给定日期的 QQQ / 纳斯达克100 是否存在【短期技术阻力】，并排除人为动作。

判定标准（必须同时遵守）：
A. 「技术阻力」指由市场自身结构造成的上方压制，例如：
   前期高点/前高附近遇阻、50日或200日均线压制、RSI 超买、
   趋势线/斐波那契阻力、双顶/头肩顶、获利了结、涨后盘整、
   放量滞涨、突破失败被拒回等。
B. 【人为动作不算】若这一轮市场变动主要由以下因素驱动，
   必须判定为 HUMAN_ACTION，不得算作技术阻力：
   美联储/FOMC/利率决议/鲍威尔讲话、CPI/非农等数据发布、
   关税/贸易战/制裁/行政令、大选/国会/政府停摆、财政刺激、
   地缘冲突/战争、OPEC、公司财报与指引、分析师上调下调、
   并购/回购/监管诉讼/反垄断、汇市或债市干预、加密事件等。
   即便新闻里同时出现「阻力」「超买」等字眼，只要主因是人为动作，
   就返回 HUMAN_ACTION。
C. 若既无技术阻力证据、也无明确人为事件，返回 NO_RESISTANCE。

仅输出 JSON，不要任何解释文字：
{"verdict": "TECHNICAL_RESISTANCE" | "HUMAN_ACTION" | "NO_RESISTANCE",
 "reasoning": "一句话中文理由",
 "evidence": ["引用的标题片段1", "引用的标题片段2"]}
"""


class RuleClassifier:
    """确定性规则分类器（离线可复现，回测默认使用）。

    判定「这一轮市场变动的主因」而不是「任意一条新闻里出现过某个词」：
      1. 只取与 as_of 最接近的 TOP_N 条头条（避免一周 250 条里必然出现的
         "fed"/"analyst" 之类的背景词造成误杀）。
      2. 统计「人为动作」与「技术阻力」两类证据的加权得分。
      3. 人为动作得分占主导 -> HUMAN_ACTION（按用户要求不算技术阻力）。
      4. 技术阻力得分占主导 -> TECHNICAL_RESISTANCE。
      5. 两边都无明确信号 -> 回退到价格结构代理。
    """

    TOP_N = 12

    def __init__(self, cfg: NewsAgentConfig | None = None):
        self.cfg = cfg or C.NA

    @staticmethod
    def _hits(text: str, kws: tuple[str, ...]) -> list[str]:
        t = text.lower()
        return [k for k in kws if k in t]

    def classify(self, headlines: list[NewsItem], as_of: date,
                 ctx: dict[str, Any]) -> tuple[bool, bool, list[str], str]:
        # 只考察最相关的头部新闻（已按日期降序），且严格 <= as_of
        top = [h for h in headlines if h.date <= as_of][: self.TOP_N]

        human_hits: list[str] = []
        tech_hits: list[str] = []
        human_titles: list[str] = []
        tech_titles: list[str] = []

        for it in top:
            t = it.title
            h = self._hits(t, self.cfg.human_action_keywords)
            r = self._hits(t, self.cfg.resistance_keywords)
            if h:
                human_hits.extend(h); human_titles.append(t)
            if r:
                tech_hits.extend(r); tech_titles.append(t)

        # 去重后的「独立证据条数」比关键词命中次数更能代表主因
        n_human = len(set(human_titles))
        n_tech = len(set(tech_titles))

        # 硬约束：人为动作是本轮变动主因 -> 不算技术阻力
        if n_human > 0 and n_human >= n_tech:
            ev = list(dict.fromkeys(human_titles))[:3]
            return (False, True, ev,
                    f"头部 {len(top)} 条新闻中 {n_human} 条指向人为/事件驱动"
                    f"（{', '.join(sorted(set(human_hits))[:4])}），"
                    f"技术证据仅 {n_tech} 条；按要求不计为技术阻力。")

        if n_tech > 0:
            ev = list(dict.fromkeys(tech_titles))[:3]
            return (True, False, ev,
                    f"头部新闻出现技术阻力证据（{', '.join(sorted(set(tech_hits))[:4])}），"
                    f"且人为动作未占主导（{n_human} vs {n_tech}）。")

        # 新闻无明确信号 -> 价格结构代理（不含未来数据）
        return self._price_structure(ctx)

    def _price_structure(self, ctx: dict[str, Any]) -> tuple[bool, bool, list[str], str]:
        """用截至 as_of 的价格结构判断短期阻力（纯历史数据，无前视）。

        语义：「刚冲完一波之后，上方是否有技术压制」。命中 >=1 即成立：
          1) 50 日均线压制：现价低于 50 日均线（反弹但仍被均线盖住）
          2) 前期套牢区：现价距 60/252 日高点 3%~20%（上方有前期密集成交区）
          3) 超买：RSI(14) > 70
          4) 滞涨：近 5 日大涨(>=4%)但最近 3 日涨幅 < 1%（冲高后停顿）

        说明：本策略的候选日往往出现在「暴跌后反弹」（VXN 高的时期），
        因此「贴近前高」不是合理判据；上方均线/套牢区才是真实阻力。
        """
        closes: list[float] = ctx.get("recent_closes", [])
        if len(closes) < 21:
            return (False, False, [], "历史数据不足，无法判定技术阻力。")
        c = closes[-1]
        ev, reasons = [], []

        ma50 = sum(closes[-50:]) / min(50, len(closes))
        if c < ma50:
            ev.append(f"现价 {c:.2f} 仍低于 50 日均线 {ma50:.2f}"
                      f"（{(c/ma50-1)*100:.2f}%），均线构成上方压制")
            reasons.append("50日均线压制")

        hi60 = max(closes[-60:]) if len(closes) >= 60 else max(closes)
        dist = hi60 / c - 1
        if 0.03 <= dist <= 0.20:
            ev.append(f"现价距 60 日高点 {hi60:.2f} 尚有 {dist*100:.1f}%，"
                      "上方存在前期密集成交/套牢区阻力")
            reasons.append("前期套牢区")

        rsi = _rsi(closes, 14)
        if rsi is not None and rsi > 70:
            ev.append(f"RSI(14)={rsi:.1f} 处于超买区")
            reasons.append("超买")

        r5 = c / closes[-6] - 1 if len(closes) > 6 else 0.0
        r3 = c / closes[-4] - 1 if len(closes) > 4 else 0.0
        if r5 >= 0.04 and r3 < 0.01:
            ev.append(f"5日冲高 {r5*100:.1f}% 但近3日仅 {r3*100:.1f}%，涨势停顿/滞涨")
            reasons.append("冲高滞涨")

        if reasons:
            return (True, False, ev[:3],
                    "新闻无明确信号，改用价格结构代理判定：" + "、".join(reasons) + "。")
        return (False, False, ev[:3],
                "新闻与价格结构均未显示短期技术阻力。")


def _rsi(closes: list[float], n: int = 14) -> float | None:
    if len(closes) < n + 1:
        return None
    gains, losses = [], []
    for i in range(-n, 0):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0)); losses.append(max(-ch, 0.0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


# ------------------------------------------------------------------ LLM 分类器
class LLMClassifier:
    """可选：调用真实 LLM。需要 api_key；未配置时自动回退 RuleClassifier。"""

    def __init__(self, cfg: NewsAgentConfig | None = None,
                 api_key: str | None = None, client: Any = None):
        self.cfg = cfg or C.NA
        self.api_key = api_key
        self.client = client
        self.fallback = RuleClassifier(cfg)

    def classify(self, headlines: list[NewsItem], as_of: date,
                 ctx: dict[str, Any]) -> tuple[bool, bool, list[str], str]:
        if not self.api_key or not headlines:
            return self.fallback.classify(headlines, as_of, ctx)
        try:
            import json as _json
            payload = {
                "as_of": as_of.isoformat(),
                "price_context": {
                    "close": ctx.get("close"),
                    "r5": ctx.get("r5"),
                    "vxn": ctx.get("vxn"),
                },
                "headlines": [h.to_dict() for h in headlines],
            }
            raw = self._call_llm(payload)
            data = _json.loads(raw)
            v = data.get("verdict", "NO_RESISTANCE")
            if v == "TECHNICAL_RESISTANCE":
                return (True, False, data.get("evidence", []), data.get("reasoning", ""))
            if v == "HUMAN_ACTION":
                return (False, True, data.get("evidence", []), data.get("reasoning", ""))
            return (False, False, data.get("evidence", []), data.get("reasoning", ""))
        except Exception as e:                       # noqa: BLE001
            res = self.fallback.classify(headlines, as_of, ctx)
            return (res[0], res[1], res[2], res[3] + f" [LLM失败回退: {e}]")

    def _call_llm(self, payload: dict) -> str:
        backend = self.cfg.llm_backend
        if backend == "openai":
            from openai import OpenAI
            cli = self.client or OpenAI(api_key=self.api_key)
            resp = cli.chat.completions.create(
                model=self.cfg.llm_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return resp.choices[0].message.content
        if backend == "anthropic":
            import anthropic
            cli = self.client or anthropic.Anthropic(api_key=self.api_key)
            resp = cli.messages.create(
                model=self.cfg.llm_model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                max_tokens=600, temperature=0,
            )
            return resp.content[0].text
        raise RuntimeError(f"unsupported llm_backend: {backend}")


# ------------------------------------------------------------------ Agent 主体
class NewsAgent:
    """对外的 Agent 门面。

    用法：
        agent = NewsAgent()
        verdict = agent.analyze(as_of=date(2024,1,5),
                                md=md, idx=t, r5=..., vxn=...)
        if verdict.has_resistance and not verdict.human_caused: ...
    """

    def __init__(self, cfg: NewsAgentConfig | None = None,
                 provider: NewsProvider | None = None,
                 classifier: Classifier | None = None,
                 api_key: str | None = None):
        self.cfg = cfg or C.NA
        self.provider = provider or CachedGDELTProvider()
        if classifier is not None:
            self.classifier = classifier
        elif self.cfg.llm_backend and api_key:
            self.classifier = LLMClassifier(self.cfg, api_key=api_key)
        else:
            self.classifier = RuleClassifier(self.cfg)
        self._cache: dict[date, NewsVerdict] = {}
        self._fund_cache: dict[date, FundamentalVerdict] = {}
        self.calls = 0

    def analyze(self, as_of: date, md=None, idx: int | None = None,
                r5: float | None = None, vxn: float | None = None) -> NewsVerdict:
        if as_of in self._cache:
            return self._cache[as_of]
        self.calls += 1

        headlines = self.provider.fetch(as_of, self.cfg)
        ctx: dict[str, Any] = {}
        if md is not None and idx is not None:
            lo = max(0, idx - 60)
            ctx = {
                "recent_closes": [float(x) for x in md.close[lo:idx + 1]],
                "close": float(md.close[idx]),
                "vxn": float(md.vxn[idx]),
                "r5": float(r5) if r5 is not None else None,
            }

        has_res, human, ev, reasoning = self.classifier.classify(headlines, as_of, ctx)
        if has_res and not human:
            verdict = "TECHNICAL_RESISTANCE"
        elif human:
            verdict = "HUMAN_ACTION"
        else:
            verdict = "NO_RESISTANCE"

        src = "gdelt-rule" if headlines else "price-structure"
        if isinstance(self.classifier, LLMClassifier) and self.cfg.llm_backend and headlines:
            src = "gdelt-llm"

        v = NewsVerdict(
            as_of=as_of,
            has_resistance=bool(has_res and not human),
            human_caused=bool(human),
            verdict=verdict,
            evidence=ev,
            reasoning=reasoning,
            source=src,
            headlines=[h.title for h in headlines[:8]],
        )
        self._cache[as_of] = v
        return v

    def fundamentals_ok(self, as_of: date) -> FundamentalVerdict:
        """Roll 前置检查：当日新闻里是否出现「基本面恶化」信号。

        判定口径（沿用现有新闻 Agent，零额外数据源）：
          * 取 as_of 及之前 lookback_days 内的新闻（只用 <= as_of，杜绝前视）
          * 命中 fundamental_bad_keywords（业绩暴雷/指引下调/评级下调/欺诈/
            诉讼/召回/裁员/减值/违约等）→ 视为基本面有问题，不 Roll
          * 无新闻或没有负面词 → 视为「基本面没问题」（默认放行）

        注意：这里刻意**不复用** C2 的 human_action_keywords，因为
        「人为动作」包含美联储/关税等宏观事件，不等于公司基本面恶化。
        """
        if as_of in self._fund_cache:
            return self._fund_cache[as_of]
        headlines = [h for h in self.provider.fetch(as_of, self.cfg)
                     if h.date <= as_of][: self.cfg.max_records]
        bad_hits: list[str] = []
        bad_titles: list[str] = []
        for it in headlines:
            t = it.title.lower()
            hits = [k for k in self.cfg.fundamental_bad_keywords if k in t]
            if hits:
                bad_hits.extend(hits); bad_titles.append(it.title)
        v = FundamentalVerdict(
            as_of=as_of,
            ok=not bad_titles,
            n_headlines=len(headlines),
            bad_hits=sorted(set(bad_hits)),
            bad_titles=list(dict.fromkeys(bad_titles))[:3],
            source="gdelt-rule" if headlines else "no-news-default-ok",
        )
        self._fund_cache[as_of] = v
        return v


def analyze_resistance(as_of: date, **kwargs) -> NewsVerdict:
    """便捷函数。"""
    return NewsAgent().analyze(as_of, **kwargs)


if __name__ == "__main__":
    # 演示：人为动作 vs 技术阻力的判定
    rc = RuleClassifier()
    d = date(2024, 1, 5)
    human_news = [NewsItem(d, "Nasdaq falls after Fed Chair Powell signals higher rates"),
                  NewsItem(d, "QQQ hits resistance at 200-day moving average")]
    tech_news = [NewsItem(d, "QQQ overbought, RSI stalls at key resistance level")]
    for label, items in (("人为动作", human_news), ("技术阻力", tech_news)):
        r = rc.classify(items, d, {"recent_closes": [100.0] * 30})
        print(label, "->", r)
