"""纯 SVG 图表生成（零外部依赖，浏览器直接渲染）。"""

from __future__ import annotations

import html
import math
from datetime import date

import numpy as np

W, H = 1000, 360
PAD_L, PAD_R, PAD_T, PAD_B = 70, 24, 24, 40


def _scale(vals: np.ndarray, lo: float, hi: float, a: float, b: float):
    if hi <= lo:
        hi = lo + 1
    return a + (np.asarray(vals, dtype=float) - lo) / (hi - lo) * (b - a)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = raw / mag
    for m in (1, 2, 2.5, 5, 10):
        if step <= m:
            step = m * mag
            break
    start = math.floor(lo / step) * step
    out = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            out.append(round(v, 10))
        v += step
    return out


def line_chart(
    series: list[tuple[str, np.ndarray, str]],
    dates: list[date],
    title: str,
    ylabel: str = "权益 ($)",
    pct: bool = False,
) -> str:
    """多条折线。series = [(label, values, color), ...]"""
    allv = np.concatenate([s[1] for s in series])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    span = hi - lo if hi > lo else 1
    lo -= span * 0.06; hi += span * 0.06
    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = H - PAD_B, PAD_T

    def X(i): return x0 + i / max(1, len(dates) - 1) * (x1 - x0)
    def Y(v): return y0 - (v - lo) / (hi - lo) * (y0 - y1)

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto;background:#fff;font-family:system-ui,-apple-system,sans-serif">',
             f'<text x="{W/2}" y="16" text-anchor="middle" font-size="14" '
             f'font-weight="600" fill="#1c2536">{html.escape(title)}</text>']

    # grid + y labels
    for t in _nice_ticks(lo, hi):
        y = Y(t)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                     f'stroke="#e9edf3" stroke-width="1"/>')
        lab = f"{t*100:.0f}%" if pct else f"{t:,.0f}"
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#7a8699">{lab}</text>')

    # x labels (every ~1/6)
    for i in range(0, len(dates), max(1, len(dates) // 6)):
        parts.append(f'<text x="{X(i):.1f}" y="{H-14}" text-anchor="middle" '
                     f'font-size="11" fill="#7a8699">{dates[i].strftime("%y/%m")}</text>')

    for label, vals, color in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')

    # legend
    lx, ly = x0 + 8, y1 + 6
    for label, _, color in series:
        parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="3" fill="{color}"/>')
        parts.append(f'<text x="{lx+17}" y="{ly+4}" font-size="11" fill="#4a5568">'
                     f'{html.escape(label)}</text>')
        lx += 22 + len(label) * 7
    parts.append(f'<text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" '
                 f'font-size="11" fill="#7a8699" text-anchor="middle">'
                 f'{html.escape(ylabel)}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def drawdown_chart(equity: np.ndarray, dates: list[date]) -> str:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    lo, hi = float(np.min(dd)), 0.0
    lo -= 0.01
    x0, x1 = PAD_L, W - PAD_R
    y0, y1 = H - PAD_B, PAD_T

    def X(i): return x0 + i / max(1, len(dates) - 1) * (x1 - x0)
    def Y(v): return y0 - (v - lo) / (hi - lo) * (y0 - y1)

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto;background:#fff;font-family:system-ui,sans-serif">',
             '<text x="500" y="16" text-anchor="middle" font-size="14" '
             'font-weight="600" fill="#1c2536">策略回撤曲线</text>']
    for t in _nice_ticks(lo, hi):
        y = Y(t)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e9edf3"/>')
        parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="#7a8699">{t*100:.0f}%</text>')
    for i in range(0, len(dates), max(1, len(dates) // 6)):
        parts.append(f'<text x="{X(i):.1f}" y="{H-14}" text-anchor="middle" '
                     f'font-size="11" fill="#7a8699">{dates[i].strftime("%y/%m")}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(dd))
    parts.append(f'<polygon points="{X(0):.1f},{Y(0):.1f} {pts} {X(len(dd)-1):.1f},{Y(0):.1f}" '
                 f'fill="#e05c5c" opacity="0.25"/>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#d64545" stroke-width="1.6"/>')
    parts.append('</svg>')
    return "\n".join(parts)
