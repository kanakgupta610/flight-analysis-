#!/usr/bin/env python3
"""Fill the <!-- data:NAME --> ... <!-- /data:NAME --> regions of index.html from data/results.json.

    python3 scripts/render_html.py            # rewrite index.html in place
    python3 scripts/render_html.py --check    # exit 1 if index.html is out of date

Everything outside the markers (design, images, animations, formulas) is left untouched.
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BLUE, AMBER, PURPLE, GREEN, RED = "#79C9E8", "#F2B45A", "#B9A2F2", "#7FD6A8", "#E8797F"
PLANE = ("M23 12 C23 11 22 10.6 20.5 10.6 L14.5 10.6 L10 3 L8 3 L10.5 10.6 L5 10.6 L3.2 7.5 L1.8 7.5 L2.8 12 "
         "L1.8 16.5 L3.2 16.5 L5 13.4 L10.5 13.4 L8 21 L10 21 L14.5 13.4 L20.5 13.4 C22 13.4 23 13 23 12 Z")
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


# ---------------------------------------------------------------- formatting (single source for every shown number)
def n(x):
    return f"{int(x):,}"


def pct(p, d=1):
    return f"{100 * p:.{d}f}%"


def pnum(p, d=1):
    return f"{100 * p:.{d}f}"


def prob(p, d=3):
    return f"{p:.{d}f}"


def esc(s):
    return html.escape(str(s), quote=True)


def svg_num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- shared chart pieces
def bar_chart(items, *, w, h, x0, y_base, y_top, ymax, step, highlight, delay0, label_size=15, value_fmt=pct,
              tick_fmt=lambda v: f"{v:g}", ylab=None, xlab=None):
    """Vertical bars in the style of the original slide 8 chart."""
    scale = (y_base - y_top) / ymax
    x1 = w - 10
    out = []
    t = 0.0
    while t <= ymax + 1e-9:
        y = y_base - t * scale
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{svg_num(y)}" y2="{svg_num(y)}" stroke="rgba(255,255,255,.06)"></line>')
        out.append(f'<text x="{x0 - 10}" y="{svg_num(y + 4)}" fill="#AFC0DD" font-size="13" text-anchor="end">{tick_fmt(round(t, 6))}</text>')
        t += step
    if ylab:
        cy = (y_base + y_top) / 2
        out.append(f'<text x="14" y="{svg_num(cy)}" fill="#DDE7F7" font-size="13" text-anchor="middle" transform="rotate(-90 14 {svg_num(cy)})">{esc(ylab)}</text>')
    band = (x1 - x0) / len(items)
    bw = band * 0.62
    for i, (label, value) in enumerate(items):
        cx = x0 + band * (i + 0.5)
        bh = value * scale
        fill = AMBER if i == highlight else BLUE + "C7"
        out.append(f'<rect x="{svg_num(cx - bw / 2)}" y="{svg_num(y_base - bh)}" width="{svg_num(bw)}" height="{svg_num(bh)}" rx="4" '
                   f'fill="{fill}" class="barY" style="animation-delay:{delay0 + 0.15 * i:.2f}s"></rect>')
        out.append(f'<text x="{svg_num(cx)}" y="{svg_num(y_base - bh - 9)}" fill="#fff" font-size="{label_size}" font-weight="700" '
                   f'text-anchor="middle" class="fade" style="animation-delay:{delay0 + 1.45:.1f}s">{value_fmt(value)}</text>')
        out.append(f'<text x="{svg_num(cx)}" y="{svg_num(y_base + 24)}" fill="#AFC0DD" font-size="13" text-anchor="middle">{esc(label)}</text>')
    if xlab:
        out.append(f'<text x="{svg_num((x0 + x1) / 2)}" y="{h - 4}" fill="#DDE7F7" font-size="13" text-anchor="middle">{esc(xlab)}</text>')
    return out


def nice_ceiling(v, step):
    return step * math.ceil(v / step + 1e-9)


# ---------------------------------------------------------------- regions
def r_chip(R):
    return ["Real data · 2019"]


def r_s1_lead(R):
    return [f'<p class="lead">A probability-based study of how weather relates to flight delays — '
            f'{n(R["overall"]["operated"])} real US departures from 2019.</p>']


def r_s3_variables(R):
    rows = [("Weather condition", "Clear, foggy, rainy, snowy, stormy — from NOAA daily weather records"),
            ("Delay status", "Delayed (≥ 15 min, DEP_DEL15) / not delayed"),
            ("Delay duration", "Departure delay in minutes (DEP_DELAY_NEW)"),
            ("Weather measurements", "Avg wind (mph), precipitation &amp; snowfall (in), max temp (°F)")]
    out = []
    for a, b in [("Variable", "Values")]:
        out += ["<tr>", f"  <th>{a}</th>", f"  <th>{b}</th>", "</tr>"]
    for a, b in rows:
        out += ["<tr>", f"  <td>{a}</td>", f'  <td style="color: #d8e1ef">{b}</td>', "</tr>"]
    return out


def r_s3_dataset(R):
    o, wj, v = R["overall"], R["weather_join"], R["validation"]
    dm = o["delay_minutes"]
    stat = lambda label, value, color="#fff": (
        f'<div><div class="label" style="margin-bottom: 4px">{label}</div>'
        f'<div class="display" style="font-size: 30px; color: {color}">{value}</div></div>')
    return [
        '<div class="glass pad" style="margin-bottom: 16px">',
        '  <div class="label">Dataset on this slide</div>',
        '  <span class="badge ok">Real data — January to December 2019</span>',
        '  <p class="small" style="margin-top: 12px">',
        '    <strong style="color: #fff">Source: </strong>BTS On-Time Reporting for every US domestic departure, joined to NOAA GHCN-Daily '
        'airport weather (Kaggle: “2019 Airline Delays w/ Weather and Airport Detail”). Delay threshold: 15 minutes.',
        '  </p>',
        '  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 14px">',
        "    " + stat("Scheduled", n(o["scheduled"])),
        "    " + stat("Cancelled", f'{n(o["cancelled"])}'),
        "    " + stat("Operated", n(o["operated"])),
        "    " + stat("Delayed ≥ 15 min", pct(o["p_delay"]), AMBER),
        "    " + stat("Median delay", f'{dm["median"]} min'),
        "    " + stat("Mean delay", f'{dm["mean"]:.1f} min'),
        '  </div>',
        f'  <p class="small" style="margin-top: 14px">{pct(o["p_cancel"])} of scheduled flights were cancelled and are left out of delay rates. '
        f'90% of departures left within {dm["p90"]} min of schedule; delayed flights waited a median {dm["median_if_delayed"]} min '
        f'(mean {dm["mean_if_delayed"]:.1f} min).</p>',
        f'  <p class="small" style="margin-top: 8px; color: #a9b7cf">Weather matched on (origin station, date) for {n(wj["matched"])} departures '
        f'({pct(wj["share_matched"])}) at {v["airports_joined"]} airports; {n(wj["flights"])} of them fall on days with a definite weather '
        f'condition. Wet days at stations that do not report snowfall are left unclassified, because rain and snow cannot be told apart. '
        f'{len(v["excluded_airports"])} airports were excluded because their listed weather station is in another city.</p>',
        '</div>',
    ]


def pick_sample_rows(sample, k=6):
    rows, used = [], set()
    for c in ["Stormy", "Rainy", "Clear", "Foggy", "Snowy"]:
        for i, s in enumerate(sample):
            if s["condition"] == c and i not in used:
                rows.append(i); used.add(i); break
    for i, s in enumerate(sample):
        if len(rows) >= k:
            break
        if i not in used and s["delayed"] and s["condition"]:
            rows.append(i); used.add(i)
    return [sample[i] for i in rows[:k]]


def r_s3_sample(R):
    out = ['<div class="glass" style="padding: 14px 10px 6px">',
           '  <div class="label" style="padding: 0 12px">Flight log · random 2019 departures</div>',
           '  <div class="tscroll"><table class="gt mono">', "    <tr>"]
    out += [f"      <th>{h}</th>" for h in ["Flight", "Weather", "Wind mph", "Delay min", "Status"]]
    out.append("    </tr>")
    for i, s in enumerate(pick_sample_rows(R["scatter_sample"])):
        badge = '<span class="badge warn">Delayed</span>' if s["delayed"] else '<span class="badge ok">On time</span>'
        out += [f'    <tr class="rv" style="animation-delay: {1.38 + 0.18 * i:.2f}s">',
                f'      <td>{esc(s["origin"])} · {s["day"]} {MONTHS[s["month"] - 1]}</td>',
                f'      <td>{s["condition"]}</td>', f'      <td>{s["awnd"]:g}</td>', f'      <td>{int(s["delay_min"])}</td>',
                f"      <td>{badge}</td>", "    </tr>"]
    out += ["  </table></div>", "</div>"]
    return out


def rc(R):
    by = {c["condition"]: c for c in R["weather_join"]["contingency"]}
    return by["Rainy"], by["Clear"]


def frac(num, den):
    return f'<span class="frac"><span class="num">{num}</span><span class="den">{den}</span></span>'


def r_s5_badge(R):
    return ['<span class="badge ok">Real 2019 counts — rainy vs clear days</span>']


def r_s5_body(R):
    rain, clear = rc(R)
    rvc = R["weather_join"]["rainy_vs_clear"]
    tot = lambda k: rain[k] + clear[k]
    row = lambda name, c: ["    <tr>", f"      <td>{name}</td>", f'      <td class="hl">{n(c["delayed"])}</td>',
                           f'      <td>{n(c["not_delayed"])}</td>', f'      <td>{n(c["total"])}</td>', "    </tr>"]
    case = lambda label, sym, c: [
        "<div>", f'  <div class="label">{label}</div>',
        f'  <div class="math">P(D | {sym}) <span class="up">=</span> {frac(n(c["delayed"]), n(c["total"]))} '
        f'<span class="up">= {prob(c["p_delay_given"])}</span></div>', "</div>"]
    big = lambda label, color, c, delay, text: [
        f'  <div class="rv" style="animation-delay: {delay}s">',
        f'    <div class="glass pad" style="text-align: center; border-top: 2px solid {color}">',
        f'      <div class="label">{label}</div>',
        f'      <div class="bignum"><span class="count" data-to="{pnum(c["p_delay_given"])}" data-delay="{delay + 0.1:.2f}" '
        f'data-dur="1.4" data-digits="1">0%</span></div>',
        f'      <p class="small">{text}</p>', "    </div>", "  </div>"]
    out = ['<div class="grid2" style="align-items: start">', "<div>",
           '  <div class="rv" style="animation-delay: 0.3s">',
           f'    <p class="lead">Take every 2019 departure from an airport on a rainy or a clear day ({n(tot("total"))} flights):</p>',
           "  </div>",
           '  <div class="rv" style="animation-delay: 0.5s">',
           '    <div class="glass" style="margin: 14px 0 22px; padding: 6px 10px">',
           '  <div class="tscroll"><table class="gt">', "    <tr>", "      <th>Weather</th>", "      <th>Delayed</th>", "      <th>Not delayed</th>",
           "      <th>Total</th>", "    </tr>",
           *row("Rainy", rain), *row("Clear", clear),
           '    <tr class="total">', "      <td>Total</td>", f'      <td>{n(tot("delayed"))}</td>',
           f'      <td>{n(tot("not_delayed"))}</td>', f'      <td>{n(tot("total"))}</td>', "    </tr>",
           "  </table></div>", "    </div>", "  </div>",
           '  <div class="rv" style="animation-delay: 1s">', *case("Case 1 — delay given rain", "R", rain), "  </div>",
           '  <div class="rv" style="animation-delay: 1.6s"><div style="margin-top: 18px">',
           *case("Case 2 — delay given clear weather", "C", clear), "  </div></div>",
           "</div>", "<div>",
           '  <div style="position: relative; display: grid; grid-template-columns: 1fr 1fr; gap: 60px; margin-top: 34px">',
           *big("Rain", BLUE, rain, 2, "of rainy-day departures delayed"),
           *big("Clear", AMBER, clear, 2.2, "of clear-day departures delayed"),
           '    <div style="position: absolute; left: 38%; right: 38%; top: 48%">',
           '      <svg viewBox="0 0 100 30" width="100%">',
           '        <path d="M0 15 C 30 0, 70 30, 100 15" fill="none" stroke="rgba(255,255,255,.7)" stroke-dasharray="1.5 4"></path>',
           '        <g>',
           '          <animateMotion path="M0 15 C 30 0, 70 30, 100 15" dur="3.5s" begin="0s" repeatCount="indefinite" rotate="auto" '
           'fill="freeze" calcMode="spline" keyTimes="0;1" keyPoints="0;1" keySplines="0.42 0 0.58 1"></animateMotion>',
           '          <g transform="scale(0.5) translate(-12,-12)">',
           f'            <path d="{PLANE}" fill="#fff" style="filter: drop-shadow(0 0 6px rgba(121,201,232,.9))"></path>',
           "          </g>", "        </g>", "      </svg>", "    </div>", "  </div>",
           '  <div class="rv glass" style="animation-delay: 3.85s;margin-top: 20px; padding: 16px 22px; display: flex; align-items: center; gap: 18px">',
           f'    <div class="display" style="font-size: 48px; color: {BLUE}">+{rvc["difference_pp"]:.1f}</div>',
           f'    <p class="small">percentage points — in 2019, departures on rainy days were delayed {rvc["difference_pp"]:.1f} percentage points '
           f'more often than on clear days (relative risk {rvc["risk_ratio"]:.2f}).</p>',
           "  </div>",
           '  <div class="rv" style="animation-delay: 4.6s">',
           '    <p class="small" style="margin-top: 18px; color: #a9b7cf">This comparison alone does not establish that rain caused the delays. '
           'Other factors — congestion, visibility and operational disruptions — may also contribute.</p>',
           "  </div>", "</div>", "</div>"]
    return out


def r_s6_lead(R):
    return ['<p class="lead" style="max-width: 900px">R calculates conditional probabilities directly from observed flight data. '
            'This example uses the real 2019 counts from Slide 5.</p>']


def r_s6_code(R):
    rain, clear = rc(R)
    num = lambda v: f"<span class='tk-num'>{v}</span>"
    s = lambda v: f"<span class='tk-str'>\"{v}\"</span>"
    lines = [
        "flights <span class='tk-op'>&lt;-</span> <span class='tk-fn'>data.frame</span>(",
        f"  weather     = <span class='tk-fn'>c</span>({s('Rainy')}, {s('Clear')}),",
        f"  delayed     = <span class='tk-fn'>c</span>({num(rain['delayed'])}, {num(clear['delayed'])}),",
        f"  not_delayed = <span class='tk-fn'>c</span>({num(rain['not_delayed'])}, {num(clear['not_delayed'])})",
        ")", "",
        "flights$probability <span class='tk-op'>&lt;-</span> <span class='tk-fn'>round</span>(",
        "  flights$delayed /",
        f"  (flights$delayed + flights$not_delayed), {num(4)})", "",
        "<span class='tk-fn'>print</span>(flights)",
        "<span class='caret'></span>",
    ]
    body = "".join(f"<span class='codeline' style='animation-delay:{1.04 + 0.24 * i:.2f}s'><span class='ln'>{i + 1}</span>{t}</span>"
                   for i, t in enumerate(lines))
    return [f"<pre>{body}</pre>"]


def r_console(rows, headers):
    """Mimic R's print.data.frame: right-aligned columns, row names, one space between."""
    cols = list(zip(*rows)) if rows else [[] for _ in headers]
    widths = [max(len(h), *(len(v) for v in col)) for h, col in zip(headers, cols)]
    rn = max(len(str(len(rows))), 1)
    lines = [" " * rn + "".join(" " + h.rjust(w) for h, w in zip(headers, widths))]
    for i, r in enumerate(rows):
        lines.append(str(i + 1).ljust(rn) + "".join(" " + v.rjust(w) for v, w in zip(r, widths)))
    return "\n".join(lines)


def r_s6_console(R):
    rain, clear = rc(R)
    p4 = lambda c: f'{round(c["delayed"] / c["total"], 4):.4f}'
    con = r_console([["Rainy", str(rain["delayed"]), str(rain["not_delayed"]), p4(rain)],
                     ["Clear", str(clear["delayed"]), str(clear["not_delayed"]), p4(clear)]],
                    ["weather", "delayed", "not_delayed", "probability"])
    bar = lambda label, c, color, d: [
        f'  <div class="rv" style="animation-delay: {d}s; margin-bottom: 18px">',
        '    <div style="display: flex; justify-content: space-between; margin-bottom: 8px">',
        f'      <span>{label}</span>',
        f'      <span class="display" style="font-size: 28px; color: {color}">{prob(c["p_delay_given"])}</span>',
        "    </div>",
        f'    <div class="hbar"><div style="width: {pnum(c["p_delay_given"])}%; background: {color}; animation-delay: {d + 0.2:.1f}s"></div></div>',
        "  </div>"]
    return ['<div class="glass pad" style="min-height: 360px">',
            '  <div class="label rv" style="animation-delay: 4s">Console output</div>',
            f'  <pre class="console rv" style="animation-delay: 4.1s">{esc(con)}</pre>',
            '  <div style="height: 24px"></div>',
            '  <div class="label rv" style="animation-delay: 4.5s">Probability dashboard</div>',
            *bar("Rainy", rain, BLUE, 5), *bar("Clear", clear, AMBER, 5.3),
            '  <p class="small rv" style="animation-delay: 5.8s; color: #a9b7cf">Counts from Slide 5, calculated from the 2019 flight and '
            'weather records by scripts/process_data.py. Bars use a 0–1 probability scale.</p>',
            "</div>"]


def r_s7_notes(R):
    cor = R["weather_join"]["correlation_with_delay_minutes"]["AWND"]
    return [
        '<div class="rv" style="animation-delay: 0.55s">',
        '  <ul class="small" style="padding-left: 18px; margin: 14px 0">',
        '    <li><strong style="color: #fff">X-axis: </strong>average daily wind speed at the origin airport (mph)</li>',
        '    <li><strong style="color: #fff">Y-axis: </strong>departure delay (minutes)</li>',
        f'    <li><strong style="color: #fff">Each point: </strong>one 2019 departure — a random sample of {len(R["scatter_sample"])} '
        f'from the {n(cor["n"])} with wind data</li>',
        "  </ul>",
        "</div>",
        '<div class="rv" style="animation-delay: 0.8s">',
        '  <p class="small">If higher wind speeds tend to occur alongside longer delays, the points show an upward pattern. '
        'If they are widely scattered, the relationship may be weak or inconsistent.</p>',
        "</div>",
        '<div class="rv" style="animation-delay: 1s">',
        '  <div class="glass editor" style="min-height: 0; margin-top: 16px; font-size: 12px; line-height: 1.6">',
        "    <pre><span class='tk-fn'>plot</span>(\n  flights$AWND,\n  flights$DEP_DELAY_NEW,\n"
        "  main = <span class='tk-str'>\"Wind Speed vs Departure Delay\"</span>,\n"
        "  xlab = <span class='tk-str'>\"Average Daily Wind Speed (mph)\"</span>,\n"
        "  ylab = <span class='tk-str'>\"Departure Delay (minutes)\"</span>,\n"
        "  pch = <span class='tk-num'>19</span>, col = <span class='tk-str'>\"steelblue\"</span>\n)</pre>",
        "  </div>",
        "</div>",
    ]


def r_s7_chart(R):
    cor = R["weather_join"]["correlation_with_delay_minutes"]["AWND"]
    pts = R["scatter_sample"]
    X0, X1, Y0, Y1 = 70, 876, 438, 18  # plot box (left, right, bottom, top)
    xmax = max(30, nice_ceiling(max(p["awnd"] for p in pts), 5))
    ymax = 180
    sx = lambda v: X0 + (X1 - X0) * v / xmax
    sy = lambda v: Y0 - (Y0 - Y1) * min(v, ymax) / ymax
    over = sum(p["delay_min"] > ymax for p in pts)
    out = ['<div class="rv" style="animation-delay: 0.4s">',
           '  <div class="glass" style="padding: 16px 18px 26px; position: relative; overflow: hidden">',
           '    <div style="display: flex; justify-content: space-between; align-items: center">',
           '      <div class="label" style="margin: 0">Wind speed vs departure delay</div>',
           f'      <span class="badge ok">Real 2019 sample · n = {len(pts)}</span>',
           "    </div>",
           '    <div style="height: 500px">',
           '      <svg viewBox="0 0 900 500" width="100%" height="100%" style="display:block">']
    for v in range(0, int(xmax) + 1, 5):
        x = svg_num(sx(v))
        out.append(f'        <line x1="{x}" x2="{x}" y1="{Y1}" y2="{Y0}" stroke="rgba(255,255,255,.06)"></line>')
        out.append(f'        <text x="{x}" y="460" fill="#AFC0DD" font-size="13" text-anchor="middle">{v}</text>')
    for v in range(0, ymax + 1, 30):
        y = svg_num(sy(v))
        out.append(f'        <line x1="{X0}" x2="{X1}" y1="{y}" y2="{y}" stroke="rgba(255,255,255,.06)"></line>')
        out.append(f'        <text x="58" y="{svg_num(sy(v) + 4)}" fill="#AFC0DD" font-size="13" text-anchor="end">{v}{"+" if v == ymax else ""}</text>')
    out.append('        <text x="485" y="486" fill="#DDE7F7" font-size="15" text-anchor="middle">Average daily wind speed (mph)</text>')
    out.append('        <text x="18" y="219" fill="#DDE7F7" font-size="15" text-anchor="middle" transform="rotate(-90 18 219.000000)">Departure delay (minutes)</text>')
    yt = svg_num(sy(15))
    out.append(f'        <line x1="{X0}" x2="{X1}" y1="{yt}" y2="{yt}" stroke="#F2B45A80" stroke-dasharray="2 5"></line>')
    out.append(f'        <text x="{X1}" y="{svg_num(sy(15) - 7)}" fill="#F2B45A" font-size="12" text-anchor="end">delay threshold: 15 min</text>')
    for i, p in enumerate(pts):
        extra = ' stroke="#F2B45A" stroke-width="1.2"' if p["delay_min"] > ymax else ""
        out.append(f'        <circle cx="{svg_num(sx(p["awnd"]))}" cy="{svg_num(sy(p["delay_min"]))}" r="4.2" fill="#79C9E8" fill-opacity="0.6"{extra} '
                   f'class="pt" style="animation-delay:{0.65 + 2.6 * i / len(pts):.2f}s; transform-box: fill-box; transform-origin: center"></circle>')
    a, b = cor["intercept"], cor["slope"]
    out.append(f'        <line x1="{X0}" y1="{svg_num(sy(a))}" x2="{X1}" y2="{svg_num(sy(a + b * xmax))}" stroke="#F2B45A" stroke-width="2.6" '
               f'class="draw" style="animation-delay:3.7s"></line>')
    out += ["      </svg>", "    </div>",
            '    <div style="position: absolute; left: 0; right: 0; bottom: 8px; height: 18px">',
            '      <div style="position: absolute; bottom: 8px; left: 0; right: 0; border-top: 1px dashed rgba(255,255,255,.18)"></div>',
            '      <div class="runner" style="position: absolute; bottom: 0">',
            '        <svg width="18" height="18" viewBox="0 0 24 24" style="transform: rotate(0deg)">',
            f'          <path d="{PLANE}" fill="#79C9E8"></path>',
            "        </svg>", "      </div>", "    </div>",
            '    <div class="rv small" style="animation-delay: 4.2s;position: absolute; top: 54px; right: 26px; max-width: 320px; padding: 10px 14px; '
            'background: rgba(6,11,23,.7); border-radius: 10px; border: 1px solid rgba(255,255,255,0.14)">',
            '      <span style="color: #F2B45A">━ </span>',
            f'      Least-squares line from all {n(cor["n"])} flights with wind data: slope = {b:.2f} min per mph, r = {cor["r"]:.2f}. '
            f'{over} sampled flights above {ymax} min are drawn on the top edge (amber ring).',
            "    </div>", "  </div>", "</div>"]
    return out


def r_s8_badge(R):
    return [f'<span class="badge ok">Real 2019 data — {n(R["weather_join"]["flights"])} departures with classified weather</span>']


def compact_table(title, headers, rows, delay, accent):
    out = [f'<div class="rv" style="animation-delay: {delay}s">',
           f'  <div class="glass" style="padding: 14px 10px 6px; height: 100%; border-top: 2px solid {accent}">',
           f'    <div class="label" style="padding: 0 12px">{title}</div>',
           '    <div class="tscroll"><table class="gt mono" style="font-size: 13px">', "      <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"]
    for r in rows:
        cells = [f"<td>{v}</td>" for v in r[:-1]] + [f'<td class="hl">{r[-1]}</td>']
        out.append("      <tr>" + "".join(cells) + "</tr>")
    out += ["    </table></div>", "  </div>", "</div>"]
    return out


def r_s8_body(R):
    wj = R["weather_join"]
    con = wj["contingency"]
    top = max(con, key=lambda c: c["p_delay_given"])
    labw = max(len(c["condition"]) for c in con)
    prop = "\n".join([" " * (labw + 2) + "FALSE".rjust(6) + "  TRUE"] +
                     [f'  {c["condition"].ljust(labw)} {prob(c["not_delayed"] / c["total"]).rjust(6)} {prob(c["p_delay_given"])}' for c in con])
    hw = max((c["ci95"][1] - c["ci95"][0]) / 2 for c in con)
    chi = wj["chi_square"]
    cor = wj["correlation_with_delay_minutes"]["AWND"]
    wind = wj["by_wind"]
    ymax = nice_ceiling(top["p_delay_given"] * 1.15, 0.05)
    bars = bar_chart([(c["condition"], c["p_delay_given"]) for c in con], w=560, h=420, x0=58, y_base=380, y_top=24,
                     ymax=ymax, step=0.05, highlight=con.index(top), delay0=1.95, ylab="Probability of delay",
                     tick_fmt=lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0")
    wmax = nice_ceiling(max(x["p_delay"] for x in wind) * 1.15, 0.05)
    wbars = bar_chart([(x["key"], x["p_delay"]) for x in wind], w=640, h=360, x0=70, y_base=298, y_top=18, ymax=wmax, step=0.05,
                      highlight=len(wind) - 1, delay0=3.1, label_size=14, xlab="Average daily wind speed (mph)",
                      tick_fmt=lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0")
    prc = wj["by_precip"]
    snow = wj["by_snow"]
    causes = R["overall"]["causes"]["share"]
    carriers = R["by_carrier"]
    ports = R["busiest_airports"][:6]
    hbar = lambda label, p, color, d: [
        f'  <div class="rv" style="animation-delay: {d:.2f}s; margin-bottom: 9px">',
        f'    <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px"><span>{label}</span>'
        f'<span style="color: {color}">{pct(p)}</span></div>',
        f'    <div class="hbar" style="height: 7px"><div style="width: {pnum(p)}%; background: {color}; animation-delay: {d + 0.2:.2f}s"></div></div>',
        "  </div>"]
    out = [
        '<div style="display: grid; grid-template-columns: 0.8fr 1.3fr 1.3fr; gap: 16px; align-items: stretch">',
        '  <div class="rv" style="animation-delay: 0.6s">',
        f'    <div class="glass pad" style="border-top: 2px solid {AMBER}; display: flex; flex-direction: column; justify-content: center">',
        '      <div class="label">Highest observed delay probability</div>',
        f'      <div class="display" style="font-size: clamp(34px,3.2vw,52px)">{top["condition"]}</div>',
        f'      <div class="bignum" style="color: {AMBER}; font-size: clamp(54px, 5vw, 84px); margin: 6px 0 10px"><span class="count" data-to="{pnum(top["p_delay_given"])}" data-delay="0.8" '
        f'data-dur="1.6" data-digits="1">0%</span></div>',
        f'      <p class="small">P(delayed | {top["condition"]}), where delayed = 15+ min late</p>',
        f'      <pre class="console" style="margin-top: 14px; font-size: 11px; color: #d8e1ef">{esc(prop)}</pre>',
        "    </div>",
        "  </div>",
        '  <div class="rv" style="animation-delay: 1.6s">',
        '    <div class="glass" style="padding: 14px 16px">',
        '      <div class="label">P(delayed | weather condition)</div>',
        '      <div style="height: 380px">',
        '        <svg viewBox="0 0 560 420" width="100%" height="100%" style="display:block">',
        *["          " + s for s in bars],
        "        </svg>",
        "      </div>",
        f'      <p class="small" style="margin-top: 6px">95% confidence intervals are within ±{math.ceil(10000 * hw) / 100:.2f} percentage points for every condition. '
        f'χ² test of independence: χ² = {n(round(chi["statistic"]))}, df = {chi["df"]}, p &lt; 0.001.</p>',
        "    </div>",
        "  </div>",
        '  <div class="rv" style="animation-delay: 2.8s">',
        '    <div class="glass" style="padding: 14px 16px">',
        '      <div class="label">Delay probability by wind speed</div>',
        '      <div style="height: 290px">',
        '        <svg viewBox="0 0 640 360" width="100%" height="100%" style="display:block">',
        *["          " + s for s in wbars],
        "        </svg>",
        "      </div>",
        f'      <p class="small" style="margin-top: 6px">Pearson r = {cor["r"]:.2f} between daily wind speed and delay minutes '
        f'(n = {n(cor["n"])}) — a very weak linear relationship, yet the delay rate rises from {pct(wind[0]["p_delay"])} below 5 mph '
        f'to {pct(wind[-1]["p_delay"])} at 20+ mph.</p>',
        "    </div>",
        "  </div>",
        "</div>",
        '<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; align-items: stretch; margin-top: 16px">',
        '  <div class="rv" style="animation-delay: 3.4s">',
        f'    <div class="glass pad" style="height: 100%; border-top: 2px solid {BLUE}">',
        '      <div class="label">P(delayed) by precipitation (in)</div>',
        *["    " + s for i, x in enumerate(prc) for s in hbar(x["key"], x["p_delay"], BLUE, 3.5 + 0.1 * i)],
        f'      <p class="small" style="margin-top: 6px; font-size: 13px">Snowfall above 3 in: {pct(snow[-1]["p_delay"])} delayed. '
        f'Airlines attribute {pct(causes["WEATHER_DELAY"])} of delay minutes directly to extreme weather; '
        f'the {pct(causes["NAS_DELAY"])} coded as air-system (NAS) delay also includes non-extreme weather.</p>',
        "    </div>",
        "  </div>",
        *["  " + s for s in compact_table(
            "Airlines · highest &amp; lowest", ["Airline", "Flights", "P(delay)"],
            [[esc(c["name"]), n(c["flights"]), pct(c["p_delay"])] for c in carriers[:3] + carriers[-3:]], 3.6, AMBER)],
        *["  " + s for s in compact_table(
            "Busiest departure airports", ["Airport", "Flights", "P(delay)"],
            [[esc(a["code"]), n(a["flights"]), pct(a["p_delay"])] for a in ports], 3.8, PURPLE)],
        "</div>",
        '<div class="rv" style="animation-delay: 4s">',
        '  <div class="glass pad" style="margin-top: 16px; padding: 14px 20px; display: flex; gap: 16px; align-items: center; border-left: 2px solid #E8797F">',
        '    <span class="label" style="margin: 0; color: #E8797F">Limitations</span>',
        f'    <p class="small">Association does not prove causation. Weather is a daily summary at the origin airport’s station, not the conditions '
        f'at departure time; {pct(1 - wj["share_matched"])} of operated flights had no compatible station; cancelled flights '
        f'({pct(R["overall"]["p_cancel"])}) are excluded. With millions of flights even small gaps are statistically significant, so compare the size of the differences.</p>',
        "  </div>",
        "</div>",
    ]
    return out


def r_s9_lead(R):
    by = {c["condition"]: c for c in R["weather_join"]["contingency"]}
    return ['<div class="rv" style="animation-delay: 0.8s">',
            '  <p class="lead" style="margin: 0 auto; max-width: 820px">Probability distributions turn historical flight and weather data into '
            'meaningful comparisons. Conditional probability estimates the observed likelihood of a delay under a specific weather condition, '
            'while scatter plots explore how weather measurements relate to delay duration.</p>',
            '  <p class="small" style="margin: 14px auto 0; max-width: 820px; color: #fff">In 2019, departures were delayed '
            f'{pct(by["Clear"]["p_delay_given"])} of the time on clear days, {pct(by["Foggy"]["p_delay_given"])} with fog, '
            f'{pct(by["Rainy"]["p_delay_given"])} on rainy days, {pct(by["Stormy"]["p_delay_given"])} with thunderstorms and '
            f'{pct(by["Snowy"]["p_delay_given"])} on snowy days.</p>',
            "</div>"]


RENDERERS = {name: fn for name, fn in [
    ("chip", r_chip), ("s1-lead", r_s1_lead), ("s3-variables", r_s3_variables), ("s3-dataset", r_s3_dataset),
    ("s3-sample", r_s3_sample), ("s5-badge", r_s5_badge), ("s5-body", r_s5_body), ("s6-lead", r_s6_lead),
    ("s6-code", r_s6_code), ("s6-console", r_s6_console), ("s7-notes", r_s7_notes), ("s7-chart", r_s7_chart),
    ("s8-badge", r_s8_badge), ("s8-body", r_s8_body), ("s9-lead", r_s9_lead)]}

MARKER = re.compile(r"^([ \t]*)<!-- data:([\w-]+) -->\n.*?^[ \t]*<!-- /data:\2 -->", re.S | re.M)


def render(page, results):
    seen = set()

    def fill(m):
        ind, name = m.group(1), m.group(2)
        if name not in RENDERERS:
            raise KeyError(f"no renderer for region {name!r}")
        seen.add(name)
        body = "\n".join(ind + "  " + line if line else "" for line in RENDERERS[name](results))
        return f"{ind}<!-- data:{name} -->\n{body}\n{ind}<!-- /data:{name} -->"

    out = MARKER.sub(fill, page)
    missing = set(RENDERERS) - seen
    if missing:
        raise SystemExit(f"markers missing from page: {sorted(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "data/results.json"))
    ap.add_argument("--html", default=str(REPO / "index.html"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    page = Path(args.html).read_text(encoding="utf-8")
    new = render(page, results)
    if args.check:
        sys.exit(0 if new == page else "index.html is out of date — run scripts/render_html.py")
    Path(args.html).write_text(new, encoding="utf-8")
    print(f"rendered {len(RENDERERS)} regions into {args.html}")


if __name__ == "__main__":
    main()
