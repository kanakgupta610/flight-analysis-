#!/usr/bin/env python3
"""Validate data/results.json and the numbers shown in index.html.

    python3 scripts/validate.py                                   # consistency + page checks
    python3 scripts/validate.py --recount ~/Downloads/archive     # also recount headline numbers from the raw CSVs

1. results.json is internally consistent (totals add up, probabilities = counts / totals, chi-square recomputes).
2. --recount: an independent, non-chunked re-implementation of the counts and the weather join.
3. index.html is rendered from the current results.json, contains no leftover simulated/demo wording, and every
   number in the visible text of the data regions can be traced to results.json (or is a listed constant).
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import render_html  # noqa: E402

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1, abs(a), abs(b))


# ---------------------------------------------------------------- 1. internal consistency
def consistency(R):
    print("results.json consistency")
    o, wj, v = R["overall"], R["weather_join"], R["validation"]["flights"]
    check(o["scheduled"] == o["cancelled"] + o["operated"], "scheduled = cancelled + operated")
    check(sum(m["flights"] for m in R["by_month"]) == o["operated"], "monthly flights sum to operated")
    check(sum(m["delayed"] for m in R["by_month"]) == o["delayed"], "monthly delayed sum to delayed")
    check(sum(c["flights"] for c in R["by_carrier"]) == o["operated"], "carrier flights sum to operated")
    check(sum(b["flights"] for b in o["delay_distribution"]) == o["operated"], "delay histogram sums to operated")
    check(sum(b["flights"] for b in o["delay_distribution"][2:]) == o["delayed"], "histogram bins >= 15 min = delayed count")
    check(close(o["p_delay"], o["delayed"] / o["operated"]), "overall P(delay) = delayed / operated")
    check(close(sum(o["causes"]["share"].values()), 1), "delay-cause shares sum to 1")
    con = wj["contingency"]
    check(sum(c["total"] for c in con) == wj["flights"] == v["joined"], "contingency total = joined flights")
    check(v["operated"] == wj["flights"] + v["no_station"] + v["no_weather_row"] + v["unclassified_weather"],
          "operated = joined + no station + no weather row + unclassified")
    for c in con:
        check(c["delayed"] + c["not_delayed"] == c["total"] and close(c["p_delay_given"], c["delayed"] / c["total"]),
              f'{c["condition"]}: delayed + not delayed = total and P(D|W) = delayed / total')
        check(c["ci95"][0] < c["p_delay_given"] < c["ci95"][1], f'{c["condition"]}: estimate inside its 95% CI')
    check(close(sum(c["p_condition"] for c in con), 1), "marginal P(W) sums to 1")
    check(close(sum(c["p_joint_delayed"] for c in con), wj["p_delay"]), "sum of joint P(W, D) = marginal P(D)")
    N = wj["flights"]; D = sum(c["delayed"] for c in con)
    chi = sum((obs - exp) ** 2 / exp for c in con
              for obs, exp in [(c["delayed"], c["total"] * D / N), (c["not_delayed"], c["total"] * (N - D) / N)])
    check(close(chi, wj["chi_square"]["statistic"], 1e-6), "chi-square statistic recomputes")
    for key in ["by_wind", "by_precip", "by_tmax"]:
        check(sum(b["flights"] for b in wj[key]) <= N, f"{key} bins do not exceed joined flights")
    check(sum(b["flights"] for b in wj["by_precip"]) == N, "precipitation bins cover every joined flight")
    cor = wj["correlation_with_delay_minutes"]["AWND"]
    check(-1 <= cor["r"] <= 1 and cor["n"] == sum(b["flights"] for b in wj["by_wind"]), "wind correlation n = wind-bin flights")
    s = R["scatter_sample"]
    check(len(s) == 500 and all(x["awnd"] is not None for x in s), "scatter sample has 500 flights with wind data")
    fc = R["validation"].get("full_data_crosscheck")
    if fc:
        gaps = [abs(R["by_month"][i]["p_delay"] - fc["by_month"][str(i + 1)]["p_delay"]) for i in range(12)]
        check(max(gaps) < 0.01, f"monthly delay rate within 1 pp of Kaggle full_data_flightdelay.csv (max gap {100 * max(gaps):.2f} pp)")


# ---------------------------------------------------------------- 2. independent recount
def recount(R, data_dir):
    import numpy as np
    import pandas as pd
    print(f"independent recount from {data_dir}")
    cols = ["MONTH", "DAY_OF_MONTH", "OP_UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "ORIGIN", "DEP_DEL15", "CANCELLED", "DEP_DELAY_NEW"]
    f = pd.concat([pd.read_csv(data_dir / f"raw_data/ONTIME_REPORTING_{m:02d}.csv", usecols=cols, engine="pyarrow")
                   for m in range(1, 13)], ignore_index=True)
    o = R["overall"]
    check(len(f) == o["scheduled"], f"scheduled flights {len(f):,}")
    check(int(f.CANCELLED.sum()) == o["cancelled"], f"cancelled {int(f.CANCELLED.sum()):,}")
    op = f[f.CANCELLED == 0]
    check(len(op) == o["operated"] and int(op.DEP_DEL15.sum()) == o["delayed"], f"operated {len(op):,} / delayed {int(op.DEP_DEL15.sum()):,}")
    check(close(op.DEP_DELAY_NEW.mean(), o["delay_minutes"]["mean"]) and op.DEP_DELAY_NEW.median() == o["delay_minutes"]["median"],
          "mean and median delay minutes")
    car = op.groupby("OP_UNIQUE_CARRIER").DEP_DEL15.mean()
    check(all(close(car[c["key"]], c["p_delay"]) for c in R["by_carrier"]), "every airline's delay rate")
    top = op.ORIGIN.value_counts().head(6)
    check(list(top.index) == [a["code"] for a in R["busiest_airports"][:6]] and
          all(int(top[a["code"]]) == a["flights"] for a in R["busiest_airports"][:6]), "six busiest airports and their flight counts")

    # weather: re-implemented classification and join
    w = pd.read_csv(data_dir / "raw_data/airport_weather_2019.csv", low_memory=False)
    w["date"] = pd.to_datetime(w.DATE, format="mixed")
    has = lambda *c: w[list(c)].notna().any(axis=1)
    w["cond"] = np.select([w.PRCP.isna(), has("WT03", "WT05", "WT10"), w.SNOW.fillna(0) > 0, w.PRCP >= 0.01, has("WT01", "WT02")],
                          [None, "Stormy", "Snowy", "Rainy", "Foggy"], "Clear")
    a = pd.read_csv(data_dir / "raw_data/airports_list.csv")
    a["NAME"] = a.NAME.replace(render_html_name_fixes())
    a = a[~a.ORIGIN_AIRPORT_ID.isin([int(k) for k in R["validation"]["excluded_airports"]])]
    op = op.assign(date=pd.to_datetime(dict(year=2019, month=op.MONTH, day=op.DAY_OF_MONTH)))
    j = op.merge(a[["ORIGIN_AIRPORT_ID", "NAME"]], on="ORIGIN_AIRPORT_ID").merge(w[["NAME", "date", "cond", "AWND"]], on=["NAME", "date"])
    j = j[j.cond.notna()]
    check(len(j) == R["weather_join"]["flights"], f"joined flights {len(j):,}")
    g = j.groupby("cond").DEP_DEL15.agg(["size", "sum"])
    for c in R["weather_join"]["contingency"]:
        check(int(g.loc[c["condition"], "size"]) == c["total"] and int(g.loc[c["condition"], "sum"]) == c["delayed"],
              f'{c["condition"]}: {int(g.loc[c["condition"], "size"]):,} flights, {int(g.loc[c["condition"], "sum"]):,} delayed')
    jw = j[j.AWND.notna()]
    r = np.corrcoef(jw.AWND, jw.DEP_DELAY_NEW)[0, 1]
    check(close(r, R["weather_join"]["correlation_with_delay_minutes"]["AWND"]["r"], 1e-6), f"wind vs delay Pearson r = {r:.4f}")


def render_html_name_fixes():
    import process_data
    return process_data.NAME_FIXES


# ---------------------------------------------------------------- 3. page checks
BANNED = ["simulated", "Simulated", "Illustrative", "illustrative", "Demo mode", "demo data", "build_html.R", "km/h",
          "Load your cleaned CSV", "No dataset loaded"]


def formatted_values(R):
    """Every way a number from results.json may legitimately be printed."""
    vals = set()

    def add(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, int):
            vals.update({str(x), f"{x:,}"})
        if isinstance(x, (int, float)):
            for d in range(0, 5):
                vals.add(f"{x:.{d}f}"); vals.add(f"{100 * x:.{d}f}")
            vals.add(f"{x:,.0f}")
            vals.add(f"{x:g}")

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            vals.update(re.findall(r"\d[\d,.]*", o))
        else:
            add(o)
    walk(R)
    # derived values the slides compute from results.json
    con = {c["condition"]: c for c in R["weather_join"]["contingency"]}
    rain, clear = con["Rainy"], con["Clear"]
    for k in ["delayed", "not_delayed", "total"]:
        add(rain[k] + clear[k])
    for c in con.values():
        add(c["not_delayed"] / c["total"]); add(round(c["delayed"] / c["total"], 4))
    add(1 - R["weather_join"]["share_of_operated"])
    add(max((c["ci95"][1] - c["ci95"][0]) / 2 for c in con.values()))
    add(round(R["weather_join"]["chi_square"]["statistic"]))
    return vals


# numbers that are part of the method, not results: threshold, CI level, year, code line numbers, R arguments,
# axis ticks, bin edges and the slide's "Unit III"/"Case 1" style labels
CONSTANTS = {"15", "95", "2019", "4", "19", "1", "2", "3", "5", "6", "7", "8", "9", "10", "11", "12", "0", "20", "0.05", "0.1",
             "0.15", "0.2", "0.25", "0.3", "0.35", "30", "60", "90", "120", "150", "180", "25", "0.001"}


def page_checks(R):
    print("index.html checks")
    page = (REPO / "index.html").read_text(encoding="utf-8")
    check(render_html.render(page, R) == page, "page is rendered from the current results.json")
    for b in BANNED:
        check(b not in page, f"no leftover text {b!r}")
    allowed = formatted_values(R) | CONSTANTS
    unknown = {}
    for name, body in re.findall(r"<!-- data:([\w-]+) -->\n(.*?)<!-- /data:\1 -->", page, re.S):
        text = html.unescape(re.sub(r"<[^>]+>", " ", body))
        for tok in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text):
            tok = tok.rstrip(",.")
            if tok not in allowed:
                unknown.setdefault(name, set()).add(tok)
    check(not unknown, "every number in the data regions traces to results.json" + (f" — untraced: {unknown}" if unknown else ""))
    check(len(re.findall(r'<section class="sec" id="s\d" data-i="\d"', page)) == 9, "nine sections present")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "data/results.json"))
    ap.add_argument("--recount", metavar="DATA_DIR", help="also recount from the raw CSVs in DATA_DIR")
    args = ap.parse_args()
    R = json.loads(Path(args.results).read_text(encoding="utf-8"))
    consistency(R)
    if args.recount:
        recount(R, Path(args.recount).expanduser())
    page_checks(R)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
