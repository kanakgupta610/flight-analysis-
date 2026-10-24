#!/usr/bin/env python3
"""Process the 2019 BTS on-time + NOAA GHCN-Daily data into a small results file.

Reads the raw Kaggle download ("2019 Airline Delays w/ Weather and Airport Detail")
in chunks, joins daily airport weather to each departure on (origin station, date)
only where the mapping is valid, and writes every number the presentation shows
to data/results.json.

    python3 scripts/process_data.py --data-dir ~/Downloads/archive

The raw CSVs never need to be inside the repository.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SEED = 2019
SCATTER_N = 500
DELAY_THRESHOLD = 15  # minutes, matches the BTS DEP_DEL15 definition

# airports_list.csv maps these airports to a weather station in a different city,
# so a day's weather there does not describe the departure airport. Excluded from the join.
EXCLUDED_AIRPORTS = {
    13871: "Omaha (Eppley) -> Grand Island, NE station",
    13487: "Midland/Odessa -> Port Arthur, TX station",
    14193: "Pensacola -> Tallahassee, FL station",
    14761: "Sanford, FL -> Tallahassee, FL station",
    13796: "Oakland -> Sacramento, CA station",
    13830: "Kahului (Maui) -> Honolulu (Oahu) station",
    15376: "Tucson -> Phoenix, AZ station",
    15370: "Tulsa -> Oklahoma City, OK station",
    13891: "Ontario, CA -> Los Angeles Intl station",
}
# Same airport, but the key in airports_list.csv is missing the "PR" of the weather file's NAME.
NAME_FIXES = {
    "SAN JUAN L M MARIN INTERNATIONAL AIRPORT, US": "SAN JUAN L M MARIN INTERNATIONAL AIRPORT, PR US",
}

CONDITIONS = ["Clear", "Foggy", "Rainy", "Snowy", "Stormy"]
WIND_BINS = [0, 5, 10, 15, 20, np.inf]
WIND_LABELS = ["0–5", "5–10", "10–15", "15–20", "20+"]
PRCP_BINS = [-0.001, 0.0, 0.1, 0.5, 1.0, np.inf]
PRCP_LABELS = ["0", "0.01–0.10", "0.11–0.50", "0.51–1.00", "> 1.00"]
SNOW_BINS = [-0.001, 0.0, 1.0, 3.0, np.inf]
SNOW_LABELS = ["0", "0.1–1.0", "1.1–3.0", "> 3.0"]
TMAX_BINS = [-np.inf, 32, 50, 70, 90, np.inf]
TMAX_LABELS = ["≤ 32", "33–50", "51–70", "71–90", "> 90"]
DELAY_BINS = [-0.5, 0.5, 14.5, 29.5, 59.5, 119.5, 179.5, np.inf]
DELAY_LABELS = ["0 (on time/early)", "1–14", "15–29", "30–59", "60–119", "120–179", "180+"]
CAUSES = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]

FLIGHT_COLS = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "ORIGIN",
               "DEP_DELAY_NEW", "DEP_DEL15", "DEP_TIME_BLK", "CANCELLED", "ARR_DELAY_NEW"] + CAUSES
WEATHER_COLS = ["STATION", "NAME", "DATE", "PRCP", "SNOW", "SNWD", "TMAX", "AWND", "WT01", "WT02", "WT03", "WT05", "WT10"]
# physically plausible ranges (inches, inches, inches, deg F, mph)
RANGES = {"PRCP": (0, 20), "SNOW": (0, 60), "SNWD": (0, 200), "TMAX": (-60, 135), "AWND": (0, 100)}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963985):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def chi2_sf(x, df):
    """Upper tail of the chi-square distribution (closed form for even df)."""
    if df % 2:
        raise ValueError("only even df supported")
    t, term, s = x / 2.0, 1.0, 1.0
    for i in range(1, df // 2):
        term *= t / i
        s += term
    # log form so huge statistics underflow gracefully to 0 instead of overflowing
    return math.exp(-t + math.log(s)) if t < 745 else 0.0


class Moments:
    """Streaming sums for Pearson correlation and least-squares line."""

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        self.n += len(x)
        self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum(); self.sxy += (x * y).sum()

    def result(self):
        n = self.n
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sxx - self.sx ** 2 / n
        vy = self.syy - self.sy ** 2 / n
        slope = cov / vx
        return {"n": n, "r": cov / math.sqrt(vx * vy), "slope": slope,
                "intercept": (self.sy - slope * self.sx) / n,
                "mean_x": self.sx / n, "mean_y": self.sy / n}


class GroupAcc:
    """Accumulates flights / delayed / delay-minute sums per key across chunks."""

    def __init__(self):
        self.parts = []

    def add(self, df, key):
        g = df.groupby(key, observed=True).agg(
            flights=("DEP_DEL15", "size"), delayed=("DEP_DEL15", "sum"), minutes=("DEP_DELAY_NEW", "sum"))
        self.parts.append(g)

    def frame(self):
        return pd.concat(self.parts).groupby(level=list(range(self.parts[0].index.nlevels)), observed=True).sum()


def summarise(g, total_flights=None):
    out = []
    for key, r in g.iterrows():
        n, k = int(r.flights), int(r.delayed)
        row = {"key": key if not isinstance(key, tuple) else list(key), "flights": n, "delayed": k,
               "p_delay": k / n, "ci95": wilson(k, n), "mean_delay_min": r.minutes / n}
        if total_flights:
            row["share_of_flights"] = n / total_flights
        out.append(row)
    return out


def quantile_from_hist(hist, q):
    c = np.cumsum(hist)
    return int(np.searchsorted(c, q * c[-1], side="left"))


# ---------------------------------------------------------------- weather
def load_weather(data_dir):
    w = pd.read_csv(data_dir / "raw_data/airport_weather_2019.csv", usecols=WEATHER_COLS, low_memory=False)
    report = {"rows": len(w), "stations": int(w.STATION.nunique())}
    # the file mixes M/D/YYYY and YYYY-MM-DD
    d1 = pd.to_datetime(w.DATE, format="%m/%d/%Y", errors="coerce")
    d2 = pd.to_datetime(w.DATE, format="%Y-%m-%d", errors="coerce")
    w["date"] = d1.fillna(d2)
    report["date_formats"] = {"M/D/YYYY": int(d1.notna().sum()), "YYYY-MM-DD": int(d2.notna().sum())}
    report["unparsed_dates"] = int(w.date.isna().sum())
    report["non_2019_dates"] = int((w.date.dt.year != 2019).sum())
    w = w[w.date.dt.year == 2019]
    report["date_range"] = [w.date.min().strftime("%Y-%m-%d"), w.date.max().strftime("%Y-%m-%d")]
    report["station_days_per_station"] = {"min": int(w.groupby("NAME").size().min()), "max": int(w.groupby("NAME").size().max())}
    report["duplicate_station_days"] = int(w.duplicated(["NAME", "date"]).sum())
    w = w.drop_duplicates(["NAME", "date"])
    report["out_of_range_set_missing"] = {}
    for c, (lo, hi) in RANGES.items():
        bad = w[c].notna() & ((w[c] < lo) | (w[c] > hi))
        report["out_of_range_set_missing"][c] = int(bad.sum())
        w.loc[bad, c] = np.nan
    report["missing_share"] = {c: float(w[c].isna().mean()) for c in RANGES}

    thunder = w.WT03.notna() | w.WT05.notna() | w.WT10.notna()
    fog = w.WT01.notna() | w.WT02.notna()
    cond = pd.Series(pd.NA, index=w.index, dtype="object")
    known = w.PRCP.notna()
    cond[known] = "Clear"
    cond[known & fog] = "Foggy"
    cond[known & (w.PRCP >= 0.01)] = "Rainy"
    cond[known & (w.SNOW > 0)] = "Snowy"
    # wet day at a station that did not report snowfall: rain and snow cannot be told apart, so leave it unclassified
    ambiguous = known & (w.PRCP >= 0.01) & w.SNOW.isna() & ~thunder
    cond[ambiguous] = pd.NA
    cond[known & thunder] = "Stormy"
    w["condition"] = pd.Categorical(cond, categories=CONDITIONS)
    report["station_days_rain_or_snow_unknown"] = int(ambiguous.sum())
    report["station_days_no_precip_reading"] = int((~known).sum())
    report["station_days_by_condition"] = {c: int((w.condition == c).sum()) for c in CONDITIONS}
    report["station_days_unclassified"] = int(w.condition.isna().sum())
    return w[["NAME", "date", "PRCP", "SNOW", "TMAX", "AWND", "condition"]], report


def load_station_map(data_dir, weather_names):
    a = pd.read_csv(data_dir / "raw_data/airports_list.csv")
    a["NAME"] = a.NAME.replace(NAME_FIXES)
    rows = []
    for r in a.itertuples():
        status = "excluded: " + EXCLUDED_AIRPORTS[r.ORIGIN_AIRPORT_ID] if r.ORIGIN_AIRPORT_ID in EXCLUDED_AIRPORTS \
            else ("joined" if r.NAME in weather_names else "excluded: station missing from weather file")
        rows.append({"airport_id": int(r.ORIGIN_AIRPORT_ID), "airport": r.DISPLAY_AIRPORT_NAME,
                     "city": r.ORIGIN_CITY_NAME, "station": r.NAME, "status": status})
    ok = a[[s["status"] == "joined" for s in rows]]
    return dict(zip(ok.ORIGIN_AIRPORT_ID, ok.NAME)), rows


# ---------------------------------------------------------------- full_data cross-check
def crosscheck_full_data(data_dir, chunksize):
    path = data_dir / "full_data_flightdelay.csv"
    if not path.exists():
        return None
    n, k, by_month = 0, 0, []
    for ch in pd.read_csv(path, usecols=["MONTH", "DEP_DEL15"], chunksize=chunksize):
        n += len(ch); k += int(ch.DEP_DEL15.sum())
        by_month.append(ch.groupby("MONTH").DEP_DEL15.agg(["size", "sum"]))
    m = pd.concat(by_month).groupby(level=0).sum()
    return {"rows": n, "p_delay": k / n,
            "by_month": {int(i): {"flights": int(r["size"]), "p_delay": r["sum"] / r["size"]} for i, r in m.iterrows()}}


# ---------------------------------------------------------------- main pass
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("FLIGHT_DATA_DIR", "~/Downloads/archive"))
    ap.add_argument("--out", default=str(REPO / "data/results.json"))
    ap.add_argument("--chunksize", type=int, default=400_000)
    ap.add_argument("--skip-crosscheck", action="store_true", help="skip the full_data_flightdelay.csv pass")
    args = ap.parse_args()
    data_dir = Path(args.data_dir).expanduser()
    files = [data_dir / f"raw_data/ONTIME_REPORTING_{m:02d}.csv" for m in range(1, 13)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing input files: {missing}")

    weather, weather_report = load_weather(data_dir)
    station_of, mapping = load_station_map(data_dir, set(weather.NAME))
    log(f"weather rows {len(weather):,}; airports joinable {len(station_of)}")
    weather = weather.rename(columns={"NAME": "station"})
    carrier_names = (pd.read_csv(data_dir / "raw_data/T3_AIR_CARRIER_SUMMARY_AIRPORT_ACTIVITY_2019.csv",
                                 usecols=["OP_UNIQUE_CARRIER", "CARRIER_NAME"])
                     .drop_duplicates("OP_UNIQUE_CARRIER").set_index("OP_UNIQUE_CARRIER").CARRIER_NAME.to_dict())

    rng = np.random.default_rng(SEED)
    totals = dict(rows=0, bad_dates=0, weekday_mismatch=0, cancelled=0, operated=0, missing_status=0, joined=0,
                  no_station=0, no_weather_row=0, matched=0, unclassified_weather=0, no_precip_reading=0,
                  rain_or_snow_unknown=0)
    acc = {k: GroupAcc() for k in ["month", "carrier", "airport", "block", "condition", "wind", "prcp", "snow",
                                   "tmax"]}
    hist_all = np.zeros(3000, dtype=np.int64)
    hist_delayed_by_cond = {c: np.zeros(3000, dtype=np.int64) for c in CONDITIONS}
    hist_bins = np.zeros(len(DELAY_LABELS), dtype=np.int64)
    cause_minutes = dict.fromkeys(CAUSES, 0.0)
    cause_flights = 0
    mom = {"AWND": Moments(), "PRCP": Moments(), "TMAX": Moments(), "SNOW": Moments()}
    sample = []
    airport_names = {}
    missing = dict.fromkeys(FLIGHT_COLS, 0)
    missing_operated = dict.fromkeys(FLIGHT_COLS, 0)
    date_min = date_max = None

    for f in files:
        log(f"reading {f.name}")
        for ch in pd.read_csv(f, usecols=FLIGHT_COLS, chunksize=args.chunksize,
                              dtype={"OP_UNIQUE_CARRIER": "category", "ORIGIN": "category", "DEP_TIME_BLK": "category"}):
            totals["rows"] += len(ch)
            for c, k in ch.isna().sum().items():
                missing[c] += int(k)
            for c, k in ch[ch.CANCELLED == 0].isna().sum().items():
                missing_operated[c] += int(k)
            date = pd.to_datetime(dict(year=2019, month=ch.MONTH, day=ch.DAY_OF_MONTH), errors="coerce")
            bad = date.isna()
            wk = ~bad & ((date.dt.dayofweek + 1) != ch.DAY_OF_WEEK)
            totals["bad_dates"] += int(bad.sum()); totals["weekday_mismatch"] += int(wk.sum())
            ch = ch.assign(date=date)[~bad & ~wk]
            lo, hi = ch.date.min(), ch.date.max()
            date_min = lo if date_min is None else min(date_min, lo)
            date_max = hi if date_max is None else max(date_max, hi)

            totals["cancelled"] += int((ch.CANCELLED == 1).sum())
            op = ch[ch.CANCELLED == 0]
            totals["missing_status"] += int(op.DEP_DEL15.isna().sum())
            op = op[op.DEP_DEL15.notna() & op.DEP_DELAY_NEW.notna()].copy()
            totals["operated"] += len(op)
            op["DEP_DEL15"] = op.DEP_DEL15.astype(np.int8)
            for code, name in op[["ORIGIN_AIRPORT_ID", "ORIGIN"]].drop_duplicates().itertuples(index=False):
                airport_names.setdefault(int(code), name)

            acc["month"].add(op, "MONTH")
            acc["carrier"].add(op, "OP_UNIQUE_CARRIER")
            acc["airport"].add(op, "ORIGIN_AIRPORT_ID")
            acc["block"].add(op, "DEP_TIME_BLK")
            mins = op.DEP_DELAY_NEW.clip(upper=len(hist_all) - 1).astype(np.int64).to_numpy()
            hist_all += np.bincount(mins, minlength=len(hist_all))
            hist_bins += np.histogram(op.DEP_DELAY_NEW, bins=DELAY_BINS)[0]
            # causes are reported (in minutes) only for flights arriving >= 15 min late
            withcause = op[op.CARRIER_DELAY.notna()]
            cause_flights += len(withcause)
            for c in CAUSES:
                cause_minutes[c] += float(withcause[c].sum())

            # ---- weather join: origin airport -> validated station, then (station, date)
            op["station"] = op.ORIGIN_AIRPORT_ID.map(station_of)
            totals["no_station"] += int(op.station.isna().sum())
            j = op[op.station.notna()].merge(weather, on=["station", "date"], how="left", indicator=True)
            totals["no_weather_row"] += int((j._merge == "left_only").sum())
            j = j[j._merge == "both"]
            totals["matched"] += len(j)
            unc = j.condition.isna()
            totals["unclassified_weather"] += int(unc.sum())
            totals["no_precip_reading"] += int((unc & j.PRCP.isna()).sum())
            totals["rain_or_snow_unknown"] += int((unc & j.PRCP.notna()).sum())
            # condition probabilities use classified days; the measurement analyses below use every matched flight
            jc = j[~unc]
            totals["joined"] += len(jc)

            acc["condition"].add(jc, "condition")
            for c in CONDITIONS:
                sel = jc.loc[(jc.condition == c) & (jc.DEP_DEL15 == 1), "DEP_DELAY_NEW"]
                hist_delayed_by_cond[c] += np.bincount(sel.clip(upper=2999).astype(np.int64), minlength=3000)
            for col, bins, labels, name in [("AWND", WIND_BINS, WIND_LABELS, "wind"), ("PRCP", PRCP_BINS, PRCP_LABELS, "prcp"),
                                            ("SNOW", SNOW_BINS, SNOW_LABELS, "snow"), ("TMAX", TMAX_BINS, TMAX_LABELS, "tmax")]:
                sub = j[j[col].notna()]
                acc[name].add(sub.assign(bin=pd.cut(sub[col], bins, labels=labels, right=True, include_lowest=True)), "bin")
                mom[col].add(sub[col].to_numpy(), sub.DEP_DELAY_NEW.to_numpy())

            # reproducible uniform sample of matched flights for the scatter plot / sample rows
            js = j[j.AWND.notna()]
            keys = rng.random(len(js))
            keep = np.argsort(keys)[:SCATTER_N]
            s = js.iloc[keep][["MONTH", "DAY_OF_MONTH", "ORIGIN", "OP_UNIQUE_CARRIER", "condition", "AWND", "PRCP",
                              "TMAX", "DEP_DELAY_NEW", "DEP_DEL15"]].assign(_k=keys[keep])
            sample.append(s)
            sample = [pd.concat(sample).nsmallest(SCATTER_N, "_k")]

    log("aggregating")
    op_n = totals["operated"]
    delayed_n = int(acc["month"].frame().delayed.sum())
    q = lambda p: quantile_from_hist(hist_all, p)
    hist_delayed = hist_all.copy(); hist_delayed[:DELAY_THRESHOLD] = 0
    qd = lambda p: quantile_from_hist(hist_delayed, p)
    total_minutes = float(acc["month"].frame().minutes.sum())

    cond = acc["condition"].frame().reindex(CONDITIONS)
    joined_n = int(cond.flights.sum()); joined_delayed = int(cond.delayed.sum())
    contingency = []
    chi2 = 0.0
    for c, r in cond.iterrows():
        row = {"condition": c, "delayed": int(r.delayed), "not_delayed": int(r.flights - r.delayed), "total": int(r.flights),
               "p_joint_delayed": r.delayed / joined_n, "p_condition": r.flights / joined_n,
               "p_delay_given": r.delayed / r.flights, "ci95": wilson(int(r.delayed), int(r.flights)),
               "mean_delay_min": r.minutes / r.flights,
               "median_delay_if_delayed": quantile_from_hist(hist_delayed_by_cond[c], 0.5)}
        for obs, col_tot in [(r.delayed, joined_delayed), (r.flights - r.delayed, joined_n - joined_delayed)]:
            exp = r.flights * col_tot / joined_n
            chi2 += (obs - exp) ** 2 / exp
        contingency.append(row)
    df_chi = (len(CONDITIONS) - 1) * 1
    by_c = {r["condition"]: r for r in contingency}

    carriers = summarise(acc["carrier"].frame(), op_n)
    for r in carriers:
        r["name"] = carrier_names.get(r["key"], r["key"])
    airports = summarise(acc["airport"].frame(), op_n)
    for r in airports:
        r["code"] = airport_names.get(int(r["key"]))

    sample = sample[0].sort_values("_k").drop(columns="_k")
    results = {
        "meta": {
            "title": "2019 US domestic departures (BTS) joined to NOAA GHCN-Daily airport weather",
            "source": "Kaggle: 2019 Airline Delays w/ Weather and Airport Detail (BTS On-Time Reporting + NOAA GHCN-Daily)",
            "period": "January–December 2019",
            "delay_definition": f"departure delay ≥ {DELAY_THRESHOLD} min (BTS DEP_DEL15 = 1)",
            "weather_resolution": "daily, at the origin airport's weather station",
            "units": {"AWND": "mph (average daily wind speed)", "PRCP": "inches", "SNOW": "inches", "TMAX": "°F"},
            "condition_rules": {
                "Stormy": "thunder (WT03), hail (WT05) or tornado (WT10) recorded that day",
                "Snowy": "snowfall > 0 in, no thunder",
                "Rainy": "precipitation ≥ 0.01 in, snowfall reported as 0, no thunder",
                "Foggy": "fog (WT01/WT02) recorded, no precipitation or thunder",
                "Clear": "none of the above (dry day, no fog or thunder)",
                "unclassified": "no precipitation reading, or a wet day at a station that did not report snowfall"},
            "seed": SEED,
        },
        "validation": {"flights": totals, "flight_dates": {"first": date_min.strftime("%Y-%m-%d"), "last": date_max.strftime("%Y-%m-%d"),
                                                           "days": int((date_max - date_min).days) + 1},
                       "flight_missing_values": {"all_rows": missing, "operated_rows": missing_operated},
                       "weather": weather_report, "station_mapping": mapping,
                       "airports_joined": len(station_of), "excluded_airports": EXCLUDED_AIRPORTS},
        "overall": {
            "scheduled": totals["rows"] - totals["bad_dates"] - totals["weekday_mismatch"],
            "cancelled": totals["cancelled"], "operated": op_n, "delayed": delayed_n, "p_delay": delayed_n / op_n,
            "ci95": wilson(delayed_n, op_n), "p_cancel": totals["cancelled"] / (totals["rows"] - totals["bad_dates"] - totals["weekday_mismatch"]),
            "airports": len(airport_names), "carriers": len(carriers),
            "delay_minutes": {"mean": total_minutes / op_n, "median": q(0.5), "p75": q(0.75), "p90": q(0.9), "p99": q(0.99),
                              "max": int(np.nonzero(hist_all)[0].max()),
                              "mean_if_delayed": float((hist_delayed * np.arange(len(hist_delayed))).sum() / hist_delayed.sum()),
                              "median_if_delayed": qd(0.5)},
            "delay_distribution": [{"bin": l, "flights": int(n), "share": n / op_n} for l, n in zip(DELAY_LABELS, hist_bins)],
            "causes": {"flights_with_cause": cause_flights,
                       "minutes": {c: cause_minutes[c] for c in CAUSES},
                       "share": {c: cause_minutes[c] / sum(cause_minutes.values()) for c in CAUSES}},
        },
        "weather_join": {
            "matched": totals["matched"], "share_matched": totals["matched"] / op_n,
            "flights": joined_n, "share_of_operated": joined_n / op_n, "delayed": joined_delayed,
            "p_delay": joined_delayed / joined_n, "contingency": contingency,
            "chi_square": {"statistic": chi2, "df": df_chi, "p_value": chi2_sf(chi2, df_chi)},
            "rainy_vs_clear": {"difference_pp": 100 * (by_c["Rainy"]["p_delay_given"] - by_c["Clear"]["p_delay_given"]),
                               "risk_ratio": by_c["Rainy"]["p_delay_given"] / by_c["Clear"]["p_delay_given"]},
            "correlation_with_delay_minutes": {k: m.result() for k, m in mom.items()},
            "by_wind": summarise(acc["wind"].frame().reindex(WIND_LABELS)),
            "by_precip": summarise(acc["prcp"].frame().reindex(PRCP_LABELS)),
            "by_snow": summarise(acc["snow"].frame().reindex(SNOW_LABELS)),
            "by_tmax": summarise(acc["tmax"].frame().reindex(TMAX_LABELS)),
        },
        "by_month": summarise(acc["month"].frame(), op_n),
        "by_time_block": summarise(acc["block"].frame(), op_n),
        "by_carrier": sorted(carriers, key=lambda r: -r["p_delay"]),
        "busiest_airports": sorted(airports, key=lambda r: -r["flights"])[:15],
        "scatter_sample": [{"month": int(r.MONTH), "day": int(r.DAY_OF_MONTH), "origin": r.ORIGIN,
                            "carrier": r.OP_UNIQUE_CARRIER, "condition": None if pd.isna(r.condition) else r.condition, "awnd": float(r.AWND),
                            "prcp": float(r.PRCP), "tmax": None if pd.isna(r.TMAX) else float(r.TMAX),
                            "delay_min": float(r.DEP_DELAY_NEW), "delayed": int(r.DEP_DEL15)}
                           for r in sample.itertuples()],
    }
    if not args.skip_crosscheck:
        log("cross-checking full_data_flightdelay.csv")
        results["validation"]["full_data_crosscheck"] = crosscheck_full_data(data_dir, 1_000_000)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    log(json.dumps({"operated": op_n, "p_delay": round(delayed_n / op_n, 4), "joined": joined_n,
                    "p_delay_given": {r["condition"]: round(r["p_delay_given"], 4) for r in contingency}}))


if __name__ == "__main__":
    main()
