# Flight Delay Analysis — Based on Weather Conditions

A nine-section presentation (`index.html`) on how weather relates to US flight delays, using real 2019 data:
BTS On-Time Reporting for every domestic departure, joined to NOAA GHCN-Daily airport weather
(Kaggle: *2019 Airline Delays w/ Weather and Airport Detail*).

## Rebuilding the numbers

The raw dataset (~4.8 GB) stays outside the repository and is never committed (see `.gitignore`).

```bash
python3 scripts/process_data.py --data-dir ~/Downloads/archive   # raw CSVs -> data/results.json (~15 s)
python3 scripts/render_html.py                                   # data/results.json -> index.html
python3 scripts/validate.py --recount ~/Downloads/archive        # consistency, independent recount, page checks
```

Requires Python 3 with pandas, numpy and pyarrow.

- `process_data.py` reads the twelve `raw_data/ONTIME_REPORTING_MM.csv` files in chunks, validates dates against the
  2019 calendar, and joins each departure to its origin airport's weather on (station, date). Airports whose listed
  weather station is in another city are excluded from the join (listed in `EXCLUDED_AIRPORTS`).
  `full_data_flightdelay.csv` is used only as a cross-check.
- `render_html.py` rewrites only the regions between `<!-- data:NAME -->` and `<!-- /data:NAME -->` markers;
  the rest of the page is left as designed.
- `validate.py` fails if any shown number cannot be traced to `data/results.json`.

## Definitions

- **Delayed**: departure delay ≥ 15 minutes (`DEP_DEL15 = 1`); cancelled flights are excluded from delay rates.
- **Weather condition** (daily, origin station; first match wins): Stormy = thunder, hail or tornado flag ·
  Snowy = snowfall > 0 · Rainy = precipitation ≥ 0.01 in with snowfall reported as 0 · Foggy = fog flag ·
  Clear = none of these. Wet days at stations that do not report snowfall (rain vs snow unknown) and days without a
  precipitation reading are left unclassified; they still count in the wind, precipitation and temperature analyses.
- Units: wind mph (average daily), precipitation and snowfall in inches, temperature °F.
