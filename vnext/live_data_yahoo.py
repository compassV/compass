"""Fetch daily bars from Yahoo for gold + macro proxies and merge into raw_data.csv.

Yahoo-only refresh. Features Yahoo doesn't cover directly (DFII10 real yield, T10YIE
breakeven inflation) are left at their last observed values from raw_data.csv, with a
gap-warning flag written to `reports/vnext_shadow/DATA_REFRESH_STATUS.json`.

Never touches Supabase, publisher, or website payloads.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "gold_core" / "data" / "raw_data.csv"
STATUS = ROOT / "reports" / "vnext_shadow" / "DATA_REFRESH_STATUS.json"

YAHOO_MAP = {
    "GC=F":   ("gold_open", "gold_high", "gold_low", "gold_close", "gold_volume"),
    "^GSPC":  (None, None, None, "spx", None),
    "^VIX":   (None, None, None, "vix", None),
    "DX-Y.NYB": (None, None, None, "dxy", None),
    "SLV":    (None, None, None, "silver", None),
    "^TNX":   (None, None, None, "tnx_nominal_10y", None),
    "^FVX":   (None, None, None, "tnx_nominal_5y", None),
}
FRED_STALE = ("DFII10", "DGS10", "DGS2", "DGS5", "T10YIE", "T5YIFR", "etf_flow_proxy",
              "is_fomc_day", "is_cpi_day")


def fetch_ticker(tkr: str, start: str) -> pd.DataFrame:
    df = yf.Ticker(tkr).history(start=start, interval="1d", auto_adjust=False, actions=False)
    if df.empty:
        raise RuntimeError(f"Yahoo empty for {tkr}")
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.floor("D")
    return df[["date", "Open", "High", "Low", "Close", "Volume"]]


def merge_yahoo_features(existing: pd.DataFrame, start_date: str) -> tuple[pd.DataFrame, dict]:
    combined = existing.set_index("date").sort_index()
    fetched = {}
    per_ticker = {}
    for tkr, mapping in YAHOO_MAP.items():
        try:
            df = fetch_ticker(tkr, start_date)
            df = df.set_index("date")
            per_ticker[tkr] = (df, mapping)
            fetched[tkr] = {"rows": int(len(df)), "start": str(df.index.min().date()), "end": str(df.index.max().date())}
        except Exception as exc:
            fetched[tkr] = {"error": str(exc)}

    # Determine union of all dates we've seen (existing + all fetched)
    all_dates = set(combined.index)
    for df, _ in per_ticker.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    combined = combined.reindex(all_dates)

    # Overlay each Yahoo series onto the corresponding raw_data column, only where existing is NaN
    for tkr, (df, mapping) in per_ticker.items():
        for src_col, dst_col in zip(("Open", "High", "Low", "Close", "Volume"), mapping):
            if dst_col is None:
                continue
            src = df[src_col].reindex(all_dates)
            if dst_col not in combined.columns:
                combined[dst_col] = src
            else:
                mask = combined[dst_col].isna()
                combined.loc[mask, dst_col] = src[mask]

    combined.index.name = "date"
    return combined.reset_index(), fetched


def carry_forward_stale(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    filled = []
    for col in FRED_STALE:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].ffill()
            filled.append(col)
    return df, filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    existing = pd.read_csv(RAW, parse_dates=["date"])
    last_date = existing["date"].max()
    print(f"Current raw_data.csv: {len(existing)} rows, last {last_date.date()}")

    start = (last_date - pd.Timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")
    updated, fetched = merge_yahoo_features(existing, start)
    updated, filled = carry_forward_stale(updated)

    new_rows_count = int((updated["date"] > last_date).sum())
    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prev_last_date": str(last_date.date()),
        "new_last_date": str(updated["date"].max().date()),
        "new_rows_appended": new_rows_count,
        "yahoo_fetched": fetched,
        "fred_stale_carried_forward": filled,
        "note": ("FRED features are not refreshed here (network unreachable from this box). Real-yield / "
                 "breakeven-inflation / calendar-event features carry forward last observed value. Vg "
                 "regime detector uses these; treat forecasts on days with stale FRED features accordingly."),
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"New Yahoo rows appended: {new_rows_count}")
    print(f"FRED-stale columns carried forward: {filled}")

    yahoo_out = ROOT / "gold_core" / "data" / "raw_data_yahoo_refreshed.csv"
    if not args.dry_run and new_rows_count > 0:
        updated.to_csv(yahoo_out, index=False)
        print(f"Wrote refreshed dataset to {yahoo_out} ({len(updated)} rows).")
        print("NOTE: raw_data.csv NOT overwritten. Vg pipeline uses raw_data_yahoo_refreshed.csv preferentially if present.")
    else:
        print("Dry-run or no new rows; no output file written")


if __name__ == "__main__":
    main()
