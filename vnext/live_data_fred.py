"""Fetch FRED macro series into raw_data.csv using the local API key.

Key location: C:\\dev\\FarACtionRadar\\v16build\\secrets\\fred.key (plaintext, one line).
Adds/refreshes: DFII10 (10y TIPS real yield), DGS10, DGS2, DGS5, T10YIE, T5YIFR.
Merges into raw_data_yahoo_refreshed.csv (creates if missing).
"""
from __future__ import annotations
import argparse
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "secrets" / "fred.key"
RAW_YAHOO = ROOT / "gold_core" / "data" / "raw_data_yahoo_refreshed.csv"
STATUS = ROOT / "reports" / "vnext_shadow" / "DATA_REFRESH_STATUS.json"

FRED_SERIES = ["DFII10", "DGS10", "DGS2", "DGS5", "T10YIE", "T5YIFR"]


def get_api_key() -> str:
    import os
    env_key = os.environ.get("FRED_API_KEY", "").strip()
    if env_key:
        return env_key
    if not KEY_FILE.exists():
        raise RuntimeError(f"FRED API key not in $FRED_API_KEY and file not found: {KEY_FILE}")
    return KEY_FILE.read_text(encoding="utf-8").strip()


def fetch_series(series_id: str, api_key: str, start: str = "2008-01-01") -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    obs = data.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].rename(columns={"value": series_id})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = get_api_key()

    if not RAW_YAHOO.exists():
        raise RuntimeError(f"Base file missing: {RAW_YAHOO}. Run vnext.live_data_yahoo first.")

    existing = pd.read_csv(RAW_YAHOO, parse_dates=["date"])
    combined = existing.set_index("date").sort_index()
    fetched = {}

    for series_id in FRED_SERIES:
        try:
            df = fetch_series(series_id, api_key, start=args.start)
            if df.empty:
                fetched[series_id] = {"error": "empty response"}
                continue
            df = df.set_index("date")
            # Overlay: use FRED value everywhere it exists, keep existing elsewhere
            aligned = df[series_id].reindex(combined.index.union(df.index))
            if series_id in combined.columns:
                combined = combined.reindex(combined.index.union(df.index))
                mask_missing = combined[series_id].isna()
                combined.loc[mask_missing, series_id] = aligned[mask_missing]
                # Also refresh recent dates unconditionally (past 60 days) since FRED is authoritative
                recent = combined.index >= (combined.index.max() - pd.Timedelta(days=60))
                combined.loc[recent, series_id] = aligned[recent].values
            else:
                combined[series_id] = aligned
            fetched[series_id] = {
                "rows": int(len(df)),
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "last_value": float(df[series_id].dropna().iloc[-1]) if not df.dropna().empty else None,
            }
        except Exception as exc:
            fetched[series_id] = {"error": str(exc)}

    combined = combined.sort_index()

    status_prev = {}
    if STATUS.exists():
        try:
            status_prev = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            pass
    status = {**status_prev,
              "fred_refreshed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "fred_fetched": fetched}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")

    for series_id, info in fetched.items():
        print(f"  {series_id}: {info}")

    if not args.dry_run:
        combined.reset_index().to_csv(RAW_YAHOO, index=False)
        print(f"Updated {RAW_YAHOO} ({len(combined)} rows)")
    else:
        print("Dry-run; file unchanged")


if __name__ == "__main__":
    main()
