"""V2-Session (Vs) features: intraday hourly features for session-anchored forecasting.

Input: hourly OHLC bars for GC=F + daily macro from raw_data_yahoo_refreshed.csv.
Output: feature DataFrame indexed by hourly timestamp with session identity + intraday
momentum + intraday vol + macro daily z-scores (forward-filled from previous close).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# UTC hours for session opens (pre-declared)
ASIA_OPEN_HOUR = 23  # Tokyo cash open ~23:00 UTC
LONDON_OPEN_HOUR = 7  # London ~07:00 UTC
NY_OPEN_HOUR = 13  # NY ~13:00 UTC (pit + electronic settle)
SESSION_HOURS = {ASIA_OPEN_HOUR: "ASIA", LONDON_OPEN_HOUR: "LONDON", NY_OPEN_HOUR: "NY"}


def session_of(hour: int) -> str:
    """Which session is this hour inside? Rough allocation."""
    if 23 <= hour or hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    return "NY"


def load_hourly_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.floor("h")
    return df.set_index("timestamp_utc").sort_index()


def build_intraday_features(bars: pd.DataFrame, macro_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday features on hourly bars, merged with daily macro."""
    b = bars.copy()
    close = b["close"].astype(float)
    b["ret_1h"] = close.pct_change()
    b["ret_4h"] = close.pct_change(4)
    b["ret_8h"] = close.pct_change(8)
    b["ret_24h"] = close.pct_change(24)
    b["ret_72h"] = close.pct_change(72)
    b["ret_168h"] = close.pct_change(168)  # 1 week
    # Realized vol
    b["vol_24h"] = close.pct_change().rolling(24).std()
    b["vol_72h"] = close.pct_change().rolling(72).std()
    # Range
    b["hl_range_pct_24h"] = (b["high"].rolling(24).max() / b["low"].rolling(24).min() - 1)
    # EMA momentum
    b["ema_short"] = close.ewm(span=24, adjust=False).mean()
    b["ema_long"] = close.ewm(span=120, adjust=False).mean()
    b["momentum_up"] = (b["ema_short"] > b["ema_long"]).astype(int)
    # Hour + session
    b["hour"] = b.index.hour
    b["session"] = b["hour"].apply(session_of)
    b["is_asia_open"] = (b["hour"] == ASIA_OPEN_HOUR).astype(int)
    b["is_london_open"] = (b["hour"] == LONDON_OPEN_HOUR).astype(int)
    b["is_ny_open"] = (b["hour"] == NY_OPEN_HOUR).astype(int)
    # Previous session close (last close of previous 8-hour window at each session open)
    b["prev_session_close"] = close.shift(8)
    b["current_vs_prev_session_close_pct"] = (close / b["prev_session_close"] - 1) * 100

    # Merge daily macro (forward-filled)
    macro = macro_daily.copy()
    macro.index = pd.to_datetime(macro.index).tz_localize("UTC") if macro.index.tz is None else macro.index
    macro_hourly = macro.reindex(b.index, method="ffill")
    for col in macro.columns:
        b[f"macro_{col}"] = macro_hourly[col]

    return b.dropna(subset=["ret_24h", "vol_24h", "prev_session_close"])


def load_macro_daily(raw_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, parse_dates=["date"]).set_index("date").sort_index()
    close = raw["gold_close"].astype(float)
    dxy = raw["dxy"].astype(float)
    vix = raw["vix"].astype(float) if "vix" in raw.columns else pd.Series(np.nan, index=raw.index)
    macro = pd.DataFrame(index=raw.index)
    # Rolling z-scores over 1 year (~252 days)
    for col_name, series in [("dxy_z252", dxy), ("vix_z252", vix), ("gold_z252", close)]:
        mu = series.rolling(252, min_periods=63).mean()
        sd = series.rolling(252, min_periods=63).std()
        macro[col_name] = (series - mu) / sd
    macro["gold_ret_63d"] = close / close.shift(63) - 1
    macro["gold_ret_20d"] = close / close.shift(20) - 1
    macro["gold_ret_5d"] = close / close.shift(5) - 1
    if "DFII10" in raw.columns:
        ry = raw["DFII10"].astype(float)
        macro["real_yield_z252"] = (ry - ry.rolling(252, min_periods=63).mean()) / ry.rolling(252, min_periods=63).std()
    return macro.dropna(how="all")


def build_targets(bars: pd.DataFrame, horizon_hours: int = 8) -> pd.DataFrame:
    """Target: forward horizon-hour return classification."""
    close = bars["close"].astype(float)
    fwd_ret = close.shift(-horizon_hours) / close - 1
    thr = 0.0015  # 15 bps threshold
    cls = np.where(fwd_ret > thr, 2, np.where(fwd_ret < -thr, 0, 1))
    return pd.DataFrame({
        "fwd_ret": fwd_ret,
        "cls": cls,
    }, index=bars.index)


def session_anchor_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """Filter to hourly rows at each session open."""
    return feat[feat["hour"].isin(SESSION_HOURS.keys())].copy()
