"""V2-Session live publisher: produces the current session-by-session forecast JSON.

Reads latest hourly bars (Yahoo), refits the intraday classifier, applies regime detector,
emits 3 session snapshots (Asia/London/NY) with prices, signals, regime state.

Publishes to web/public/data/gold_v2.json (this replaces the daily Vg output for launch).
Never touches Supabase / broker / trading paths.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from vnext.session_engine.features import (
    load_hourly_bars, load_macro_daily, build_intraday_features,
    build_targets, SESSION_HOURS,
)
from vnext.session_engine.regime import annotate_session_regime

ROOT = Path(__file__).resolve().parents[2]
BARS = ROOT / "reports" / "vnext_data" / "GC_F_1H_2024-09-13_TO_2026-09-12.csv"
RAW = ROOT / "gold_core" / "data" / "raw_data_yahoo_refreshed.csv"
DRIFT_FLAG = ROOT / "reports" / "vnext_shadow" / "DRIFT_KILL_FLAG.json"
OUT_JSON = ROOT / "data" / "gold_v2.json"
TRACK_RECORD_JSON = ROOT / "data" / "gold_v2_track_record.json"

CLASS_MAP = {0: "BEAR", 1: "FLAT", 2: "BULL"}
FEATURE_COLS = [
    "ret_1h", "ret_4h", "ret_8h", "ret_24h", "ret_72h", "ret_168h",
    "vol_24h", "vol_72h", "hl_range_pct_24h", "momentum_up",
    "current_vs_prev_session_close_pct", "hour",
    "macro_dxy_z252", "macro_vix_z252", "macro_gold_z252",
    "macro_gold_ret_63d", "macro_gold_ret_20d", "macro_gold_ret_5d",
    "macro_real_yield_z252",
]
HORIZON_HOURS = 8


def refresh_hourly_bars() -> pd.DataFrame:
    """Refetch hourly bars from Yahoo (max 730 days)."""
    tk = yf.Ticker("GC=F")
    df = tk.history(period="730d", interval="60m", auto_adjust=True, actions=False, prepost=True)
    df = df.reset_index()
    df["timestamp_utc"] = pd.to_datetime(df["Datetime"], utc=True).dt.floor("h")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.drop(columns=[c for c in df.columns if c not in ("timestamp_utc", "open", "high", "low", "close", "volume")])
    df = df.set_index("timestamp_utc").sort_index()
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=30, random_state=42,
        )),
    ])


def fit_and_predict(feat: pd.DataFrame, target: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    available = [c for c in FEATURE_COLS if c in feat.columns]
    frame = feat[available].join(target[["cls"]]).dropna(subset=["cls"] + ["ret_168h", "macro_dxy_z252"])
    with contextlib.redirect_stdout(io.StringIO()):
        pipe = build_pipeline()
        pipe.fit(frame[available].values, frame["cls"].astype(int).values)
    return pipe, available


def session_snapshot(feat: pd.DataFrame, pipe: Pipeline, features: list[str], session_hour: int, session_name: str) -> dict:
    # Find most recent bar at this session hour (within last 3 days)
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=3)
    session_rows = feat[(feat["hour"] == session_hour) & (feat.index >= cutoff)]
    if session_rows.empty:
        return {"session": session_name, "state": "NO_RECENT_DATA"}
    latest = session_rows.iloc[-1:]
    probs = pipe.predict_proba(latest[features].values)[0]
    pred_class = int(probs.argmax())
    direction = CLASS_MAP[pred_class]
    regime_on = bool(latest["session_regime_on"].iloc[0])
    emit = direction in {"BULL", "BEAR"} and regime_on

    # Previous session close = close from 8 bars earlier (roughly previous session close)
    ts = latest.index[0]
    prev_ts = ts - pd.Timedelta(hours=8)
    prev_close = float(feat.loc[prev_ts, "close"]) if prev_ts in feat.index else None
    current_price = float(latest["close"].iloc[0])
    current_open = float(latest["open"].iloc[0])

    return {
        "session": session_name,
        "session_open_utc": ts.isoformat().replace("+00:00", "Z"),
        "session_open_price": current_open,
        "current_price": current_price,
        "previous_session_close": prev_close,
        "move_vs_previous_close_pct": (current_price / prev_close - 1) * 100 if prev_close else None,
        "signal": direction if emit else "NO_SIGNAL",
        "raw_direction": direction,
        "prob_bull": float(probs[2]),
        "prob_bear": float(probs[0]),
        "prob_flat": float(probs[1]),
        "regime_on": regime_on,
        "regime_signals": {
            "trend_168h_pct": float(latest["ret_168h"].iloc[0]) * 100,
            "momentum_up": bool(int(latest["momentum_up"].iloc[0])),
            "dxy_z252": float(latest["macro_dxy_z252"].iloc[0]) if not pd.isna(latest["macro_dxy_z252"].iloc[0]) else None,
            "real_yield_z252": float(latest["macro_real_yield_z252"].iloc[0]) if "macro_real_yield_z252" in latest.columns and not pd.isna(latest["macro_real_yield_z252"].iloc[0]) else None,
        },
        "forecast_horizon_hours": HORIZON_HOURS,
    }


def track_record_summary() -> dict:
    sim_path = ROOT / "reports" / "VNEXT_VS_SESSION_SIMULATION_2026-09-13.json"
    if not sim_path.exists():
        return {"available": False}
    d = json.loads(sim_path.read_text(encoding="utf-8"))
    vs = d["vs_full"]
    return {
        "available": True,
        "methodology": (
            "V2-Session (Vs) walk-forward on Yahoo GC=F hourly bars covering 2024-11 to 2026-09. "
            "HistGradientBoostingClassifier trained on intraday hourly features + daily macro z-scores. "
            "Emits at three session opens: 23:00 UTC (Asia), 07:00 UTC (London), 13:00 UTC (NY). "
            "Regime filter (pre-declared): 168-hour trend > +1% AND 24h EMA > 120h EMA AND (DXY z252 < 0 OR real yield z252 < 0). "
            "Target: 8-hour forward direction. Realistic cost sensitivity applied."
        ),
        "n_holdout_emits": vs["n"],
        "coverage": f"{d['coverage_start']} to {d['coverage_end']}",
        "mean_return_pct_per_emit": vs["mean_pct"],
        "median_return_pct_per_emit": vs["median_pct"],
        "win_rate": vs["win_rate"],
        "std_return_pct": vs["std_pct"],
        "bootstrap_ci95_mean_pct": [vs["ci95_lo"], vs["ci95_hi"]],
        "bootstrap_ci_excludes_zero": vs["excludes_zero"],
        "block_bootstrap_20h_ci95": [d["block_bootstrap_20"]["lo"], d["block_bootstrap_20"]["hi"]],
        "cost_5bps_mean_pct": d["cost_sensitivity"]["5"]["mean_pct"],
        "cost_5bps_ci_excludes_zero": d["cost_sensitivity"]["5"]["excludes_zero"],
        "by_session": {k: {"n": v["n"], "mean_pct": v.get("mean_pct"), "win_rate": v.get("win_rate")} for k, v in d["by_session"].items()},
        "monte_carlo_12mo_median_terminal_pct": d["monte_carlo"]["12"]["terminal_pct"]["p50"] - 100 if "12" in d["monte_carlo"] else None,
        "monte_carlo_12mo_prob_loss_pct": d["monte_carlo"]["12"]["prob_loss_pct"] if "12" in d["monte_carlo"] else None,
        "historical_emits_per_year": d.get("emits_per_year_historical"),
        "notes": (
            "This V2-Session engine emits up to 3 times per day (once per session open). Regime filter "
            "keeps it silent when the intraday trend isn't clearly up. Historical evidence covers ~2 years "
            "of walk-forward simulation; performance in unseen regimes (bear markets, sideways) is unknown. "
            "All returns are directional 8-hour realized returns after regime + Model filter. Realistic costs "
            "up to 8 bps preserve statistical significance; 15 bps breaks it."
        ),
    }


def load_drift_state() -> str:
    if not DRIFT_FLAG.exists():
        return "NOT_INITIALIZED"
    try:
        return json.loads(DRIFT_FLAG.read_text(encoding="utf-8")).get("state", "UNKNOWN")
    except Exception:
        return "READ_ERROR"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-refresh-yahoo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading hourly bars...")
    if args.no_refresh_yahoo:
        bars = load_hourly_bars(BARS)
    else:
        try:
            bars = refresh_hourly_bars()
            print(f"Refreshed from Yahoo: {len(bars)} bars, last {bars.index.max()}")
            # Save a fresh snapshot
            bars.reset_index().to_csv(BARS.with_name("GC_F_1H_LATEST_YAHOO.csv"), index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            print(f"Yahoo hourly refresh failed: {e}. Falling back to frozen file.")
            bars = load_hourly_bars(BARS)

    macro = load_macro_daily(RAW)
    feat = build_intraday_features(bars, macro)
    feat = annotate_session_regime(feat)
    print(f"Feature frame: {len(feat)} hourly rows, last: {feat.index.max()}")

    targets = build_targets(feat, horizon_hours=HORIZON_HOURS)
    pipe, features_used = fit_and_predict(feat, targets)
    print(f"Fitted classifier with {len(features_used)} features")

    sessions = {}
    for hour, session_name in SESSION_HOURS.items():
        sessions[session_name] = session_snapshot(feat, pipe, features_used, hour, session_name)

    drift = load_drift_state()
    for s in sessions.values():
        if isinstance(s, dict) and drift == "KILL_ENGAGED":
            s["signal"] = "PAUSED_DRIFT_ALERT"
        if isinstance(s, dict):
            s["drift_state"] = drift

    payload = {
        "schema_version": 2,
        "product": "Compass V2 (Vs) — gold session-anchored directional forecast (Asia / London / NY)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sessions": sessions,
        "track_record": track_record_summary(),
        "disclaimers": {
            "not_trading_advice": (
                "This forecast is a directional model output for informational purposes only. It is not "
                "personal investment advice. Users are responsible for their own trading decisions and "
                "risk management. Historical performance does not guarantee future results."
            ),
            "regime_dependence": (
                "The Vs signal only fires when the intraday regime is on (trending uptrend + momentum + "
                "macro tailwind). Outside that regime it emits NO_SIGNAL for the session by design."
            ),
            "short_history": (
                "V2-Session evidence covers approximately 2 years of Yahoo hourly gold data (2024-11 to "
                "2026-09). This period was a strong trending bull market for gold. Performance in bear, "
                "range-bound, or high-volatility periods is not represented in the historical evidence."
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(payload, indent=2, default=str))
    else:
        OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Wrote {OUT_JSON}")
        for name, s in sessions.items():
            if isinstance(s, dict) and "signal" in s:
                print(f"  {name}: {s['signal']} regime={s.get('regime_on')} 168h_trend={s.get('regime_signals', {}).get('trend_168h_pct'):+.2f}%")


if __name__ == "__main__":
    main()
