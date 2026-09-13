"""V2-Compass BTC live publisher: produces the current session-by-session forecast JSON for BTC.

Mirrors vnext/session_engine/publisher.py for BTC-USD. Writes web/public/data/btc_v2.json.
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
    build_intraday_features, load_macro_daily,
    build_targets, SESSION_HOURS,
)
from vnext.session_engine.regime import annotate_session_regime

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "gold_core" / "data" / "raw_data_yahoo_refreshed.csv"
DRIFT_FLAG = ROOT / "reports" / "vnext_shadow" / "DRIFT_KILL_FLAG.json"
OUT_JSON = ROOT / "data" / "btc_v2.json"
SIM_JSON = ROOT / "reports" / "VNEXT_VS_BTC_SIMULATION_2026-09-13.json"

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


def fetch_btc_hourly() -> pd.DataFrame:
    tk = yf.Ticker("BTC-USD")
    df = tk.history(period="730d", interval="60m", auto_adjust=True, actions=False, prepost=True)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else "Date"
    df["timestamp_utc"] = pd.to_datetime(df[tcol], utc=True).dt.floor("h")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    keep = [c for c in ("timestamp_utc", "open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].set_index("timestamp_utc").sort_index()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=30, random_state=42)),
    ])


def fit(feat: pd.DataFrame, target: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    available = [c for c in FEATURE_COLS if c in feat.columns]
    frame = feat[available].join(target[["cls"]]).dropna(subset=["cls", "ret_168h", "macro_dxy_z252"])
    with contextlib.redirect_stdout(io.StringIO()):
        pipe = build_pipeline()
        pipe.fit(frame[available].values, frame["cls"].astype(int).values)
    return pipe, available


def session_snapshot(feat: pd.DataFrame, pipe: Pipeline, features: list[str], session_hour: int, session_name: str) -> dict:
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=3)
    rows = feat[(feat["hour"] == session_hour) & (feat.index >= cutoff)]
    if rows.empty:
        return {"session": session_name, "state": "NO_RECENT_DATA"}
    latest = rows.iloc[-1:]
    probs = pipe.predict_proba(latest[features].values)[0]
    pred_class = int(probs.argmax())
    direction = CLASS_MAP[pred_class]
    regime_on = bool(latest["session_regime_on"].iloc[0])
    emit = direction in {"BULL", "BEAR"} and regime_on
    ts = latest.index[0]
    prev_ts = ts - pd.Timedelta(hours=8)
    prev_close = float(feat.loc[prev_ts, "close"]) if prev_ts in feat.index else None
    return {
        "session": session_name,
        "session_open_utc": ts.isoformat().replace("+00:00", "Z"),
        "session_open_price": float(latest["open"].iloc[0]),
        "current_price": float(latest["close"].iloc[0]),
        "previous_session_close": prev_close,
        "move_vs_previous_close_pct": (float(latest["close"].iloc[0]) / prev_close - 1) * 100 if prev_close else None,
        "signal": direction if emit else "NO_SIGNAL",
        "raw_direction": direction,
        "prob_bull": float(probs[2]), "prob_bear": float(probs[0]), "prob_flat": float(probs[1]),
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
    if not SIM_JSON.exists():
        return {"available": False, "note": "Run vnext.session_engine.btc_walkforward first"}
    d = json.loads(SIM_JSON.read_text(encoding="utf-8"))
    vs = d["vs_full"]
    return {
        "available": True,
        "methodology": (
            "V2-Compass BTC session engine walk-forward on Yahoo BTC-USD hourly (24/7 trading). "
            "Same intraday features + macro tailwind regime as gold. HistGradientBoostingClassifier. "
            "Emits at 23:00 UTC (Asia), 07:00 UTC (London), 13:00 UTC (NY). 8-hour forward direction target."
        ),
        "n_holdout_emits": vs["n"],
        "coverage": f"{d['coverage_start']} to {d['coverage_end']}",
        "mean_return_pct_per_emit": vs["mean_pct"],
        "median_return_pct_per_emit": vs["median_pct"],
        "win_rate": vs["win_rate"],
        "bootstrap_ci95_mean_pct": [vs["ci95_lo"], vs["ci95_hi"]],
        "bootstrap_ci_excludes_zero": vs["excludes_zero"],
        "block_bootstrap_20h_ci95": [d["block_bootstrap_20"]["lo"], d["block_bootstrap_20"]["hi"]],
        "cost_5bps_mean_pct": d["cost_sensitivity"]["5"]["mean_pct"],
        "cost_5bps_ci_excludes_zero": d["cost_sensitivity"]["5"]["excludes_zero"],
        "by_session": {k: {"n": v["n"], "mean_pct": v.get("mean_pct"), "win_rate": v.get("win_rate")} for k, v in d["by_session"].items()},
        "monte_carlo_12mo_median_terminal_pct": (d["monte_carlo"]["12"]["terminal_pct"]["p50"] - 100) if "12" in d["monte_carlo"] else None,
        "historical_emits_per_year": d.get("emits_per_year_historical"),
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Fetching BTC hourly...")
    bars = fetch_btc_hourly()
    macro = load_macro_daily(RAW)
    feat = build_intraday_features(bars, macro)
    feat = annotate_session_regime(feat)
    targets = build_targets(feat, horizon_hours=HORIZON_HOURS)
    pipe, feats = fit(feat, targets)
    print(f"Fitted; features={len(feats)}, last bar {feat.index.max()}")

    sessions = {}
    for hour, name in SESSION_HOURS.items():
        sessions[name] = session_snapshot(feat, pipe, feats, hour, name)

    drift = load_drift_state()
    for s in sessions.values():
        if isinstance(s, dict) and drift == "KILL_ENGAGED":
            s["signal"] = "PAUSED_DRIFT_ALERT"
        if isinstance(s, dict):
            s["drift_state"] = drift

    payload = {
        "schema_version": 2,
        "product": "Compass V2 (Vs) — BTC session-anchored directional forecast (Asia / London / NY)",
        "instrument": "BTC-USD",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sessions": sessions,
        "track_record": track_record_summary(),
        "disclaimers": {
            "not_trading_advice": (
                "This forecast is a directional model output for informational purposes only. It is not "
                "personal investment advice. Users are responsible for their own trading decisions and risk."
            ),
            "regime_dependence": (
                "The Vs BTC signal only fires when the intraday regime is on (168h trend + momentum + macro tailwind). "
                "On off-regime sessions no signal is published."
            ),
            "short_history": (
                "V2-Compass BTC evidence covers ~2 years of Yahoo hourly data (2024-11 to 2026-09), a strong "
                "bull period for BTC. Performance in bear or sideways regimes is not represented."
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
                trend = s.get("regime_signals", {}).get("trend_168h_pct", 0)
                print(f"  {name}: {s['signal']} regime={s.get('regime_on')} 168h_trend={trend:+.2f}%")


if __name__ == "__main__":
    main()
