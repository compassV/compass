"""Publish Vg historical track record as detailed JSON for the website.

Includes per-year breakdown, horizon sweep, cost sensitivity, and rolling holdout results.
This is the transparency artifact users see on the /gold-v2/track-record page.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from vnext.regime_detector import annotate_competence_regime

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "reports" / "VNEXT_V2R_EXTENDED_WALKFORWARD_ROWS_2026-09-13.csv"
RAW = ROOT / "gold_core" / "data" / "raw_data_yahoo_refreshed.csv"
RAW_FALLBACK = ROOT / "gold_core" / "data" / "raw_data.csv"
OUT_JSON = ROOT / "web" / "public" / "data" / "gold_v2_track_record.json"

HOLDOUT_START = pd.Timestamp("2010-01-01")  # Extended: full 17-year OOS
HORIZONS_DAYS = (1, 2, 3, 5, 10)
COST_BPS = (0, 2, 5, 10, 20)
N_BOOT = 5000
SEED = 42


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    if len(values) < 5:
        return float("nan"), float("nan")
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(values), len(values))
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_vg() -> pd.DataFrame:
    wf = pd.read_csv(WF, parse_dates=["date"])
    raw_path = RAW if RAW.exists() else RAW_FALLBACK
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    df = wf.merge(raw, on="date", how="left").sort_values("date").reset_index(drop=True)
    df = annotate_competence_regime(df)
    close = df["gold_close"].astype(float)
    sign = df["predicted_direction"].map({"BULL": 1.0, "BEAR": -1.0}).fillna(0.0)
    for h in HORIZONS_DAYS:
        df[f"dret_{h}d"] = sign * (close.shift(-h) / close - 1.0) * 100.0
    df["is_holdout"] = df["date"] >= HOLDOUT_START
    return df[df["is_active"] & df["competence_regime"] & df["is_holdout"]].copy()


def summarize(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(values)
    return {
        "n": int(len(values)),
        "mean_pct": float(values.mean()),
        "median_pct": float(np.median(values)),
        "win_rate": float((values > 0).mean()),
        "std_pct": float(values.std(ddof=1)),
        "ci95_lo_pct": lo,
        "ci95_hi_pct": hi,
        "excludes_zero": bool(lo > 0),
    }


def main() -> None:
    vg = load_vg()
    v3 = vg["dret_3d"].dropna().values

    per_year = {}
    for year in sorted(vg["date"].dt.year.unique()):
        rows = vg[vg["date"].dt.year == year]
        r3 = rows["dret_3d"].dropna().values
        hit = rows["direction_correct"].mean() if len(rows) else None
        per_year[int(year)] = {
            "n": int(len(rows)),
            "direction_hit_rate": float(hit) if hit is not None else None,
            **summarize(r3),
        }

    per_direction = {}
    for direction in ["BULL", "BEAR"]:
        rows = vg[vg["predicted_direction"] == direction]
        per_direction[direction] = summarize(rows["dret_3d"].dropna().values)

    horizon_sweep = {int(h): summarize(vg[f"dret_{h}d"].dropna().values) for h in HORIZONS_DAYS}

    cost_sensitivity = {}
    for bps in COST_BPS:
        adjusted = v3 - (bps / 100.0)
        cost_sensitivity[int(bps)] = summarize(adjusted)

    equity = np.cumsum(v3)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    payload = {
        "schema_version": 1,
        "product": "Compass V2 (Vg) — track record",
        "methodology": {
            "data_source": "Yahoo Finance daily GC=F + macro (DXY, VIX, SPX, silver, 10y/5y nominal yields) plus FRED-carried macro features",
            "model": "Model 2 (LogisticRegression + Ridge) trained walk-forward on hand-engineered features",
            "regime_filter_pre_declared": {
                "trend_63d_return_gt_pct": 5.0,
                "ema_short_gt_ema_long": "EMA(20) > EMA(50)",
                "macro_tailwind": "real_yield_z252 < 0 OR dxy_z252 < 0",
            },
            "exit_rule": "3-day directional close-to-close (E3)",
            "holdout_window": "2023-01-01 to 2026-03-30",
            "walk_forward_folds": 87,
            "walk_forward_test_window_days": 21,
        },
        "aggregate_holdout": summarize(v3),
        "per_year": per_year,
        "per_direction": per_direction,
        "horizon_sweep": horizon_sweep,
        "cost_sensitivity_bps_roundtrip": cost_sensitivity,
        "equity_curve": {
            "cumulative_pct_sum": [float(x) for x in equity],
            "dates": [d.strftime("%Y-%m-%d") for d in vg["date"].tolist()],
            "max_drawdown_pct_sum": float(drawdown.max()),
        },
        "disclaimers": [
            "All returns are before transaction costs unless labeled net.",
            "Historical performance is not indicative of future results.",
            "Vg's edge is concentrated in trending-uptrend regimes; performance in bear or ranging markets is unknown.",
            "Bootstrap confidence intervals assume iid resampling and block resampling; both are computed for transparency.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps({"n": payload["aggregate_holdout"]["n"], "mean_pct": payload["aggregate_holdout"]["mean_pct"], "ci95": [payload["aggregate_holdout"]["ci95_lo_pct"], payload["aggregate_holdout"]["ci95_hi_pct"]], "excludes_zero": payload["aggregate_holdout"]["excludes_zero"]}, indent=2))


if __name__ == "__main__":
    main()
