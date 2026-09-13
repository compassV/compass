"""V2-R candidate Vg: pre-declared regime detector for gating Model 2 emits.

Spec locked in VNEXT_V2R_RD_PLAN_2026-09-12.md. Do not modify thresholds.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


TREND_THRESHOLD_63D_RET = 0.05
EMA_SHORT = 20
EMA_LONG = 50
MACRO_Z_WINDOW = 252
MACRO_Z_MINP = 63


def annotate_competence_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add competence_regime column. Input must have gold_close, dxy, DFII10, date sorted asc."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["gold_close"].astype(float)
    df["ret_63d"] = close / close.shift(63) - 1.0
    df["ema_short"] = close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_long"] = close.ewm(span=EMA_LONG, adjust=False).mean()
    df["trend_up"] = df["ret_63d"] > TREND_THRESHOLD_63D_RET
    df["momentum_up"] = df["ema_short"] > df["ema_long"]

    dxy = df["dxy"].astype(float)
    dxy_z = (dxy - dxy.rolling(MACRO_Z_WINDOW, min_periods=MACRO_Z_MINP).mean()) / dxy.rolling(MACRO_Z_WINDOW, min_periods=MACRO_Z_MINP).std()
    ry = df["DFII10"].astype(float)
    ry_z = (ry - ry.rolling(MACRO_Z_WINDOW, min_periods=MACRO_Z_MINP).mean()) / ry.rolling(MACRO_Z_WINDOW, min_periods=MACRO_Z_MINP).std()
    df["dxy_z252"] = dxy_z
    df["real_yield_z252"] = ry_z
    df["macro_tailwind"] = (ry_z < 0) | (dxy_z < 0)

    df["competence_regime"] = df["trend_up"] & df["momentum_up"] & df["macro_tailwind"]
    return df
