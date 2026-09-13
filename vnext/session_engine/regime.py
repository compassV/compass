"""V2-Session regime detector (pre-declared, thresholds locked).

Emits ON when all three hold at the current hourly bar:
  - Intraday trend up: 168-hour (1 week) return > +1%
  - Intraday momentum: 24h EMA > 120h EMA
  - Macro tailwind: dxy z-score < 0 OR real_yield z-score < 0

These thresholds are frozen for the entire simulation battery. Post-hoc tuning invalidates results.
"""
from __future__ import annotations
import pandas as pd


TREND_168H_THRESHOLD = 0.01  # 1% over the previous 168 hours = 1 week
EMA_SHORT_HOURS = 24
EMA_LONG_HOURS = 120


def annotate_session_regime(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    f["trend_up_168h"] = f["ret_168h"] > TREND_168H_THRESHOLD
    dxy_favorable = f["macro_dxy_z252"] < 0
    ry_favorable = f["macro_real_yield_z252"] < 0 if "macro_real_yield_z252" in f.columns else pd.Series(False, index=f.index)
    f["macro_tailwind"] = dxy_favorable | ry_favorable
    f["session_regime_on"] = f["trend_up_168h"] & (f["momentum_up"] == 1) & f["macro_tailwind"]
    return f
