"""
Composite score computation.
Algorithm (per the document):
  1. Cross-sectionally rank each signal.
  2. Z-score the ranks.
  3. Average z-scores within each pillar.
  4. Combine pillars: 40% Momentum + 35% Value + 25% Quality.
  5. Assign decile (1 = top, 10 = bottom).
"""
import logging

import pandas as pd

from config import MOMENTUM_WEIGHT, VALUE_WEIGHT, QUALITY_WEIGHT

logger = logging.getLogger(__name__)


def _zscore_ranks(series: pd.Series) -> pd.Series:
    """Rank cross-sectionally (pct), then z-score. NaN stays NaN."""
    ranked = series.rank(pct=True, na_option="keep")
    mu = ranked.mean()
    sigma = ranked.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=series.index)
    return (ranked - mu) / sigma


def compute_composite_scores(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the composite factor score for every ticker.

    Returns the input DataFrame enriched with:
      momentum_score, value_score, quality_score, composite, decile
    Sorted descending by composite (rank 1 = best).
    """
    df = signals_df.copy()

    # ── Momentum sub-score ────────────────────────────────────────────
    m_cols = [c for c in ("M1", "M2", "M3", "M4") if c in df.columns]
    if m_cols:
        m_z = pd.concat([_zscore_ranks(df[c]) for c in m_cols], axis=1, keys=m_cols)
        df["momentum_score"] = m_z.mean(axis=1)
    else:
        df["momentum_score"] = 0.0

    # ── Value sub-score ───────────────────────────────────────────────
    v_cols = [c for c in ("V1", "V2", "V3", "V4") if c in df.columns]
    if v_cols:
        v_z = pd.concat([_zscore_ranks(df[c]) for c in v_cols], axis=1, keys=v_cols)
        df["value_score"] = v_z.mean(axis=1)
    else:
        df["value_score"] = 0.0

    # ── Quality sub-score ─────────────────────────────────────────────
    q_cols = [c for c in ("Q1", "Q2", "Q3", "Q4") if c in df.columns]
    if q_cols:
        q_z = pd.concat([_zscore_ranks(df[c]) for c in q_cols], axis=1, keys=q_cols)
        df["quality_score"] = q_z.mean(axis=1)
    else:
        df["quality_score"] = 0.0

    # ── Composite ─────────────────────────────────────────────────────
    df["composite"] = (
        MOMENTUM_WEIGHT * df["momentum_score"]
        + VALUE_WEIGHT   * df["value_score"]
        + QUALITY_WEIGHT * df["quality_score"]
    )

    # Decile: 1 = top 10%, 10 = bottom 10%
    rank_asc = df["composite"].rank(ascending=False, method="first")
    df["decile"] = pd.cut(rank_asc, bins=10, labels=False) + 1

    return df.sort_values("composite", ascending=False)
