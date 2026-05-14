"""
Layer 3: Position construction + Layer 4: Risk overlay.

Position construction rules:
  - Target 15–20 stocks, equal-weighted.
  - Hard cap: ≤3 names per GICS sector.
  - Buy list: top 20 by composite after LLM overrides.
  - Sell rule: any holding outside top 40 is sold.
  - Cash drag: if fewer than MIN qualify, hold remainder in SGOV/BIL.

Risk overlay:
  - Market regime filter (Faber/Antonacci): if SPY 10-month return < 0
    AND SPY < 200-day MA → cut equity to 50%, rest in SGOV.
  - Momentum-crash filter (Daniel-Moskowitz): if 24-month market return < 0
    AND 3-month vol > 80th pct of trailing 5-yr distribution → halve
    momentum weight, double value weight for this month.
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    CASH_TICKERS,
    CRASH_LOOKBACK_MONTHS,
    MAX_PER_SECTOR,
    PORTFOLIO_MIN,
    PORTFOLIO_TARGET,
    SELL_RANK_THRESHOLD,
    SPY_LOOKBACK_MONTHS,
    SPY_MA_DAYS,
    VOL_PERCENTILE,
    VOL_WINDOW_DAYS,
)
from strategy.llm import apply_overrides

logger = logging.getLogger(__name__)


# ── Risk overlay ──────────────────────────────────────────────────────────────


def check_market_regime() -> dict:
    """
    Compute Layer 4 risk-overlay signals from SPY price history.

    Returns:
        equity_fraction      — fraction of portfolio to hold in equities (0.5 or 1.0)
        momentum_weight_adj  — multiplier for momentum pillar weight (0.5 or 1.0)
        value_weight_adj     — multiplier for value pillar weight (1.0 or 2.0)
        regime_triggered     — bool, True if market regime filter fired
        crash_triggered      — bool, True if momentum-crash filter fired
    """
    result = {
        "equity_fraction": 1.0,
        "momentum_weight_adj": 1.0,
        "value_weight_adj": 1.0,
        "regime_triggered": False,
        "crash_triggered": False,
    }

    try:
        hist = yf.Ticker("SPY").history(period="6y")["Close"].dropna()
    except Exception as exc:
        logger.warning("Could not fetch SPY history for regime check: %s", exc)
        return result

    # ── Market regime filter ───────────────────────────────────────────
    trading_days_10mo = SPY_LOOKBACK_MONTHS * 21
    if len(hist) >= trading_days_10mo + SPY_MA_DAYS:
        current = hist.iloc[-1]
        mo10_ago = hist.iloc[-trading_days_10mo]
        ma200 = hist.tail(SPY_MA_DAYS).mean()

        if (current / mo10_ago - 1) < 0 and current < ma200:
            logger.warning("Market regime filter triggered — cutting equity to 50%%")
            result["equity_fraction"] = 0.5
            result["regime_triggered"] = True

    # ── Momentum-crash filter ──────────────────────────────────────────
    trading_days_24mo = CRASH_LOOKBACK_MONTHS * 21
    if len(hist) >= trading_days_24mo + VOL_WINDOW_DAYS:
        current = hist.iloc[-1]
        mo24_ago = hist.iloc[-trading_days_24mo]
        market_return_24mo = current / mo24_ago - 1

        returns = hist.pct_change().dropna()
        vol_3mo = returns.tail(VOL_WINDOW_DAYS).std() * np.sqrt(252)
        rolling_vol = returns.rolling(VOL_WINDOW_DAYS).std().dropna() * np.sqrt(252)
        pct_rank = float((rolling_vol < vol_3mo).mean() * 100)

        if market_return_24mo < 0 and pct_rank > VOL_PERCENTILE:
            logger.warning(
                "Momentum-crash filter triggered (vol pct=%.1f%%) — "
                "halving momentum weight, doubling value weight",
                pct_rank,
            )
            result["momentum_weight_adj"] = 0.5
            result["value_weight_adj"] = 2.0
            result["crash_triggered"] = True

    return result


# ── Position construction ─────────────────────────────────────────────────────


def construct_portfolio(
    scores_df: pd.DataFrame,
    llm_scores: dict,
    fundamentals: dict,
    prev_holdings: set | None = None,
) -> dict:
    """
    Build the target portfolio for this month.

    Args:
        scores_df:     DataFrame with 'composite' column, indexed by ticker.
        llm_scores:    Output of strategy.llm.score_top_candidates().
        fundamentals:  {ticker: {sector: ..., ...}} from strategy.data.
        prev_holdings: Set of currently held tickers (for sell-rule check).

    Returns dict with:
        holdings       — ordered list of tickers to hold
        weights        — {ticker: weight} (sum ≤ 1.0; remainder is cash)
        sector_counts  — {sector: count}
        llm_exits      — tickers excluded by risk-flag rule
        llm_boosts     — tickers elevated by post-earnings rule
        llm_half_weight— tickers held at half weight
        regime         — risk overlay state dict
        sells          — tickers to sell from previous portfolio
    """
    regime = check_market_regime()
    overrides = apply_overrides(scores_df, llm_scores)

    ranked = scores_df["composite"].sort_values(ascending=False).copy()

    # Apply risk-flag exits
    ranked = ranked.drop(labels=overrides["exits"], errors="ignore")

    # Boost: push qualified tickers above the current top
    for ticker in overrides["boosts"]:
        if ticker in ranked.index:
            ranked[ticker] = ranked.max() + 0.01
    ranked = ranked.sort_values(ascending=False)

    # Select top candidates respecting sector cap
    selected: list[str] = []
    sector_counts: dict[str, int] = {}
    half_weight_set = set(overrides["half_weight"])

    for ticker in ranked.index:
        if len(selected) >= PORTFOLIO_TARGET:
            break
        sector = fundamentals.get(ticker, {}).get("sector", "Unknown")
        if sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        selected.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(selected) < PORTFOLIO_MIN:
        logger.warning(
            "Only %d stocks qualify (min=%d). Cash buffer added.", len(selected), PORTFOLIO_MIN
        )

    # Equal-weight, with half-weight for LLM-flagged positions
    total_units = sum(0.5 if t in half_weight_set else 1.0 for t in selected)
    equity_weights: dict[str, float] = {}
    for t in selected:
        unit = 0.5 if t in half_weight_set else 1.0
        equity_weights[t] = (unit / total_units) * regime["equity_fraction"]

    weights = dict(equity_weights)

    # Cash remainder (SGOV or BIL)
    cash_fraction = 1.0 - sum(weights.values())
    if cash_fraction > 0.005:
        weights[CASH_TICKERS[0]] = round(cash_fraction, 6)

    # Sell rule: previous holdings outside top-40 composite rank
    sells: list[str] = []
    if prev_holdings:
        top40 = set(scores_df["composite"].nlargest(SELL_RANK_THRESHOLD).index)
        sells = [t for t in prev_holdings if t not in top40]

    return {
        "holdings":        selected,
        "weights":         weights,
        "sector_counts":   sector_counts,
        "llm_exits":       overrides["exits"],
        "llm_boosts":      overrides["boosts"],
        "llm_half_weight": overrides["half_weight"],
        "regime":          regime,
        "sells":           sells,
    }
