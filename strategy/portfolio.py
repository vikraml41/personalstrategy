"""
Portfolio construction + risk overlay.
No LLM — all rules are technical/quantitative.
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    CASH_TICKERS,
    CRASH_LOOKBACK_MONTHS,
    ENTRY_MAX_EXTENSION,
    ENTRY_RSI_MAX,
    MAX_PER_SECTOR,
    PORTFOLIO_MIN,
    PORTFOLIO_TARGET,
    SELL_RANK_THRESHOLD,
    SPY_LOOKBACK_MONTHS,
    SPY_MA_DAYS,
    VOL_PERCENTILE,
    VOL_WINDOW_DAYS,
)
from strategy.technical import compute_entry_quality

logger = logging.getLogger(__name__)


def check_market_regime() -> dict:
    """SPY-based regime filter. Returns equity_fraction and flags."""
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
        logger.warning("SPY fetch failed: %s", exc)
        return result

    trading_days_10mo = SPY_LOOKBACK_MONTHS * 21
    if len(hist) >= trading_days_10mo + SPY_MA_DAYS:
        current  = hist.iloc[-1]
        mo10_ago = hist.iloc[-trading_days_10mo]
        ma200    = hist.tail(SPY_MA_DAYS).mean()
        if (current / mo10_ago - 1) < 0 and current < ma200:
            logger.warning("Regime filter triggered — cutting equity to 50%%")
            result["equity_fraction"] = 0.5
            result["regime_triggered"] = True

    trading_days_24mo = CRASH_LOOKBACK_MONTHS * 21
    if len(hist) >= trading_days_24mo + VOL_WINDOW_DAYS:
        current      = hist.iloc[-1]
        mo24_ago     = hist.iloc[-trading_days_24mo]
        market_ret   = current / mo24_ago - 1
        returns      = hist.pct_change().dropna()
        vol_3mo      = returns.tail(VOL_WINDOW_DAYS).std() * np.sqrt(252)
        rolling_vol  = returns.rolling(VOL_WINDOW_DAYS).std().dropna() * np.sqrt(252)
        pct_rank     = float((rolling_vol < vol_3mo).mean() * 100)
        if market_ret < 0 and pct_rank > VOL_PERCENTILE:
            logger.warning("Crash filter triggered")
            result["momentum_weight_adj"] = 0.5
            result["value_weight_adj"]    = 2.0
            result["crash_triggered"]     = True

    return result


def construct_portfolio(
    scores_df: pd.DataFrame,
    fundamentals: dict,
    prices: pd.DataFrame,
    prev_portfolio: dict | None = None,
) -> dict:
    """
    Build the target portfolio.

    - Applies entry quality filter (RSI + MA50 extension) to top candidates.
    - Applies sector cap.
    - Equal-weights selected stocks.
    - Returns sells list from previous portfolio.
    """
    regime = check_market_regime()
    prev_holdings = set(prev_portfolio.get("holdings", [])) if prev_portfolio else set()

    ranked = scores_df["composite"].sort_values(ascending=False).copy()

    # Apply entry quality filter — skip stocks that are too extended or overbought
    entry_rejected: list[str] = []
    qualified: list[str] = []
    for ticker in ranked.index:
        if ticker in prices.columns:
            if compute_entry_quality(prices[ticker].dropna(),
                                     rsi_max=ENTRY_RSI_MAX,
                                     ext_max=ENTRY_MAX_EXTENSION):
                qualified.append(ticker)
            else:
                entry_rejected.append(ticker)
        else:
            qualified.append(ticker)  # no price data → don't penalise

    if entry_rejected:
        logger.info("Entry quality filter rejected %d stocks (overbought/extended): %s",
                    len(entry_rejected), entry_rejected[:10])

    # Re-rank qualified only
    ranked = ranked[ranked.index.isin(qualified)]

    # Sector cap
    selected: list[str]      = []
    sector_counts: dict[str, int] = {}

    for ticker in ranked.index:
        if len(selected) >= PORTFOLIO_TARGET:
            break
        sector = fundamentals.get(ticker, {}).get("sector", "Unknown")
        if sector_counts.get(sector, 0) >= MAX_PER_SECTOR:
            continue
        selected.append(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(selected) < PORTFOLIO_MIN:
        logger.warning("Only %d stocks qualify (min=%d). Cash added.", len(selected), PORTFOLIO_MIN)

    # Equal weights scaled by equity fraction
    eq_weight = regime["equity_fraction"] / max(len(selected), 1)
    weights   = {t: round(eq_weight, 6) for t in selected}

    cash_frac = 1.0 - sum(weights.values())
    if cash_frac > 0.005:
        weights[CASH_TICKERS[0]] = round(cash_frac, 6)

    # Sell rule
    top_n  = set(scores_df["composite"].nlargest(SELL_RANK_THRESHOLD).index)
    sells  = [t for t in prev_holdings if t not in top_n]

    return {
        "holdings":        selected,
        "weights":         weights,
        "sector_counts":   sector_counts,
        "entry_rejected":  entry_rejected,
        "regime":          regime,
        "sells":           sells,
    }
