"""
Signal computation for all three pillars:
  Momentum  — M1 (12-1 return), M2 (6-mo return), M3 (52-wk high distance), M4 (TS binary)
  Value     — V1 (EBITDA/EV), V2 (FCF/EV), V3 (B/M), V4 (Sales/Price)
  Quality   — Q1 (Piotroski F-Score), Q2 (gross profitability), Q3 (ROA proxy), Q4 (-NetDebt/EBITDA)
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Momentum ─────────────────────────────────────────────────────────────────


def compute_momentum_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute M1–M4 for every ticker in prices.
    prices: DataFrame (dates × tickers), adjusted close.
    Returns DataFrame indexed by ticker with columns M1–M4 and M_composite.
    """
    if len(prices) < 252:
        raise ValueError(f"Need ≥252 rows of price history, got {len(prices)}")

    latest   = prices.iloc[-1]
    mo1_ago  = prices.iloc[-21]    # t-1 month  (~21 trading days)
    mo6_ago  = prices.iloc[-126]   # t-6 months (~126 trading days)
    mo12_ago = prices.iloc[-252]   # t-12 months

    # M1: 12-1 month return (skip last month to avoid short-term reversal)
    m1 = (mo1_ago / mo12_ago) - 1

    # M2: 6-month return
    m2 = (latest / mo6_ago) - 1

    # M3: distance to 52-week high (0–1; higher = closer to high = bullish)
    high_52w = prices.tail(252).max()
    m3 = latest / high_52w

    # M4: binary time-series signal — is own 12-month return positive?
    m4 = ((latest / mo12_ago - 1) > 0).astype(float)

    df = pd.DataFrame({"M1": m1, "M2": m2, "M3": m3, "M4": m4})
    df.index.name = "ticker"

    # Simple unweighted composite used only for pre-filtering
    df["M_composite"] = df[["M1", "M2", "M3", "M4"]].mean(axis=1)
    return df


# ── Piotroski F-Score ─────────────────────────────────────────────────────────


def _row(df: pd.DataFrame, keyword: str, col: int):
    """Find the first row whose label contains keyword and return the value at column col."""
    for label in df.index:
        if keyword.lower() in str(label).lower():
            vals = df.loc[label]
            if col < len(vals):
                v = vals.iloc[col]
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
    return None


def compute_piotroski_fscore(fund: dict) -> Optional[int]:
    """
    Compute the 9-point Piotroski F-Score from yfinance financial statements.
    Returns None when insufficient data is available.
    """
    fin = fund.get("_financials")
    bs  = fund.get("_balance_sheet")
    cf  = fund.get("_cashflow")

    if fin is None or bs is None or cf is None:
        return None
    if fin.empty or bs.empty or cf.empty:
        return None
    if fin.shape[1] < 2 or bs.shape[1] < 2:
        return None

    try:
        ta_cur  = _row(bs,  "total assets",  0)
        ta_prev = _row(bs,  "total assets",  1)
        ni_cur  = _row(fin, "net income",    0)
        ni_prev = _row(fin, "net income",    1)
        gp_cur  = _row(fin, "gross profit",  0)
        gp_prev = _row(fin, "gross profit",  1)
        rev_cur = _row(fin, "total revenue", 0)
        rev_prev= _row(fin, "total revenue", 1)
        ltd_cur = _row(bs,  "long term debt",0)
        ltd_prev= _row(bs,  "long term debt",1)
        cfo_cur = _row(cf,  "operating",     0)
        cr      = fund.get("current_ratio")

        def roa(ni, ta):
            return ni / ta if ni is not None and ta and ta != 0 else None

        roa_cur  = roa(ni_cur,  ta_cur)
        roa_prev = roa(ni_prev, ta_prev)

        score = 0

        # Profitability
        if roa_cur  is not None and roa_cur > 0:                         score += 1
        if cfo_cur  is not None and cfo_cur > 0:                         score += 1
        if roa_cur  is not None and roa_prev is not None and roa_cur > roa_prev: score += 1
        if (cfo_cur is not None and ni_cur is not None and ta_cur and ta_cur != 0
                and (cfo_cur / ta_cur) > (ni_cur / ta_cur)):             score += 1

        # Leverage / liquidity
        if (ltd_cur is not None and ltd_prev is not None
                and ta_cur and ta_prev and ta_cur != 0 and ta_prev != 0
                and (ltd_cur / ta_cur) < (ltd_prev / ta_prev)):          score += 1
        if cr is not None and cr > 1:                                    score += 1
        score += 1  # share dilution — yfinance lacks prior-year shares; give benefit of doubt

        # Operating efficiency
        if (gp_cur is not None and rev_cur and rev_cur != 0
                and gp_prev is not None and rev_prev and rev_prev != 0
                and (gp_cur / rev_cur) > (gp_prev / rev_prev)):         score += 1
        if (rev_cur is not None and ta_cur and ta_cur != 0
                and rev_prev is not None and ta_prev and ta_prev != 0
                and (rev_cur / ta_cur) > (rev_prev / ta_prev)):         score += 1

        return score
    except Exception as exc:
        logger.debug("Piotroski error: %s", exc)
        return None


# ── Value ─────────────────────────────────────────────────────────────────────


def compute_value_signals(tickers: list[str], fundamentals: dict) -> pd.DataFrame:
    """Compute V1–V4 value signals. Higher = cheaper = better."""
    records = []
    for ticker in tickers:
        fund = fundamentals.get(ticker, {})

        ev_ebitda = fund.get("enterprise_to_ebitda")
        v1 = (1.0 / ev_ebitda) if ev_ebitda and ev_ebitda > 0 else None

        fcf = fund.get("free_cashflow")
        ev  = fund.get("enterprise_value")
        v2  = (fcf / ev) if fcf and ev and ev > 0 else None

        pb = fund.get("price_to_book")
        v3 = (1.0 / pb) if pb and pb > 0 else None

        ps = fund.get("price_to_sales")
        v4 = (1.0 / ps) if ps and ps > 0 else None

        records.append({"ticker": ticker, "V1": v1, "V2": v2, "V3": v3, "V4": v4})

    return pd.DataFrame(records).set_index("ticker")


# ── Quality ───────────────────────────────────────────────────────────────────


def compute_quality_signals(tickers: list[str], fundamentals: dict) -> pd.DataFrame:
    """Compute Q1–Q4 quality/trap-avoidance signals. Higher = better."""
    records = []
    for ticker in tickers:
        fund = fundamentals.get(ticker, {})

        # Q1: Piotroski F-Score (0–9, use directly)
        q1 = compute_piotroski_fscore(fund)

        # Q2: Gross profitability = gross profit / total assets (Novy-Marx)
        gp = fund.get("gross_profits")
        ta = fund.get("total_assets")
        q2 = (gp / ta) if gp and ta and ta > 0 else None

        # Q3: ROA proxy for earnings stability (higher / more stable = better)
        q3 = fund.get("return_on_assets")

        # Q4: Net debt / EBITDA — lower leverage is better, so negate
        total_debt = fund.get("total_debt") or 0
        total_cash = fund.get("total_cash") or 0
        ebitda     = fund.get("ebitda")
        q4 = (-(total_debt - total_cash) / ebitda) if ebitda and ebitda > 0 else None

        records.append({"ticker": ticker, "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4})

    return pd.DataFrame(records).set_index("ticker")


# ── Combined ──────────────────────────────────────────────────────────────────


def compute_all_signals(
    prices: pd.DataFrame,
    fundamentals: dict,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Compute and merge all signals for the provided tickers.
    Returns DataFrame indexed by ticker.
    """
    mom  = compute_momentum_signals(prices)
    mom  = mom[mom.index.isin(tickers)]

    val  = compute_value_signals(tickers, fundamentals)
    qual = compute_quality_signals(tickers, fundamentals)

    return mom.join(val, how="left").join(qual, how="left")
