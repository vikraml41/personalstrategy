"""
Data pipeline: batch price download, fundamental fetching, and news retrieval.
All data is sourced from yfinance (free tier).
"""
import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Prices ─────────────────────────────────────────────────────────────────


def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """
    Batch-download adjusted close prices for all tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    Drops any ticker with fewer than 252 trading days of history.
    """
    logger.info("Downloading prices for %d tickers (period=%s)...", len(tickers), period)
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker returns a flat DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    valid = [col for col in prices.columns if prices[col].notna().sum() >= 252]
    logger.info("%d tickers have 252+ days of history", len(valid))
    return prices[valid].copy()


# ── Fundamentals ────────────────────────────────────────────────────────────


def _safe_get(info: dict, key: str):
    val = info.get(key)
    return val if val not in (None, "None", "N/A") else None


def fetch_fundamentals(tickers: list[str], delay: float = 0.4) -> dict:
    """
    Fetch fundamental data for a list of tickers via yfinance.
    Returns {ticker: {field: value, ...}}.
    Also attaches raw financial statement DataFrames under '_financials',
    '_balance_sheet', and '_cashflow' keys for Piotroski scoring.
    """
    fundamentals: dict = {}
    for i, ticker in enumerate(tickers):
        if i and i % 50 == 0:
            logger.info("  fundamentals: %d/%d fetched", i, len(tickers))
            time.sleep(3)
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            entry: dict = {
                "sector": _safe_get(info, "sector") or "Unknown",
                "industry": _safe_get(info, "industry") or "",
                "market_cap": _safe_get(info, "marketCap"),
                "enterprise_value": _safe_get(info, "enterpriseValue"),
                "enterprise_to_ebitda": _safe_get(info, "enterpriseToEbitda"),
                "price_to_book": _safe_get(info, "priceToBook"),
                "price_to_sales": _safe_get(info, "priceToSalesTrailing12Months"),
                "free_cashflow": _safe_get(info, "freeCashflow"),
                "total_debt": _safe_get(info, "totalDebt"),
                "total_cash": _safe_get(info, "totalCash"),
                "gross_profits": _safe_get(info, "grossProfits"),
                "total_assets": _safe_get(info, "totalAssets"),
                "return_on_assets": _safe_get(info, "returnOnAssets"),
                "current_ratio": _safe_get(info, "currentRatio"),
                "shares_outstanding": _safe_get(info, "sharesOutstanding"),
                "operating_cashflow": _safe_get(info, "operatingCashflow"),
                "net_income": _safe_get(info, "netIncomeToCommon"),
                "total_revenue": _safe_get(info, "totalRevenue"),
                "ebitda": _safe_get(info, "ebitda"),
            }

            # Annual statements for Piotroski F-Score
            try:
                entry["_financials"] = t.financials
                entry["_balance_sheet"] = t.balance_sheet
                entry["_cashflow"] = t.cashflow
            except Exception as exc:
                logger.debug("No statements for %s: %s", ticker, exc)

            fundamentals[ticker] = entry
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Failed to fetch fundamentals for %s: %s", ticker, exc)
            fundamentals[ticker] = {}

    return fundamentals


# ── News ─────────────────────────────────────────────────────────────────────


def fetch_news_for_tickers(tickers: list[str]) -> dict:
    """
    Fetch the most recent news headlines for each ticker via yfinance.
    Returns {ticker: [headline, ...]}.
    """
    news_data: dict = {}
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).news or []
            headlines = [
                item.get("title", "")
                for item in items[:10]
                if item.get("title")
            ]
            news_data[ticker] = headlines
            time.sleep(0.2)
        except Exception as exc:
            logger.warning("News fetch failed for %s: %s", ticker, exc)
            news_data[ticker] = []
    return news_data
