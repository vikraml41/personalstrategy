"""
Universe management: S&P 500 stocks + core ETFs.
Caches the list locally; refreshes every 30 days.
"""
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CORE_ETFS = [
    "SPY", "QQQ", "IWM",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

UNIVERSE_CACHE = Path("outputs/universe_cache.json")


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituent tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, flavor="lxml")
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def get_universe() -> list[str]:
    """
    Return investment universe: S&P 500 stocks + core ETFs.
    Uses a cached list if it is less than 30 days old.
    """
    if UNIVERSE_CACHE.exists():
        cache = json.loads(UNIVERSE_CACHE.read_text())
        cached_date = date.fromisoformat(cache.get("date", "2000-01-01"))
        if (date.today() - cached_date).days < 30:
            logger.info("Using cached universe (%d tickers)", len(cache["tickers"]))
            return cache["tickers"]

    logger.info("Fetching S&P 500 universe from Wikipedia...")
    try:
        stocks = get_sp500_tickers()
        logger.info("Fetched %d S&P 500 tickers", len(stocks))
    except Exception as exc:
        logger.error("Failed to fetch S&P 500: %s", exc)
        stocks = []

    all_tickers = list(dict.fromkeys(stocks + CORE_ETFS))

    UNIVERSE_CACHE.parent.mkdir(exist_ok=True)
    UNIVERSE_CACHE.write_text(
        json.dumps({"tickers": all_tickers, "date": date.today().isoformat()}, indent=2)
    )
    logger.info("Universe: %d tickers", len(all_tickers))
    return all_tickers
