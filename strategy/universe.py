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
    """
    Fetch S&P 500 constituent tickers from Wikipedia.
    Tries up to 3 times with increasing timeouts before giving up.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for attempt in range(3):
        try:
            tables = pd.read_html(url, flavor="lxml", storage_options={"timeout": 15 + attempt * 10})
            df = tables[0]
            tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            if len(tickers) > 400:   # sanity check — S&P 500 should have ~503
                return tickers
            logger.warning("Wikipedia returned only %d tickers (attempt %d)", len(tickers), attempt + 1)
        except Exception as exc:
            logger.warning("Wikipedia fetch attempt %d failed: %s", attempt + 1, exc)
    return []


def get_universe() -> list[str]:
    """
    Return investment universe: S&P 500 stocks + core ETFs.
    Uses a cached list if it is less than 30 days old AND has >100 stock tickers.
    """
    if UNIVERSE_CACHE.exists():
        cache = json.loads(UNIVERSE_CACHE.read_text())
        cached_date = date.fromisoformat(cache.get("date", "2000-01-01"))
        n_stocks = len([t for t in cache.get("tickers", []) if t not in set(CORE_ETFS)])
        if (date.today() - cached_date).days < 30 and n_stocks > 100:
            logger.info("Using cached universe (%d tickers, %d stocks)", len(cache["tickers"]), n_stocks)
            return cache["tickers"]
        elif n_stocks <= 100:
            logger.warning("Cached universe only has %d stocks — refreshing.", n_stocks)

    logger.info("Fetching S&P 500 universe from Wikipedia...")
    stocks = get_sp500_tickers()

    if not stocks:
        logger.error("Could not fetch S&P 500 tickers. Only ETFs will be available — results will be poor.")
    else:
        logger.info("Fetched %d S&P 500 tickers", len(stocks))

    all_tickers = list(dict.fromkeys(stocks + CORE_ETFS))

    UNIVERSE_CACHE.parent.mkdir(exist_ok=True)
    UNIVERSE_CACHE.write_text(
        json.dumps({"tickers": all_tickers, "date": date.today().isoformat()}, indent=2)
    )
    logger.info("Universe: %d tickers (%d stocks + %d ETFs)", len(all_tickers), len(stocks), len(CORE_ETFS))
    return all_tickers
