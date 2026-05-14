"""
Monthly rebalance — runs on the first trading day of each month.

Pipeline (per Part 8 "Next Actions" ordering):
  1. Universe       — S&P 500 + core ETFs (cached 30 days)
  2. Prices         — 2-year adjusted close via yfinance batch download
  3. Momentum       — M1–M4 for all tickers; pre-filter to top 300
  4. Fundamentals   — fetch value & quality data for top-300 candidates
  5. All signals    — M1–M4, V1–V4, Q1–Q4
  6. Composite      — z-scored ranks, weighted combination
  7. LLM override   — score top-50 with Claude API
  8. Portfolio      — sector-capped, equal-weighted, risk-overlaid
  9. Persist        — rankings, portfolio, trade list → outputs/
"""
import logging
import sys
from datetime import date

from strategy.data import fetch_fundamentals, fetch_news_for_tickers, fetch_prices
from strategy.llm import score_top_candidates
from strategy.portfolio import construct_portfolio
from strategy.report import (
    load_current_portfolio,
    print_portfolio_summary,
    save_portfolio,
    save_rankings,
    save_trades,
)
from strategy.scorer import compute_composite_scores
from strategy.signals import compute_all_signals, compute_momentum_signals
from strategy.universe import CORE_ETFS, get_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pre-filter: how many tickers get fundamental analysis
MOMENTUM_PREFILTER = 300
# How many get LLM scoring
LLM_TOP_N = 50


def main() -> None:
    today = date.today()
    month_str = today.strftime("%Y-%m")
    logger.info("── Monthly rebalance  %s ────────────────────", month_str)

    # ── 1. Universe ───────────────────────────────────────────────────
    logger.info("Step 1/9  Getting universe...")
    tickers = get_universe()
    logger.info("Universe: %d tickers", len(tickers))

    # ── 2. Prices ─────────────────────────────────────────────────────
    logger.info("Step 2/9  Downloading price data...")
    prices = fetch_prices(tickers, period="2y")
    if prices.empty:
        logger.error("No price data returned. Aborting.")
        sys.exit(1)
    logger.info("Prices: %d tickers × %d days", prices.shape[1], prices.shape[0])

    # ── 3. Momentum pre-filter ────────────────────────────────────────
    logger.info("Step 3/9  Computing momentum for pre-filter...")
    try:
        mom_df = compute_momentum_signals(prices)
    except ValueError as exc:
        logger.error("Momentum computation failed: %s", exc)
        sys.exit(1)

    _etf_set = set(CORE_ETFS)
    # ETFs are fallback only — exclude them from scored candidates
    stocks_only = mom_df[~mom_df.index.isin(_etf_set)]
    top_candidates = (
        stocks_only["M_composite"]
        .nlargest(MOMENTUM_PREFILTER)
        .index.tolist()
    )
    logger.info("Pre-filter: top %d stocks by momentum (ETFs excluded from scoring)", len(top_candidates))

    # ── 4. Fundamentals ───────────────────────────────────────────────
    logger.info("Step 4/9  Fetching fundamentals (this takes a few minutes)...")
    fundamentals = fetch_fundamentals(top_candidates)
    n_ok = sum(1 for v in fundamentals.values() if v.get("market_cap"))
    logger.info("Fundamentals fetched for %d/%d tickers", n_ok, len(top_candidates))

    # ── 5. All signals ────────────────────────────────────────────────
    logger.info("Step 5/9  Computing all signals (M, V, Q)...")
    signals_df = compute_all_signals(prices, fundamentals, top_candidates)

    # ── 6. Composite score ────────────────────────────────────────────
    logger.info("Step 6/9  Computing composite scores...")
    scores_df = compute_composite_scores(signals_df)
    logger.info("Top 5:\n%s", scores_df[["composite", "decile"]].head())

    # ── 7. LLM override ───────────────────────────────────────────────
    logger.info("Step 7/9  Running LLM scoring on top %d...", LLM_TOP_N)
    top50 = scores_df.head(LLM_TOP_N).index.tolist()
    news_data = fetch_news_for_tickers(top50)
    llm_scores = score_top_candidates(top50, news_data)

    # ── 8. Portfolio construction ─────────────────────────────────────
    logger.info("Step 8/9  Constructing portfolio...")
    prev = load_current_portfolio()
    prev_holdings = set(prev.get("holdings", [])) if prev else set()

    portfolio = construct_portfolio(scores_df, llm_scores, fundamentals, prev_holdings)

    new_holdings = set(portfolio["holdings"])
    buys  = sorted(new_holdings - prev_holdings)
    sells = sorted(portfolio["sells"])

    # ── 9. Persist ────────────────────────────────────────────────────
    logger.info("Step 9/9  Saving outputs...")
    save_rankings(scores_df, month_str)
    save_portfolio(portfolio)
    save_trades(
        {
            "date":  today.isoformat(),
            "month": month_str,
            "buys":  buys,
            "sells": sells,
            "holdings_after": sorted(new_holdings),
        },
        month_str,
    )

    # ── Human-readable summary ─────────────────────────────────────────
    print_portfolio_summary(portfolio)
    logger.info("Buys  (%d): %s", len(buys),  buys)
    logger.info("Sells (%d): %s", len(sells), sells)
    logger.info("Monthly rebalance complete.")


if __name__ == "__main__":
    main()
