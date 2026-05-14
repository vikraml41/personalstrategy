"""
Weekly LLM event scan — runs every Sunday evening.

What it does:
  1. Load the most recent monthly rankings.
  2. Fetch the last 7 days of news headlines for the top-50 ranked stocks.
  3. Score each ticker with the LLM (sentiment / surprise / risk_flag).
  4. Apply override rules and identify any mid-month risk-flag exits.
  5. Persist scores and overrides to outputs/.

This run NEVER adds new buy entries — only risk-flag exits are acted on
mid-month (per the cadence spec in Part 4 of the research document).
"""
import logging
import sys
from datetime import date

from strategy.data import fetch_news_for_tickers
from strategy.llm import apply_overrides, score_news_batch
from strategy.report import (
    load_current_portfolio,
    load_latest_rankings,
    print_portfolio_summary,
    save_llm_scores,
    save_portfolio,
)
from config import LLM_SCAN_TOP_N

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("── Weekly LLM scan  %s ──────────────────────", date.today())

    # ── 1. Load rankings ──────────────────────────────────────────────
    rankings = load_latest_rankings()
    if rankings is None:
        logger.error("No rankings found. Run run_monthly.py first.")
        sys.exit(1)

    top_tickers = rankings[:LLM_SCAN_TOP_N]
    logger.info("Scanning top %d tickers with LLM", len(top_tickers))

    # ── 2. Fetch news ─────────────────────────────────────────────────
    logger.info("Fetching news headlines...")
    news_data = fetch_news_for_tickers(top_tickers)
    n_with_news = sum(1 for h in news_data.values() if h)
    logger.info("%d/%d tickers have recent news", n_with_news, len(top_tickers))

    # ── 3. LLM scoring ────────────────────────────────────────────────
    logger.info("Scoring with LLM...")
    llm_scores = score_news_batch(news_data)

    # ── 4. Apply overrides ────────────────────────────────────────────
    # Build a minimal scores structure (just ranks) so apply_overrides works
    import pandas as pd
    rank_series = pd.Series(
        {t: float(len(rankings) - i) for i, t in enumerate(rankings[:LLM_SCAN_TOP_N])}
    )
    scores_stub = pd.DataFrame({"composite": rank_series})
    overrides = apply_overrides(scores_stub, llm_scores)

    # ── 5. Mid-month exits ────────────────────────────────────────────
    today_str = date.today().isoformat()
    save_llm_scores(llm_scores, overrides, today_str)

    if overrides["exits"]:
        logger.warning("RISK-FLAG EXITS this week: %s", overrides["exits"])
        # Update current portfolio to remove risk-flag exits
        portfolio = load_current_portfolio()
        if portfolio:
            removed = [t for t in overrides["exits"] if t in portfolio.get("holdings", [])]
            if removed:
                portfolio["holdings"] = [t for t in portfolio["holdings"] if t not in removed]
                for t in removed:
                    portfolio["weights"].pop(t, None)
                portfolio["llm_exits"] = portfolio.get("llm_exits", []) + removed
                save_portfolio(portfolio)
                logger.info("Removed from portfolio: %s", removed)

    if overrides["boosts"]:
        logger.info("Post-earnings boosts identified (take effect next monthly rebal): %s",
                    overrides["boosts"])
    if overrides["half_weight"]:
        logger.info("Half-weight candidates: %s", overrides["half_weight"])

    # ── Summary ───────────────────────────────────────────────────────
    portfolio = load_current_portfolio()
    if portfolio:
        print_portfolio_summary(portfolio)

    logger.info("Weekly scan complete.")


if __name__ == "__main__":
    main()
