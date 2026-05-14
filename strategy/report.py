"""
Output persistence layer.
All results are saved as JSON files under outputs/ and committed by
the GitHub Actions workflow so every run is version-controlled.
"""
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved %s", path)


# ── Rankings ──────────────────────────────────────────────────────────────────


def save_rankings(scores_df: pd.DataFrame, month_str: str) -> None:
    """Persist top-100 composite rankings for the given month (YYYY-MM)."""
    cols = [c for c in ("composite", "momentum_score", "value_score", "quality_score", "decile") if c in scores_df.columns]
    top100 = scores_df.head(100)[cols].copy()
    top100.index.name = "ticker"
    records = top100.reset_index().to_dict(orient="records")
    _dump(OUTPUTS / f"rankings_{month_str}.json", records)


def load_latest_rankings() -> Optional[list[str]]:
    """Return tickers from the most recent rankings file, in rank order."""
    files = sorted(OUTPUTS.glob("rankings_*.json"), reverse=True)
    if not files:
        return None
    data = json.loads(files[0].read_text())
    return [item["ticker"] for item in data]


# ── Portfolio ─────────────────────────────────────────────────────────────────


def save_portfolio(portfolio: dict) -> None:
    _dump(OUTPUTS / "portfolio_current.json", portfolio)


def load_current_portfolio() -> Optional[dict]:
    path = OUTPUTS / "portfolio_current.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── Trades ────────────────────────────────────────────────────────────────────


def save_trades(trades: dict, month_str: str) -> None:
    _dump(OUTPUTS / f"trades_{month_str}.json", trades)


# ── LLM scores ────────────────────────────────────────────────────────────────


def save_llm_scores(llm_scores: dict, overrides: dict, date_str: str) -> None:
    _dump(
        OUTPUTS / f"llm_scores_{date_str}.json",
        {"date": date_str, "scores": llm_scores, "overrides": overrides},
    )


# ── Human-readable summary ────────────────────────────────────────────────────


def print_portfolio_summary(portfolio: dict) -> None:
    holdings = portfolio.get("holdings", [])
    weights  = portfolio.get("weights", {})
    regime   = portfolio.get("regime", {})

    print("\n" + "=" * 60)
    print(f"PORTFOLIO SUMMARY  ({date.today()})")
    print("=" * 60)
    print(f"Stocks held : {len(holdings)}")
    print(f"Regime filter triggered : {regime.get('regime_triggered', False)}")
    print(f"Crash filter triggered  : {regime.get('crash_triggered', False)}")
    print(f"Equity fraction : {regime.get('equity_fraction', 1.0):.0%}")
    print()
    print(f"{'Ticker':<8}  {'Weight':>7}  {'Half-wt':>8}")
    print("-" * 30)
    half = set(portfolio.get("llm_half_weight", []))
    for t in holdings:
        flag = "  *" if t in half else ""
        print(f"{t:<8}  {weights.get(t, 0):>7.2%}{flag}")
    cash = portfolio.get("weights", {}).get("SGOV") or portfolio.get("weights", {}).get("BIL")
    if cash:
        print(f"{'SGOV/BIL':<8}  {cash:>7.2%}  (cash buffer)")
    print()
    if portfolio.get("llm_exits"):
        print("LLM risk-flag exits :", portfolio["llm_exits"])
    if portfolio.get("llm_boosts"):
        print("LLM post-earnings boosts:", portfolio["llm_boosts"])
    if portfolio.get("sells"):
        print("Sell (fell out of top-40):", portfolio["sells"])
    print("=" * 60 + "\n")
