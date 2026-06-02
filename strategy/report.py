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


# ── Human-readable summary ────────────────────────────────────────────────────


def print_portfolio_summary(portfolio: dict) -> None:
    holdings     = portfolio.get("holdings", [])
    weights      = portfolio.get("weights", {})
    entry_prices = portfolio.get("entry_prices", {})
    peak_prices  = portfolio.get("peak_prices", {})
    regime       = portfolio.get("regime", {})

    print("\n" + "=" * 70)
    print(f"PORTFOLIO SUMMARY  ({date.today()})")
    print("=" * 70)
    print(f"Stocks held : {len(holdings)}")
    print(f"Regime filter triggered : {regime.get('regime_triggered', False)}")
    print(f"Crash filter triggered  : {regime.get('crash_triggered', False)}")
    print(f"Equity fraction : {regime.get('equity_fraction', 1.0):.0%}")
    print()
    print(f"{'Ticker':<8}  {'Weight':>7}  {'Entry':>8}  {'Peak':>8}")
    print("-" * 40)
    for t in holdings:
        ep   = entry_prices.get(t, 0)
        peak = peak_prices.get(t, 0)
        print(f"{t:<8}  {weights.get(t, 0):>7.2%}  {ep:>8.2f}  {peak:>8.2f}")
    print("=" * 70 + "\n")
