"""
Daily portfolio check — runs Mon–Fri after market close.

What it does:
  1. Load current portfolio state.
  2. Download today's prices + volume for all holdings.
  3. For each holding: check stops, classify the move.
  4. Update peak prices (ratchets up, never down).
  5. Exit any positions that hit a stop or distribution signal.
  6. Check if a full rebalance should be triggered.
  7. Email the user only when action is required.
  8. Write /tmp/rebalance_needed if a rebalance should run.
"""
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from config import (
    HARD_STOP_PCT,
    REBALANCE_EXIT_TRIGGER,
    REBALANCE_MAX_WEEKS,
    TRAILING_STOP_PCT,
)
from strategy.data import fetch_prices_and_volume
from strategy.report import load_current_portfolio, save_portfolio
from strategy.technical import classify_position

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUTS = Path("outputs")


def _weeks_since(date_str: str) -> float:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days / 7
    except Exception:
        return 0.0


def _build_daily_html(classifications: dict, exits: list[str], rebalance_triggered: bool, portfolio: dict) -> str:
    today_str = date.today().isoformat()
    portfolio_size = portfolio.get("portfolio_size", 0)
    shares_map     = portfolio.get("shares", {})

    # Sell orders
    sell_rows = ""
    for ticker in exits:
        sh  = shares_map.get(ticker, 0)
        cl  = classifications.get(ticker, {})
        sell_rows += f"""
        <tr style="background:#fff5f5">
          <td style="padding:10px 8px;font-weight:700;color:#dc3545">{ticker}</td>
          <td style="padding:10px 8px;color:#dc3545;font-weight:600">SELL</td>
          <td style="padding:10px 8px">Sell all <strong>{sh:.3f} shares</strong> at market</td>
          <td style="padding:10px 8px;color:#666;font-size:0.85em">{cl.get('reason','')}</td>
        </tr>"""

    # Hold/watch rows
    hold_rows = ""
    for ticker, cl in sorted(classifications.items()):
        if ticker in exits:
            continue
        action = cl.get("action", "HOLD")
        gain   = cl.get("gain_pct", 0)
        gain_str = f"{gain*100:+.1f}%"
        color = "#28a745" if action == "HOLD" else "#ffc107"
        icon  = "✅" if action == "HOLD" else "⚠️"
        sh    = shares_map.get(ticker, 0)
        hard_stop = cl.get("hard_stop", 0)
        hold_rows += f"""
        <tr>
          <td style="padding:8px;font-weight:700;color:#1a1a2e">{ticker}</td>
          <td style="padding:8px">
            <span style="color:{color};font-weight:600">{icon} {action}</span>
          </td>
          <td style="padding:8px;font-size:0.85em">{gain_str} | {sh:.3f} sh | stop ${hard_stop:.2f}</td>
          <td style="padding:8px;color:#666;font-size:0.82em">{cl.get('reason','')}</td>
        </tr>"""

    rebal_banner = ""
    if rebalance_triggered:
        rebal_banner = """
        <div style="background:#fff3cd;border-left:4px solid #ffc107;padding:13px 16px;
                    border-radius:0 8px 8px 0;margin:8px 0">
          <strong style="color:#856404">🔄 Full rebalance triggered — new rankings running separately.</strong>
        </div>"""

    sell_section = ""
    if exits:
        sell_section = f"""
        <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.08)">
          <h2 style="color:#dc3545;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
            🚨 Action Required — Execute in Fidelity
          </h2>
          <table style="width:100%;border-collapse:collapse;font-size:0.9em">
            <thead>
              <tr style="background:#f8d7da;border-bottom:2px solid #eee">
                <th style="padding:10px 8px;text-align:left">Ticker</th>
                <th style="padding:10px 8px;text-align:left">Action</th>
                <th style="padding:10px 8px;text-align:left">Order</th>
                <th style="padding:10px 8px;text-align:left">Reason</th>
              </tr>
            </thead>
            <tbody>{sell_rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;color:#333;max-width:720px;margin:0 auto;padding:0">

  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:24px 20px">
    <div style="font-size:0.82em;opacity:0.7;margin-bottom:4px">Momentum × Value Strategy</div>
    <div style="font-size:1.5em;font-weight:700">{"🚨 Portfolio Alert" if exits else "📊 Daily Check"}</div>
    <div style="font-size:1em;opacity:0.85;margin-top:4px">{today_str}</div>
  </div>

  {sell_section}
  {rebal_banner}

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
              box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
      Portfolio Status
    </h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <thead>
        <tr style="background:#f8f9fa;border-bottom:2px solid #eee">
          <th style="padding:8px;text-align:left">Ticker</th>
          <th style="padding:8px;text-align:left">Status</th>
          <th style="padding:8px;text-align:left">P&amp;L | Shares | Stop</th>
          <th style="padding:8px;text-align:left">Note</th>
        </tr>
      </thead>
      <tbody>{hold_rows}</tbody>
    </table>
  </div>

  <div style="text-align:center;color:#aaa;font-size:0.8em;padding:20px">
    Generated {today_str} · Stops checked daily after market close
  </div>
</body>
</html>"""


def main() -> None:
    logger.info("── Daily check  %s ──────────────────────────", date.today())

    portfolio = load_current_portfolio()
    if not portfolio:
        logger.info("No portfolio found — nothing to check.")
        return

    holdings   = portfolio.get("holdings", [])
    if not holdings:
        logger.info("Portfolio is empty.")
        return

    entry_prices = portfolio.get("entry_prices", {})
    peak_prices  = portfolio.get("peak_prices", {})
    shares_map   = portfolio.get("shares", {})

    # ── Download prices + volume ──────────────────────────────────────────
    logger.info("Downloading prices for %d holdings...", len(holdings))
    prices, volumes = fetch_prices_and_volume(holdings, period="3mo")

    # ── Classify each position ────────────────────────────────────────────
    classifications: dict = {}
    updated_peaks:  dict  = dict(peak_prices)
    exits: list[str]      = []

    for ticker in holdings:
        if ticker not in prices.columns:
            logger.warning("No price data for %s — skipping", ticker)
            continue

        entry = entry_prices.get(ticker)
        peak  = peak_prices.get(ticker)

        if not entry:
            logger.warning("No entry price for %s — skipping stop check", ticker)
            continue

        vol_series = volumes[ticker].dropna() if ticker in volumes.columns else prices[ticker].dropna() * 0 + 1e6

        cl = classify_position(
            prices  = prices[ticker].dropna(),
            volumes = vol_series,
            ticker  = ticker,
            entry_price   = entry,
            peak_price    = peak or entry,
            hard_stop_pct = HARD_STOP_PCT,
            trail_stop_pct= TRAILING_STOP_PCT,
        )
        classifications[ticker] = cl
        updated_peaks[ticker]   = cl["updated_peak"]

        logger.info("%s  %s  %+.1f%%  RSI=%.0f  reason=%s",
                    ticker, cl["action"], cl.get("gain_pct", 0)*100,
                    cl.get("rsi", 0), cl["reason"][:60])

        if cl["action"] == "SELL":
            exits.append(ticker)

    # ── Apply exits ───────────────────────────────────────────────────────
    if exits:
        logger.warning("EXITS triggered: %s", exits)
        portfolio["holdings"]              = [t for t in holdings if t not in exits]
        portfolio["exits_since_rebalance"] = portfolio.get("exits_since_rebalance", 0) + len(exits)
        # Remove from shares, entry/peak prices
        for t in exits:
            portfolio.get("shares", {}).pop(t, None)
            portfolio.get("entry_prices", {}).pop(t, None)
            portfolio.get("weights", {}).pop(t, None)
        portfolio.setdefault("historical_exits", []).extend([
            {"ticker": t, "date": date.today().isoformat(),
             "reason": classifications[t].get("reason", "")} for t in exits
        ])

    # Update peak prices
    portfolio["peak_prices"] = updated_peaks

    # ── Check rebalance triggers ──────────────────────────────────────────
    last_rebal  = portfolio.get("last_rebalance_date", "2026-01-01")
    weeks_since = _weeks_since(last_rebal)
    exit_count  = portfolio.get("exits_since_rebalance", 0)

    rebalance_triggered = (
        weeks_since >= REBALANCE_MAX_WEEKS
        or exit_count >= REBALANCE_EXIT_TRIGGER
    )

    if rebalance_triggered:
        logger.info("Rebalance trigger: %.1f weeks since last, %d exits since last", weeks_since, exit_count)
        Path("/tmp/rebalance_needed").touch()

    # ── Persist updated portfolio ─────────────────────────────────────────
    save_portfolio(portfolio)

    # ── Email only if action needed ───────────────────────────────────────
    action_needed = bool(exits) or rebalance_triggered
    if action_needed:
        html = _build_daily_html(classifications, exits, rebalance_triggered, portfolio)
        with open("/tmp/daily_report.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Daily report written to /tmp/daily_report.html (%d exits)", len(exits))
    else:
        logger.info("No action needed today — no email sent.")

    logger.info("Daily check complete.")


if __name__ == "__main__":
    main()
