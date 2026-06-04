"""
Daily portfolio check — runs Mon–Fri after market close.

Always emails a full daily snapshot. Only modifies portfolio holdings
when a stop is hit or a distribution signal fires.
"""
import logging
from datetime import date, datetime
from pathlib import Path

import yfinance as yf

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


def _weeks_since(date_str: str) -> float:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days / 7
    except Exception:
        return 0.0


def _spy_context(prices) -> dict:
    """Return SPY daily change and 1-month return for market context."""
    try:
        if "SPY" in prices.columns:
            spy = prices["SPY"].dropna()
        else:
            spy = yf.download("SPY", period="2mo", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
        if len(spy) < 2:
            return {}
        day_chg  = float(spy.iloc[-1] / spy.iloc[-2] - 1)
        mo_chg   = float(spy.iloc[-1] / spy.iloc[-21] - 1) if len(spy) >= 21 else 0.0
        return {"price": float(spy.iloc[-1]), "day_chg": day_chg, "mo_chg": mo_chg}
    except Exception:
        return {}


def _build_html(
    classifications: dict,
    exits: list[str],
    rebalance_triggered: bool,
    portfolio: dict,
    prices,
    spy: dict,
) -> str:
    today_str    = date.today().isoformat()
    shares_map   = portfolio.get("shares", {})
    entry_prices = portfolio.get("entry_prices", {})
    holdings     = portfolio.get("holdings", [])

    # ── Compute portfolio totals ──────────────────────────────────────────
    total_value   = 0.0
    total_cost    = 0.0
    total_day_chg = 0.0
    for t in holdings:
        sh  = shares_map.get(t, 0)
        ep  = entry_prices.get(t, 0)
        if t in prices.columns and sh:
            curr = float(prices[t].iloc[-1])
            prev = float(prices[t].iloc[-2]) if len(prices[t].dropna()) >= 2 else curr
            total_value   += curr * sh
            total_cost    += ep   * sh
            total_day_chg += (curr - prev) * sh
    cash         = portfolio.get("cash", 0)
    total_value += cash
    total_gain   = total_value - total_cost - cash  # gain on invested portion
    total_gain_pct = total_gain / total_cost if total_cost else 0

    # ── Market bar ────────────────────────────────────────────────────────
    spy_color  = "#28a745" if spy.get("day_chg", 0) >= 0 else "#dc3545"
    spy_sign   = "+" if spy.get("day_chg", 0) >= 0 else ""
    spy_mo_sign= "+" if spy.get("mo_chg", 0) >= 0 else ""
    market_bar = f"""
    <div style="background:#16213e;color:white;padding:10px 20px;display:flex;gap:24px;flex-wrap:wrap;font-size:0.88em">
      <span>📈 <strong>SPY</strong> ${spy.get('price',0):.2f}
        <span style="color:{spy_color};font-weight:700"> {spy_sign}{spy.get('day_chg',0)*100:.2f}% today</span>
        <span style="opacity:0.6"> · {spy_mo_sign}{spy.get('mo_chg',0)*100:.1f}% 1mo</span>
      </span>
      <span>💼 <strong>Portfolio</strong> ${total_value:,.2f}
        <span style="color:{'#28a745' if total_day_chg >= 0 else '#dc3545'};font-weight:700">
          {'+' if total_day_chg >= 0 else ''}${total_day_chg:,.2f} today</span>
        <span style="color:{'#28a745' if total_gain >= 0 else '#dc3545'}">
          · {'+' if total_gain >= 0 else ''}${total_gain:,.2f} total ({total_gain_pct*100:+.1f}%)</span>
      </span>
    </div>"""

    # ── Action alert (only shown if exits needed) ─────────────────────────
    action_section = ""
    if exits:
        sell_rows = ""
        for t in exits:
            sh  = shares_map.get(t, 0)
            cl  = classifications.get(t, {})
            sell_rows += f"""
            <tr style="background:#fff5f5">
              <td style="padding:10px 8px;font-weight:700;color:#dc3545">{t}</td>
              <td style="padding:10px 8px">Sell all <strong>{sh:.3f} shares</strong> at market open</td>
              <td style="padding:10px 8px;color:#666;font-size:0.85em">{cl.get('reason','')}</td>
            </tr>"""
        action_section = f"""
        <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.08)">
          <h2 style="color:#dc3545;border-bottom:2px solid #fee;padding-bottom:8px;margin-top:0">
            🚨 Action Required — Execute in Fidelity at Open
          </h2>
          <table style="width:100%;border-collapse:collapse;font-size:0.9em">
            <thead>
              <tr style="background:#f8d7da;border-bottom:2px solid #eee">
                <th style="padding:10px 8px;text-align:left">Ticker</th>
                <th style="padding:10px 8px;text-align:left">Order</th>
                <th style="padding:10px 8px;text-align:left">Reason</th>
              </tr>
            </thead>
            <tbody>{sell_rows}</tbody>
          </table>
        </div>"""

    if rebalance_triggered and not exits:
        action_section += """
        <div style="background:#fff3cd;border-left:4px solid #ffc107;padding:13px 16px;
                    border-radius:0 8px 8px 0;margin:8px 4px">
          <strong style="color:#856404">🔄 Rebalance triggered — new rankings running, expect a separate email shortly.</strong>
        </div>"""

    # ── Holdings table ────────────────────────────────────────────────────
    rows = ""
    for t in holdings:
        cl      = classifications.get(t, {})
        action  = cl.get("action", "HOLD")
        gain    = cl.get("gain_pct", 0)
        rsi     = cl.get("rsi", 50)
        sh      = shares_map.get(t, 0)
        ep      = entry_prices.get(t, 0)
        hard_stop = cl.get("hard_stop", 0)
        trail_stop= cl.get("trail_stop", 0)

        curr = float(prices[t].iloc[-1]) if t in prices.columns else ep
        prev = float(prices[t].iloc[-2]) if t in prices.columns and len(prices[t].dropna()) >= 2 else curr
        day_chg = (curr / prev - 1) if prev else 0
        curr_val = curr * sh

        # Colour coding
        if action == "SELL":
            row_bg, status_color, icon = "#fff5f5", "#dc3545", "🚨"
        elif action == "WATCH":
            row_bg, status_color, icon = "#fffbf0", "#ffc107", "⚠️"
        else:
            row_bg, status_color, icon = "white", "#28a745", "✅"

        gain_color = "#28a745" if gain >= 0 else "#dc3545"
        day_color  = "#28a745" if day_chg >= 0 else "#dc3545"

        # Stop buffer (how far above hard stop)
        stop_buf = ((curr / hard_stop) - 1) * 100 if hard_stop else 0
        stop_str = f"${hard_stop:.2f} ({stop_buf:.1f}% away)"

        rows += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #f0f0f0">
          <td style="padding:10px 8px;font-weight:700;color:#1a1a2e;white-space:nowrap">{t}</td>
          <td style="padding:10px 8px;text-align:right">
            ${curr:.2f}<br>
            <span style="color:{day_color};font-size:0.82em">{'+' if day_chg >= 0 else ''}{day_chg*100:.2f}%</span>
          </td>
          <td style="padding:10px 8px;text-align:right">
            <span style="color:{gain_color};font-weight:600">{gain*100:+.1f}%</span><br>
            <span style="color:#888;font-size:0.82em">${curr_val:,.0f}</span>
          </td>
          <td style="padding:10px 8px;text-align:center">
            <span style="font-size:0.85em;color:#555">RSI {rsi:.0f}</span>
          </td>
          <td style="padding:10px 8px;font-size:0.82em;color:#888">{stop_str}</td>
          <td style="padding:10px 8px">
            <span style="color:{status_color};font-weight:600;font-size:0.88em">{icon} {action}</span><br>
            <span style="color:#666;font-size:0.78em">{cl.get('reason','')[:60]}</span>
          </td>
        </tr>"""

    # ── Cash row ──────────────────────────────────────────────────────────
    if cash:
        rows += f"""
        <tr style="background:#f8f9fa;border-bottom:1px solid #f0f0f0">
          <td style="padding:10px 8px;font-weight:700;color:#888">CASH</td>
          <td style="padding:10px 8px;text-align:right;color:#888">${cash:,.2f}</td>
          <td colspan="4" style="padding:10px 8px;color:#aaa;font-size:0.85em">Money market — awaiting deployment</td>
        </tr>"""

    title = "🚨 Portfolio Alert" if exits else "📊 Daily Portfolio Update"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;color:#333;max-width:740px;margin:0 auto;padding:0">

  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:22px 20px 10px">
    <div style="font-size:0.82em;opacity:0.7;margin-bottom:4px">Momentum × Value Strategy</div>
    <div style="font-size:1.5em;font-weight:700">{title}</div>
    <div style="font-size:0.95em;opacity:0.85;margin-top:2px">{today_str}</div>
  </div>
  {market_bar}

  {action_section}

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
              box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
      Holdings
    </h2>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.88em">
        <thead>
          <tr style="background:#f8f9fa;border-bottom:2px solid #eee">
            <th style="padding:10px 8px;text-align:left">Ticker</th>
            <th style="padding:10px 8px;text-align:right">Price / Day</th>
            <th style="padding:10px 8px;text-align:right">Total G/L</th>
            <th style="padding:10px 8px;text-align:center">RSI</th>
            <th style="padding:10px 8px;text-align:left">Hard Stop</th>
            <th style="padding:10px 8px;text-align:left">Signal</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </div>

  <div style="text-align:center;color:#aaa;font-size:0.8em;padding:16px">
    Checked daily after market close · Hard stop -12% · Trailing stop -15% from peak
  </div>
</body>
</html>"""


def main() -> None:
    logger.info("── Daily check  %s ──────────────────────────", date.today())

    portfolio = load_current_portfolio()
    if not portfolio:
        logger.info("No portfolio found.")
        return

    holdings = portfolio.get("holdings", [])
    if not holdings:
        logger.info("Portfolio is empty.")
        return

    entry_prices = portfolio.get("entry_prices", {})
    peak_prices  = portfolio.get("peak_prices", {})

    # ── Download prices + volume (include SPY for market context) ─────────
    tickers_to_fetch = list(set(holdings + ["SPY"]))
    logger.info("Downloading prices for %d tickers...", len(tickers_to_fetch))
    prices, volumes = fetch_prices_and_volume(tickers_to_fetch, period="3mo")

    spy = _spy_context(prices)
    logger.info("SPY: %.2f%% today, %.1f%% 1mo", spy.get("day_chg", 0)*100, spy.get("mo_chg", 0)*100)

    # ── Classify each position ────────────────────────────────────────────
    classifications: dict = {}
    updated_peaks:   dict = dict(peak_prices)
    exits: list[str]      = []

    for ticker in holdings:
        if ticker not in prices.columns:
            logger.warning("No price data for %s", ticker)
            continue

        entry = entry_prices.get(ticker)
        peak  = peak_prices.get(ticker)
        if not entry:
            logger.warning("No entry price for %s — skipping stop check", ticker)
            continue

        vol_series = volumes[ticker].dropna() if ticker in volumes.columns else prices[ticker].dropna() * 0 + 1e6

        cl = classify_position(
            prices         = prices[ticker].dropna(),
            volumes        = vol_series,
            ticker         = ticker,
            entry_price    = entry,
            peak_price     = peak or entry,
            hard_stop_pct  = HARD_STOP_PCT,
            trail_stop_pct = TRAILING_STOP_PCT,
        )
        classifications[ticker] = cl
        updated_peaks[ticker]   = cl["updated_peak"]

        logger.info("%s  %s  %+.1f%%  RSI=%.0f  %s",
                    ticker, cl["action"], cl.get("gain_pct", 0)*100,
                    cl.get("rsi", 0), cl["reason"][:70])

        if cl["action"] == "SELL":
            exits.append(ticker)

    # ── Apply exits ───────────────────────────────────────────────────────
    if exits:
        logger.warning("EXITS triggered: %s", exits)
        portfolio["holdings"]              = [t for t in holdings if t not in exits]
        portfolio["exits_since_rebalance"] = portfolio.get("exits_since_rebalance", 0) + len(exits)
        for t in exits:
            portfolio.get("shares", {}).pop(t, None)
            portfolio.get("entry_prices", {}).pop(t, None)
            portfolio.get("weights", {}).pop(t, None)
        portfolio.setdefault("historical_exits", []).extend([
            {"ticker": t, "date": date.today().isoformat(),
             "reason": classifications[t].get("reason", "")} for t in exits
        ])

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
        logger.info("Rebalance trigger: %.1f weeks elapsed, %d exits since last", weeks_since, exit_count)
        Path("/tmp/rebalance_needed").touch()

    # ── Persist ───────────────────────────────────────────────────────────
    save_portfolio(portfolio)

    # ── Always write daily report ─────────────────────────────────────────
    html = _build_html(classifications, exits, rebalance_triggered, portfolio, prices, spy)
    with open("/tmp/daily_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Daily report written (%d exits, rebalance=%s)", len(exits), rebalance_triggered)


if __name__ == "__main__":
    main()
