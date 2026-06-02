"""
Monthly/triggered rebalance.

Pipeline:
  1. Universe  — S&P 500 (cached 30 days)
  2. Prices    — 2-year adjusted close
  3. Momentum pre-filter — top 300 stocks
  4. Fundamentals — value & quality data
  5. All signals — M, V, Q
  6. Composite — z-scored rank combination
  7. Entry quality filter — reject overbought/extended stocks at buy time
  8. Portfolio construction — sector-capped, equal-weighted, regime-overlaid
  9. Persist — rankings, portfolio, trade list
 10. HTML report with exact Fidelity trade orders
"""
import logging
import sys
from datetime import date
from pathlib import Path

from config import PORTFOLIO_SIZE, PORTFOLIO_TARGET
from strategy.charts import generate_portfolio_chart
from strategy.data import fetch_fundamentals, fetch_prices, fetch_prices_and_volume
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

MOMENTUM_PREFILTER = 300


def _build_html(buys, sells, portfolio, scores_df, prices_short):
    today_str      = date.today().isoformat()
    month_str      = date.today().strftime("%Y-%m")
    holdings       = portfolio.get("holdings", [])
    shares_map     = portfolio.get("shares", {})
    entry_prices   = portfolio.get("entry_prices", {})
    psize          = portfolio.get("portfolio_size", PORTFOLIO_SIZE)
    n_pos          = max(len(holdings), 1)
    target_per_pos = psize / n_pos

    sell_rows = ""
    for ticker in sells:
        sh = shares_map.get(ticker, 0)
        sell_rows += f"""
        <tr style="background:#fff5f5">
          <td style="padding:10px 8px;font-weight:700;color:#dc3545">{ticker}</td>
          <td style="padding:10px 8px;color:#dc3545;font-weight:600">SELL</td>
          <td style="padding:10px 8px">Sell all <strong>{sh:.3f} shares</strong> at market open</td>
        </tr>"""

    buy_rows = ""
    for ticker in buys:
        cp  = float(prices_short[ticker].iloc[-1]) if ticker in prices_short.columns else 0
        est = round(target_per_pos / cp, 3) if cp > 0 else 0
        lim = round(cp * 1.005, 2)
        buy_rows += f"""
        <tr style="background:#f0fff4">
          <td style="padding:10px 8px;font-weight:700;color:#28a745">{ticker}</td>
          <td style="padding:10px 8px;color:#28a745;font-weight:600">BUY</td>
          <td style="padding:10px 8px">Buy ~<strong>{est} shares</strong>
            · limit ${lim:.2f} · (≈${target_per_pos:.0f})</td>
        </tr>"""

    keep_rows = ""
    for ticker in holdings:
        if ticker in buys or ticker in sells:
            continue
        sh   = shares_map.get(ticker, 0)
        ep   = entry_prices.get(ticker, 0)
        cp   = float(prices_short[ticker].iloc[-1]) if ticker in prices_short.columns else ep
        gain = f"{(cp/ep - 1)*100:+.1f}%" if ep else "—"
        keep_rows += f"""
        <tr>
          <td style="padding:8px;font-weight:700;color:#1a1a2e">{ticker}</td>
          <td style="padding:8px;color:#555;font-weight:600">HOLD</td>
          <td style="padding:8px;color:#666">Keep {sh:.3f} shares · {gain} · no action</td>
        </tr>"""

    regime  = portfolio.get("regime", {})
    eq_pct  = f"{regime.get('equity_fraction', 1.0):.0%}"
    reg_clr = "#dc3545" if regime.get("regime_triggered") else "#28a745"
    cra_clr = "#dc3545" if regime.get("crash_triggered")  else "#28a745"
    reg_txt = "TRIGGERED ⚠️" if regime.get("regime_triggered") else "OFF ✅"
    cra_txt = "TRIGGERED ⚠️" if regime.get("crash_triggered")  else "OFF ✅"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;color:#333;max-width:720px;margin:0 auto;padding:0">

  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:28px 24px">
    <div style="font-size:0.85em;opacity:0.7;margin-bottom:4px">Momentum × Value Strategy</div>
    <div style="font-size:1.6em;font-weight:700">📈 Portfolio Rebalance</div>
    <div style="font-size:1.1em;opacity:0.85;margin-top:4px">{month_str} · Account ${psize:,.0f}</div>
  </div>

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
              box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
      🛒 Trade Orders — Execute in Fidelity
    </h2>
    <p style="color:#666;font-size:0.85em;margin:0 0 12px">
      Target per position: <strong>~${target_per_pos:,.0f}</strong>
      · Positions: <strong>{n_pos}</strong>
    </p>
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <thead>
        <tr style="background:#f8f9fa;border-bottom:2px solid #eee">
          <th style="padding:10px 8px;text-align:left">Ticker</th>
          <th style="padding:10px 8px;text-align:left">Action</th>
          <th style="padding:10px 8px;text-align:left">Order Details</th>
        </tr>
      </thead>
      <tbody>
        {sell_rows}
        {buy_rows}
        {keep_rows}
      </tbody>
    </table>
  </div>

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
              box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
      ⚡ Risk Overlay
    </h2>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <div style="flex:1;min-width:120px;text-align:center;padding:14px;background:#f8f9fa;border-radius:8px">
        <div style="font-size:1.6em;font-weight:700">{eq_pct}</div>
        <div style="color:#666;font-size:0.82em;margin-top:3px">Equity Fraction</div>
      </div>
      <div style="flex:1;min-width:120px;text-align:center;padding:14px;background:#f8f9fa;border-radius:8px">
        <div style="font-size:1em;font-weight:700;color:{reg_clr}">{reg_txt}</div>
        <div style="color:#666;font-size:0.82em;margin-top:3px">Regime Filter</div>
      </div>
      <div style="flex:1;min-width:120px;text-align:center;padding:14px;background:#f8f9fa;border-radius:8px">
        <div style="font-size:1em;font-weight:700;color:{cra_clr}">{cra_txt}</div>
        <div style="color:#666;font-size:0.82em;margin-top:3px">Crash Filter</div>
      </div>
    </div>
  </div>

  <div style="text-align:center;color:#aaa;font-size:0.8em;padding:20px">
    Generated {today_str} · Momentum × Value Strategy
  </div>
</body>
</html>"""


def main() -> None:
    today     = date.today()
    month_str = today.strftime("%Y-%m")
    logger.info("── Rebalance  %s ─────────────────────────────", month_str)

    # 1. Universe
    logger.info("Step 1/8  Getting universe...")
    tickers = get_universe()
    logger.info("Universe: %d tickers", len(tickers))

    # 2. Prices (2y for signal computation)
    logger.info("Step 2/8  Downloading 2y prices...")
    prices_2y = fetch_prices(tickers, period="2y")
    if prices_2y.empty:
        logger.error("No price data. Aborting.")
        sys.exit(1)
    logger.info("Prices: %d tickers × %d days", prices_2y.shape[1], prices_2y.shape[0])

    # 3. Momentum pre-filter
    logger.info("Step 3/8  Momentum pre-filter...")
    try:
        mom_df = compute_momentum_signals(prices_2y)
    except ValueError as exc:
        logger.error("Momentum failed: %s", exc)
        sys.exit(1)

    _etf_set    = set(CORE_ETFS)
    stocks_only = mom_df[~mom_df.index.isin(_etf_set)]
    top_candidates = stocks_only["M_composite"].nlargest(MOMENTUM_PREFILTER).index.tolist()
    if not top_candidates:
        logger.error("No candidates after momentum filter. Aborting.")
        sys.exit(1)
    logger.info("Pre-filter: %d stocks", len(top_candidates))

    # 4. Fundamentals
    logger.info("Step 4/8  Fetching fundamentals...")
    fundamentals = fetch_fundamentals(top_candidates)
    n_ok = sum(1 for v in fundamentals.values() if v.get("market_cap"))
    logger.info("Fundamentals OK for %d/%d", n_ok, len(top_candidates))

    # 5. All signals
    logger.info("Step 5/8  Computing signals (M, V, Q)...")
    signals_df = compute_all_signals(prices_2y, fundamentals, top_candidates)

    # 6. Composite score
    logger.info("Step 6/8  Computing composite scores...")
    scores_df = compute_composite_scores(signals_df)
    logger.info("Top 5:\n%s", scores_df[["composite", "decile"]].head())

    # 7. Short-term prices for entry quality filter
    logger.info("Step 7/8  Entry quality check (RSI + MA extension)...")
    top_scored   = scores_df.head(100).index.tolist()
    prices_short, _ = fetch_prices_and_volume(top_scored, period="3mo")

    # 8. Portfolio construction (entry quality applied inside)
    logger.info("Step 8/8  Constructing portfolio...")
    prev      = load_current_portfolio()
    portfolio = construct_portfolio(scores_df, fundamentals, prices_short, prev)

    new_holdings  = set(portfolio["holdings"])
    prev_holdings = set(prev.get("holdings", [])) if prev else set()
    buys          = sorted(new_holdings - prev_holdings)
    sells         = sorted(portfolio["sells"])

    # Carry forward position tracking for held stocks
    portfolio["shares"]       = {}
    portfolio["entry_prices"] = {}
    portfolio["peak_prices"]  = {}
    portfolio["entry_dates"]  = {}
    if prev:
        for t in portfolio["holdings"]:
            portfolio["shares"][t]       = prev.get("shares", {}).get(t, 0)
            portfolio["entry_prices"][t] = prev.get("entry_prices", {}).get(t, 0)
            portfolio["peak_prices"][t]  = prev.get("peak_prices", {}).get(t, 0)
            portfolio["entry_dates"][t]  = prev.get("entry_dates", {}).get(t, today.isoformat())

    # Initialize entry data for new buys
    n_pos = max(len(portfolio["holdings"]), 1)
    psize = prev.get("portfolio_size", PORTFOLIO_SIZE) if prev else PORTFOLIO_SIZE
    target_per_pos = psize / n_pos
    for ticker in buys:
        cp = float(prices_short[ticker].iloc[-1]) if ticker in prices_short.columns else 0
        portfolio["shares"][ticker]       = round(target_per_pos / cp, 3) if cp > 0 else 0
        portfolio["entry_prices"][ticker] = cp
        portfolio["peak_prices"][ticker]  = cp
        portfolio["entry_dates"][ticker]  = today.isoformat()

    portfolio["last_rebalance_date"]   = today.isoformat()
    portfolio["exits_since_rebalance"] = 0
    portfolio["portfolio_size"]        = psize
    portfolio["cash"]                  = prev.get("cash", 0) if prev else 0

    trades = {
        "date": today.isoformat(), "month": month_str,
        "buys": buys, "sells": sells,
        "holdings_after": sorted(new_holdings),
    }

    save_rankings(scores_df, month_str)
    save_portfolio(portfolio)
    save_trades(trades, month_str)

    # Chart (attached to email)
    generate_portfolio_chart(scores_df, portfolio, prices_2y)

    # HTML report with exact trade orders
    html = _build_html(buys, sells, portfolio, scores_df, prices_short)
    with open("/tmp/monthly_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print_portfolio_summary(portfolio)
    logger.info("Buys  (%d): %s", len(buys),  buys)
    logger.info("Sells (%d): %s", len(sells), sells)
    logger.info("Rebalance complete.")


if __name__ == "__main__":
    main()
