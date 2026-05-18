"""
Weekly LLM event scan — runs every Sunday evening.

What it does:
  1. Load the most recent monthly rankings.
  2. Fetch the last 7 days of news headlines for the top-50 ranked stocks.
  3. Score each ticker with the LLM (sentiment / surprise / risk_flag).
  4. Apply override rules and identify any mid-month risk-flag exits.
  5. Generate an HTML email report and save to /tmp/weekly_report.html.

This run NEVER adds new buy entries — only risk-flag exits are acted on
mid-month (per the cadence spec in Part 4 of the research document).
"""
import logging
import sys
from datetime import date

import pandas as pd

from config import LLM_SCAN_TOP_N
from strategy.data import fetch_news_for_tickers
from strategy.llm import apply_overrides, score_news_batch
from strategy.report import (
    load_current_portfolio,
    load_latest_rankings,
    print_portfolio_summary,
    save_llm_scores,
    save_portfolio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_html(llm_scores: dict, overrides: dict, portfolio: dict | None) -> str:
    exits     = overrides.get("exits", [])
    boosts    = overrides.get("boosts", [])
    holdings  = portfolio.get("holdings", []) if portfolio else []
    today_str = date.today().isoformat()

    # ── Alerts ────────────────────────────────────────────────────────
    alerts_html = ""
    if exits:
        alerts_html += (
            f'<div style="background:#f8d7da;border-left:4px solid #dc3545;'
            f'padding:13px 16px;border-radius:0 8px 8px 0;margin:8px 0">'
            f'<strong style="color:#721c24">⚠️ RISK-FLAG EXITS — SELL NOW:</strong>'
            f'<span style="color:#721c24"> {", ".join(exits)}</span></div>'
        )
    else:
        alerts_html = (
            '<div style="background:#d4edda;border-left:4px solid #28a745;'
            'padding:13px 16px;border-radius:0 8px 8px 0;margin:8px 0">'
            '<strong style="color:#155724">✅ No risk-flag exits this week. Hold all positions.</strong></div>'
        )
    if boosts:
        alerts_html += (
            f'<div style="background:#fff3cd;border-left:4px solid #ffc107;'
            f'padding:13px 16px;border-radius:0 8px 8px 0;margin:8px 0">'
            f'<strong style="color:#856404">🚀 Post-earnings candidates (apply next rebal):</strong>'
            f'<span style="color:#856404"> {", ".join(boosts)}</span></div>'
        )

    # ── Sentiment table — only show current holdings ──────────────────
    rows_html = ""
    display_tickers = [t for t in holdings if t in llm_scores]
    if not display_tickers:
        display_tickers = list(llm_scores.keys())[:15]

    for ticker in display_tickers:
        s = llm_scores[ticker]
        sentiment  = s.get("sentiment",  50)
        surprise   = s.get("surprise",   0)
        risk_flag  = s.get("risk_flag",  0)
        rationale  = s.get("rationale",  "")
        n_news     = s.get("n_headlines", 0)

        # Colour-code sentiment
        if sentiment >= 70:
            sent_color, sent_bg = "#155724", "#d4edda"
        elif sentiment <= 30:
            sent_color, sent_bg = "#721c24", "#f8d7da"
        else:
            sent_color, sent_bg = "#555", "#f8f9fa"

        risk_color = "#dc3545" if risk_flag > 50 else ("#ffc107" if risk_flag > 25 else "#28a745")
        news_note  = f"{n_news} headlines" if n_news else "no news found"

        rows_html += f"""
        <tr>
          <td style="padding:10px 8px;font-weight:700;color:#1a1a2e;white-space:nowrap">{ticker}</td>
          <td style="padding:10px 8px;text-align:center">
            <span style="background:{sent_bg};color:{sent_color};padding:3px 9px;
                         border-radius:12px;font-size:0.85em;font-weight:600">{sentiment:.0f}</span>
          </td>
          <td style="padding:10px 8px;text-align:center;color:#555;font-size:0.9em">{surprise:.0f}</td>
          <td style="padding:10px 8px;text-align:center">
            <span style="color:{risk_color};font-weight:600;font-size:0.9em">{risk_flag:.0f}</span>
          </td>
          <td style="padding:10px 8px;color:#666;font-size:0.82em;font-style:italic">{rationale or news_note}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;color:#333;max-width:720px;margin:0 auto;padding:0">

  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:24px 20px">
    <div style="font-size:0.82em;opacity:0.7;margin-bottom:4px">Momentum × Value × LLM Strategy</div>
    <div style="font-size:1.5em;font-weight:700">🔍 Weekly LLM Scan</div>
    <div style="font-size:1em;opacity:0.85;margin-top:4px">{today_str}</div>
  </div>

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
              box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
      Mid-Week Action
    </h2>
    {alerts_html}
    <p style="color:#888;font-size:0.82em;margin:10px 0 0">
      New buy entries are only added at the monthly rebalance. Mid-week exits fire immediately.
    </p>
  </div>

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;
              box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">
      Sentiment Scores — Current Holdings
    </h2>
    <p style="color:#666;font-size:0.85em;margin:0 0 12px">
      Each score is the LLM's read of the past 7 days of company-specific news.
      <strong>Sentiment</strong> = bullish/bearish (50 = neutral).
      <strong>Surprise</strong> = new info not yet priced in.
      <strong>Risk</strong> = probability of a material adverse event.
    </p>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:0.9em">
        <thead>
          <tr style="background:#f8f9fa;border-bottom:2px solid #eee">
            <th style="padding:10px 8px;text-align:left;color:#1a1a2e">Ticker</th>
            <th style="padding:10px 8px;text-align:center;color:#1a1a2e">Sentiment</th>
            <th style="padding:10px 8px;text-align:center;color:#1a1a2e">Surprise</th>
            <th style="padding:10px 8px;text-align:center;color:#1a1a2e">Risk</th>
            <th style="padding:10px 8px;text-align:left;color:#1a1a2e">Note</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <div style="text-align:center;color:#aaa;font-size:0.8em;padding:20px">
    Generated {today_str} · Next full rebalance: 1st of month
  </div>
</body>
</html>"""


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
    rank_series = pd.Series(
        {t: float(len(rankings) - i) for i, t in enumerate(rankings[:LLM_SCAN_TOP_N])}
    )
    scores_stub = pd.DataFrame({"composite": rank_series})
    overrides = apply_overrides(scores_stub, llm_scores)

    today_str = date.today().isoformat()
    save_llm_scores(llm_scores, overrides, today_str)

    # ── 5. Mid-month exits ────────────────────────────────────────────
    portfolio = load_current_portfolio()
    if overrides["exits"]:
        logger.warning("RISK-FLAG EXITS this week: %s", overrides["exits"])
        if portfolio:
            removed = [t for t in overrides["exits"] if t in portfolio.get("holdings", [])]
            if removed:
                portfolio["holdings"] = [t for t in portfolio["holdings"] if t not in removed]
                for t in removed:
                    portfolio["weights"].pop(t, None)
                portfolio["llm_exits"] = portfolio.get("llm_exits", []) + removed
                save_portfolio(portfolio)
                logger.info("Removed from portfolio: %s", removed)

    # ── 6. HTML report ────────────────────────────────────────────────
    html = _build_html(llm_scores, overrides, portfolio)
    with open("/tmp/weekly_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    if portfolio:
        print_portfolio_summary(portfolio)
    logger.info("Weekly scan complete. %d tickers had news.", n_with_news)


if __name__ == "__main__":
    main()
