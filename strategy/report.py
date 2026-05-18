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


# ── HTML email report ─────────────────────────────────────────────────────────


def generate_html_report(
    portfolio: dict,
    trades: dict,
    scores_df: pd.DataFrame,
    theses: dict,
    chart_b64: Optional[str],
) -> str:
    """
    Build a rich HTML email body with factor scores, theses, and an embedded chart.
    """
    holdings = portfolio.get("holdings", [])
    weights  = portfolio.get("weights", {})
    regime   = portfolio.get("regime", {})
    buys     = trades.get("buys", [])
    sells    = trades.get("sells", [])
    month    = trades.get("month", str(date.today())[:7])
    half_set = set(portfolio.get("llm_half_weight", []))
    exits    = portfolio.get("llm_exits", [])
    boosts   = portfolio.get("llm_boosts", [])

    # ── Stock cards ───────────────────────────────────────────────────
    cards_html = ""
    for ticker in holdings:
        row  = scores_df.loc[ticker] if ticker in scores_df.index else {}
        m    = float(row.get("momentum_score", 0)) if hasattr(row, "get") else 0.0
        v    = float(row.get("value_score",    0)) if hasattr(row, "get") else 0.0
        q    = float(row.get("quality_score",  0)) if hasattr(row, "get") else 0.0
        comp = float(row.get("composite",      0)) if hasattr(row, "get") else 0.0
        wt   = weights.get(ticker, 0)
        thesis = theses.get(ticker, "")

        new_badge  = '<span style="background:#d4edda;color:#155724;padding:2px 7px;border-radius:12px;font-size:0.75em;font-weight:bold;margin-left:6px">NEW</span>' if ticker in buys else ""
        half_badge = '<span style="background:#fff3cd;color:#856404;padding:2px 7px;border-radius:12px;font-size:0.75em;margin-left:4px">½ wt</span>' if ticker in half_set else ""

        def _score_badge(label, val, color):
            sign = "+" if val >= 0 else ""
            return (
                f'<span style="background:{color}22;color:{color};padding:3px 9px;'
                f'border-radius:12px;font-size:0.78em;font-weight:600">'
                f'{label} {sign}{val:.2f}</span>'
            )

        cards_html += f"""
        <div style="border-bottom:1px solid #f0f0f0;padding:16px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
            <span style="font-size:1.25em;font-weight:700;color:#1a1a2e">{ticker}</span>
            <span>{new_badge}{half_badge}</span>
            <span style="color:#666;font-size:0.95em;font-weight:600">{wt:.1%}</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:5px;margin:7px 0">
            {_score_badge("📈 Mom", m, "#e94560")}
            {_score_badge("💎 Val", v, "#0f9b8e")}
            {_score_badge("⭐ Qual", q, "#f5a623")}
            {_score_badge("🎯 Total", comp, "#6c63ff")}
          </div>
          <div style="color:#555;font-size:0.9em;line-height:1.6;margin-top:6px">{thesis}</div>
        </div>"""

    # ── Risk overlay ──────────────────────────────────────────────────
    eq_pct  = f"{regime.get('equity_fraction', 1.0):.0%}"
    reg_clr = "#dc3545" if regime.get("regime_triggered") else "#28a745"
    cra_clr = "#dc3545" if regime.get("crash_triggered")  else "#28a745"
    reg_txt = "TRIGGERED ⚠️" if regime.get("regime_triggered") else "OFF ✅"
    cra_txt = "TRIGGERED ⚠️" if regime.get("crash_triggered")  else "OFF ✅"

    # ── Chart ─────────────────────────────────────────────────────────
    # Gmail blocks base64 inline images — chart is sent as an email attachment instead.
    chart_section = ""
    if chart_b64:
        chart_section = """
        <div style="background:white;margin:8px 4px;padding:16px 20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08)">
          <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">📊 Charts</h2>
          <p style="color:#666;font-size:0.9em;margin:0">
            Factor score breakdown and 3-month price performance chart attached as
            <strong>portfolio_chart.png</strong>. Tap the attachment below to view.
          </p>
        </div>"""

    # ── LLM alerts ────────────────────────────────────────────────────
    alerts_html = ""
    if exits:
        alerts_html += f'<p style="color:#721c24;background:#f8d7da;padding:10px;border-radius:6px">⚠️ <strong>LLM risk-flag exits:</strong> {", ".join(exits)}</p>'
    if boosts:
        alerts_html += f'<p style="color:#155724;background:#d4edda;padding:10px;border-radius:6px">🚀 <strong>Post-earnings boosts:</strong> {", ".join(boosts)}</p>'

    buys_str  = ", ".join(buys)  if buys  else "None"
    sells_str = ", ".join(sells) if sells else "None"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;color:#333;max-width:720px;margin:0 auto;padding:0">

  <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:28px 24px;margin:0">
    <div style="font-size:0.85em;opacity:0.7;margin-bottom:4px">Momentum × Value × LLM Strategy</div>
    <div style="font-size:1.6em;font-weight:700">📈 Monthly Rebalance</div>
    <div style="font-size:1.1em;opacity:0.85;margin-top:4px">{month}</div>
  </div>

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">🛒 Action Required</h2>
    <div style="background:#d4edda;border-left:4px solid #28a745;padding:13px 16px;border-radius:0 8px 8px 0;margin:8px 0">
      <strong style="color:#155724">🟢 BUY ({len(buys)}):</strong>
      <span style="color:#155724"> {buys_str}</span>
    </div>
    <div style="background:#f8d7da;border-left:4px solid #dc3545;padding:13px 16px;border-radius:0 8px 8px 0;margin:8px 0">
      <strong style="color:#721c24">🔴 SELL ({len(sells)}):</strong>
      <span style="color:#721c24"> {sells_str}</span>
    </div>
    {alerts_html}
  </div>

  {chart_section}

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">📋 Portfolio Holdings ({len(holdings)} stocks)</h2>
    {cards_html}
  </div>

  <div style="background:white;margin:8px 4px;padding:20px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08)">
    <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;margin-top:0">⚡ Risk Overlay</h2>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <div style="flex:1;min-width:120px;text-align:center;padding:14px;background:#f8f9fa;border-radius:8px">
        <div style="font-size:1.6em;font-weight:700">{eq_pct}</div>
        <div style="color:#666;font-size:0.82em;margin-top:3px">Equity Fraction</div>
      </div>
      <div style="flex:1;min-width:120px;text-align:center;padding:14px;background:#f8f9fa;border-radius:8px">
        <div style="font-size:1.1em;font-weight:700;color:{reg_clr}">{reg_txt}</div>
        <div style="color:#666;font-size:0.82em;margin-top:3px">Regime Filter</div>
      </div>
      <div style="flex:1;min-width:120px;text-align:center;padding:14px;background:#f8f9fa;border-radius:8px">
        <div style="font-size:1.1em;font-weight:700;color:{cra_clr}">{cra_txt}</div>
        <div style="color:#666;font-size:0.82em;margin-top:3px">Crash Filter</div>
      </div>
    </div>
  </div>

  <div style="text-align:center;color:#aaa;font-size:0.8em;padding:20px">
    Generated {date.today()} · Momentum × Value × LLM Strategy
  </div>

</body>
</html>"""
