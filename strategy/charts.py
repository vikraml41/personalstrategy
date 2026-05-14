"""
Chart generation for the monthly email report.
Produces a two-panel PNG (base64-encoded) suitable for embedding in HTML:
  Left  — horizontal stacked bar showing momentum / value / quality factor scores
  Right — 3-month price performance of holdings vs SPY benchmark
"""
import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for server/CI use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Dark theme palette
BG       = "#1a1a2e"
PANEL_BG = "#16213e"
MOM_CLR  = "#e94560"
VAL_CLR  = "#0f9b8e"
QUAL_CLR = "#f5a623"
SPY_CLR  = "#ffffff"
GRID_CLR = "#2a2a4a"
TEXT_CLR = "#e0e0e0"


def _apply_dark_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(color=GRID_CLR, linestyle="--", linewidth=0.5, alpha=0.6)


def generate_portfolio_chart(
    scores_df: pd.DataFrame,
    portfolio: dict,
    prices: pd.DataFrame,
) -> str | None:
    """
    Build and return a base64-encoded PNG string for the email.
    Returns None if chart generation fails.
    """
    holdings = portfolio.get("holdings", [])
    if not holdings:
        return None

    try:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, max(4, len(holdings) * 0.55))
        )
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(wspace=0.35)

        # ── Left panel: factor score breakdown ───────────────────────
        hold_scores = scores_df.reindex(holdings).fillna(0)

        m = hold_scores.get("momentum_score", pd.Series(0, index=holdings)).values
        v = hold_scores.get("value_score",    pd.Series(0, index=holdings)).values
        q = hold_scores.get("quality_score",  pd.Series(0, index=holdings)).values

        y = np.arange(len(holdings))
        h = 0.6

        # Separate positive and negative bars so they stack correctly
        def _safe_bar(ax, y, vals, left, color, label):
            ax.barh(y, vals, left=left, height=h, color=color,
                    alpha=0.85, label=label, edgecolor="none")

        _safe_bar(ax1, y, m, np.zeros(len(m)), MOM_CLR,  "Momentum")
        _safe_bar(ax1, y, v, m,                VAL_CLR,  "Value")
        _safe_bar(ax1, y, q, m + v,            QUAL_CLR, "Quality")

        ax1.set_yticks(y)
        ax1.set_yticklabels(holdings, fontsize=9, color=TEXT_CLR)
        ax1.set_xlabel("Factor z-score contribution", fontsize=9)
        ax1.set_title("Factor Score Breakdown", fontsize=11, fontweight="bold", pad=10)
        ax1.axvline(0, color=TEXT_CLR, linewidth=0.8, alpha=0.4)
        ax1.legend(
            loc="lower right", fontsize=8,
            facecolor=PANEL_BG, labelcolor=TEXT_CLR, framealpha=0.8,
        )
        _apply_dark_style(ax1)

        # ── Right panel: 3-month price performance ────────────────────
        tickers_to_plot = [t for t in holdings if t in prices.columns]
        if "SPY" in prices.columns:
            tickers_to_plot_with_spy = tickers_to_plot + ["SPY"]
        else:
            tickers_to_plot_with_spy = tickers_to_plot

        recent = prices[tickers_to_plot_with_spy].tail(63).copy()
        first_valid = recent.apply(lambda s: s.first_valid_index())
        normalized = pd.DataFrame(index=recent.index)
        for col in recent.columns:
            base_idx = first_valid[col]
            if base_idx is not None:
                base = recent.loc[base_idx, col]
                if base and base != 0:
                    normalized[col] = (recent[col] / base - 1) * 100

        cmap = plt.cm.get_cmap("tab10", len(tickers_to_plot))
        for i, ticker in enumerate(tickers_to_plot):
            if ticker in normalized.columns:
                ax2.plot(
                    normalized.index, normalized[ticker],
                    linewidth=1.6, color=cmap(i), alpha=0.85, label=ticker,
                )

        if "SPY" in normalized.columns:
            ax2.plot(
                normalized.index, normalized["SPY"],
                linewidth=2.2, color=SPY_CLR, linestyle="--",
                alpha=0.9, label="SPY", zorder=10,
            )

        ax2.axhline(0, color=TEXT_CLR, linewidth=0.6, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax2.set_xlabel("Date", fontsize=9)
        ax2.set_ylabel("Return", fontsize=9)
        ax2.set_title("3-Month Price Performance vs SPY", fontsize=11, fontweight="bold", pad=10)
        ax2.legend(
            loc="upper left", fontsize=7, ncol=2,
            facecolor=PANEL_BG, labelcolor=TEXT_CLR, framealpha=0.8,
        )
        _apply_dark_style(ax2)

        # ── Encode ────────────────────────────────────────────────────
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)
        plt.close("all")
        return None
