"""
LLM event-override layer (Pillar C).

The LLM is a *feature extractor*, not the strategy core.
It converts unstructured text (news, earnings calls, filing diffs) into
three numeric scores per stock: sentiment, surprise, risk_flag.

Prompt structure follows Part 5 of the research document exactly.
For output stability each item is scored LLM_STABILITY_RUNS times at
temperature LLM_STABILITY_TEMPERATURE and the scores are averaged.
"""
import json
import logging
import time
from typing import Optional

import anthropic

from config import (
    CRASH_LOOKBACK_MONTHS,
    LLM_HALF_WEIGHT_THRESHOLD,
    LLM_MODEL,
    LLM_RISK_EXCLUDE_THRESHOLD,
    LLM_SENTIMENT_BOOST_THRESHOLD,
    LLM_STABILITY_RUNS,
    LLM_STABILITY_TEMPERATURE,
    LLM_SURPRISE_BOOST_THRESHOLD,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial analyst. Output JSON only, no prose.\n"
    'For each input, produce {"sentiment": int 0-100, "surprise": int 0-100, '
    '"risk_flag": int 0-100, "rationale": string (max 30 words)}.\n'
    "Definitions:\n"
    "- sentiment: 50 = neutral; >70 clearly bullish; <30 clearly bearish.\n"
    "- surprise: degree to which this conveys new information not already priced in.\n"
    "- risk_flag: probability of material adverse event "
    "(litigation, restatement, guidance cut, exec departure, going concern)."
)


# ── Core scoring ─────────────────────────────────────────────────────────────


def _score_once(client: anthropic.Anthropic, text: str, temperature: float) -> Optional[dict]:
    """Call the LLM once and return parsed JSON, or None on failure."""
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=150,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = response.content[0].text.strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for LLM output")
        return None
    except Exception as exc:
        logger.warning("LLM call error: %s", exc)
        return None


def score_text(client: anthropic.Anthropic, text: str) -> dict:
    """
    Score a single text item.
    Runs LLM_STABILITY_RUNS times and averages numeric fields for stability.
    Falls back to neutral scores if all calls fail.
    """
    n = LLM_STABILITY_RUNS
    temp = LLM_STABILITY_TEMPERATURE if n > 1 else LLM_TEMPERATURE
    results = []

    for attempt in range(n):
        parsed = _score_once(client, text, temp)
        if parsed:
            results.append(parsed)
        if attempt < n - 1:
            time.sleep(0.5)

    if not results:
        return {"sentiment": 50, "surprise": 0, "risk_flag": 0, "rationale": "scoring failed"}

    return {
        "sentiment":  sum(r.get("sentiment",  50) for r in results) / len(results),
        "surprise":   sum(r.get("surprise",    0) for r in results) / len(results),
        "risk_flag":  sum(r.get("risk_flag",   0) for r in results) / len(results),
        "rationale":  results[0].get("rationale", ""),
    }


# ── Batch scoring ─────────────────────────────────────────────────────────────


def score_news_batch(news_data: dict) -> dict:
    """
    Score news headlines for multiple tickers.

    Args:
        news_data: {ticker: [headline, ...]}
    Returns:
        {ticker: {sentiment, surprise, risk_flag, rationale, n_headlines}}
    """
    client = anthropic.Anthropic()
    scores: dict = {}

    for i, (ticker, headlines) in enumerate(news_data.items()):
        if not headlines:
            scores[ticker] = {
                "sentiment": 50, "surprise": 0, "risk_flag": 0,
                "rationale": "no news available", "n_headlines": 0,
            }
            continue

        combined = "\n".join(f"- {h}" for h in headlines[:7])
        text = f"Company: {ticker}\nRecent news headlines:\n{combined}"

        result = score_text(client, text)
        result["n_headlines"] = len(headlines)
        scores[ticker] = result

        if i and i % 10 == 0:
            logger.info("  LLM scored %d/%d tickers", i, len(news_data))

    return scores


def score_top_candidates(tickers: list[str], news_data: dict) -> dict:
    """Score the top-N candidates for the monthly rebalance."""
    subset = {t: news_data.get(t, []) for t in tickers}
    return score_news_batch(subset)


# ── Override rules ────────────────────────────────────────────────────────────


def apply_overrides(scores_df, llm_scores: dict) -> dict:
    """
    Apply the three LLM override rules from Layer 2 of the architecture.

    Rule 1: top-decile + risk_flag > threshold  →  exclude this month
    Rule 2: 2nd–5th decile + sentiment high + surprise high  →  boost to buy
    Rule 3: top-decile + sentiment < threshold  →  still buy but half weight

    Returns {'exits': [...], 'boosts': [...], 'half_weight': [...]}
    """
    composite = scores_df["composite"] if hasattr(scores_df, "columns") else scores_df
    n = len(composite.dropna())
    if n == 0:
        return {"exits": [], "boosts": [], "half_weight": []}

    # Percentile rank: 0.0 = best, 1.0 = worst
    pct_rank = composite.rank(ascending=False, pct=True)

    exits: list[str] = []
    boosts: list[str] = []
    half_weight: list[str] = []

    for ticker, llm in llm_scores.items():
        if ticker not in pct_rank.index:
            continue
        rank = pct_rank[ticker]  # 0–1, lower = better

        if rank <= 0.10 and llm.get("risk_flag", 0) > LLM_RISK_EXCLUDE_THRESHOLD:
            exits.append(ticker)
        elif (
            0.10 < rank <= 0.50
            and llm.get("sentiment", 50) > LLM_SENTIMENT_BOOST_THRESHOLD
            and llm.get("surprise",   0)  > LLM_SURPRISE_BOOST_THRESHOLD
        ):
            boosts.append(ticker)
        elif rank <= 0.10 and llm.get("sentiment", 50) < LLM_HALF_WEIGHT_THRESHOLD:
            half_weight.append(ticker)

    return {"exits": exits, "boosts": boosts, "half_weight": half_weight}
