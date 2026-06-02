"""
Technical indicator computation for stop-loss management and move classification.
All functions operate on pandas Series of daily adjusted close prices.
"""
import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder's RSI. Returns float 0-100, or 50.0 if insufficient data."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float]:
    """Returns (macd_line, signal_line). Positive macd > signal = bullish."""
    if len(prices) < slow + signal:
        return 0.0, 0.0
    ema_fast   = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow   = prices.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def compute_ma50_extension(prices: pd.Series) -> float:
    """How far current price is above/below 50d MA. 0.05 = 5% above."""
    if len(prices) < 50:
        return 0.0
    ma50 = prices.tail(50).mean()
    if ma50 == 0:
        return 0.0
    return float(prices.iloc[-1] / ma50 - 1)


def compute_rel_volume(volumes: pd.Series, lookback: int = 20) -> float:
    """Today's volume relative to 20-day average. 2.0 = double normal."""
    if len(volumes) < lookback + 1:
        return 1.0
    avg_vol = volumes.iloc[-(lookback+1):-1].mean()
    if avg_vol == 0:
        return 1.0
    return float(volumes.iloc[-1] / avg_vol)


def compute_entry_quality(prices: pd.Series, rsi_max: float = 70, ext_max: float = 0.10) -> bool:
    """
    Returns True if this stock is a good entry right now.
    Rejects stocks that are overbought (RSI > rsi_max) or too extended (> ext_max above MA50).
    """
    rsi = compute_rsi(prices)
    ext = compute_ma50_extension(prices)
    return rsi <= rsi_max and ext <= ext_max


def classify_position(
    prices: pd.Series,
    volumes: pd.Series,
    ticker: str,
    entry_price: float,
    peak_price: float,
    hard_stop_pct: float = 0.12,
    trail_stop_pct: float = 0.15,
) -> dict:
    """
    Classify what action to take on a current holding.

    Returns:
        action        — "SELL" | "WATCH" | "HOLD"
        reason        — human-readable explanation
        updated_peak  — new peak price (>=old peak)
        hard_stop     — hard stop price level
        trail_stop    — current trailing stop level
        gain_pct      — % gain/loss from entry
        rsi           — current RSI
    """
    current = float(prices.iloc[-1])
    updated_peak = max(peak_price if peak_price else entry_price, current)

    hard_stop  = entry_price * (1 - hard_stop_pct)
    trail_stop = updated_peak * (1 - trail_stop_pct)
    gain_pct   = (current / entry_price) - 1

    # ── Stop checks (highest priority) ──────────────────────────────────
    if current <= hard_stop:
        return {
            "action": "SELL", "stop_type": "hard",
            "reason": f"Hard stop hit: {gain_pct*100:.1f}% from entry (stop was ${hard_stop:.2f})",
            "updated_peak": updated_peak, "hard_stop": hard_stop,
            "trail_stop": trail_stop, "gain_pct": gain_pct,
            "rsi": compute_rsi(prices),
        }

    if current <= trail_stop and gain_pct > 0:
        drawdown_from_peak = (current / updated_peak) - 1
        return {
            "action": "SELL", "stop_type": "trail",
            "reason": f"Trailing stop hit: {drawdown_from_peak*100:.1f}% from peak of ${updated_peak:.2f}",
            "updated_peak": updated_peak, "hard_stop": hard_stop,
            "trail_stop": trail_stop, "gain_pct": gain_pct,
            "rsi": compute_rsi(prices),
        }

    # ── Technical signals ────────────────────────────────────────────────
    rsi           = compute_rsi(prices)
    macd, sig     = compute_macd(prices)
    ext           = compute_ma50_extension(prices)
    rel_vol       = compute_rel_volume(volumes)
    macd_healthy  = macd > sig

    # Crowding / distribution: overbought RSI + abnormal volume
    crowded   = rsi > 73 and rel_vol > 2.0
    # Repriced: big gain + extended above MA + momentum fading
    repriced  = gain_pct > 0.18 and ext > 0.12 and not macd_healthy
    # Healthy: MACD positive, not over-extended
    healthy   = macd_healthy and ext < 0.12 and rsi < 70

    if gain_pct >= 0:
        if crowded:
            return {
                "action": "SELL",
                "reason": f"Distribution signal: RSI {rsi:.0f} + {rel_vol:.1f}× normal volume. General population moving in.",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
        elif repriced:
            return {
                "action": "WATCH",
                "reason": f"Move may be repriced: +{gain_pct*100:.0f}%, {ext*100:.0f}% above MA50, MACD fading. Tighten trailing stop.",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
        elif healthy:
            return {
                "action": "HOLD",
                "reason": f"Momentum healthy: +{gain_pct*100:.1f}%, MACD positive, RSI {rsi:.0f}",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
        else:
            return {
                "action": "HOLD",
                "reason": f"+{gain_pct*100:.1f}%, RSI {rsi:.0f}, no distribution signal",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
    else:
        # Losing position
        pct_to_stop = (current / hard_stop - 1) * 100
        if rel_vol > 1.5 and gain_pct < -0.06:
            return {
                "action": "WATCH",
                "reason": f"Volume-confirmed breakdown: {gain_pct*100:.1f}%, {rel_vol:.1f}× normal vol. {pct_to_stop:.1f}% to hard stop.",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
        elif gain_pct < -0.08:
            return {
                "action": "WATCH",
                "reason": f"Deep pullback: {gain_pct*100:.1f}%, {pct_to_stop:.1f}% to hard stop (${hard_stop:.2f})",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
        else:
            return {
                "action": "HOLD",
                "reason": f"Normal pullback: {gain_pct*100:.1f}%, low-volume, likely recovers",
                "updated_peak": updated_peak, "hard_stop": hard_stop,
                "trail_stop": trail_stop, "gain_pct": gain_pct, "rsi": rsi,
            }
