# Signal weights (must sum to 1.0)
MOMENTUM_WEIGHT = 0.40
VALUE_WEIGHT    = 0.35
QUALITY_WEIGHT  = 0.25

# Portfolio construction
PORTFOLIO_TARGET    = 10   # target number of stocks
PORTFOLIO_MIN       = 7    # hold cash ETF if fewer qualify
MAX_PER_SECTOR      = 2    # hard GICS sector cap
SELL_RANK_THRESHOLD = 20   # sell any holding that drops below this composite rank

# Universe filters
MARKET_CAP_FLOOR = 2_000_000_000  # $2B minimum
MIN_HISTORY_DAYS = 252             # 1 year of price history required

# Risk overlay
SPY_LOOKBACK_MONTHS  = 10
SPY_MA_DAYS          = 200
CRASH_LOOKBACK_MONTHS= 24
VOL_WINDOW_DAYS      = 63
VOL_PERCENTILE       = 80

# ── Portfolio management ──────────────────────────────────────────────────────
PORTFOLIO_SIZE         = 4427.43   # total account value in dollars
HARD_STOP_PCT          = 0.12      # -12% from entry price → immediate exit
TRAILING_STOP_PCT      = 0.15      # -15% from peak price → exit (protects gains)

# Exit / profit-taking signals (rules-based, no LLM)
RSI_CROWD_THRESHOLD    = 73        # RSI above this = overbought/crowded
REL_VOL_CROWD          = 2.5       # daily volume > 2.5× 20d avg = distribution signal
MA50_EXTEND_THRESHOLD  = 0.12      # >12% above 50d MA = extended, tighten stops
MACD_FAST              = 12
MACD_SLOW              = 26
MACD_SIGNAL            = 9
RSI_PERIOD             = 14

# Entry quality filter (applied at stock selection)
ENTRY_MAX_EXTENSION    = 0.10      # skip stocks >10% above 50d MA at buy time
ENTRY_RSI_MAX          = 70        # skip stocks with RSI >70 at buy time

# Dynamic rebalance triggers
REBALANCE_MAX_WEEKS    = 6         # force rebalance after 6 weeks regardless
REBALANCE_EXIT_TRIGGER = 3         # trigger rebalance after 3+ exits since last rebalance

# Cash fallback
CASH_TICKERS = ["SGOV", "BIL"]
