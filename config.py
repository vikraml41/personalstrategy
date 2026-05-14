# Signal weights (must sum to 1.0)
MOMENTUM_WEIGHT = 0.40
VALUE_WEIGHT = 0.35
QUALITY_WEIGHT = 0.25

# Portfolio construction
PORTFOLIO_TARGET = 20       # target number of stocks
PORTFOLIO_MIN = 15          # if fewer qualify, hold remainder in cash ETF
MAX_PER_SECTOR = 3          # hard sector cap (GICS level)
SELL_RANK_THRESHOLD = 40    # sell any holding that falls out of top 40

# Universe filters
MARKET_CAP_FLOOR = 2_000_000_000   # $2B minimum
MIN_HISTORY_DAYS = 252              # 1 year of price history required

# Risk overlay thresholds
SPY_LOOKBACK_MONTHS = 10    # months for SPY trend signal
SPY_MA_DAYS = 200           # moving average window for regime filter
CRASH_LOOKBACK_MONTHS = 24  # months for momentum-crash detection
VOL_WINDOW_DAYS = 63        # ~3 months rolling vol window
VOL_PERCENTILE = 80         # top 20th percentile triggers crash filter

# LLM override thresholds (0-100 scale)
LLM_RISK_EXCLUDE_THRESHOLD = 70     # risk_flag > this in top decile → exclude
LLM_SENTIMENT_BOOST_THRESHOLD = 80  # sentiment > this to qualify for boost
LLM_SURPRISE_BOOST_THRESHOLD = 60   # surprise > this to qualify for boost
LLM_HALF_WEIGHT_THRESHOLD = 30      # sentiment < this → half weight

# LLM scan parameters
LLM_SCAN_TOP_N = 50         # score top N stocks weekly
NEWS_LOOKBACK_DAYS = 7      # days of headlines to fetch
FILING_LOOKBACK_DAYS = 90   # window for recent earnings/10-Q/10-K

# Cash fallback tickers (held when equity exposure is reduced)
CASH_TICKERS = ["SGOV", "BIL"]

# Anthropic model settings
LLM_MODEL = "claude-opus-4-7"
LLM_TEMPERATURE = 0.0
LLM_STABILITY_RUNS = 3         # run each item N times and average
LLM_STABILITY_TEMPERATURE = 0.2
