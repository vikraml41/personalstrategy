"""
Universe management: S&P 500 stocks + core ETFs.
Caches the list locally; refreshes every 30 days.
"""
import json
import logging
import requests
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CORE_ETFS = [
    "SPY", "QQQ", "IWM",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

UNIVERSE_CACHE = Path("outputs/universe_cache.json")

# Public GitHub dataset — works reliably from GitHub Actions runners
_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/main/data/constituents.csv"
)

# Hardcoded fallback — last-resort if all network sources fail
_SP500_FALLBACK = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM",
    "ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE",
    "AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON",
    "APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP",
    "AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI","BAX","BDX","WRB","BBY",
    "BIO","TECH","BIIB","BLK","BX","BA","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR",
    "BXP","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CAT","CBOE",
    "CBRE","CDW","CE","COR","CNC","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB",
    "CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL",
    "CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST",
    "CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DELL","DAL","DVN",
    "DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DTE","DUK","DD",
    "EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM",
    "EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC","EXPE","EXPD",
    "EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI",
    "FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
    "GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC",
    "HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB",
    "HUM","HBAN","HII","IBM","IEX","IDXX","ITW","ILMN","INCY","IR","PODD","INTC","ICE",
    "IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JKHY","J","JBL","JPM",
    "K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH",
    "LRCX","LW","LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB",
    "MTB","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR",
    "MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE",
    "NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY",
    "OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PANW","PARA","PH",
    "PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG",
    "PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM",
    "DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP",
    "ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS",
    "SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SYF","SNPS",
    "SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN",
    "TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UDR",
    "ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ",
    "VRTX","VTRS","VICI","V","VMC","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC",
    "WFC","WELL","WST","WDC","WMB","WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]


def get_sp500_tickers() -> list[str]:
    """
    Fetch S&P 500 tickers. Tries sources in order:
      1. Public GitHub CSV dataset (works from GitHub Actions)
      2. Wikipedia (works from local/non-cloud environments)
      3. Hardcoded fallback list
    """
    # Source 1: public GitHub CSV (most reliable on Actions runners)
    try:
        resp = requests.get(_SP500_CSV_URL, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        col = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
        if col:
            tickers = df[col].str.strip().str.replace(".", "-", regex=False).tolist()
            if len(tickers) > 400:
                logger.info("Fetched %d tickers from GitHub dataset", len(tickers))
                return tickers
    except Exception as exc:
        logger.warning("GitHub CSV fetch failed: %s", exc)

    # Source 2: Wikipedia
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", flavor="lxml"
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            logger.info("Fetched %d tickers from Wikipedia", len(tickers))
            return tickers
    except Exception as exc:
        logger.warning("Wikipedia fetch failed: %s", exc)

    # Source 3: hardcoded fallback
    logger.warning("All live sources failed — using hardcoded S&P 500 fallback list (%d tickers)", len(_SP500_FALLBACK))
    return list(_SP500_FALLBACK)


def get_universe() -> list[str]:
    """
    Return investment universe: S&P 500 stocks + core ETFs.
    Uses a cached list if it is less than 30 days old AND has >100 stock tickers.
    """
    if UNIVERSE_CACHE.exists():
        cache = json.loads(UNIVERSE_CACHE.read_text())
        cached_date = date.fromisoformat(cache.get("date", "2000-01-01"))
        n_stocks = len([t for t in cache.get("tickers", []) if t not in set(CORE_ETFS)])
        if (date.today() - cached_date).days < 30 and n_stocks > 100:
            logger.info("Using cached universe (%d tickers, %d stocks)", len(cache["tickers"]), n_stocks)
            return cache["tickers"]
        elif n_stocks <= 100:
            logger.warning("Cached universe only has %d stocks — refreshing.", n_stocks)

    stocks = get_sp500_tickers()
    all_tickers = list(dict.fromkeys(stocks + CORE_ETFS))

    UNIVERSE_CACHE.parent.mkdir(exist_ok=True)
    UNIVERSE_CACHE.write_text(
        json.dumps({"tickers": all_tickers, "date": date.today().isoformat()}, indent=2)
    )
    logger.info("Universe: %d tickers (%d stocks + %d ETFs)", len(all_tickers), len(stocks), len(CORE_ETFS))
    return all_tickers
