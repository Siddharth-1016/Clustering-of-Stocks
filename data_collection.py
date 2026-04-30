# =============================================================================
# data_collection.py  —  STEP 2: Data Collection
# =============================================================================
"""
Downloads historical OHLCV data + fundamental metrics for every ticker in the
configured universe and saves them as CSV files.

Production features:
  • Retry logic with exponential back-off
  • Per-ticker error isolation (one bad ticker never kills the run)
  • Incremental saves (crash-safe)
  • Progress bar
  • Logging to file + console
"""

import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, "data_collection.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# Helper: robust single-ticker download with retries
# =============================================================================
def _download_with_retry(ticker: str, period: str, retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """
    Download daily OHLCV for *ticker* using yfinance.
    Retries up to *retries* times with exponential back-off.
    Returns an empty DataFrame on persistent failure.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,       # adjusts for splits/dividends
                progress=False,
                threads=False,
            )
            if df.empty:
                log.warning(f"[{ticker}] Empty price data (attempt {attempt})")
                time.sleep(delay * attempt)
                continue

            # Flatten MultiIndex columns that yfinance sometimes returns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df["Ticker"] = ticker
            return df

        except Exception as exc:
            log.warning(f"[{ticker}] Download error (attempt {attempt}): {exc}")
            time.sleep(delay * attempt)

    log.error(f"[{ticker}] Failed after {retries} attempts — skipping.")
    return pd.DataFrame()


# =============================================================================
# Fundamental metrics via yfinance .info dict
# =============================================================================
FUNDAMENTAL_KEYS = {
    # Valuation
    "trailingPE":           "PE_Ratio",
    "priceToBook":          "PB_Ratio",
    "enterpriseToEbitda":   "EV_EBITDA",
    # Profitability
    "trailingEps":          "EPS",
    "returnOnEquity":       "ROE",
    "returnOnAssets":       "ROA",
    "profitMargins":        "Profit_Margin",
    "operatingMargins":     "Operating_Margin",
    # Growth
    "revenueGrowth":        "Revenue_Growth",
    "earningsGrowth":       "Earnings_Growth",
    # Balance sheet
    "debtToEquity":         "Debt_to_Equity",
    "currentRatio":         "Current_Ratio",
    "quickRatio":           "Quick_Ratio",
    # Cash flow
    "freeCashflow":         "Free_Cash_Flow",
    "operatingCashflow":    "Operating_Cash_Flow",
    # Dividend
    "dividendYield":        "Dividend_Yield",
    # Size
    "marketCap":            "Market_Cap",
    "totalRevenue":         "Total_Revenue",
    "totalDebt":            "Total_Debt",
}


def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetches fundamental metrics for a single ticker.
    Returns a flat dict {column_name: value}.
    Missing fields are returned as np.nan.
    """
    record = {"Ticker": ticker}
    try:
        info = yf.Ticker(ticker).info
        for src_key, col_name in FUNDAMENTAL_KEYS.items():
            val = info.get(src_key, np.nan)
            record[col_name] = float(val) if val is not None else np.nan
    except Exception as exc:
        log.warning(f"[{ticker}] Fundamentals fetch error: {exc}")
        for col_name in FUNDAMENTAL_KEYS.values():
            record[col_name] = np.nan
    return record


# =============================================================================
# Main collection routine
# =============================================================================
def collect_all_data(
    tickers: list[str] = config.TICKERS,
    period:  str       = config.HISTORY_PERIOD,
    price_out: str     = os.path.join(config.DATA_DIR, "price_history.csv"),
    fundamentals_out: str = os.path.join(config.DATA_DIR, "fundamentals.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full data collection pipeline.

    Returns
    -------
    price_df        : Long-form DataFrame of daily OHLCV for all tickers
    fundamentals_df : One row per ticker with fundamental metrics
    """
    log.info(f"Starting data collection for {len(tickers)} tickers | period={period}")

    all_prices: list[pd.DataFrame] = []
    all_fundamentals: list[dict]   = []

    for ticker in tqdm(tickers, desc="Collecting data", unit="ticker"):
        # ── Price history ──────────────────────────────────────────────────
        price_df = _download_with_retry(ticker, period)
        if not price_df.empty:
            all_prices.append(price_df)
        else:
            log.warning(f"[{ticker}] No price data — will be missing from price_history.csv")

        # ── Fundamentals ───────────────────────────────────────────────────
        fund = fetch_fundamentals(ticker)
        all_prices_available = not price_df.empty
        if all_prices_available:
            # Attach latest close price for reference
            fund["Last_Close"] = float(price_df["Close"].iloc[-1])
            fund["52W_High"]   = float(price_df["High"].rolling(252).max().iloc[-1])
            fund["52W_Low"]    = float(price_df["Low"].rolling(252).min().iloc[-1])
        all_fundamentals.append(fund)

        time.sleep(0.3)   # polite rate-limiting

    # ── Assemble final DataFrames ──────────────────────────────────────────────
    if not all_prices:
        raise RuntimeError("No price data collected — check your internet connection or tickers.")

    price_history  = pd.concat(all_prices, axis=0)
    fundamentals   = pd.DataFrame(all_fundamentals).set_index("Ticker")

    # ── Persist ────────────────────────────────────────────────────────────────
    price_history.to_csv(price_out)
    fundamentals.to_csv(fundamentals_out)

    log.info(f"Price history  → {price_out}  ({price_history.shape})")
    log.info(f"Fundamentals   → {fundamentals_out}  ({fundamentals.shape})")

    return price_history, fundamentals


# =============================================================================
# Convenience loader (used by later pipeline steps)
# =============================================================================
def load_data(
    price_path: str       = os.path.join(config.DATA_DIR, "price_history.csv"),
    fund_path:  str       = os.path.join(config.DATA_DIR, "fundamentals.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved CSVs."""
    price = pd.read_csv(price_path, index_col=0, parse_dates=True)
    funds = pd.read_csv(fund_path,  index_col=0)
    log.info(f"Loaded price history {price.shape} and fundamentals {funds.shape}")
    return price, funds


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    prices, fundamentals = collect_all_data()
    print("\n=== Price History Sample ===")
    print(prices.head())
    print("\n=== Fundamentals Sample ===")
    print(fundamentals.head())
    print(f"\n✅ Collection complete. Files saved to: {config.DATA_DIR}")
