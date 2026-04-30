# =============================================================================
# config.py — Central Configuration for Stock Clustering ML Project
# =============================================================================
# All tuneable parameters live here so no magic numbers scatter across files.

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

for _d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────
# S&P 500 tickers (top 50 by market cap — extend as needed)
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM",  "JNJ",  "V",    "UNH",  "XOM",  "PG",   "MA",   "HD",
    "LLY",  "MRK",  "ABBV", "CVX",  "PEP",  "KO",   "COST", "AVGO",
    "MCD",  "WMT",  "BAC",  "CRM",  "ACN",  "LIN",  "TMO",  "CSCO",
    "ABT",  "ADBE", "NKE",  "DIS",  "TXN",  "NEE",  "PM",   "DHR",
    "RTX",  "AMGN", "QCOM", "LOW",  "HON",  "UPS",  "MDT",  "BMY",
    "SBUX", "INTU"
]

# NIFTY 50 tickers (NSE — uncomment to use instead)
# NIFTY50_TICKERS = [
#     "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
#     "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
#     ... etc
# ]

TICKERS = SP500_TICKERS          # ← switch here to use NIFTY

# ── Date Range ────────────────────────────────────────────────────────────────
HISTORY_YEARS  = 2               # years of daily price history to download
HISTORY_PERIOD = f"{HISTORY_YEARS}y"

# ── Feature Engineering ───────────────────────────────────────────────────────
SMA_WINDOWS     = [20, 50, 200]
EMA_WINDOWS     = [12, 26]
RSI_WINDOW      = 14
BOLLINGER_WINDOW = 20
MOMENTUM_WINDOW = 10
VOLATILITY_WINDOW = 20

# ── Clustering ────────────────────────────────────────────────────────────────
N_CLUSTERS          = 3          # BUY / MAYBE / NOT BUY
KMEANS_RANDOM_STATE = 42
MAX_K_SEARCH        = 10         # max k for elbow/silhouette search
DBSCAN_EPS          = 0.5
DBSCAN_MIN_SAMPLES  = 3

# ── Preprocessing ─────────────────────────────────────────────────────────────
OUTLIER_Z_THRESHOLD   = 3.0      # |z| > this → replace with median
MISSING_VALUE_STRATEGY = "median" # 'median' | 'mean' | 'drop'
TEST_SIZE             = 0.2      # used only if we ever add supervised eval

# ── Cluster → Label mapping (set AFTER interpreting clusters) ─────────────────
# Keys = cluster IDs (int), values = human labels
# Populated dynamically in cluster_interpretation.py
CLUSTER_LABEL_MAP: dict = {}

# ── Colours used across all plots ─────────────────────────────────────────────
CLUSTER_COLORS = {
    "BUY":       "#00C853",   # green
    "MAYBE BUY": "#FFD600",   # amber
    "NOT BUY":   "#D50000",   # red
}
PALETTE = ["#00C853", "#FFD600", "#D50000"]
