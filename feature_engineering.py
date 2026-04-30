# =============================================================================
# feature_engineering.py  —  STEP 3: Feature Engineering Pipeline
# =============================================================================
"""
Computes all technical indicators from price history, merges with fundamentals,
cleans missing values, removes outliers, and scales features.

WHY each feature matters:
  PE_Ratio        — Valuation: low P/E can mean undervalued or value trap
  PB_Ratio        — Valuation vs book value; <1 = trading below assets
  EPS             — Absolute earnings power
  ROE             — How efficiently equity is deployed; >15% is strong
  ROA             — Asset efficiency; higher = better capital allocation
  Profit_Margin   — Pricing power + cost control
  Revenue_Growth  — Top-line momentum; key for growth stocks
  Debt_to_Equity  — Financial leverage risk; >2 is elevated
  Dividend_Yield  — Income generation; matters for defensive portfolios
  Free_Cash_Flow  — Real cash generation vs accounting earnings
  SMA_20/50/200   — Price trend at short/medium/long timeframes
  EMA_12/26       — Exponential weight → more responsive to recent prices
  RSI             — Overbought (>70) / oversold (<30) momentum signal
  MACD            — Trend-following momentum; signal line crossovers
  Bollinger_%B    — Position within recent volatility range
  Volatility      — Risk proxy; high vol = higher risk premium required
  Momentum        — Price rate-of-change; captures trend persistence
  Volume_Trend    — Money flow; rising price + rising volume = conviction
  52W_High_Dist   — Distance from peak; <5% often signals strength
  52W_Low_Dist    — Distance from trough; >50% signals recovery
"""

import logging
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
import ta                             # Technical Analysis library

import config
from data_collection import load_data

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# =============================================================================
# SECTION A — Technical Indicators
# =============================================================================
def compute_technical_indicators(price_df: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """
    Given a DataFrame of daily OHLCV for ONE ticker, compute all technical
    indicators and return a Series representing the LATEST snapshot.

    All indicators are summarised to a single scalar (latest value, or a
    statistical summary) so each ticker becomes one row in the feature matrix.
    """
    df = price_df.copy().sort_index()

    if len(df) < 210:           # need at least ~200 trading days for SMA-200
        log.warning(f"[{ticker}] Insufficient history ({len(df)} rows) — skipping technicals.")
        return None

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    features: dict = {}

    # ── Simple Moving Averages ─────────────────────────────────────────────
    for w in config.SMA_WINDOWS:
        sma = close.rolling(w).mean()
        # Ratio: price vs SMA captures above/below trend
        features[f"Price_SMA{w}_Ratio"] = float(close.iloc[-1] / sma.iloc[-1]) if sma.iloc[-1] != 0 else np.nan

    # ── Exponential Moving Averages ────────────────────────────────────────
    for w in config.EMA_WINDOWS:
        ema = close.ewm(span=w, adjust=False).mean()
        features[f"Price_EMA{w}_Ratio"] = float(close.iloc[-1] / ema.iloc[-1]) if ema.iloc[-1] != 0 else np.nan

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi_obj = ta.momentum.RSIIndicator(close=close, window=config.RSI_WINDOW)
    features["RSI"] = float(rsi_obj.rsi().iloc[-1])

    # ── MACD ───────────────────────────────────────────────────────────────
    macd_obj   = ta.trend.MACD(close=close)
    macd_val   = macd_obj.macd().iloc[-1]
    macd_sig   = macd_obj.macd_signal().iloc[-1]
    macd_diff  = macd_obj.macd_diff().iloc[-1]
    features["MACD"]            = float(macd_val)
    features["MACD_Signal"]     = float(macd_sig)
    features["MACD_Hist"]       = float(macd_diff)   # histogram; positive = bullish crossover

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(
        close=close, window=config.BOLLINGER_WINDOW, window_dev=2
    )
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_mid   = bb.bollinger_mavg().iloc[-1]
    bb_band_width = (bb_upper - bb_lower) / bb_mid if bb_mid != 0 else np.nan
    bb_pct_b = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else np.nan
    features["BB_Bandwidth"] = float(bb_band_width)  # volatility proxy
    features["BB_PctB"]      = float(bb_pct_b)       # 0=at lower band, 1=at upper band

    # ── Volatility (annualised) ────────────────────────────────────────────
    daily_returns  = close.pct_change().dropna()
    vol_20d        = daily_returns.tail(20).std() * np.sqrt(252)
    vol_full       = daily_returns.std() * np.sqrt(252)
    features["Volatility_20D"]  = float(vol_20d)
    features["Volatility_Full"] = float(vol_full)

    # ── Momentum ───────────────────────────────────────────────────────────
    # Rate of change over last N days
    if len(close) > config.MOMENTUM_WINDOW:
        momentum = (close.iloc[-1] - close.iloc[-(config.MOMENTUM_WINDOW + 1)]) / close.iloc[-(config.MOMENTUM_WINDOW + 1)]
        features["Momentum_10D"] = float(momentum)

    # 1-month, 3-month, 6-month, 12-month returns
    for days, label in [(21, "1M"), (63, "3M"), (126, "6M"), (252, "12M")]:
        if len(close) > days:
            ret = (close.iloc[-1] - close.iloc[-(days + 1)]) / close.iloc[-(days + 1)]
            features[f"Return_{label}"] = float(ret)
        else:
            features[f"Return_{label}"] = np.nan

    # ── Volume trend ───────────────────────────────────────────────────────
    vol_ma_20  = volume.rolling(20).mean().iloc[-1]
    vol_latest = volume.iloc[-1]
    features["Volume_Ratio"] = float(vol_latest / vol_ma_20) if vol_ma_20 != 0 else np.nan

    # ── Average True Range (ATR) ───────────────────────────────────────────
    atr_obj = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    atr_val = atr_obj.average_true_range().iloc[-1]
    features["ATR_Pct"] = float(atr_val / close.iloc[-1]) if close.iloc[-1] != 0 else np.nan

    # ── 52-Week High / Low Distance ────────────────────────────────────────
    rolling_252 = close.rolling(252)
    high_52w    = rolling_252.max().iloc[-1]
    low_52w     = rolling_252.min().iloc[-1]
    cur         = close.iloc[-1]
    features["Dist_52W_High"] = float((high_52w - cur) / high_52w) if high_52w != 0 else np.nan
    features["Dist_52W_Low"]  = float((cur - low_52w)  / low_52w)  if low_52w  != 0 else np.nan

    # ── OBV (On-Balance Volume) trend ──────────────────────────────────────
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    obv_slope = float(np.polyfit(np.arange(20), obv.iloc[-20:].values, 1)[0])
    features["OBV_Slope"] = obv_slope

    return pd.Series(features, name=ticker)


# =============================================================================
# SECTION B — Build Full Technical Feature Matrix
# =============================================================================
def build_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates over each ticker in price_df and builds a (n_tickers × n_features)
    DataFrame of technical indicators.
    """
    tickers = price_df["Ticker"].unique() if "Ticker" in price_df.columns else [price_df.index.name]
    rows: list[pd.Series] = []

    for ticker in tickers:
        ticker_prices = price_df[price_df["Ticker"] == ticker].drop(columns=["Ticker"], errors="ignore")
        series = compute_technical_indicators(ticker_prices, ticker)
        if series is not None:
            rows.append(series)

    tech_df = pd.DataFrame(rows)
    tech_df.index.name = "Ticker"
    log.info(f"Technical features shape: {tech_df.shape}")
    return tech_df


# =============================================================================
# SECTION C — Merge Fundamentals + Technicals
# =============================================================================
def merge_features(
    fundamentals: pd.DataFrame,
    technicals:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner join on ticker index so we only keep tickers that have BOTH
    fundamental and technical data available.
    """
    merged = fundamentals.join(technicals, how="inner")
    log.info(f"Merged feature matrix: {merged.shape}")
    return merged


# =============================================================================
# SECTION D — Data Cleaning
# =============================================================================
def handle_missing_values(df: pd.DataFrame, strategy: str = config.MISSING_VALUE_STRATEGY) -> pd.DataFrame:
    """
    Strategy:
      'median' — fill with column median (robust to outliers)
      'mean'   — fill with column mean
      'drop'   — drop columns with > 30% missing, then drop remaining rows
    """
    # Report missing %
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 0.50].index.tolist()
    if high_missing:
        log.warning(f"Dropping {len(high_missing)} columns with >50% missing: {high_missing}")
        df = df.drop(columns=high_missing)

    if strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == "drop":
        col_thresh = int(0.70 * len(df))
        df = df.dropna(axis=1, thresh=col_thresh)
        df = df.dropna(axis=0)
    else:
        raise ValueError(f"Unknown missing strategy: {strategy}")

    log.info(f"After missing value handling: {df.shape}")
    return df


def remove_outliers(df: pd.DataFrame, threshold: float = config.OUTLIER_Z_THRESHOLD) -> pd.DataFrame:
    """
    For each numeric column, values with |z-score| > threshold are replaced
    by the column median.  We do NOT drop rows — just winsorise in place.

    This preserves dataset size while removing extreme distortions that would
    otherwise pull cluster centroids.
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_median = df_clean[col].median()
        z_scores   = np.abs(stats.zscore(df_clean[col].fillna(col_median)))
        outlier_mask = z_scores > threshold
        n_outliers   = outlier_mask.sum()
        if n_outliers > 0:
            df_clean.loc[outlier_mask, col] = col_median
            log.debug(f"[outlier] {col}: replaced {n_outliers} values")

    log.info("Outlier winsorisation complete.")
    return df_clean


# =============================================================================
# SECTION E — Feature Scaling
# =============================================================================
def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    """
    RobustScaler is preferred over StandardScaler for financial data because:
      • Financial distributions are fat-tailed (non-Gaussian)
      • Outliers are common even after winsorisation
      • RobustScaler uses median/IQR — unaffected by extremes

    Returns (scaled_df, fitted_scaler) so the scaler can be reused at
    prediction time on new tickers.
    """
    scaler    = RobustScaler()
    scaled_np = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_np, index=df.index, columns=df.columns)
    log.info(f"Features scaled with RobustScaler. Shape: {scaled_df.shape}")
    return scaled_df, scaler


# =============================================================================
# SECTION F — Full Pipeline (orchestrator)
# =============================================================================
def run_feature_pipeline(
    price_df:     pd.DataFrame,
    fundamentals: pd.DataFrame,
    save_path:    str = os.path.join(config.DATA_DIR, "features.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    End-to-end feature engineering.

    Returns
    -------
    raw_features    : Merged, cleaned (but unscaled) features — for EDA
    scaled_features : Scaled features ready for clustering
    scaler          : Fitted RobustScaler for later use in prediction
    """
    log.info("=== Feature Engineering Pipeline START ===")

    # 1. Technical indicators
    technicals = build_technical_features(price_df)

    # 2. Merge
    merged = merge_features(fundamentals, technicals)

    # 3. Select only numeric columns
    numeric_merged = merged.select_dtypes(include=[np.number])

    # 4. Missing values
    clean = handle_missing_values(numeric_merged)

    # 5. Outlier removal
    clean = remove_outliers(clean)

    # Save raw (for EDA)
    clean.to_csv(save_path)
    log.info(f"Raw features saved → {save_path}")

    # 6. Scale
    scaled, scaler = scale_features(clean)

    log.info("=== Feature Engineering Pipeline COMPLETE ===")
    return clean, scaled, scaler


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    prices, fundamentals = load_data()
    raw_features, scaled_features, scaler = run_feature_pipeline(prices, fundamentals)

    print("\n=== Feature Matrix (raw, unscaled) ===")
    print(raw_features.head())
    print(f"\nShape: {raw_features.shape}")
    print(f"Columns ({len(raw_features.columns)}):\n{list(raw_features.columns)}")
