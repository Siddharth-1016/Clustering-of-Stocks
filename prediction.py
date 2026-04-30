# =============================================================================
# prediction.py  —  STEP 8: Prediction Pipeline
# =============================================================================
"""
predict_stock_category(ticker)
  • Fetches live data for a NEW ticker via yfinance
  • Runs it through the exact same feature engineering pipeline
  • Aligns features to those the model was trained on
  • Uses trained KMeans to assign a cluster
  • Maps cluster → BUY / MAYBE BUY / NOT BUY
  • Returns a rich PredictionResult object

This module is designed to be imported by the Streamlit app (app.py) as well
as called from the command line.
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import config
from data_collection import _download_with_retry, fetch_fundamentals
from feature_engineering import compute_technical_indicators, handle_missing_values

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# =============================================================================
# PredictionResult dataclass
# =============================================================================
@dataclass
class PredictionResult:
    ticker:       str
    category:     str                        # 'BUY' | 'MAYBE BUY' | 'NOT BUY'
    cluster_id:   int
    confidence:   float                      # distance-based confidence 0–1
    key_metrics:  dict  = field(default_factory=dict)
    raw_features: Optional[pd.Series] = None
    error:        Optional[str] = None

    @property
    def emoji(self) -> str:
        return {"BUY": "🟢", "MAYBE BUY": "🟡", "NOT BUY": "🔴"}.get(self.category, "⚪")

    def __str__(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"  {self.emoji}  {self.ticker}  →  {self.category}",
            f"  Cluster ID : {self.cluster_id}",
            f"  Confidence : {self.confidence:.1%}",
            f"{'='*50}",
            "  Key Metrics:",
        ]
        for k, v in self.key_metrics.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                lines.append(f"    {k:<28} {v:.3f}" if isinstance(v, float) else f"    {k:<28} {v}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)


# =============================================================================
# Feature alignment helper
# =============================================================================
def align_features(
    feature_series: pd.Series,
    trained_columns: list[str],
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Ensures the new ticker's feature vector has exactly the same columns
    in the same order as the training data.
    Extra columns are dropped; missing columns are filled with fill_value.
    """
    aligned = pd.Series(fill_value, index=trained_columns, dtype=float)
    for col in trained_columns:
        if col in feature_series.index:
            aligned[col] = feature_series[col]
    return aligned


# =============================================================================
# Confidence from cluster centroid distance
# =============================================================================
def _centroid_confidence(X_point: np.ndarray, km_model) -> float:
    """
    Computes how confidently the model assigned this point.
    Logic: closer to its cluster centroid relative to other centroids = higher confidence.

    Returns value in [0, 1].
    """
    from sklearn.metrics.pairwise import euclidean_distances
    point    = X_point.reshape(1, -1)
    dists    = euclidean_distances(point, km_model.cluster_centers_)[0]
    assigned = np.argmin(dists)
    min_dist = dists[assigned]

    other_dists = np.delete(dists, assigned)
    second_min  = other_dists.min() if len(other_dists) > 0 else min_dist + 1e-9

    # Confidence: 1 when min_dist → 0, 0 when min_dist → second_min
    confidence = max(0.0, min(1.0, 1 - (min_dist / (min_dist + second_min))))
    return float(confidence)


# =============================================================================
# Load artefacts
# =============================================================================
def load_prediction_artefacts() -> tuple:
    """
    Loads:
      - Trained KMeans model
      - RobustScaler
      - Training feature columns
      - Label map (cluster id → investment label)
    """
    km_path    = os.path.join(config.MODEL_DIR, "kmeans_model.pkl")
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    cols_path  = os.path.join(config.MODEL_DIR, "feature_columns.pkl")
    label_path = os.path.join(config.MODEL_DIR, "label_map.pkl")

    for p in [km_path, scaler_path, cols_path, label_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required model file not found: {p}\n"
                "Please run train_pipeline.py first."
            )

    km_model   = joblib.load(km_path)
    scaler     = joblib.load(scaler_path)
    feat_cols  = joblib.load(cols_path)
    label_map  = joblib.load(label_path)
    return km_model, scaler, feat_cols, label_map


# =============================================================================
# Main prediction function
# =============================================================================
def predict_stock_category(ticker: str) -> PredictionResult:
    """
    End-to-end prediction for a single ticker.

    Steps:
      1. Download price history (2y)
      2. Fetch fundamentals
      3. Compute technical indicators
      4. Merge & align features to training schema
      5. Scale with fitted RobustScaler
      6. Assign cluster via KMeans.predict()
      7. Map cluster → investment label
      8. Compute confidence
      9. Return PredictionResult

    Parameters
    ----------
    ticker : str  e.g. "AAPL", "MSFT", "RELIANCE.NS"

    Returns
    -------
    PredictionResult
    """
    ticker = ticker.upper().strip()
    log.info(f"Predicting: {ticker}")

    try:
        # ── Load artefacts ─────────────────────────────────────────────────
        km_model, scaler, feat_cols, label_map = load_prediction_artefacts()

        # ── 1. Price history ───────────────────────────────────────────────
        price_df = _download_with_retry(ticker, config.HISTORY_PERIOD)
        if price_df.empty:
            return PredictionResult(
                ticker=ticker, category="UNKNOWN", cluster_id=-1,
                confidence=0.0, error=f"Could not fetch price data for {ticker}"
            )

        # ── 2. Fundamentals ────────────────────────────────────────────────
        fund_dict = fetch_fundamentals(ticker)
        fund_series = pd.Series(fund_dict).drop("Ticker", errors="ignore")

        # ── 3. Technical indicators ────────────────────────────────────────
        tech_series = compute_technical_indicators(price_df, ticker)
        if tech_series is None:
            return PredictionResult(
                ticker=ticker, category="UNKNOWN", cluster_id=-1,
                confidence=0.0, error=f"Insufficient price history for {ticker}"
            )

        # ── 4. Merge & align ───────────────────────────────────────────────
        combined = pd.concat([fund_series, tech_series])
        aligned  = align_features(combined, feat_cols)

        # Fill any remaining NaN with median (training-time values stored in scaler)
        aligned = aligned.fillna(0.0)   # fallback; scaler handles the rest

        # ── 5. Scale ───────────────────────────────────────────────────────
        X_scaled = scaler.transform(aligned.values.reshape(1, -1))

        # ── 6. Cluster assignment ──────────────────────────────────────────
        cluster_id = int(km_model.predict(X_scaled)[0])

        # ── 7. Label ───────────────────────────────────────────────────────
        category = label_map.get(cluster_id, "UNKNOWN")

        # ── 8. Confidence ──────────────────────────────────────────────────
        confidence = _centroid_confidence(X_scaled[0], km_model)

        # ── 9. Key metrics for display ─────────────────────────────────────
        key_metrics_keys = [
            "PE_Ratio", "PB_Ratio", "ROE", "ROA", "Profit_Margin",
            "Revenue_Growth", "Debt_to_Equity", "Dividend_Yield",
            "RSI", "Return_12M", "Return_6M", "Volatility_20D",
            "Price_SMA200_Ratio", "MACD_Hist", "Free_Cash_Flow",
        ]
        key_metrics = {}
        for k in key_metrics_keys:
            val = combined.get(k, np.nan)
            if not (isinstance(val, float) and np.isnan(val)):
                key_metrics[k] = float(val) if isinstance(val, (int, float, np.floating)) else val

        return PredictionResult(
            ticker=ticker,
            category=category,
            cluster_id=cluster_id,
            confidence=confidence,
            key_metrics=key_metrics,
            raw_features=aligned,
        )

    except FileNotFoundError as e:
        return PredictionResult(ticker=ticker, category="ERROR", cluster_id=-1,
                                confidence=0.0, error=str(e))
    except Exception as e:
        log.exception(f"Unexpected error predicting {ticker}: {e}")
        return PredictionResult(ticker=ticker, category="ERROR", cluster_id=-1,
                                confidence=0.0, error=str(e))


# =============================================================================
# Batch prediction
# =============================================================================
def predict_multiple(tickers: list[str]) -> pd.DataFrame:
    """
    Predict investment category for a list of tickers.
    Returns a DataFrame summary.
    """
    results = []
    for ticker in tickers:
        res = predict_stock_category(ticker)
        results.append({
            "Ticker":     res.ticker,
            "Category":   res.category,
            "Cluster_ID": res.cluster_id,
            "Confidence": res.confidence,
            **{k: v for k, v in res.key_metrics.items() if isinstance(v, float)},
            "Error": res.error or "",
        })
        time.sleep(0.3)  # polite rate-limiting

    return pd.DataFrame(results).set_index("Ticker")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Quick demo — predict a handful of stocks
    demo_tickers = ["AAPL", "MSFT", "AMZN"]
    for t in demo_tickers:
        result = predict_stock_category(t)
        print(result)
