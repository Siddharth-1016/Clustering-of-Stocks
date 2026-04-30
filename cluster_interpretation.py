# =============================================================================
# cluster_interpretation.py  —  STEP 6: Cluster Interpretation  ★ KEY STEP ★
# =============================================================================
"""
This is the MOST IMPORTANT step in the pipeline.

After unsupervised clustering we have groups with no names — just 0, 1, 2.
This module maps those numbers to investment labels by:

  1. Computing the MEAN of each feature per cluster
  2. Scoring each cluster on a composite "investment quality" score
  3. Ranking clusters and assigning BUY / MAYBE BUY / NOT BUY
  4. Printing an interpretable report

WHY THIS SCORING LOGIC:
  A stock cluster is labelled BUY when it shows:
    • Strong profitability  → high ROE, ROA, profit margin
    • Reasonable valuation  → lower P/E, P/B relative to others
    • Positive momentum     → RSI in 50-70 range, positive MACD, +ve 12M return
    • Low financial risk    → lower debt-to-equity, higher current ratio
    • Strong cash flow      → positive Free Cash Flow
    • Trending up           → price above SMA-200 (ratio > 1)
  
  NOT BUY shows the opposite profile.
  MAYBE BUY sits in-between.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


# =============================================================================
# Scoring weights — tuned by a quant researcher
# =============================================================================
# Each entry: (feature_name, weight, direction)
#   direction = +1 → higher is better (e.g. ROE)
#   direction = -1 → lower is better (e.g. Debt_to_Equity, PE_Ratio when very high)
SCORING_CRITERIA = [
    # ── Fundamental quality ────────────────────────────────────────────────
    ("ROE",                  3.0,  +1),   # Return on Equity: core profitability
    ("ROA",                  2.5,  +1),   # Return on Assets: capital efficiency
    ("Profit_Margin",        2.5,  +1),   # Pricing power
    ("Revenue_Growth",       2.0,  +1),   # Top-line momentum
    ("Earnings_Growth",      2.0,  +1),   # Bottom-line momentum
    ("Free_Cash_Flow",       1.5,  +1),   # Cash generation (normalised below)
    ("Operating_Margin",     1.5,  +1),   # Operational efficiency
    ("Current_Ratio",        1.0,  +1),   # Liquidity buffer
    ("Debt_to_Equity",       2.0,  -1),   # Leverage risk (lower = better)
    ("PE_Ratio",             0.5,  -1),   # Valuation: penalise extreme overvaluation
    ("PB_Ratio",             0.5,  -1),   # Book value premium
    # ── Technical momentum ────────────────────────────────────────────────
    ("Return_12M",           2.5,  +1),   # 1-year price momentum
    ("Return_6M",            1.5,  +1),   # 6-month momentum
    ("Return_3M",            1.0,  +1),   # 3-month momentum
    ("RSI",                  1.0,  +1),   # RSI > 50 = bullish (scaled 0-1)
    ("MACD_Hist",            1.5,  +1),   # Positive histogram = bullish momentum
    ("Price_SMA200_Ratio",   2.0,  +1),   # Above SMA-200 = long-term uptrend
    ("Price_SMA50_Ratio",    1.5,  +1),   # Above SMA-50 = medium-term uptrend
    ("Momentum_10D",         1.0,  +1),   # Short-term price acceleration
    ("Dist_52W_High",        1.0,  -1),   # Closer to 52W high → stronger (lower dist = better)
    ("Dist_52W_Low",         1.0,  +1),   # Far from 52W low → safer
    ("Volume_Ratio",         0.5,  +1),   # Volume confirmation
    ("Volatility_20D",       1.5,  -1),   # Risk-adjusted: lower vol preferred in BUY
]


# =============================================================================
# Helper: rank-normalise values to [0, 1] across clusters
# =============================================================================
def _rank_normalise(series: pd.Series) -> pd.Series:
    """Rank-normalise a series to [0, 1]."""
    r = series.rank(pct=True)
    return r


# =============================================================================
# Main interpreter
# =============================================================================
def interpret_clusters(
    raw_features: pd.DataFrame,
    labels: np.ndarray,
    n_clusters: int = config.N_CLUSTERS,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Assigns investment labels to cluster IDs.

    Parameters
    ----------
    raw_features  : Unscaled feature DataFrame (tickers as index)
    labels        : Cluster assignments from KMeans (same order as raw_features)
    n_clusters    : Expected number of clusters

    Returns
    -------
    labelled_df   : raw_features with 'Cluster' and 'Category' columns added
    label_map     : {cluster_id (int): 'BUY' | 'MAYBE BUY' | 'NOT BUY'}
    cluster_stats : Per-cluster mean statistics (for visualisation)
    """
    df = raw_features.copy()
    df["Cluster"] = labels

    # ── Cluster means ─────────────────────────────────────────────────────
    cluster_means = df.groupby("Cluster").mean(numeric_only=True)

    # ── Build composite investment score per cluster ───────────────────────
    scores: dict[int, float] = {c: 0.0 for c in range(n_clusters)}

    total_weight = sum(abs(w) for _, w, _ in SCORING_CRITERIA)

    for feature, weight, direction in SCORING_CRITERIA:
        if feature not in cluster_means.columns:
            continue

        col_values = cluster_means[feature]

        # RSI special handling: best score near 55, penalise <30 or >70
        if feature == "RSI":
            # Convert RSI to a 0-1 "healthy" score
            col_values = col_values.apply(
                lambda x: 1.0 - abs(x - 55) / 45 if not np.isnan(x) else 0.5
            )
            direction = +1   # already re-mapped to higher-is-better

        # Rank normalise across clusters (0 = worst cluster, 1 = best)
        if col_values.nunique() > 1:
            ranked = _rank_normalise(col_values)
        else:
            ranked = pd.Series([0.5] * len(col_values), index=col_values.index)

        for cluster_id in scores:
            contribution = direction * ranked.get(cluster_id, 0.5) * weight
            scores[cluster_id] += contribution

    # Normalise scores to [0, 1] range for readability
    score_series = pd.Series(scores)
    score_min, score_max = score_series.min(), score_series.max()
    if score_max > score_min:
        score_series = (score_series - score_min) / (score_max - score_min)

    # ── Assign labels by score rank ────────────────────────────────────────
    sorted_clusters = score_series.sort_values(ascending=False)
    investment_labels = ["BUY", "MAYBE BUY", "NOT BUY"]
    label_map: dict[int, str] = {}
    for rank, cluster_id in enumerate(sorted_clusters.index):
        label_map[int(cluster_id)] = investment_labels[rank]

    log.info("=== CLUSTER INTERPRETATION ===")
    for cid, lbl in label_map.items():
        log.info(f"  Cluster {cid} → {lbl}  (composite score = {score_series[cid]:.3f})")

    # ── Attach labels to main DataFrame ───────────────────────────────────
    df["Category"] = df["Cluster"].map(label_map)

    # ── Update global config map ───────────────────────────────────────────
    config.CLUSTER_LABEL_MAP.update(label_map)

    return df, label_map, cluster_means


# =============================================================================
# Interpretability report
# =============================================================================
def print_cluster_report(
    labelled_df: pd.DataFrame,
    cluster_means: pd.DataFrame,
    label_map: dict,
) -> None:
    """
    Prints a rich textual interpretation of each cluster, covering:
      - Count of stocks
      - Avg key metrics
      - Investment rationale
    """
    REPORT_FEATURES = [
        "PE_Ratio", "PB_Ratio", "ROE", "ROA", "Profit_Margin",
        "Revenue_Growth", "Debt_to_Equity", "Dividend_Yield",
        "RSI", "Return_12M", "Return_6M", "Volatility_20D",
        "Price_SMA200_Ratio", "MACD_Hist", "Free_Cash_Flow",
    ]
    available = [f for f in REPORT_FEATURES if f in cluster_means.columns]

    print("\n" + "═" * 70)
    print("       CLUSTER INTERPRETATION REPORT")
    print("═" * 70)

    for cid in sorted(label_map.keys()):
        cat   = label_map[cid]
        emoji = {"BUY": "🟢", "MAYBE BUY": "🟡", "NOT BUY": "🔴"}.get(cat, "⚪")
        stocks = labelled_df[labelled_df["Cluster"] == cid].index.tolist()
        n      = len(stocks)

        print(f"\n{emoji}  CLUSTER {cid}  →  {cat}  ({n} stocks)")
        print("-" * 60)

        means = cluster_means.loc[cid]
        for feat in available:
            if feat in means:
                val = means[feat]
                if not np.isnan(val):
                    print(f"  {feat:<28} {val:>10.3f}")

        print(f"\n  Stocks: {', '.join(sorted(stocks)[:20])}", end="")
        if len(stocks) > 20:
            print(f" ... +{len(stocks)-20} more")
        else:
            print()

    print("\n" + "═" * 70)


# =============================================================================
# Key differentiating features per cluster
# =============================================================================
def get_feature_importance_per_cluster(
    labelled_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uses the difference-from-global-mean to rank which features most
    strongly characterise each cluster (positive or negative deviation).

    Returns a DataFrame: clusters × features, values = z-score-like deviation.
    """
    numeric_df = labelled_df.select_dtypes(include=[np.number]).drop(columns=["Cluster"], errors="ignore")
    global_mean = numeric_df.mean()
    global_std  = numeric_df.std().replace(0, 1)  # avoid division by zero

    cluster_means = labelled_df.groupby("Cluster")[numeric_df.columns].mean()
    cluster_z     = (cluster_means - global_mean) / global_std

    # Map cluster ids to labels
    cluster_z.index = cluster_z.index.map(config.CLUSTER_LABEL_MAP)
    return cluster_z


# =============================================================================
# Export
# =============================================================================
def save_labelled_data(labelled_df: pd.DataFrame) -> None:
    path = os.path.join(config.DATA_DIR, "labelled_stocks.csv")
    labelled_df.to_csv(path)
    log.info(f"Labelled dataset saved → {path}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from data_collection import load_data
    from feature_engineering import run_feature_pipeline
    from clustering import compare_models, save_models

    prices, fundamentals = load_data()
    raw, scaled, scaler  = run_feature_pipeline(prices, fundamentals)
    comparison, km_model, km_labels, agg_model, _ = compare_models(scaled.values)
    save_models(km_model, agg_model)

    labelled, label_map, cluster_means = interpret_clusters(raw, km_labels)
    print_cluster_report(labelled, cluster_means, label_map)
    save_labelled_data(labelled)

    print("\n=== Category Distribution ===")
    print(labelled["Category"].value_counts())
