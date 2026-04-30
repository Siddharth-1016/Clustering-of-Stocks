# =============================================================================
# eda.py  —  STEP 4: Exploratory Data Analysis
# =============================================================================
"""
Produces all EDA artefacts:
  1. Summary statistics table
  2. Missing-value heatmap
  3. Feature distribution plots
  4. Correlation heatmap
  5. PCA variance plot + 2-D scatter
  6. Pairplot of top features

All plots saved to outputs/ for reproducibility.
"""

import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

import config

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})
SAVE = lambda name: plt.savefig(os.path.join(config.OUTPUT_DIR, name), bbox_inches="tight")


# =============================================================================
# 1. Summary statistics
# =============================================================================
def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() with skewness and kurtosis."""
    stats = df.describe().T
    stats["skewness"] = df.skew(numeric_only=True)
    stats["kurtosis"] = df.kurtosis(numeric_only=True)
    stats["missing%"] = (df.isnull().mean() * 100).round(2)
    stats.to_csv(os.path.join(config.OUTPUT_DIR, "summary_stats.csv"))
    log.info("Summary statistics saved.")
    return stats


# =============================================================================
# 2. Missing value heatmap
# =============================================================================
def plot_missing(df: pd.DataFrame) -> None:
    """
    Visualises the missing data pattern.
    Each cell is coloured if the value is NaN.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        df.isnull().T, cbar=False, cmap="viridis",
        yticklabels=True, xticklabels=False, ax=ax
    )
    ax.set_title("Missing Value Map  (yellow = NaN)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Stocks")
    ax.set_ylabel("Features")
    plt.tight_layout()
    SAVE("01_missing_values.png")
    plt.close()
    log.info("Plot saved: 01_missing_values.png")


# =============================================================================
# 3. Feature distributions
# =============================================================================
def plot_distributions(df: pd.DataFrame, max_cols: int = 24) -> None:
    """
    Histograms + KDE for every numeric feature (up to max_cols).
    Shows the shape of each distribution — important for choosing the
    right scaler and detecting highly skewed features.
    """
    cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    n    = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=25, color="#4e79a7", alpha=0.75, edgecolor="white")
        axes[i].set_title(col, fontsize=9, fontweight="bold")
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):     # hide empty subplots
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    SAVE("02_distributions.png")
    plt.close()
    log.info("Plot saved: 02_distributions.png")


# =============================================================================
# 4. Correlation heatmap
# =============================================================================
def plot_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation heatmap.

    Investment insight: highly correlated features (e.g. SMA20 ~ SMA50) can
    be dropped or merged via PCA to reduce dimensionality without losing signal.
    """
    corr = df.corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle

    fig, ax = plt.subplots(figsize=(max(14, len(corr) // 2), max(12, len(corr) // 2)))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="grey",
        cbar_kws={"shrink": 0.7}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    SAVE("03_correlation_heatmap.png")
    plt.close()
    log.info("Plot saved: 03_correlation_heatmap.png")
    return corr


# =============================================================================
# 5. PCA analysis
# =============================================================================
def pca_analysis(scaled_df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """
    Fits PCA on scaled features.
    Plots:
      (a) Explained variance ratio (scree plot)
      (b) 2-D scatter of first two PCs coloured by ticker name

    Returns PCA-transformed DataFrame (first 2 PCs).
    """
    n_comp = min(n_components, scaled_df.shape[1], scaled_df.shape[0])
    pca    = PCA(n_components=n_comp, random_state=42)
    pcs    = pca.fit_transform(scaled_df.values)

    # ── (a) Scree plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(
        range(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
        color="#4e79a7", alpha=0.85
    )
    axes[0].plot(
        range(1, n_comp + 1),
        np.cumsum(pca.explained_variance_ratio_) * 100,
        "o-", color="#e15759", linewidth=2
    )
    axes[0].axhline(80, linestyle="--", color="grey", linewidth=1)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("PCA — Scree Plot", fontweight="bold")

    # ── (b) 2-D scatter ────────────────────────────────────────────────────
    pc_df = pd.DataFrame(
        pcs[:, :2], index=scaled_df.index, columns=["PC1", "PC2"]
    )
    axes[1].scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.7, s=60, color="#4e79a7", edgecolors="white", linewidths=0.5)

    for ticker, row in pc_df.iterrows():
        axes[1].annotate(ticker, (row["PC1"], row["PC2"]), fontsize=6, alpha=0.7)

    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_title("PCA — 2D Stock Landscape", fontweight="bold")

    plt.suptitle("Principal Component Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    SAVE("04_pca_analysis.png")
    plt.close()
    log.info("Plot saved: 04_pca_analysis.png")

    # ── Feature loadings (top contributors to PC1 & PC2) ──────────────────
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        index=scaled_df.columns,
        columns=["PC1_loading", "PC2_loading"]
    )
    loadings["PC1_abs"] = loadings["PC1_loading"].abs()
    loadings["PC2_abs"] = loadings["PC2_loading"].abs()
    loadings = loadings.sort_values("PC1_abs", ascending=False)
    loadings.to_csv(os.path.join(config.OUTPUT_DIR, "pca_loadings.csv"))
    log.info("PCA loadings saved.")

    return pc_df


# =============================================================================
# 6. Top-feature pairplot
# =============================================================================
def plot_pairplot(df: pd.DataFrame, top_n: int = 6) -> None:
    """
    Pairplot of the top N most informative features (by variance).
    Useful to see natural separations in the data before clustering.
    """
    variances   = df.var(numeric_only=True).sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()

    # Add fundamental + technical mix
    must_haves = ["PE_Ratio", "ROE", "RSI", "Return_12M", "Volatility_20D"]
    for f in must_haves:
        if f in df.columns and f not in top_features:
            top_features.append(f)
    top_features = top_features[:top_n]

    g = sns.pairplot(
        df[top_features].dropna(),
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 30},
        diag_kws={"fill": True},
    )
    g.fig.suptitle("Pairplot — Top Features", y=1.02, fontsize=13, fontweight="bold")
    SAVE("05_pairplot.png")
    plt.close()
    log.info("Plot saved: 05_pairplot.png")


# =============================================================================
# Orchestrator
# =============================================================================
def run_eda(raw_features: pd.DataFrame, scaled_features: pd.DataFrame) -> pd.DataFrame:
    """Runs all EDA steps. Returns PCA-transformed DataFrame."""
    log.info("=== EDA START ===")
    summary_stats(raw_features)
    plot_missing(raw_features)
    plot_distributions(raw_features)
    corr = plot_correlation(raw_features)
    pc_df = pca_analysis(scaled_features)
    plot_pairplot(raw_features)
    log.info(f"=== EDA COMPLETE — all plots in {config.OUTPUT_DIR} ===")
    return pc_df


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from feature_engineering import run_feature_pipeline
    from data_collection import load_data

    prices, fundamentals = load_data()
    raw, scaled, scaler  = run_feature_pipeline(prices, fundamentals)
    pc_df = run_eda(raw, scaled)
    print("\nPCA result sample:")
    print(pc_df.head())
