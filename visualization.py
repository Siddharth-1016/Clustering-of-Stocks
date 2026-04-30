# =============================================================================
# visualization.py  —  STEP 7: Visualization Dashboard
# =============================================================================
"""
Creates all post-clustering visualisations:
  1. PCA cluster plot (coloured by BUY / MAYBE / NOT BUY)
  2. Radar chart of cluster averages across key metrics
  3. Feature deviation heatmap (what drives each cluster)
  4. Top stocks per category table
  5. Distribution of key metrics by category
  6. Interactive Plotly dashboard (saved as HTML)
"""

import logging
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

import config

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
sns.set_theme(style="darkgrid", font_scale=1.1)
SAVE = lambda name: plt.savefig(os.path.join(config.OUTPUT_DIR, name), bbox_inches="tight", dpi=130)

COLOR_MAP  = {"BUY": "#00C853", "MAYBE BUY": "#FFD600", "NOT BUY": "#D50000"}
EMOJI_MAP  = {"BUY": "🟢", "MAYBE BUY": "🟡", "NOT BUY": "🔴"}


# =============================================================================
# 1. PCA Cluster Plot
# =============================================================================
def plot_pca_clusters(
    scaled_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
) -> None:
    """
    Reduces all features to 2 PCs and produces a publication-quality
    scatter plot coloured by investment category.
    """
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(scaled_df.values)

    categories = labelled_df.loc[scaled_df.index, "Category"]

    fig, ax = plt.subplots(figsize=(13, 9))
    for cat in ["BUY", "MAYBE BUY", "NOT BUY"]:
        mask  = categories == cat
        idx   = np.where(mask.values)[0]
        color = COLOR_MAP[cat]
        ax.scatter(pcs[idx, 0], pcs[idx, 1], color=color, s=90,
                   label=f"{EMOJI_MAP[cat]} {cat}", alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=3)
        for i in idx:
            ax.annotate(
                scaled_df.index[i], (pcs[i, 0], pcs[i, 1]),
                fontsize=6.5, color=color, alpha=0.85,
                xytext=(3, 3), textcoords="offset points"
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Stock Investment Clusters — PCA 2D View", fontsize=14, fontweight="bold", pad=12)
    ax.legend(title="Category", fontsize=10, title_fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    SAVE("10_pca_clusters.png")
    plt.close()
    log.info("Plot saved: 10_pca_clusters.png")


# =============================================================================
# 2. Radar Chart
# =============================================================================
RADAR_FEATURES = [
    "ROE", "ROA", "Profit_Margin", "Revenue_Growth",
    "Return_12M", "RSI", "Price_SMA200_Ratio",
    "Momentum_10D", "MACD_Hist", "Volatility_20D",
]

def plot_radar(cluster_means: pd.DataFrame, label_map: dict) -> None:
    """
    Radar / spider chart showing normalised average of key metrics per category.
    Each axis = one feature, normalised to [0,1] across clusters so all metrics
    are directly comparable.
    """
    available = [f for f in RADAR_FEATURES if f in cluster_means.columns]
    n_features = len(available)
    if n_features < 3:
        log.warning("Not enough features for radar chart.")
        return

    # Normalise each feature to [0,1] across clusters
    means_sub = cluster_means[available].copy()
    for col in available:
        mn, mx = means_sub[col].min(), means_sub[col].max()
        if mx > mn:
            means_sub[col] = (means_sub[col] - mn) / (mx - mn)
        else:
            means_sub[col] = 0.5

    # Map cluster ids to labels
    means_sub.index = means_sub.index.map(label_map)

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for cat in ["BUY", "MAYBE BUY", "NOT BUY"]:
        if cat not in means_sub.index:
            continue
        values = means_sub.loc[cat].tolist()
        values += values[:1]
        color  = COLOR_MAP[cat]
        ax.plot(angles, values, linewidth=2.5, color=color, label=f"{EMOJI_MAP[cat]} {cat}")
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in available], fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="grey")
    ax.set_title("Cluster Profiles — Key Metrics (Normalised)", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    SAVE("11_radar_chart.png")
    plt.close()
    log.info("Plot saved: 11_radar_chart.png")


# =============================================================================
# 3. Feature Deviation Heatmap
# =============================================================================
def plot_feature_heatmap(cluster_z: pd.DataFrame, top_n: int = 20) -> None:
    """
    Shows which features deviate most from the global mean for each cluster.
    Red = above average, Blue = below average.
    This is the most direct answer to "WHAT makes each cluster different?"
    """
    # Take top N features by max absolute deviation across any cluster
    max_abs = cluster_z.abs().max(axis=0)
    top_features = max_abs.nlargest(top_n).index
    plot_df = cluster_z[top_features].T

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        plot_df, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-2, vmax=2,
        linewidths=0.5, linecolor="grey",
        cbar_kws={"label": "Deviation from global mean (z-score)", "shrink": 0.7},
        ax=ax
    )
    ax.set_title(f"Top {top_n} Distinguishing Features per Investment Category",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Investment Category")
    ax.set_ylabel("Feature")
    plt.xticks(rotation=15)
    plt.tight_layout()
    SAVE("12_feature_heatmap.png")
    plt.close()
    log.info("Plot saved: 12_feature_heatmap.png")


# =============================================================================
# 4. Top stocks per category
# =============================================================================
def get_top_stocks(
    labelled_df: pd.DataFrame,
    score_features: list[str] = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Within each category, rank stocks by composite score of key quality metrics.
    Returns dict{category: top_n DataFrame}.
    """
    if score_features is None:
        score_features = [
            "ROE", "ROA", "Profit_Margin", "Revenue_Growth",
            "Return_12M", "MACD_Hist"
        ]
    available = [f for f in score_features if f in labelled_df.columns]

    # Min-max normalise and average
    df_norm = labelled_df[available].copy()
    for col in available:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        if mx > mn:
            df_norm[col] = (df_norm[col] - mn) / (mx - mn)
        else:
            df_norm[col] = 0.5
    labelled_df = labelled_df.copy()
    labelled_df["_composite_score"] = df_norm.mean(axis=1)

    result = {}
    for cat in ["BUY", "MAYBE BUY", "NOT BUY"]:
        subset = labelled_df[labelled_df["Category"] == cat].sort_values(
            "_composite_score", ascending=(cat == "NOT BUY")
        )
        result[cat] = subset[["Category", "_composite_score"] + available].head(top_n)

    return result


def plot_top_stocks_table(top_stocks: dict, display_cols: list[str] = None) -> None:
    """Renders top stocks as a formatted matplotlib table image."""
    if display_cols is None:
        display_cols = ["ROE", "Profit_Margin", "Revenue_Growth", "Return_12M"]

    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.6)

    for row_idx, (cat, df) in enumerate(top_stocks.items()):
        ax = fig.add_subplot(gs[row_idx])
        ax.axis("off")

        available_cols = [c for c in display_cols if c in df.columns]
        table_data = df[available_cols].round(3)
        table_data.insert(0, "Ticker", table_data.index)

        tbl = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)

        header_color = COLOR_MAP[cat]
        for j in range(len(table_data.columns)):
            tbl[(0, j)].set_facecolor(header_color)
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")

        ax.set_title(f"{EMOJI_MAP[cat]}  Top Stocks — {cat}", fontsize=11,
                     fontweight="bold", color=COLOR_MAP[cat], pad=6)

    fig.suptitle("Top Stocks by Investment Category", fontsize=14, fontweight="bold", y=1.01)
    SAVE("13_top_stocks_table.png")
    plt.close()
    log.info("Plot saved: 13_top_stocks_table.png")


# =============================================================================
# 5. Distribution plots by category
# =============================================================================
def plot_category_distributions(labelled_df: pd.DataFrame) -> None:
    """
    Box plots of key metrics split by category.
    Makes it immediately clear HOW DIFFERENT the categories are.
    """
    features = [
        "PE_Ratio", "ROE", "Revenue_Growth",
        "Return_12M", "Volatility_20D", "Debt_to_Equity"
    ]
    available = [f for f in features if f in labelled_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    category_order = ["BUY", "MAYBE BUY", "NOT BUY"]
    palette = [COLOR_MAP[c] for c in category_order]

    for i, feat in enumerate(available):
        sns.boxplot(
            data=labelled_df, x="Category", y=feat,
            order=category_order, palette=palette,
            width=0.5, fliersize=4, ax=axes[i]
        )
        axes[i].set_title(feat, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", labelsize=9)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Key Metric Distributions by Investment Category",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    SAVE("14_category_distributions.png")
    plt.close()
    log.info("Plot saved: 14_category_distributions.png")


# =============================================================================
# 6. Interactive Plotly dashboard (saved as HTML)
# =============================================================================
def build_interactive_dashboard(
    scaled_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
) -> None:
    """
    Produces a self-contained HTML file with:
      - 3-D PCA scatter
      - Category pie chart
      - Interactive hover showing key metrics
    """
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(scaled_df.values)

    categories = labelled_df.loc[scaled_df.index, "Category"]
    hover_cols = ["PE_Ratio", "ROE", "Revenue_Growth", "Return_12M", "Volatility_20D"]
    avail_hover = [c for c in hover_cols if c in labelled_df.columns]

    fig = go.Figure()
    for cat in ["BUY", "MAYBE BUY", "NOT BUY"]:
        mask = (categories == cat).values
        idx  = np.where(mask)[0]
        tickers_in_cat = scaled_df.index[idx]

        hover_texts = []
        for ticker in tickers_in_cat:
            row = labelled_df.loc[ticker]
            lines = [f"<b>{ticker}</b> — {cat}"]
            for c in avail_hover:
                val = row.get(c, np.nan)
                if not np.isnan(val):
                    lines.append(f"{c}: {val:.2f}")
            hover_texts.append("<br>".join(lines))

        fig.add_trace(go.Scatter3d(
            x=pcs[idx, 0], y=pcs[idx, 1], z=pcs[idx, 2],
            mode="markers+text",
            marker=dict(size=6, color=COLOR_MAP[cat], opacity=0.85, line=dict(width=0.5, color="white")),
            text=list(tickers_in_cat),
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
            name=f"{EMOJI_MAP[cat]} {cat}",
        ))

    fig.update_layout(
        title=dict(text="Stock Investment Clusters — 3D PCA View", font=dict(size=16)),
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
        ),
        legend=dict(title="Category"),
        template="plotly_dark",
        height=700,
    )

    out_path = os.path.join(config.OUTPUT_DIR, "interactive_dashboard.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    log.info(f"Interactive dashboard saved → {out_path}")


# =============================================================================
# Orchestrator
# =============================================================================
def run_visualizations(
    scaled_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    cluster_means: pd.DataFrame,
    cluster_z: pd.DataFrame,
    label_map: dict,
) -> None:
    log.info("=== Visualization Dashboard START ===")
    plot_pca_clusters(scaled_df, labelled_df)
    plot_radar(cluster_means, label_map)
    plot_feature_heatmap(cluster_z)
    top_stocks = get_top_stocks(labelled_df)
    plot_top_stocks_table(top_stocks)
    plot_category_distributions(labelled_df)
    build_interactive_dashboard(scaled_df, labelled_df)
    log.info(f"=== Visualization Dashboard COMPLETE — outputs in {config.OUTPUT_DIR} ===")
    return top_stocks


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from data_collection import load_data
    from feature_engineering import run_feature_pipeline
    from clustering import compare_models, save_models
    from cluster_interpretation import (
        interpret_clusters, get_feature_importance_per_cluster, save_labelled_data
    )

    prices, fundamentals = load_data()
    raw, scaled, scaler  = run_feature_pipeline(prices, fundamentals)
    _, km_model, km_labels, agg_model, _ = compare_models(scaled.values)
    save_models(km_model, agg_model)

    labelled, label_map, cluster_means = interpret_clusters(raw, km_labels)
    cluster_z = get_feature_importance_per_cluster(labelled)
    save_labelled_data(labelled)

    run_visualizations(scaled, raw, labelled, cluster_means, cluster_z, label_map)
    print(f"\n✅ All visualizations saved to {config.OUTPUT_DIR}")
