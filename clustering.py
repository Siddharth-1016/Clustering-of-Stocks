# =============================================================================
# clustering.py  —  STEP 5: Clustering Model Training
# =============================================================================
"""
Implements and compares three clustering algorithms:
  1. KMeans            — Fast, well-understood, assumes spherical clusters
  2. Agglomerative     — Hierarchical; no centroid assumption; dendrogram insight
  3. DBSCAN            — Density-based; detects noise/outliers naturally

Evaluation metrics used:
  • Elbow method (KMeans inertia)     — where does WCSS plateau?
  • Silhouette score                  — [-1,1]; higher = better separation
  • Davies–Bouldin index              — lower = better; ratio of scatter to separation
  • Calinski–Harabasz score           — higher = denser, well-separated clusters

Best model + labels are persisted via joblib.
"""

import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

import config

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
sns.set_theme(style="darkgrid", font_scale=1.1)
SAVE = lambda name: plt.savefig(os.path.join(config.OUTPUT_DIR, name), bbox_inches="tight")


# =============================================================================
# Metric helpers
# =============================================================================
def _eval_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all clustering quality metrics for a given labelling."""
    unique = np.unique(labels[labels != -1])   # exclude DBSCAN noise (-1)
    if len(unique) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    mask = labels != -1
    return {
        "silhouette":         round(silhouette_score(X[mask], labels[mask]), 4),
        "davies_bouldin":     round(davies_bouldin_score(X[mask], labels[mask]), 4),
        "calinski_harabasz":  round(calinski_harabasz_score(X[mask], labels[mask]), 4),
    }


# =============================================================================
# 1. KMeans — with Elbow + Silhouette search
# =============================================================================
def train_kmeans(
    X: np.ndarray,
    k_range: range = range(2, config.MAX_K_SEARCH + 1),
    random_state: int = config.KMEANS_RANDOM_STATE,
) -> tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Trains KMeans for every k in k_range, collects Elbow + Silhouette curves,
    and selects the best k (defaults to config.N_CLUSTERS = 3).

    Returns
    -------
    best_model  : Fitted KMeans with optimal k
    labels      : Cluster assignments (np.ndarray)
    eval_df     : DataFrame of metrics at each k
    """
    inertias, silhouettes, db_scores, ch_scores = [], [], [], []

    log.info("Training KMeans across k values …")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=500)
        km.fit(X)
        labels_k = km.labels_
        inertias.append(km.inertia_)
        metrics = _eval_labels(X, labels_k)
        silhouettes.append(metrics["silhouette"])
        db_scores.append(metrics["davies_bouldin"])
        ch_scores.append(metrics["calinski_harabasz"])
        log.info(f"  k={k}: inertia={km.inertia_:.1f}  sil={metrics['silhouette']:.3f}  DB={metrics['davies_bouldin']:.3f}")

    eval_df = pd.DataFrame({
        "k": list(k_range), "inertia": inertias,
        "silhouette": silhouettes, "davies_bouldin": db_scores,
        "calinski_harabasz": ch_scores,
    })

    # ── Elbow + Silhouette plots ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(k_range, inertias, "o-", color="#4e79a7", linewidth=2)
    axes[0].axvline(config.N_CLUSTERS, linestyle="--", color="#e15759", label=f"k={config.N_CLUSTERS}")
    axes[0].set_title("Elbow Method — Inertia", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].legend()

    axes[1].plot(k_range, silhouettes, "s-", color="#59a14f", linewidth=2)
    axes[1].axvline(config.N_CLUSTERS, linestyle="--", color="#e15759")
    axes[1].set_title("Silhouette Score", fontweight="bold")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score (higher = better)")

    axes[2].plot(k_range, db_scores, "^-", color="#f28e2b", linewidth=2)
    axes[2].axvline(config.N_CLUSTERS, linestyle="--", color="#e15759")
    axes[2].set_title("Davies–Bouldin Index", fontweight="bold")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Score (lower = better)")

    plt.suptitle("KMeans Cluster Selection Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    SAVE("06_kmeans_selection.png")
    plt.close()

    # Final model at chosen N_CLUSTERS
    best_model = KMeans(
        n_clusters=config.N_CLUSTERS,
        random_state=random_state,
        n_init=30, max_iter=500
    )
    best_model.fit(X)
    return best_model, best_model.labels_, eval_df


# =============================================================================
# 2. Agglomerative (Hierarchical) Clustering
# =============================================================================
def train_agglomerative(
    X: np.ndarray,
    n_clusters: int = config.N_CLUSTERS,
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """
    Ward linkage agglomerative clustering.
    Also plots the truncated dendrogram for structural insight.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X)

    # ── Dendrogram (truncated) ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    Z = linkage(X, method="ward")
    dendrogram(
        Z, truncate_mode="lastp", p=30,
        leaf_rotation=90, leaf_font_size=9,
        show_contracted=True, ax=ax,
        color_threshold=Z[-(n_clusters - 1), 2],
    )
    ax.set_title("Hierarchical Clustering — Dendrogram (truncated)", fontweight="bold")
    ax.set_xlabel("Sample index or (cluster size)")
    ax.set_ylabel("Distance")
    ax.axhline(Z[-(n_clusters - 1), 2], linestyle="--", color="#e15759",
               label=f"Cut for k={n_clusters}")
    ax.legend()
    plt.tight_layout()
    SAVE("07_dendrogram.png")
    plt.close()

    metrics = _eval_labels(X, labels)
    log.info(f"Agglomerative k={n_clusters}: sil={metrics['silhouette']:.3f}  DB={metrics['davies_bouldin']:.3f}")
    return model, labels


# =============================================================================
# 3. DBSCAN
# =============================================================================
def train_dbscan(
    X: np.ndarray,
    eps: float     = config.DBSCAN_EPS,
    min_samples: int = config.DBSCAN_MIN_SAMPLES,
) -> tuple[DBSCAN, np.ndarray]:
    """
    DBSCAN doesn't require specifying k.
    Label -1 = noise point (stock doesn't fit any cluster).

    Best for detecting truly outlier stocks that defy normal groupings.
    """
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    metrics    = _eval_labels(X, labels)
    log.info(
        f"DBSCAN: eps={eps}, min_samples={min_samples} → "
        f"{n_clusters} clusters, {n_noise} noise points | sil={metrics['silhouette']:.3f}"
    )
    return model, labels


# =============================================================================
# 4. Silhouette subplot visualisation
# =============================================================================
def plot_silhouette(X: np.ndarray, labels: np.ndarray, title: str = "KMeans") -> None:
    """
    Silhouette plot per cluster.  Thin clusters or negatives indicate poor fit.
    """
    n_clusters  = len(np.unique(labels[labels != -1]))
    sil_samples = silhouette_samples(X[labels != -1], labels[labels != -1])
    avg_sil     = silhouette_score(X[labels != -1], labels[labels != -1])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    colors  = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        ith_sil = np.sort(sil_samples[labels[labels != -1] == i])
        size_i  = ith_sil.shape[0]
        y_upper = y_lower + size_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                         facecolor=colors[i], alpha=0.75)
        ax.text(-0.05, y_lower + 0.5 * size_i, f"Cluster {i}", fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(avg_sil, linestyle="--", color="red", label=f"Avg = {avg_sil:.3f}")
    ax.set_title(f"Silhouette Plot — {title}", fontweight="bold")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    plt.tight_layout()
    SAVE(f"08_silhouette_{title.lower().replace(' ', '_')}.png")
    plt.close()


# =============================================================================
# 5. Comparison summary
# =============================================================================
def compare_models(X: np.ndarray) -> pd.DataFrame:
    """
    Trains all three models and returns a comparison table.
    Returns the best (KMeans) labels as the primary labelling.
    """
    log.info("\n=== MODEL COMPARISON ===")

    km_model, km_labels, km_eval = train_kmeans(X)
    agg_model, agg_labels        = train_agglomerative(X)
    db_model, db_labels          = train_dbscan(X)

    rows = [
        {"Model": "KMeans",          **_eval_labels(X, km_labels)},
        {"Model": "Agglomerative",   **_eval_labels(X, agg_labels)},
        {"Model": "DBSCAN",          **_eval_labels(X, db_labels)},
    ]
    comparison = pd.DataFrame(rows).set_index("Model")
    log.info("\n" + comparison.to_string())

    # Silhouette plots
    plot_silhouette(X, km_labels,  "KMeans")
    plot_silhouette(X[agg_labels != -1], agg_labels[agg_labels != -1], "Agglomerative")

    return comparison, km_model, km_labels, agg_model, agg_labels


# =============================================================================
# 6. PCA visualisation of clusters
# =============================================================================
def plot_clusters_pca(
    X: np.ndarray,
    labels: np.ndarray,
    ticker_index: pd.Index,
    label_map: dict = None,
    title: str = "KMeans Clusters (PCA 2D)",
    filename: str = "09_cluster_pca.png",
) -> None:
    """
    Projects the feature space to 2 PCs and colours each stock by cluster.
    Annotates tickers for easy identification.
    """
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    label_names = [label_map.get(l, f"Cluster {l}") if label_map else f"Cluster {l}" for l in labels]
    color_map   = {
        "BUY":       "#00C853",
        "MAYBE BUY": "#FFD600",
        "NOT BUY":   "#D50000",
    }
    default_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

    fig, ax = plt.subplots(figsize=(14, 9))
    for lbl in sorted(set(label_names)):
        mask = [n == lbl for n in label_names]
        pcs_sub = pcs[mask]
        col = color_map.get(lbl, default_colors[list(set(label_names)).index(lbl) % len(default_colors)])
        ax.scatter(pcs_sub[:, 0], pcs_sub[:, 1], label=lbl, color=col,
                   s=80, alpha=0.85, edgecolors="white", linewidths=0.6)

    for i, (ticker, ln) in enumerate(zip(ticker_index, label_names)):
        col = color_map.get(ln, "grey")
        ax.annotate(ticker, (pcs[i, 0], pcs[i, 1]), fontsize=7, color=col, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(title="Category", framealpha=0.9)
    plt.tight_layout()
    SAVE(filename)
    plt.close()
    log.info(f"Plot saved: {filename}")


# =============================================================================
# 7. Persist models
# =============================================================================
def save_models(km_model: KMeans, agg_model: AgglomerativeClustering) -> None:
    joblib.dump(km_model,  os.path.join(config.MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(agg_model, os.path.join(config.MODEL_DIR, "agglomerative_model.pkl"))
    log.info(f"Models saved to {config.MODEL_DIR}")


def load_kmeans() -> KMeans:
    path = os.path.join(config.MODEL_DIR, "kmeans_model.pkl")
    return joblib.load(path)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from data_collection import load_data
    from feature_engineering import run_feature_pipeline

    prices, fundamentals = load_data()
    raw, scaled, scaler  = run_feature_pipeline(prices, fundamentals)
    X = scaled.values

    comparison, km_model, km_labels, agg_model, agg_labels = compare_models(X)
    save_models(km_model, agg_model)

    print("\n=== Clustering Comparison ===")
    print(comparison)

    print("\n=== KMeans Cluster Counts ===")
    unique, counts = np.unique(km_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} stocks")
