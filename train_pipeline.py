# =============================================================================
# train_pipeline.py  —  Master Orchestrator
# =============================================================================
"""
Run this file ONCE to:
  1. Collect data
  2. Engineer features
  3. Run EDA
  4. Train + evaluate clustering models
  5. Interpret clusters
  6. Build visualizations
  7. Persist all artefacts (model, scaler, columns, label_map)

After this completes, app.py (Streamlit) can be launched immediately.

Usage:
  python train_pipeline.py
  python train_pipeline.py --skip-collection   (if data already downloaded)
  python train_pipeline.py --skip-eda          (skip EDA plots)
"""

import argparse
import logging
import os
import time

import joblib
import pandas as pd

import config
from cluster_interpretation import (
    get_feature_importance_per_cluster,
    interpret_clusters,
    print_cluster_report,
    save_labelled_data,
)
from clustering import compare_models, plot_clusters_pca, save_models
from data_collection import collect_all_data, load_data
from eda import run_eda
from feature_engineering import run_feature_pipeline
from visualization import run_visualizations

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.OUTPUT_DIR, "train_pipeline.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def run(skip_collection: bool = False, skip_eda: bool = False) -> None:
    t0 = time.time()
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║     STOCK CLUSTERING ML PIPELINE START       ║")
    log.info("╚══════════════════════════════════════════════╝")

    # ── STEP 1: Data Collection ────────────────────────────────────────────
    if skip_collection:
        log.info("── STEP 1: Loading existing data (--skip-collection) ──")
        prices, fundamentals = load_data()
    else:
        log.info("── STEP 1: Data Collection ──")
        prices, fundamentals = collect_all_data()

    # ── STEP 2: Feature Engineering ────────────────────────────────────────
    log.info("── STEP 2: Feature Engineering ──")
    raw_features, scaled_features, scaler = run_feature_pipeline(prices, fundamentals)

    # ── STEP 3: EDA ────────────────────────────────────────────────────────
    if not skip_eda:
        log.info("── STEP 3: Exploratory Data Analysis ──")
        run_eda(raw_features, scaled_features)

    # ── STEP 4: Clustering ─────────────────────────────────────────────────
    log.info("── STEP 4: Clustering Models ──")
    X = scaled_features.values
    comparison, km_model, km_labels, agg_model, agg_labels = compare_models(X)
    log.info("\n" + comparison.to_string())
    save_models(km_model, agg_model)

    # ── STEP 5: Cluster Interpretation ────────────────────────────────────
    log.info("── STEP 5: Cluster Interpretation ──")
    labelled_df, label_map, cluster_means = interpret_clusters(raw_features, km_labels)
    print_cluster_report(labelled_df, cluster_means, label_map)
    cluster_z = get_feature_importance_per_cluster(labelled_df)
    save_labelled_data(labelled_df)

    # ── STEP 6: Visualizations ─────────────────────────────────────────────
    log.info("── STEP 6: Visualizations ──")
    plot_clusters_pca(X, km_labels, scaled_features.index, label_map,
                      title="Stock Clusters after KMeans")
    top_stocks = run_visualizations(
        scaled_features, raw_features, labelled_df, cluster_means, cluster_z, label_map
    )

    # ── STEP 7: Persist prediction artefacts ──────────────────────────────
    log.info("── STEP 7: Saving Prediction Artefacts ──")
    joblib.dump(scaler,                  os.path.join(config.MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(scaled_features.columns), os.path.join(config.MODEL_DIR, "feature_columns.pkl"))
    joblib.dump(label_map,               os.path.join(config.MODEL_DIR, "label_map.pkl"))
    joblib.dump(raw_features,            os.path.join(config.MODEL_DIR, "raw_features.pkl"))
    joblib.dump(scaled_features,         os.path.join(config.MODEL_DIR, "scaled_features.pkl"))
    joblib.dump(labelled_df,             os.path.join(config.MODEL_DIR, "labelled_df.pkl"))
    joblib.dump(cluster_means,           os.path.join(config.MODEL_DIR, "cluster_means.pkl"))
    joblib.dump(cluster_z,               os.path.join(config.MODEL_DIR, "cluster_z.pkl"))

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  Pipeline COMPLETE in {elapsed:.1f}s")
    log.info(f"  Stocks processed : {len(labelled_df)}")
    log.info(f"  Feature columns  : {len(scaled_features.columns)}")
    log.info(f"  Label map        : {label_map}")
    log.info(f"  Category counts  :\n{labelled_df['Category'].value_counts().to_string()}")
    log.info(f"  Outputs          : {config.OUTPUT_DIR}")
    log.info(f"  Models           : {config.MODEL_DIR}")
    log.info(f"{'='*60}")

    print("\n🚀 Launch the Streamlit app:")
    print("   streamlit run app.py")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Clustering ML Pipeline")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip data collection and load existing CSVs")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plots")
    args = parser.parse_args()
    run(skip_collection=args.skip_collection, skip_eda=args.skip_eda)
