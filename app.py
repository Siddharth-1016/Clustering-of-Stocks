# =============================================================================
# app.py  —  STEP 9: Streamlit Web Application
# =============================================================================
"""
Launch with:   streamlit run app.py

Features:
  • Single ticker analysis with BUY / MAYBE BUY / NOT BUY verdict
  • Key financial metrics displayed as visual cards
  • Live radar chart for new ticker vs cluster average
  • Cluster PCA map with new ticker highlighted
  • Full universe overview with category filters
  • Batch analysis: paste multiple tickers
"""

import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

import config
from prediction import predict_stock_category, PredictionResult

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="Stock Cluster AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    /* Main app styling */
    .main { background-color: #0e1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #4e79a7;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateX(3px); }
    .metric-label { font-size: 11px; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 22px; font-weight: 700; color: #e8eaf0; margin-top: 2px; }

    /* Verdict banner */
    .verdict-buy      { background: linear-gradient(135deg, #00401a, #00C853); border-radius: 16px; padding: 24px; text-align: center; }
    .verdict-maybe    { background: linear-gradient(135deg, #3d3200, #FFD600); border-radius: 16px; padding: 24px; text-align: center; }
    .verdict-notbuy   { background: linear-gradient(135deg, #400000, #D50000); border-radius: 16px; padding: 24px; text-align: center; }
    .verdict-unknown  { background: linear-gradient(135deg, #1a1a2e, #4a4a6a); border-radius: 16px; padding: 24px; text-align: center; }
    .verdict-ticker   { font-size: 32px; font-weight: 800; color: white; letter-spacing: 2px; }
    .verdict-label    { font-size: 48px; font-weight: 900; color: white; margin: 8px 0; }
    .verdict-conf     { font-size: 14px; color: rgba(255,255,255,0.8); }

    /* Section headers */
    .section-header { font-size: 18px; font-weight: 700; color: #4e79a7; margin: 20px 0 10px; border-bottom: 1px solid #252b3b; padding-bottom: 6px; }

    /* Sidebar */
    .sidebar .sidebar-content { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cache: load model artefacts once
# =============================================================================
@st.cache_resource(show_spinner="Loading models…")
def load_artefacts():
    try:
        km_model       = joblib.load(os.path.join(config.MODEL_DIR, "kmeans_model.pkl"))
        scaler         = joblib.load(os.path.join(config.MODEL_DIR, "scaler.pkl"))
        feat_cols      = joblib.load(os.path.join(config.MODEL_DIR, "feature_columns.pkl"))
        label_map      = joblib.load(os.path.join(config.MODEL_DIR, "label_map.pkl"))
        labelled_df    = joblib.load(os.path.join(config.MODEL_DIR, "labelled_df.pkl"))
        scaled_df      = joblib.load(os.path.join(config.MODEL_DIR, "scaled_features.pkl"))
        cluster_means  = joblib.load(os.path.join(config.MODEL_DIR, "cluster_means.pkl"))
        config.CLUSTER_LABEL_MAP.update(label_map)
        return km_model, scaler, feat_cols, label_map, labelled_df, scaled_df, cluster_means
    except FileNotFoundError:
        return None


# =============================================================================
# Helpers
# =============================================================================
COLOR_MAP  = {"BUY": "#00C853", "MAYBE BUY": "#FFD600", "NOT BUY": "#D50000"}
EMOJI_MAP  = {"BUY": "🟢", "MAYBE BUY": "🟡", "NOT BUY": "🔴"}
CSS_CLASS  = {"BUY": "verdict-buy", "MAYBE BUY": "verdict-maybe", "NOT BUY": "verdict-notbuy"}

METRIC_LABELS = {
    "PE_Ratio":           "P/E Ratio",
    "PB_Ratio":           "P/B Ratio",
    "ROE":                "Return on Equity",
    "ROA":                "Return on Assets",
    "Profit_Margin":      "Profit Margin",
    "Revenue_Growth":     "Revenue Growth",
    "Earnings_Growth":    "Earnings Growth",
    "Debt_to_Equity":     "Debt / Equity",
    "Dividend_Yield":     "Dividend Yield",
    "RSI":                "RSI (14)",
    "Return_12M":         "12-Month Return",
    "Return_6M":          "6-Month Return",
    "Volatility_20D":     "Volatility (20D)",
    "Price_SMA200_Ratio": "Price / SMA-200",
    "MACD_Hist":          "MACD Histogram",
}

def fmt(val, pct=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if pct:
        return f"{val*100:.1f}%"
    return f"{val:.2f}"

def metric_card(label: str, value, unit: str = "") -> str:
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}{unit}</div>
    </div>"""

def verdict_banner(result: PredictionResult) -> str:
    css = CSS_CLASS.get(result.category, "verdict-unknown")
    emoji = EMOJI_MAP.get(result.category, "⚪")
    return f"""<div class="{css}">
        <div class="verdict-ticker">{result.ticker}</div>
        <div class="verdict-label">{emoji} {result.category}</div>
        <div class="verdict-conf">Confidence: {result.confidence:.1%} &nbsp;|&nbsp; Cluster #{result.cluster_id}</div>
    </div>"""


# =============================================================================
# Plot: Radar for new ticker vs its cluster average
# =============================================================================
RADAR_FEATURES = [
    "ROE", "ROA", "Profit_Margin", "Revenue_Growth",
    "Return_12M", "RSI", "Price_SMA200_Ratio",
    "Momentum_10D", "MACD_Hist", "Volatility_20D",
]

def build_radar_chart(result: PredictionResult, cluster_means: pd.DataFrame, label_map: dict) -> go.Figure:
    cat = result.category
    cluster_id = result.cluster_id

    available = [f for f in RADAR_FEATURES if
                 f in cluster_means.columns and
                 f in (result.raw_features.index if result.raw_features is not None else [])]
    if not available:
        return None

    # Normalise both to [0,1]
    cat_means = cluster_means.loc[cluster_id, available] if cluster_id in cluster_means.index else pd.Series()
    ticker_vals = result.raw_features[available] if result.raw_features is not None else pd.Series()

    combined = pd.DataFrame({"cluster": cat_means, "ticker": ticker_vals}).dropna()
    mn = combined.min().min(); mx = combined.max().max()
    if mx == mn:
        return None
    combined_norm = (combined - mn) / (mx - mn)

    angles = RADAR_FEATURES[:len(combined_norm)]
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=combined_norm["cluster"].tolist() + [combined_norm["cluster"].iloc[0]],
        theta=angles + [angles[0]],
        fill="toself",
        name=f"{EMOJI_MAP.get(cat, '')} {cat} Avg",
        line=dict(color=COLOR_MAP.get(cat, "#4e79a7"), width=2),
        fillcolor=COLOR_MAP.get(cat, "#4e79a7").replace("#", "rgba(") + ",0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=combined_norm["ticker"].tolist() + [combined_norm["ticker"].iloc[0]],
        theta=angles + [angles[0]],
        fill="toself",
        name=result.ticker,
        line=dict(color="#ffffff", width=2.5, dash="dot"),
        fillcolor="rgba(255,255,255,0.05)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=9)),
            bgcolor="#1a1f2e",
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        title=dict(text=f"{result.ticker} vs {cat} Cluster Average", font=dict(size=13)),
        legend=dict(font=dict(size=10)),
        height=400,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


# =============================================================================
# Plot: PCA map with all stocks + new ticker highlighted
# =============================================================================
def build_pca_map(scaled_df: pd.DataFrame, labelled_df: pd.DataFrame,
                  highlight_ticker: str = None,
                  new_point: np.ndarray = None) -> go.Figure:
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(scaled_df.values)

    categories = labelled_df.loc[scaled_df.index, "Category"] if "Category" in labelled_df.columns else pd.Series("UNKNOWN", index=scaled_df.index)

    fig = go.Figure()
    for cat in ["BUY", "MAYBE BUY", "NOT BUY"]:
        mask = categories == cat
        idx  = np.where(mask.values)[0]
        fig.add_trace(go.Scatter(
            x=pcs[idx, 0], y=pcs[idx, 1],
            mode="markers+text",
            marker=dict(size=8, color=COLOR_MAP[cat], opacity=0.8, line=dict(width=0.5, color="#0e1117")),
            text=list(scaled_df.index[idx]),
            textfont=dict(size=8, color=COLOR_MAP[cat]),
            textposition="top center",
            name=f"{EMOJI_MAP[cat]} {cat}",
            hovertemplate="%{text}<br>Category: " + cat + "<extra></extra>",
        ))

    # New ticker highlight
    if new_point is not None:
        pc_new = pca.transform(new_point.reshape(1, -1))
        fig.add_trace(go.Scatter(
            x=[pc_new[0, 0]], y=[pc_new[0, 1]],
            mode="markers+text",
            marker=dict(size=18, color="white", symbol="star", line=dict(width=2, color="gold")),
            text=[f"★ {highlight_ticker}"],
            textfont=dict(size=11, color="gold"),
            textposition="top center",
            name=f"★ {highlight_ticker} (NEW)",
            hovertemplate=f"<b>{highlight_ticker}</b> — NEW PREDICTION<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="All Stocks — PCA Cluster Map", font=dict(size=13)),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e",
        font=dict(color="white"),
        xaxis=dict(title=f"PC1", gridcolor="#252b3b"),
        yaxis=dict(title=f"PC2", gridcolor="#252b3b"),
        legend=dict(font=dict(size=10), bgcolor="#1a1f2e", bordercolor="#252b3b"),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


# =============================================================================
# Universe overview table
# =============================================================================
def build_universe_table(labelled_df: pd.DataFrame) -> go.Figure:
    disp_cols = ["Category", "PE_Ratio", "ROE", "Profit_Margin",
                 "Revenue_Growth", "Return_12M", "Volatility_20D"]
    avail = [c for c in disp_cols if c in labelled_df.columns]
    df = labelled_df[avail].copy().round(3)
    df.index.name = "Ticker"
    df_display = df.reset_index()

    cell_colors = [
        [COLOR_MAP.get(cat, "#252b3b") if col == "Category" else "#1a1f2e"
         for cat in df_display.get("Category", [])]
        for col in df_display.columns
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in df_display.columns],
            fill_color="#252b3b",
            font=dict(color="white", size=11),
            align="center",
        ),
        cells=dict(
            values=[df_display[c] for c in df_display.columns],
            fill_color=cell_colors,
            font=dict(color="white", size=10),
            align="center",
            height=26,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117", height=600, margin=dict(t=20, b=20, l=10, r=10)
    )
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 Stock Cluster AI")
        st.markdown("*Powered by ML Clustering*")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🔍 Analyze Stock", "🗺️ Universe Map", "📋 All Stocks", "ℹ️ About"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown(
            "**Model**: KMeans (k=3)\n\n"
            "**Features**: 35+ fundamentals + technicals\n\n"
            "**Universe**: S&P 500 top 50",
            unsafe_allow_html=False
        )

    # ── Load artefacts ─────────────────────────────────────────────────────
    artefacts = load_artefacts()
    if artefacts is None:
        st.error(
            "⚠️ Model artefacts not found.\n\n"
            "Please run the training pipeline first:\n\n"
            "```bash\npython train_pipeline.py\n```"
        )
        st.stop()

    km_model, scaler, feat_cols, label_map, labelled_df, scaled_df, cluster_means = artefacts

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1: Analyze a single stock
    # ══════════════════════════════════════════════════════════════════════
    if page == "🔍 Analyze Stock":
        st.markdown("## 🔍 Analyze a Stock")
        st.markdown("Enter any stock ticker to get an AI-powered investment category.")

        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_input(
                "Stock Ticker",
                placeholder="e.g. AAPL, MSFT, TSLA, RELIANCE.NS",
                label_visibility="collapsed"
            )
        with col2:
            analyze_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")

        if analyze_btn and ticker_input:
            ticker = ticker_input.strip().upper()
            with st.spinner(f"Fetching data and analyzing {ticker}…"):
                result = predict_stock_category(ticker)

            if result.error and result.category in ("ERROR", "UNKNOWN"):
                st.error(f"❌ {result.error}")
            else:
                # ── Verdict banner ────────────────────────────────────────
                st.markdown(verdict_banner(result), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Key metrics cards ─────────────────────────────────────
                st.markdown('<div class="section-header">📌 Key Financial Metrics</div>', unsafe_allow_html=True)

                metrics_to_show = {
                    "row1": [
                        ("PE_Ratio", False), ("PB_Ratio", False), ("ROE", True), ("ROA", True)
                    ],
                    "row2": [
                        ("Profit_Margin", True), ("Revenue_Growth", True), ("Debt_to_Equity", False), ("Dividend_Yield", True)
                    ],
                    "row3": [
                        ("RSI", False), ("Return_12M", True), ("Volatility_20D", True), ("Price_SMA200_Ratio", False)
                    ],
                }

                for row_key, cols_cfg in metrics_to_show.items():
                    row_cols = st.columns(4)
                    for col, (metric, is_pct) in zip(row_cols, cols_cfg):
                        val = result.key_metrics.get(metric)
                        display_val = fmt(val, pct=is_pct) if val is not None else "—"
                        label = METRIC_LABELS.get(metric, metric)
                        col.markdown(metric_card(label, display_val), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Two-column: Radar + PCA ───────────────────────────────
                left, right = st.columns(2)

                with left:
                    st.markdown('<div class="section-header">📡 Feature Profile (vs Cluster)</div>', unsafe_allow_html=True)
                    radar_fig = build_radar_chart(result, cluster_means, label_map)
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.info("Radar chart unavailable for this ticker.")

                with right:
                    st.markdown('<div class="section-header">🗺️ Position in Cluster Map</div>', unsafe_allow_html=True)
                    new_point = result.raw_features.values if result.raw_features is not None else None
                    pca_fig = build_pca_map(
                        scaled_df, labelled_df,
                        highlight_ticker=ticker,
                        new_point=scaler.transform(new_point.reshape(1, -1))[0] if new_point is not None else None,
                    )
                    st.plotly_chart(pca_fig, use_container_width=True)

                # ── Interpretation box ────────────────────────────────────
                st.markdown('<div class="section-header">💡 Interpretation</div>', unsafe_allow_html=True)
                cat = result.category
                interpretations = {
                    "BUY": (
                        "✅ This stock exhibits **strong fundamental and technical signals**. "
                        "It shows solid profitability metrics (ROE, ROA, Profit Margin), positive price momentum, "
                        "and healthy financial structure. Consider for **long positions** with appropriate risk management."
                    ),
                    "MAYBE BUY": (
                        "⚠️ This stock shows **mixed signals**. Some metrics are attractive but others raise concerns. "
                        "It may be in a transition phase — improving fundamentals but lagging technical momentum, "
                        "or vice versa. Consider **watchlisting** and revisiting on next earnings release."
                    ),
                    "NOT BUY": (
                        "❌ This stock exhibits **weak fundamental and/or technical signals**. "
                        "Signs may include declining margins, high leverage, negative momentum, or poor returns. "
                        "**Avoid or consider short** with proper position sizing and stop-losses."
                    ),
                }
                st.info(interpretations.get(cat, "No interpretation available."))

                # ── Warning ───────────────────────────────────────────────
                st.warning(
                    "⚠️ **Disclaimer**: This is an AI-generated analysis for educational purposes only. "
                    "It does NOT constitute financial advice. Always do your own research (DYOR) and "
                    "consult a qualified financial advisor before making investment decisions."
                )

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2: Universe Map
    # ══════════════════════════════════════════════════════════════════════
    elif page == "🗺️ Universe Map":
        st.markdown("## 🗺️ Stock Universe Cluster Map")
        st.markdown("All trained stocks plotted in PCA space, coloured by investment category.")

        pca_fig = build_pca_map(scaled_df, labelled_df)
        st.plotly_chart(pca_fig, use_container_width=True)

        # Category distribution pie
        cat_counts = labelled_df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        pie_fig = px.pie(
            cat_counts, names="Category", values="Count",
            color="Category",
            color_discrete_map=COLOR_MAP,
            title="Category Distribution",
            hole=0.45,
        )
        pie_fig.update_layout(
            paper_bgcolor="#0e1117", font=dict(color="white"), height=350
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Category Counts")
            for _, row in cat_counts.iterrows():
                e = EMOJI_MAP.get(row["Category"], "⚪")
                st.markdown(f"{e} **{row['Category']}**: {row['Count']} stocks")
        with col2:
            st.plotly_chart(pie_fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3: All Stocks Table
    # ══════════════════════════════════════════════════════════════════════
    elif page == "📋 All Stocks":
        st.markdown("## 📋 Full Stock Universe")

        cat_filter = st.multiselect(
            "Filter by Category",
            options=["BUY", "MAYBE BUY", "NOT BUY"],
            default=["BUY", "MAYBE BUY", "NOT BUY"],
        )

        filtered = labelled_df[labelled_df["Category"].isin(cat_filter)] if "Category" in labelled_df.columns else labelled_df
        st.markdown(f"Showing **{len(filtered)}** stocks")

        table_fig = build_universe_table(filtered)
        st.plotly_chart(table_fig, use_container_width=True)

        # Download button
        disp_cols = ["Category", "PE_Ratio", "ROE", "Profit_Margin",
                     "Revenue_Growth", "Return_12M", "Volatility_20D"]
        avail = [c for c in disp_cols if c in filtered.columns]
        csv_data = filtered[avail].round(4).to_csv()
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_data, file_name="stock_clusters.csv", mime="text/csv"
        )

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4: About
    # ══════════════════════════════════════════════════════════════════════
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This Project")
        st.markdown("""
        ### 🧠 Methodology
        
        This project uses **unsupervised machine learning** to cluster stocks into three investment
        categories based on their financial fingerprint — no historical labels required.
        
        #### Pipeline Overview
        ```
        Raw Data (yfinance)
            ↓
        Feature Engineering (35+ indicators)
            ↓
        Outlier Removal + RobustScaler
            ↓
        KMeans Clustering (k=3)
            ↓
        Cluster Interpretation (scoring logic)
            ↓
        BUY / MAYBE BUY / NOT BUY
        ```
        
        #### Features Used
        **Fundamentals**: P/E, P/B, EPS, ROE, ROA, Debt/Equity, Revenue Growth,
        Profit Margin, Free Cash Flow, Dividend Yield
        
        **Technical**: SMA-20/50/200, EMA-12/26, RSI, MACD, Bollinger Bands,
        Volatility, Momentum (1M/3M/6M/12M returns), Volume Trend, ATR
        
        #### Model
        - **Algorithm**: KMeans (optimal k selected via Elbow + Silhouette)
        - **Scaling**: RobustScaler (handles fat-tailed financial distributions)
        - **Validation**: Silhouette score, Davies-Bouldin index
        
        #### Disclaimer
        > ⚠️ This tool is for **educational and research purposes only**.
        > It does NOT provide financial advice. Past cluster behaviour does not
        > guarantee future returns. Always consult a qualified financial advisor.
        """)


# =============================================================================
if __name__ == "__main__":
    main()
