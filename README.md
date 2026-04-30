# 📊 Stock Cluster AI — ML-Powered Investment Classification

> **Cluster stocks into 🟢 BUY / 🟡 MAYBE BUY / 🔴 NOT BUY using unsupervised machine learning on 35+ financial features.**

---

## 🏗️ Project Architecture

```
stock_cluster_ml/
├── config.py                  ← Central configuration (tickers, hyperparameters)
├── data_collection.py         ← STEP 2: yfinance data + fundamentals
├── feature_engineering.py     ← STEP 3: Technical indicators + scaling
├── eda.py                     ← STEP 4: Exploratory data analysis
├── clustering.py              ← STEP 5: KMeans, Agglomerative, DBSCAN
├── cluster_interpretation.py  ← STEP 6: ★ Map clusters → BUY/MAYBE/NOT BUY
├── visualization.py           ← STEP 7: PCA plots, radar, heatmaps, Plotly
├── prediction.py              ← STEP 8: predict_stock_category(ticker)
├── train_pipeline.py          ← Master orchestrator (run this first)
├── app.py                     ← STEP 9: Streamlit web app
├── model_improvements.py      ← STEP 10: Future enhancement ideas
├── requirements.txt
│
├── data/                      ← Auto-created: CSVs of raw + engineered data
├── models/                    ← Auto-created: Saved .pkl model artefacts
└── outputs/                   ← Auto-created: All plots + logs
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full training pipeline
```bash
# Full run (downloads data, trains models, saves everything)
python train_pipeline.py

# Skip data download (if already collected)
python train_pipeline.py --skip-collection

# Skip EDA plots (faster iteration)
python train_pipeline.py --skip-collection --skip-eda
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. Predict a single stock (CLI)
```python
from prediction import predict_stock_category

result = predict_stock_category("AAPL")
print(result)
# Output:
# ==================================================
#   🟢  AAPL  →  BUY
#   Cluster ID : 0
#   Confidence : 82.3%
# ==================================================
#   Key Metrics:
#     PE_Ratio                      28.500
#     ROE                            1.472
#     Profit_Margin                  0.263
#     Revenue_Growth                 0.081
#     RSI                           58.230
#     Return_12M                     0.312
# ==================================================
```

---

## 🧠 How It Works

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
├──────────────┬──────────────────────────────────────────────┤
│  Data        │  yfinance: 2 years OHLCV + fundamentals      │
│  Collection  │  Universe: S&P 500 top 50 stocks             │
├──────────────┼──────────────────────────────────────────────┤
│  Feature     │  • 11 fundamental metrics                    │
│  Engineering │  • 24+ technical indicators                  │
│              │  • RobustScaler normalization                 │
│              │  • Z-score outlier winsorization              │
├──────────────┼──────────────────────────────────────────────┤
│  Clustering  │  • KMeans (k=3, selected via Elbow/Silhouette)│
│              │  • Agglomerative (Ward linkage, dendrogram)  │
│              │  • DBSCAN (noise detection)                  │
├──────────────┼──────────────────────────────────────────────┤
│  Interpret   │  • Composite scoring (20 weighted criteria)  │
│  Clusters    │  • Rank: BUY > MAYBE BUY > NOT BUY           │
└──────────────┴──────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION PHASE                         │
│  New ticker → same feature pipeline → KMeans.predict()     │
│  → cluster_id → label_map → BUY / MAYBE BUY / NOT BUY      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 Features Used (35+)

### Fundamental Metrics (from yfinance .info)
| Feature | Description | Investment Signal |
|---------|-------------|-------------------|
| PE_Ratio | Price-to-Earnings | Lower = potentially undervalued |
| PB_Ratio | Price-to-Book | <1 = trading below assets |
| EPS | Earnings Per Share | Absolute earning power |
| ROE | Return on Equity | >15% = strong capital efficiency |
| ROA | Return on Assets | Higher = better asset utilization |
| Profit_Margin | Net income / Revenue | Pricing power proxy |
| Revenue_Growth | YoY revenue change | Top-line momentum |
| Debt_to_Equity | Leverage ratio | >2 = elevated risk |
| Dividend_Yield | Annual div / Price | Income generation |
| Free_Cash_Flow | Operating CF - CapEx | Real cash generation |

### Technical Indicators (computed from price history)
| Feature | Description | Investment Signal |
|---------|-------------|-------------------|
| Price_SMA200_Ratio | Close / 200-day MA | >1 = long-term uptrend |
| RSI | Relative Strength Index | 50-70 = bullish momentum |
| MACD_Hist | MACD histogram | Positive = bullish crossover |
| BB_PctB | Bollinger Band %B | 0.8+ = strong trend |
| Return_12M | 1-year price return | Long-term momentum |
| Volatility_20D | Annualised vol | Risk proxy |
| Volume_Ratio | Volume vs 20D avg | Trend confirmation |
| Dist_52W_High | Distance from 52W high | Strength indicator |

---

## 🔍 Cluster Interpretation Logic

```python
# Each cluster gets a COMPOSITE SCORE based on weighted criteria:

SCORING_CRITERIA = [
    ("ROE",              3.0,  +1),   # higher ROE = better
    ("Debt_to_Equity",   2.0,  -1),   # higher D/E = worse
    ("Return_12M",       2.5,  +1),   # higher return = better
    ("Price_SMA200_Ratio", 2.0, +1),  # above SMA200 = better
    # ... 20 criteria total
]

# Clusters are ranked by score → mapped to labels:
#   Rank 1 (highest score) → 🟢 BUY
#   Rank 2 (middle score)  → 🟡 MAYBE BUY  
#   Rank 3 (lowest score)  → 🔴 NOT BUY
```

---

## 📊 Outputs Generated

After running `train_pipeline.py`, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `01_missing_values.png` | Missing data heatmap |
| `02_distributions.png` | Feature distribution plots |
| `03_correlation_heatmap.png` | Feature correlation matrix |
| `04_pca_analysis.png` | PCA scree + 2D scatter |
| `05_pairplot.png` | Top feature relationships |
| `06_kmeans_selection.png` | Elbow + Silhouette curves |
| `07_dendrogram.png` | Hierarchical clustering tree |
| `08_silhouette_kmeans.png` | Silhouette per cluster |
| `09_cluster_pca.png` | Clusters in PCA space |
| `10_pca_clusters.png` | Colour-coded investment categories |
| `11_radar_chart.png` | Cluster profile radar chart |
| `12_feature_heatmap.png` | Feature deviation heatmap |
| `13_top_stocks_table.png` | Top stocks per category |
| `14_category_distributions.png` | Metric distributions by category |
| `interactive_dashboard.html` | **Interactive 3D Plotly dashboard** |

---

## 🎛️ Configuration

Edit `config.py` to customize:

```python
# Switch between S&P 500 and NIFTY 50
TICKERS = SP500_TICKERS   # or NIFTY50_TICKERS

# Number of clusters (default: 3)
N_CLUSTERS = 3

# History: how many years of price data
HISTORY_YEARS = 2

# Outlier threshold (z-score)
OUTLIER_Z_THRESHOLD = 3.0
```

---

## 📈 Model Performance

Typical results on S&P 500 top 50:

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.28–0.45 |
| Davies-Bouldin | 0.90–1.30 |
| Calinski-Harabasz | 120–180 |

*Note: Financial clustering naturally produces moderate silhouette scores because stocks exist on a continuous spectrum, not in perfectly separated groups.*

---

## 🔮 Future Improvements

See `model_improvements.py` for detailed sketches of:

1. **FinBERT Sentiment Analysis** — News tone as a feature
2. **LSTM Price Forecasting** — Forward-looking dimension
3. **Markowitz Portfolio Optimization** — Optimal capital allocation
4. **Backtesting** — Validate BUY cluster outperformance
5. **Weekly Auto-retraining** — Keep the model fresh
6. **Ensemble Labelling** — More robust cluster assignments

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.
> It does **NOT** constitute financial advice.
> Always do your own research and consult a qualified financial advisor
> before making any investment decisions.
> Past ML cluster performance does not guarantee future returns.

---

## 🛠️ Tech Stack

- **Data**: `yfinance`, `pandas`, `numpy`
- **Technical Analysis**: `ta` (Technical Analysis Library)
- **ML**: `scikit-learn` (KMeans, Agglomerative, DBSCAN, PCA, RobustScaler)
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **App**: `streamlit`
- **Serialization**: `joblib`

---

*Built by a Senior AI/ML Engineer & Quant Researcher*
