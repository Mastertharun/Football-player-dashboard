# ⚽ Transfermarkt Player Intelligence Dashboard

A full-stack football analytics dashboard built with **Streamlit**, powered by the [Kaggle Transfermarkt dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores). Features KMeans clustering, PCA visualization, market value tracking, transfer analysis, and interactive player scouting.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=flat-square&logo=plotly)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KMeans+PCA-F7931E?style=flat-square&logo=scikit-learn)

---

## 📸 Preview

> Dashboard includes 6 fully interactive tabs with dark-themed UI, KPI cards, and 20+ chart types.

---

## 🗂️ Dataset

Download from Kaggle: [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores)

Place the following CSV files inside a `data/` folder at the project root:

```
data/
├── players.csv
├── appearances.csv
├── player_valuations.csv
├── transfers.csv
├── clubs.csv
└── competitions.csv
```

---

## 🚀 Features

### 🎯 Tab 1 — Cluster Analysis
- PCA scatter plot of player clusters in 2D feature space
- Radar chart fingerprints per playing style
- Cluster composition by league (stacked bar)
- Donut chart with progress bars for cluster share

### 📊 Tab 2 — Player Performance
- Goals vs Assists scatter by position
- Avg Goals/Game violin plot by position
- Minutes played vs Goals with OLS trendline
- Top 20 players by Goals + Assists (stacked bar)
- Season-wise G+A trend line chart

### 💰 Tab 3 — Market Value & Transfers
- Market value box plot by league
- Market value vs performance scatter
- Market value timeline for top 8 players
- Top clubs by transfer spend
- Transfer fee vs market value scatter

### 🌡️ Tab 4 — Heatmaps
- Full feature correlation matrix (Seaborn)
- Position × Cluster heatmap
- League × Cluster heatmap
- Normalized cluster mean stats heatmap

### 📈 Tab 5 — Distributions
- Age distribution by league
- Market value distribution (log scale)
- Height distribution by sub-position
- International caps by nationality
- League → Sub-Position → Cluster sunburst
- Transfer fee distribution by season

### 🔍 Tab 6 — Scout & Compare
- Searchable player scouting table with gradient styling
- Multi-player comparison radar chart (normalized)
- Side-by-side percentile bar comparison

---

## 🧠 ML — KMeans Clustering

Players are clustered using **KMeans** on 5 features:

| Feature | Description |
|---|---|
| `avg_goals_pg` | Average goals per game |
| `avg_assists_pg` | Average assists per game |
| `avg_minutes` | Average minutes played |
| `market_value_m` | Market value in €M |
| `total_yellow` | Total yellow cards |

Features are scaled with **StandardScaler** and reduced to 2D with **PCA** for visualization. Number of clusters is configurable via sidebar slider (3–10).

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `streamlit` | App framework & UI |
| `pandas` | Data loading & transformation |
| `plotly` | Interactive charts |
| `seaborn` + `matplotlib` | Heatmaps |
| `scikit-learn` | KMeans, PCA, StandardScaler |
| `statsmodels` | OLS trendlines via Plotly Express |
| `numpy` | Numerical operations |

---

## ⚙️ Local Setup

**1. Clone the repo:**
```bash
git clone https://github.com/YOUR_USERNAME/player-dashboard.git
cd player-dashboard
```

**2. Create and activate virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download dataset from Kaggle and place CSVs in `data/` folder**

**5. Run the app:**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
seaborn
matplotlib
scikit-learn
statsmodels
```

Generate with:
```bash
pip freeze > requirements.txt
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your code to GitHub (ensure `data/` CSVs are included or use Kaggle API)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Click **Deploy**

---

## 📁 Project Structure

```
player-dashboard/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Ignores venv, cache, secrets
└── data/                   # Kaggle CSVs (not tracked in git if large)
    ├── players.csv
    ├── appearances.csv
    ├── player_valuations.csv
    ├── transfers.csv
    ├── clubs.csv
    └── competitions.csv
```

---

## 🙏 Credits

- Dataset: [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) on Kaggle
- Data sourced from [Transfermarkt](https://www.transfermarkt.com)

---

## 📄 License

MIT License — free to use, modify, and distribute.
