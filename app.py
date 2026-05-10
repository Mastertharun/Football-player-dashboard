"""
⚽ Transfermarkt Player Performance Dashboard
Real Kaggle schema: players.csv · appearances.csv · player_valuations.csv
                    transfers.csv · clubs.csv · competitions.csv
Tech: Streamlit · Pandas · Plotly · Seaborn · Scikit-learn KMeans
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings, os
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Transfermarkt Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:      #070d12;
  --bg2:     #0d1821;
  --card:    #111f2e;
  --border:  #1c3047;
  --accent1: #00d4aa;   /* teal */
  --accent2: #f7c844;   /* gold */
  --accent3: #e84393;   /* pink */
  --text:    #d6eaf8;
  --muted:   #5d8aa8;
}

html, body, [class*="css"] {
  font-family:'Inter',sans-serif;
  background:var(--bg);
  color:var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#050c14 0%,#0b1828 100%);
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] * { color:#b8d4e8 !important; }
section[data-testid="stSidebar"] label { color:var(--accent1) !important; font-weight:600 !important; font-size:.78rem !important; letter-spacing:.08em !important; text-transform:uppercase !important; }

/* ── KPI card ── */
.kpi {
  background:linear-gradient(135deg,#111f2e 0%,#0d1821 100%);
  border:1px solid var(--border);
  border-radius:14px;
  padding:16px 20px;
  text-align:center;
  position:relative;
  overflow:hidden;
  transition:transform .2s,box-shadow .2s;
}
.kpi::after {
  content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--accent1),var(--accent2));
}
.kpi:hover { transform:translateY(-3px); box-shadow:0 8px 24px #00d4aa18; }
.kpi-label { font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:6px; }
.kpi-val   { font-family:'Exo 2',sans-serif;font-size:2.1rem;font-weight:800;color:var(--accent1);line-height:1; }
.kpi-sub   { font-size:.68rem;color:#3a6e8a;margin-top:3px; }

/* ── Section headers ── */
.sh {
  font-family:'Exo 2',sans-serif;
  font-size:1.2rem;font-weight:700;
  letter-spacing:.06em;
  color:var(--accent1);
  border-left:4px solid var(--accent1);
  padding-left:12px;
  margin:22px 0 14px;
}

/* ── Page title ── */
.hero-title {
  font-family:'Exo 2',sans-serif;
  font-size:3rem;font-weight:800;letter-spacing:.04em;
  background:linear-gradient(135deg,#00d4aa 0%,#f7c844 55%,#e84393 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  line-height:1.1;
}
.hero-sub {
  color:var(--muted);font-size:.82rem;letter-spacing:.12em;
  text-transform:uppercase;margin-top:5px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap:6px;background:transparent; }
.stTabs [data-baseweb="tab"] {
  background:#111f2e;border:1px solid #1c3047;border-radius:8px;
  color:#5d8aa8;font-weight:600;padding:8px 16px;letter-spacing:.04em;
}
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,#152536,#0d1821);
  border-color:var(--accent1);color:var(--accent1) !important;
}

/* ── Plotly containers ── */
.stPlotlyChart { border:1px solid #1c3047;border-radius:12px;overflow:hidden; }

/* ── Progress bars ── */
.prog-wrap { margin-bottom:10px; }
.prog-label { display:flex;justify-content:space-between;margin-bottom:3px; }
.prog-bar-bg { background:#1c3047;border-radius:4px;height:5px; }
.prog-bar-fill { height:5px;border-radius:4px; }

#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1rem}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY BASE THEME
# ══════════════════════════════════════════════════════════════════════════════
PLT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(7,13,18,0.7)',
    font=dict(family='Inter', color='#b8d4e8', size=11),
    # title=dict(font=dict(family='Exo 2', size=17, color='#00d4aa')),
    xaxis=dict(gridcolor='#1c3047', linecolor='#1c3047', zerolinecolor='#1c3047'),
    yaxis=dict(gridcolor='#1c3047', linecolor='#1c3047', zerolinecolor='#1c3047'),
    legend=dict(bgcolor='rgba(13,24,33,.85)', bordercolor='#1c3047', borderwidth=1),
    margin=dict(l=44,r=16,t=52,b=40),
)

CLUSTER_COLORS = {
    'Creative Playmakers':'#f7c844',
    'Clinical Strikers'  :'#e84393',
    'Box-to-Box Engines' :'#00d4aa',
    'Defensive Shields'  :'#6c9ecf',
    'Flying Wingers'     :'#a8e06a',
    'Sweeper Keepers'    :'#c084fc',
}

POS_COLORS = {
    'Goalkeeper':'#c084fc','Centre-Back':'#6c9ecf','Left-Back':'#60a5fa',
    'Right-Back':'#93c5fd','Defensive Midfield':'#34d399',
    'Central Midfield':'#00d4aa','Attacking Midfield':'#f7c844',
    'Left Winger':'#a8e06a','Right Winger':'#86efac',
    'Centre-Forward':'#e84393','Second Striker':'#fca5a5',
}

LEAGUE_COLORS = {
    'Premier League':'#38003c','La Liga':'#ff4b44','Bundesliga':'#d00027',
    'Serie A':'#008fd7','Ligue 1':'#091c3e',
}
LEAGUE_ACC = {
    'Premier League':'#00ff85','La Liga':'#f4a020','Bundesliga':'#e8c22a',
    'Serie A':'#008fd7','Ligue 1':'#d5283a',
}

# def download_kaggle_data():
#     required_files = [
#         'data/players.csv',
#         'data/appearances.csv', 
#         'data/player_valuations.csv',
#         'data/transfers.csv',
#         'data/clubs.csv',
#         'data/competitions.csv',
#     ]
    
#     # Check if all required files already exist
#     if all(os.path.exists(f) for f in required_files):
#         return

#     try:
#         os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle']['username']
#         os.environ['KAGGLE_KEY']      = st.secrets['kaggle']['key']
#     except Exception:
#         pass

#     try:
#         import kaggle
#         os.makedirs('data', exist_ok=True)
#         kaggle.api.authenticate()

#         # Download only the 6 files you need, not the whole dataset
#         files_to_download = [
#             'players.csv',
#             'appearances.csv',
#             'player_valuations.csv',
#             'transfers.csv',
#             'clubs.csv',
#             'competitions.csv',
#         ]

#         for filename in files_to_download:
#             if not os.path.exists(f'data/{filename}'):
#                 with st.spinner(f'📥 Downloading {filename}...'):
#                     kaggle.api.dataset_download_file(
#                         'davidcariboo/player-scores',
#                         file_name=filename,
#                         path='data/',
#                         force=False
#                     )
#                     # Unzip if downloaded as .zip
#                     zip_path = f'data/{filename}.zip'
#                     if os.path.exists(zip_path):
#                         import zipfile
#                         with zipfile.ZipFile(zip_path, 'r') as z:
#                             z.extractall('data/')
#                         os.remove(zip_path)
#                 st.success(f'✅ {filename} ready!')

#         st.success('✅ All data downloaded!')
#         st.rerun()

#     except Exception as e:
#         st.error(f'❌ Kaggle download failed: {e}')
#         st.stop()

def download_kaggle_data():
    required_files = [
        'data/players.csv',
        'data/appearances.csv',
        'data/player_valuations.csv',
        'data/transfers.csv',
        'data/clubs.csv',
        'data/competitions.csv',
    ]

    if all(os.path.exists(f) for f in required_files):
        return

    # Direct download URLs from your GitHub Release
    # Replace YOUR_USERNAME and YOUR_REPO with your actual values
    BASE_URL = "https://github.com/Mastertharun/Football-player-dashboard/releases/download/v1.0-data"
    
    files = [
        'players.csv',
        'appearances.csv',
        'player_valuations.csv',
        'transfers.csv',
        'clubs.csv',
        'competitions.csv',
    ]

    import requests
    os.makedirs('data', exist_ok=True)

    for filename in files:
        filepath = f'data/{filename}'
        if not os.path.exists(filepath):
            with st.spinner(f'📥 Downloading {filename}...'):
                try:
                    url = f'{BASE_URL}/{filename}'
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success(f'✅ {filename} done!')
                except Exception as e:
                    st.error(f'❌ Failed to download {filename}: {e}')
                    st.stop()

    st.success('✅ All data ready!')
    st.rerun()

def plt_override(**kwargs):
    """Merge PLT with overrides, handling nested dicts like margin."""
    cfg = PLT.copy()
    cfg.update(kwargs)
    return cfg

def hex_to_rgba(hex_color, alpha=0.15):
    """Convert #rrggbb hex to rgba(r,g,b,alpha) string for Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD DATA IF MISSING
# ══════════════════════════════════════════════════════════════════════════════
download_kaggle_data()  # ← ADD THIS LINE

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_all(n_clusters=6):
    players  = pd.read_csv('data/players.csv')
    apps     = pd.read_csv('data/appearances.csv')
    comps    = pd.read_csv('data/competitions.csv')
    clubs    = pd.read_csv('data/clubs.csv')
    vals     = pd.read_csv('data/player_valuations.csv')
    transfers= pd.read_csv('data/transfers.csv')

    # ── Aggregate appearances ──────────────────────────────────────────────
    agg = apps.groupby('player_id').agg(
        games_played   =('appearance_id','count'),
        total_goals    =('goals','sum'),
        total_assists  =('assists','sum'),
        total_yellow   =('yellow_cards','sum'),
        total_red      =('red_cards','sum'),
        total_minutes  =('minutes_played','sum'),
        avg_goals_pg   =('goals','mean'),
        avg_assists_pg =('assists','mean'),
        avg_minutes    =('minutes_played','mean'),
    ).reset_index()

    df = players.merge(agg, on='player_id', how='left')
    for c in ['total_goals','total_assists','games_played','total_minutes',
              'avg_goals_pg','avg_assists_pg','avg_minutes','total_yellow','total_red']:
        df[c] = df[c].fillna(0)

    # League name
    comp_map = dict(zip(comps.competition_id, comps.name))
    df['league_name'] = df['current_club_domestic_competition_id'].map(comp_map)

    # Age
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    # df['age'] = ((pd.Timestamp('2024-12-01') - df['date_of_birth']).dt.days / 365.25).astype(int)

    df['age'] = ((pd.Timestamp('2024-12-01') - df['date_of_birth']).dt.days / 365.25)
    df['age'] = df['age'].replace([np.inf, -np.inf], np.nan)  # kill inf values
    df['age'] = df['age'].fillna(0).astype(int)   

    # Market value in millions
    df['market_value_m'] = (df['market_value_in_eur'] / 1e6).round(2)
    df['g_plus_a']       = df['total_goals'] + df['total_assists']

    # ── KMeans clustering ─────────────────────────────────────────────────
    feat = ['avg_goals_pg','avg_assists_pg','avg_minutes','market_value_m','total_yellow']
    X = df[feat].fillna(0)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_sc)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    df['pca_x'] = coords[:,0].round(3)
    df['pca_y'] = coords[:,1].round(3)

    style_names = ['Creative Playmakers','Clinical Strikers','Box-to-Box Engines',
                   'Defensive Shields','Flying Wingers','Sweeper Keepers',
                   'Cluster G','Cluster H','Cluster I','Cluster J']
    df['cluster_name'] = df['cluster'].map({i: style_names[i] for i in range(n_clusters)})

    # ── Latest valuations ─────────────────────────────────────────────────
    latest_val = vals.sort_values('date').groupby('player_id').last()[['market_value_in_eur']].reset_index()
    latest_val.columns = ['player_id','latest_market_value_in_eur']
    df = df.merge(latest_val, on='player_id', how='left')

    # ── Transfer spend per club ────────────────────────────────────────────
    club_spend = transfers.groupby('to_club_name')['transfer_fee'].sum().reset_index()
    club_spend.columns = ['club','total_spend']
    club_income = transfers.groupby('from_club_name')['transfer_fee'].sum().reset_index()
    club_income.columns = ['club','total_income']

    return df, apps, vals, transfers, clubs, comps, club_spend, club_income, pca

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:\'Exo 2\',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(135deg,#00d4aa,#f7c844);-webkit-background-clip:text;-webkit-text-fill-color:transparent">⚽ TM Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#3a6e8a;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase">Transfermarkt Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")

    n_clusters = st.slider("🔵 K-Means Clusters", 3, 10, 6)

    df_all, apps, vals, transfers, clubs, comps, club_spend, club_income, pca = load_all(n_clusters)

    # Dynamic cluster color map
    c_colors = {nm: list(CLUSTER_COLORS.values())[i % 6]
                for i, nm in enumerate(df_all['cluster_name'].unique())}

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    leagues = st.multiselect("League", sorted(df_all.league_name.dropna().unique()),
                             default=sorted(df_all.league_name.dropna().unique()))
    positions = st.multiselect("Position", sorted(df_all.position.unique()),
                               default=sorted(df_all.position.unique()))
    styles = st.multiselect("Playing Style", sorted(df_all.cluster_name.unique()),
                            default=sorted(df_all.cluster_name.unique()))
    age_r = st.slider("Age", int(df_all.age.min()), int(df_all.age.max()),
                      (int(df_all.age.min()), int(df_all.age.max())))
    mv_r  = st.slider("Market Value (€M)", 0.0, float(df_all.market_value_m.max()),
                      (0.0, float(df_all.market_value_m.max())))

    st.markdown("---")
    st.caption("📁 Data: Kaggle · davidcariboo/player-scores\nColumns: players · appearances · player_valuations · transfers · clubs · competitions")

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df_all.league_name.isin(leagues) &
    df_all.position.isin(positions) &
    df_all.cluster_name.isin(styles) &
    df_all.age.between(*age_r) &
    df_all.market_value_m.between(*mv_r)
)
df = df_all[mask].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">⚽ Transfermarkt<br>Player Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Kaggle · davidcariboo/player-scores · KMeans Clustering · Playing Style Analysis</div>', unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k = st.columns(6)
kpis = [
    ("Players Analyzed", f"{len(df):,}", "filtered"),
    ("Avg Market Value", f"€{df.market_value_m.mean():.1f}M", "per player"),
    ("Total Goals", f"{int(df.total_goals.sum()):,}", "across appearances"),
    ("Total Assists", f"{int(df.total_assists.sum()):,}", "across appearances"),
    ("Avg Age", f"{df.age.mean():.1f}", "years old"),
    ("Style Clusters", f"{df.cluster_name.nunique()}", "playing types"),
]
for col, (label, val, sub) in zip(k, kpis):
    with col:
        st.markdown(f"""<div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
t1,t2,t3,t4,t5,t6 = st.tabs([
    "🎯 Cluster Analysis","📊 Player Performance","💰 Market & Transfers",
    "🌡️ Heatmaps","📈 Distributions","🔍 Scout & Compare"
])

# ╔════════════════════════════════════════════════════════╗
# ║  TAB 1 — CLUSTER ANALYSIS                             ║
# ╚════════════════════════════════════════════════════════╝
with t1:
    c_left, c_right = st.columns([3,1])
    with c_left:
        st.markdown('<div class="sh">PCA Cluster Map — Playing Styles</div>', unsafe_allow_html=True)
        fig = px.scatter(
            df, x='pca_x', y='pca_y', color='cluster_name',
            color_discrete_map=c_colors,
            size='market_value_m', size_max=26, opacity=.82,
            hover_data=dict(name=True, position=True, current_club_name=True,
                            league_name=True, total_goals=True, total_assists=True,
                            market_value_m=True, pca_x=False, pca_y=False),
            title='Player Clusters in PCA Feature Space',
        )
        fig.update_traces(marker_line_width=.4, marker_line_color='#000')
        fig.update_layout(**PLT, height=520,
            xaxis_title='PC1 — Attacking tendency →',
            yaxis_title='PC2 — Value / discipline →',
            legend_title_text='Playing Style')
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="sh">Cluster Share</div>', unsafe_allow_html=True)
        cc = df.cluster_name.value_counts().reset_index()
        cc.columns=['style','count']
        fig_p = px.pie(cc, values='count', names='style',
                       color='style', color_discrete_map=c_colors, hole=.58)
        fig_p.update_layout(**plt_override(height=240, showlegend=False, margin=dict(l=4,r=4,t=8,b=4)))
        fig_p.update_traces(textinfo='label+percent', textfont_size=9)
        st.plotly_chart(fig_p, use_container_width=True)

        for _, row in cc.iterrows():
            col = c_colors.get(row['style'],'#00d4aa')
            pct = row['count']/len(df)*100
            st.markdown(f"""<div class="prog-wrap">
              <div class="prog-label">
                <span style="font-size:.7rem;color:{col};font-weight:700">{row['style']}</span>
                <span style="font-size:.7rem;color:#3a6e8a">{row['count']}</span>
              </div>
              <div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{pct}%;background:{col}"></div></div>
            </div>""", unsafe_allow_html=True)

    # Radar fingerprints
    st.markdown('<div class="sh">Cluster Fingerprints — Radar</div>', unsafe_allow_html=True)
    radar_feats  = ['avg_goals_pg','avg_assists_pg','avg_minutes','market_value_m','total_yellow']
    radar_labels = ['Goals/Game','Assists/Game','Avg Min','Value €M','Yellow Cards']
    cmeans = df.groupby('cluster_name')[radar_feats].mean()
    for col in radar_feats:
        cmeans[col] = (cmeans[col]-cmeans[col].min())/(cmeans[col].max()-cmeans[col].min()+1e-9)

    fig_r = go.Figure()
    for cname, clr in c_colors.items():
        if cname not in cmeans.index: continue
        v = cmeans.loc[cname].tolist(); v.append(v[0])
        l = radar_labels + [radar_labels[0]]
        fig_r.add_trace(go.Scatterpolar(r=v, theta=l, name=cname, fill='toself',
    fillcolor=hex_to_rgba(clr, 0.13), line=dict(color=clr, width=2.2)))
        
    fig_r.update_layout(**plt_override(height=440,
        polar=dict(bgcolor='rgba(7,13,18,.85)',
            radialaxis=dict(visible=True, range=[0,1], gridcolor='#1c3047',
                            tickfont=dict(size=8,color='#5d8aa8')),
            angularaxis=dict(gridcolor='#1c3047', tickfont=dict(size=11,color='#b8d4e8'))),
        title='Normalized Stat Profiles per Style Cluster'))

    st.plotly_chart(fig_r, use_container_width=True)

    # Cluster × league breakdown
    st.markdown('<div class="sh">Cluster Composition by League</div>', unsafe_allow_html=True)
    cl_lg = df.groupby(['cluster_name','league_name']).size().reset_index(name='count')
    fig_cl = px.bar(cl_lg, x='cluster_name', y='count', color='league_name',
                    barmode='stack',
                    color_discrete_sequence=['#00ff85','#f4a020','#e8c22a','#008fd7','#d5283a'])
    fig_cl.update_layout(**PLT, height=360, xaxis_title='', yaxis_title='Players', legend_title='League')
    st.plotly_chart(fig_cl, use_container_width=True)

# ╔════════════════════════════════════════════════════════╗
# ║  TAB 2 — PLAYER PERFORMANCE                           ║
# ╚════════════════════════════════════════════════════════╝
with t2:
    r1a, r1b = st.columns(2)
    with r1a:
        st.markdown('<div class="sh">Goals vs Assists — by Position</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='total_goals', y='total_assists',
            color='position', color_discrete_map=POS_COLORS,
            size='market_value_m', size_max=22, opacity=.8,
            hover_data=['name','current_club_name','league_name','games_played','market_value_m'])
        fig.update_layout(**PLT, height=370, xaxis_title='Total Goals (appearances)', yaxis_title='Total Assists')
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        st.markdown('<div class="sh">Avg Goals/Game — Violin by Position</div>', unsafe_allow_html=True)
        fig = px.violin(df[df.position!='Goalkeeper'], x='position', y='avg_goals_pg',
            color='position', color_discrete_map=POS_COLORS, box=True, points='outliers')
        fig.update_layout(**plt_override(height=370, showlegend=False,
            xaxis=dict(**PLT['xaxis'], tickangle=-35), yaxis_title='Avg Goals/Game'))
        st.plotly_chart(fig, use_container_width=True)

    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown('<div class="sh">Minutes Played vs Goals — by Cluster</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='total_minutes', y='total_goals',
            color='cluster_name', color_discrete_map=c_colors,
            trendline='ols', trendline_scope='overall',
            trendline_color_override='#00d4aa',
            hover_data=['name','position','current_club_name'], opacity=.75)
        fig.update_layout(**PLT, height=360,
            xaxis_title='Total Minutes Played', yaxis_title='Total Goals')
        st.plotly_chart(fig, use_container_width=True)

    with r2b:
        st.markdown('<div class="sh">Top 20 Players — Goals + Assists</div>', unsafe_allow_html=True)
        top20 = df.nlargest(20,'g_plus_a')[['name','total_goals','total_assists','cluster_name']].copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top20['name'], x=top20['total_goals'], name='Goals',
            orientation='h', marker_color='#e84393', marker_line_width=0))
        fig.add_trace(go.Bar(y=top20['name'], x=top20['total_assists'], name='Assists',
            orientation='h', marker_color='#00d4aa', marker_line_width=0))
        fig.update_layout(**plt_override(
            barmode='stack', height=500,
            xaxis_title='Goals + Assists', yaxis_title='',
            legend=dict(**PLT['legend'], orientation='h', y=1.04, x=0)))
        st.plotly_chart(fig, use_container_width=True)

    # Per-season goals trend
    st.markdown('<div class="sh">Goals & Assists Trend — by Season</div>', unsafe_allow_html=True)
    apps_copy = apps.copy()
    apps_copy['date'] = pd.to_datetime(apps_copy['date'], errors='coerce')
    apps_copy['season'] = apps_copy['date'].dt.year.astype(str)
    season_agg = apps_copy[apps_copy['player_id'].isin(df['player_id'])].groupby('season').agg(
        goals=('goals','sum'), assists=('assists','sum'), appearances=('appearance_id','count')
    ).reset_index()
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=season_agg.season, y=season_agg.goals, name='Goals',
        line=dict(color='#e84393',width=2.5), mode='lines+markers', marker_size=8))
    fig_s.add_trace(go.Scatter(x=season_agg.season, y=season_agg.assists, name='Assists',
        line=dict(color='#00d4aa',width=2.5), mode='lines+markers', marker_size=8))
    fig_s.update_layout(**plt_override(
    height=320,
    title='Total G+A per Season (filtered players)',
    xaxis_title='Season', yaxis_title='Total'))
    st.plotly_chart(fig_s, use_container_width=True)

# ╔════════════════════════════════════════════════════════╗
# ║  TAB 3 — MARKET VALUE & TRANSFERS                     ║
# ╚════════════════════════════════════════════════════════╝
with t3:
    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="sh">Market Value by League — Box</div>', unsafe_allow_html=True)
        fig = px.box(df, x='league_name', y='market_value_m',
            color='league_name',
            color_discrete_sequence=['#00ff85','#f4a020','#e8c22a','#008fd7','#d5283a'],
            points='outliers')
        fig.update_layout(**PLT, height=360, showlegend=False,
            xaxis_title='', yaxis_title='Market Value (€M)')
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        st.markdown('<div class="sh">Market Value vs Performance — Scatter</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='g_plus_a', y='market_value_m',
            color='cluster_name', color_discrete_map=c_colors,
            size='games_played', size_max=20,
            hover_data=['name','position','league_name'],
            trendline='ols', trendline_color_override='#f7c844', opacity=.78)
        fig.update_layout(**PLT, height=360,
            xaxis_title='Goals + Assists', yaxis_title='Market Value (€M)')
        st.plotly_chart(fig, use_container_width=True)

    # Market value over time (valuations)
    st.markdown('<div class="sh">Market Value Trend — player_valuations.csv</div>', unsafe_allow_html=True)
    vals_c = vals.copy()
    vals_c['date'] = pd.to_datetime(vals_c['date'], errors='coerce')

    # Pick top 8 by current value among filtered players
    top_pids = df.nlargest(8,'market_value_m')['player_id'].tolist()
    top_names = dict(zip(df['player_id'], df['name']))
    vals_top = vals_c[vals_c['player_id'].isin(top_pids)].copy()
    vals_top['player_name'] = vals_top['player_id'].map(top_names)
    vals_top['value_m'] = vals_top['market_value_in_eur']/1e6

    fig_vt = px.line(vals_top, x='date', y='value_m', color='player_name',
        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_vt.update_layout(**plt_override(
    height=380,
    title='Market Value Timeline — Top 8 Players',
    xaxis_title='Date', yaxis_title='Value (€M)'))
    st.plotly_chart(fig_vt, use_container_width=True)

    tr1, tr2 = st.columns(2)
    with tr1:
        st.markdown('<div class="sh">Top Clubs by Transfer Spend</div>', unsafe_allow_html=True)
        top_spend = club_spend.nlargest(15,'total_spend').copy()
        top_spend['spend_m'] = (top_spend['total_spend']/1e6).round(1)
        fig = px.bar(top_spend, x='spend_m', y='club', orientation='h',
            color='spend_m', color_continuous_scale='teal',
            text='spend_m', labels={'spend_m':'€M'})
        fig.update_traces(texttemplate='€%{text:.0f}M', textposition='outside', marker_line_width=0)
        fig.update_layout(**PLT, height=440, showlegend=False,
            xaxis_title='Total Transfer Spend (€M)', yaxis_title='',
            coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tr2:
        st.markdown('<div class="sh">Transfer Fee vs Market Value — transfers.csv</div>', unsafe_allow_html=True)
        tr_plot = transfers[transfers['transfer_fee']>0].copy()
        tr_plot['fee_m'] = tr_plot['transfer_fee']/1e6
        tr_plot['mv_m']  = tr_plot['market_value_in_eur']/1e6
        fig = px.scatter(tr_plot.sample(min(500,len(tr_plot))), x='mv_m', y='fee_m',
            opacity=0.6, color_discrete_sequence=['#f7c844'],
            trendline='ols', trendline_color_override='#e84393',
            hover_data=['from_club_name','to_club_name','transfer_season'],
            labels={'mv_m':'Market Value (€M)','fee_m':'Transfer Fee (€M)'})
        fig.update_layout(**plt_override(
            height=440,
            title='Fee Paid vs Market Value at Transfer'))
        st.plotly_chart(fig, use_container_width=True)

# ╔════════════════════════════════════════════════════════╗
# ║  TAB 4 — HEATMAPS (Seaborn)                           ║
# ╚════════════════════════════════════════════════════════╝
with t4:
    st.markdown('<div class="sh">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    hm_cols = ['total_goals','total_assists','avg_goals_pg','avg_assists_pg',
               'total_minutes','avg_minutes','total_yellow','total_red',
               'market_value_m','age','height_in_cm','international_caps','international_goals']
    corr = df[hm_cols].corr()
    fig_h, ax = plt.subplots(figsize=(13,8))
    fig_h.patch.set_facecolor('#070d12')
    ax.set_facecolor('#070d12')
    mask = np.triu(np.ones_like(corr,dtype=bool))
    cmap = sns.diverging_palette(175, 330, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=0,
                vmin=-1, vmax=1, linewidths=.5, linecolor='#070d12',
                annot_kws={'size':8,'weight':'bold','color':'white'}, ax=ax,
                cbar_kws={'shrink':.75})
    ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right',color='#00d4aa',fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0,color='#00d4aa',fontsize=9)
    ax.set_title('Kaggle Columns Correlation Matrix', color='#00d4aa', fontsize=15, fontweight='bold', pad=14)
    plt.tight_layout(); st.pyplot(fig_h, use_container_width=True); plt.close()

    h1, h2 = st.columns(2)
    with h1:
        st.markdown('<div class="sh">Position × Cluster Heatmap</div>', unsafe_allow_html=True)
        piv = df.groupby(['position','cluster_name']).size().unstack(fill_value=0)
        fig_ph, ax2 = plt.subplots(figsize=(10,5))
        fig_ph.patch.set_facecolor('#070d12')
        ax2.set_facecolor('#070d12')
        sns.heatmap(piv, annot=True, fmt='d', cmap='YlGnBu', linewidths=.4,
                    linecolor='#070d12', annot_kws={'size':9,'weight':'bold'},ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(),rotation=30,ha='right',color='#00d4aa',fontsize=9)
        ax2.set_yticklabels(ax2.get_yticklabels(),rotation=0,color='#b8d4e8',fontsize=9)
        ax2.set_title('Position × Playing Style Cluster', color='#00d4aa', fontsize=13, fontweight='bold')
        ax2.set_xlabel(''); ax2.set_ylabel('')
        plt.tight_layout(); st.pyplot(fig_ph, use_container_width=True); plt.close()

    with h2:
        st.markdown('<div class="sh">League × Cluster Heatmap</div>', unsafe_allow_html=True)
        piv2 = df.groupby(['league_name','cluster_name']).size().unstack(fill_value=0)
        fig_lh, ax3 = plt.subplots(figsize=(10,4))
        fig_lh.patch.set_facecolor('#070d12')
        ax3.set_facecolor('#070d12')
        sns.heatmap(piv2, annot=True, fmt='d', cmap='plasma', linewidths=.4,
                    linecolor='#070d12', annot_kws={'size':10,'weight':'bold','color':'white'}, ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(),rotation=30,ha='right',color='#00d4aa',fontsize=9)
        ax3.set_yticklabels(ax3.get_yticklabels(),rotation=0,color='#b8d4e8',fontsize=10)
        ax3.set_title('League × Playing Style Cluster', color='#00d4aa', fontsize=13, fontweight='bold')
        ax3.set_xlabel(''); ax3.set_ylabel('')
        plt.tight_layout(); st.pyplot(fig_lh, use_container_width=True); plt.close()

    # Cluster mean stats heatmap
    st.markdown('<div class="sh">Cluster Mean Stats (Seaborn)</div>', unsafe_allow_html=True)
    stat_c = ['avg_goals_pg','avg_assists_pg','avg_minutes','market_value_m','total_yellow','age','international_caps']
    stat_l = ['Goals/G','Assists/G','Avg Min','Value €M','Yellow Cds','Age','Intl Caps']
    cstat = df.groupby('cluster_name')[stat_c].mean().round(2)
    cstat_n = (cstat-cstat.min())/(cstat.max()-cstat.min()+1e-9)
    cstat_n.columns = stat_l
    cstat.columns = stat_l
    fig_cs, ax4 = plt.subplots(figsize=(13,4))
    fig_cs.patch.set_facecolor('#070d12')
    ax4.set_facecolor('#070d12')
    sns.heatmap(cstat_n, annot=cstat, fmt='.2f', cmap='magma', linewidths=.4,
                linecolor='#070d12', annot_kws={'size':9,'weight':'bold','color':'white'}, ax=ax4)
    ax4.set_xticklabels(ax4.get_xticklabels(),rotation=0,color='#00d4aa',fontsize=9)
    ax4.set_yticklabels(ax4.get_yticklabels(),rotation=0,color='#b8d4e8',fontsize=9)
    ax4.set_title('Normalized Cluster Profiles (raw values shown)', color='#00d4aa', fontsize=14, fontweight='bold', pad=12)
    ax4.set_xlabel(''); ax4.set_ylabel('')
    plt.tight_layout(); st.pyplot(fig_cs, use_container_width=True); plt.close()

# ╔════════════════════════════════════════════════════════╗
# ║  TAB 5 — DISTRIBUTIONS                                ║
# ╚════════════════════════════════════════════════════════╝
with t5:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="sh">Age Distribution by League</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='age', color='league_name', nbins=18,
            barmode='overlay', opacity=.75,
            color_discrete_sequence=['#00ff85','#f4a020','#e8c22a','#008fd7','#d5283a'])
        fig.update_layout(**PLT, height=320, xaxis_title='Age', yaxis_title='Players')
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        st.markdown('<div class="sh">Market Value Distribution (log)</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='market_value_m', color='sub_position', nbins=35,
            barmode='overlay', opacity=.7, log_y=True,
            color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(**PLT, height=320, xaxis_title='Market Value (€M)', yaxis_title='Count (log)')
        st.plotly_chart(fig, use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        st.markdown('<div class="sh">Height Distribution by Sub-Position</div>', unsafe_allow_html=True)
        fig = px.box(df, x='sub_position', y='height_in_cm',
            color='sub_position', color_discrete_sequence=px.colors.qualitative.Safe,
            points='outliers')
        fig.update_layout(**PLT, height=340, showlegend=False,
            xaxis_title='', yaxis_title='Height (cm)')
        st.plotly_chart(fig, use_container_width=True)

    with d4:
        st.markdown('<div class="sh">International Caps by Nationality (Top 12)</div>', unsafe_allow_html=True)
        nat = df.groupby('country_of_citizenship')['international_caps'].mean().nlargest(12).reset_index()
        fig = px.bar(nat, x='country_of_citizenship', y='international_caps',
            color='international_caps', color_continuous_scale='teal')
        fig.update_layout(**plt_override(
            height=340, showlegend=False,
            xaxis_title='', yaxis_title='Avg Intl Caps',
            coloraxis_showscale=False,
            xaxis=dict(**PLT['xaxis'], tickangle=-30)))

        st.plotly_chart(fig, use_container_width=True)

    # Sunburst — League → Sub-Position → Cluster
    st.markdown('<div class="sh">League → Sub-Position → Style Cluster — Sunburst</div>', unsafe_allow_html=True)
    # fig_sun = px.sunburst(df, path=['league_name','sub_position','cluster_name'],
    #     color_discrete_sequence=px.colors.qualitative.Dark24, maxdepth=3)
    df_sun = df.copy()
    df_sun['league_name']  = df_sun['league_name'].fillna('Unknown League')
    df_sun['sub_position'] = df_sun['sub_position'].fillna('Unknown Position')
    df_sun['cluster_name'] = df_sun['cluster_name'].fillna('Unknown Style')
    fig_sun = px.sunburst(df_sun, path=['league_name','sub_position','cluster_name'],
                          
    color_discrete_sequence=px.colors.qualitative.Dark24, maxdepth=3)
    fig_sun.update_layout(**PLT, height=520)
    fig_sun.update_traces(insidetextorientation='radial')
    st.plotly_chart(fig_sun, use_container_width=True)

    # Transfer fee distribution
    st.markdown('<div class="sh">Transfer Fee Distribution — transfers.csv</div>', unsafe_allow_html=True)
    tr_nonzero = transfers[transfers['transfer_fee']>500_000].copy()
    tr_nonzero['fee_m'] = tr_nonzero['transfer_fee']/1e6
    fig_tf = px.histogram(tr_nonzero, x='fee_m', nbins=50,
        color='transfer_season', barmode='overlay', opacity=.7,
        color_discrete_sequence=px.colors.qualitative.Antique)
    fig_tf.update_layout(**PLT, height=320,
        xaxis_title='Transfer Fee (€M)', yaxis_title='Count', legend_title='Season')
    st.plotly_chart(fig_tf, use_container_width=True)

# ╔════════════════════════════════════════════════════════╗
# ║  TAB 6 — SCOUT & COMPARE                              ║
# ╚════════════════════════════════════════════════════════╝
with t6:
    st.markdown('<div class="sh">🔍 Player Scouting Table</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns([2,1,1])
    with s1:
        q = st.text_input("Search name / club", placeholder="Haaland, PSG, Goalkeeper…")
    with s2:
        sort_c = st.selectbox("Sort by", ['market_value_m','total_goals','total_assists',
                                          'avg_goals_pg','avg_assists_pg','games_played','age'])
    with s3:
        top_n = st.slider("Show top N", 10, 100, 25)

    res = df.copy()
    if q:
        q_low = q.lower()
        res = res[res.apply(lambda r: q_low in str(r['name']).lower() or
                                       q_low in str(r['current_club_name']).lower() or
                                       q_low in str(r['position']).lower(), axis=1)]

    disp_cols = ['name','age','position','current_club_name','league_name','cluster_name',
                 'total_goals','total_assists','games_played','avg_goals_pg','avg_assists_pg',
                 'market_value_m','international_caps']
    table = res.nlargest(top_n, sort_c)[disp_cols].copy()
    table.columns = ['Name','Age','Position','Club','League','Style','Goals','Assists',
                     'Games','G/G','A/G','Value(€M)','Intl Caps']
    st.dataframe(
        table.style
            .background_gradient(cmap='YlGnBu', subset=['Goals','Assists','G/G','Value(€M)'])
            .format({'G/G':'{:.2f}','A/G':'{:.2f}','Value(€M)':'{:.1f}'}),
        use_container_width=True, height=400
    )

    # ── Comparison radar ──────────────────────────────────────────────────
    st.markdown('<div class="sh">Player Comparison</div>', unsafe_allow_html=True)
    sel = st.multiselect("Select up to 5 players", options=res['name'].tolist(), max_selections=5)

    if len(sel) >= 2:
        comp_f = ['total_goals','total_assists','avg_goals_pg','avg_assists_pg','avg_minutes','market_value_m','international_caps']
        comp_l = ['Goals','Assists','G/Game','A/Game','Avg Min','Value €M','Intl Caps']
        comp_df = df[df.name.isin(sel)].drop_duplicates('name')
        norms = comp_df[comp_f].copy()
        for c in comp_f:
            mn,mx = df[c].min(),df[c].max()
            norms[c] = (norms[c]-mn)/(mx-mn+1e-9)

        pal = ['#e84393','#00d4aa','#f7c844','#a8e06a','#c084fc']
        fig_cmp = go.Figure()
        for idx,(_, row) in enumerate(comp_df.iterrows()):
            vals = norms.loc[row.name, comp_f].tolist(); vals.append(vals[0])
            fig_cmp.add_trace(go.Scatterpolar(r=vals, theta=comp_l+[comp_l[0]],
                name=row['name'], fill='toself',
                fillcolor=hex_to_rgba(pal[idx%5], 0.16),
                line=dict(color=pal[idx%5], width=2.5)))
            
        fig_cmp.update_layout(**plt_override(
            height=450,
            polar=dict(bgcolor='rgba(7,13,18,.9)',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='#1c3047',
                                tickfont=dict(size=8, color='#3a6e8a')),
                angularaxis=dict(gridcolor='#1c3047', tickfont=dict(size=12, color='#b8d4e8'))),
            title='Player Comparison Radar (Normalized to dataset range)'))
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Side-by-side bar comparison
        st.markdown('<div class="sh">Percentile Comparison</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        for idx, nm in enumerate(sel):
            row = df[df.name==nm].iloc[0]
            raw = [row[c] for c in comp_f]
            pct = [(row[c]-df[c].min())/(df[c].max()-df[c].min()+1e-9)*100 for c in comp_f]
            fig_bar.add_trace(go.Bar(name=nm, x=comp_l, y=[round(p,1) for p in pct],
                marker_color=pal[idx%5],
                text=[f"{round(v,1)}" for v in raw],
                textposition='outside', textfont_size=9))
        fig_bar.update_layout(**PLT, barmode='group', height=380,
            yaxis_title='Percentile Score (vs all players)', xaxis_title='')
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""<div style='text-align:center;padding:30px 0 8px;color:#1c3047;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase'>
⚽ Transfermarkt Dashboard · players.csv · appearances.csv · player_valuations.csv · transfers.csv · clubs.csv · competitions.csv
· Built with Streamlit · Pandas · Plotly · Seaborn · Scikit-learn KMeans+PCA
</div>""", unsafe_allow_html=True)