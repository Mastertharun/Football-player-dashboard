"""
generate_data.py
────────────────
Generates a Transfermarkt-style player dataset and runs K-Means clustering.
Run once before launching the dashboard:
    python generate_data.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)
n = 350

positions = ['GK','CB','LB','RB','CDM','CM','CAM','LW','RW','ST','CF']
pos_weights = [0.07,0.10,0.07,0.07,0.08,0.12,0.10,0.08,0.08,0.13,0.10]
leagues = ['Premier League','La Liga','Bundesliga','Serie A','Ligue 1']
clubs_by_league = {
    'Premier League':['Man City','Arsenal','Liverpool','Chelsea','Tottenham','Man United','Newcastle','Aston Villa'],
    'La Liga':['Real Madrid','Barcelona','Atletico Madrid','Real Sociedad','Athletic Bilbao','Villarreal'],
    'Bundesliga':['Bayern Munich','Dortmund','Leverkusen','RB Leipzig','Union Berlin','Freiburg'],
    'Serie A':['Napoli','Inter','Lazio','Roma','AC Milan','Juventus','Atalanta'],
    'Ligue 1':['PSG','Marseille','Lyon','Monaco','Lens','Rennes','Lille'],
}
nationalities = ['England','Spain','Germany','France','Italy','Brazil','Argentina',
                 'Portugal','Netherlands','Belgium','Senegal','Morocco','Nigeria','Croatia','Uruguay']
first_names = ['Liam','Mateo','Kai','Paulo','Vinicius','Erling','Kylian','Jude',
               'Pedri','Gavi','Fede','Antoine','Marcus','Harry','Virgil','Kevin',
               'Phil','Bukayo','Gabriel','Darwin','Romelu','Trent','Mason','Jack']
last_names = ['Smith','Rodriguez','Müller','Fernandez','Da Silva','Bellingham',
              'Valverde','Saka','Jesus','Nuñez','Kane','Rashford','Griezmann',
              'de Bruyne','Foden','Mount','Salah','Mané','Hernandez','Torres']

names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]
ages = np.random.randint(18, 36, n)
positions_arr = np.random.choice(positions, n, p=pos_weights)
leagues_arr = np.random.choice(leagues, n)
clubs_arr = [np.random.choice(clubs_by_league[l]) for l in leagues_arr]
nationalities_arr = np.random.choice(nationalities, n)

all_goals, all_assists = [], []
for pos in positions_arr:
    if pos == 'GK':
        all_goals.append(0); all_assists.append(int(np.random.poisson(0.3)))
    elif pos in ['CB','LB','RB']:
        all_goals.append(int(np.random.poisson(1.2))); all_assists.append(int(np.random.poisson(2.5)))
    elif pos in ['CDM','CM']:
        all_goals.append(int(np.random.poisson(3))); all_assists.append(int(np.random.poisson(4)))
    elif pos in ['CAM','LW','RW']:
        all_goals.append(int(np.random.poisson(9))); all_assists.append(int(np.random.poisson(8)))
    else:
        all_goals.append(int(np.random.poisson(17))); all_assists.append(int(np.random.poisson(5)))

games_played = np.random.randint(12, 38, n)
minutes_played = games_played * np.random.randint(55, 95, n)
pass_accuracy = np.clip(np.random.normal(82, 7, n), 55, 97).round(1)
dribbles = np.random.poisson(1.8, n).astype(float)
tackles = np.random.poisson(2.1, n).astype(float)
shots = np.random.poisson(1.5, n).astype(float)
key_passes = np.random.poisson(1.8, n).astype(float)
aerial = np.random.poisson(2.0, n).astype(float)

for i, pos in enumerate(positions_arr):
    if pos in ['CB','CDM']:
        dribbles[i] *= 0.5; tackles[i] *= 1.8; aerial[i] *= 1.6
    elif pos in ['LW','RW','CAM']:
        dribbles[i] *= 2.2; shots[i] *= 1.5
    elif pos == 'GK':
        dribbles[i] = 0; shots[i] = 0

df = pd.DataFrame({
    'name': names, 'age': ages, 'position': positions_arr,
    'league': leagues_arr, 'club': clubs_arr, 'nationality': nationalities_arr,
    'games_played': games_played, 'minutes_played': minutes_played,
    'goals': all_goals, 'assists': all_assists,
    'pass_accuracy': pass_accuracy, 'dribbles_completed': dribbles.round(1),
    'tackles_won': tackles.round(1), 'shots_on_target': shots.round(1),
    'key_passes': key_passes.round(1), 'aerial_duels_won': aerial.round(1),
    'yellow_cards': np.random.poisson(2.1, n).astype(int),
    'red_cards': np.random.poisson(0.15, n).astype(int),
    'market_value_m': np.clip(np.random.lognormal(3.0, 1.1, n), 0.5, 200).round(1),
    'avg_rating': np.clip(np.random.normal(7.0, 0.6, n), 5.5, 9.5).round(2),
    'fouls_committed': np.random.poisson(1.8, n).astype(float).round(1),
    'interceptions': np.random.poisson(1.4, n).astype(float).round(1),
    'clean_sheets': np.where(positions_arr == 'GK', np.random.randint(3,20,n), np.random.randint(0,3,n)),
    'saves': np.where(positions_arr == 'GK', np.random.randint(40,130,n), 0),
})

# K-Means Clustering
features = ['goals','assists','pass_accuracy','dribbles_completed','tackles_won',
            'shots_on_target','key_passes','aerial_duels_won','interceptions']
scaler = StandardScaler()
X = scaler.fit_transform(df[features].fillna(0))

km = KMeans(n_clusters=6, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
df['pca_x'] = coords[:,0].round(4)
df['pca_y'] = coords[:,1].round(4)

style_map = {
    'GK':'Sweeper Keepers','CB':'Rock-Solid Defenders','LB':'Attacking Fullbacks',
    'RB':'Attacking Fullbacks','CDM':'Defensive Shields','CM':'Box-to-Box Engines',
    'CAM':'Creative Playmakers','LW':'Flying Wingers','RW':'Flying Wingers',
    'ST':'Clinical Strikers','CF':'False Nines'
}
cluster_style_names = ['Creative Playmakers','Clinical Strikers','Box-to-Box Engines',
                       'Defensive Shields','Flying Wingers','Sweeper Keepers & Defenders']
df['style'] = df['position'].map(style_map)
df['cluster_name'] = df['cluster'].map({i: cluster_style_names[i] for i in range(6)})

df.to_csv('data/players_clustered.csv', index=False)
print(f"✅ Generated {len(df)} players with clustering.")
print(df['cluster_name'].value_counts())
