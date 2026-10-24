"""Voegt de categoriecode (Categorie, bijv. MC2, LN3, MSA, M45) toe aan ritten_theta.parquet en schat per rijder het geboortejaar.
Leeftijd op 1 juli van het seizoen per code: pupillen YF..YA = 7..12, junior C1/C2 = 13/14, B1/B2 = 15/16, A1/A2 = 17/18,
neo N1..N4 = 19..22 (op een jaar nauwkeurig), senior SA = 23..29 en masters 30, 35, ... (band van 5 jaar).
Geboortejaar = mediaan van seizoen - leeftijd over de nauwkeurige codes; anders het midden van het mogelijke bereik uit de banden.
Schrijft ritten_cat.parquet. Gebruik: python prep_cat.py <map met de csv en ritten_theta.parquet>"""
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/schaatsen/schaatsmodel')
import numpy as np, pandas as pd, do_analysis as da
w = sys.argv[1] if len(sys.argv) > 1 else '.'
df = da.laad_dataset(os.path.join(w, 'speedskating_historie_racedatum.csv'), verbose=False)
df = df.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd'])[['skater_id', 'dag', 'afstand', 'tijd', 'categorie']]
df['afstand'] = df['afstand'].astype(str); df['categorie'] = df['categorie'].astype(str)
R = pd.read_parquet(os.path.join(w, 'ritten_theta.parquet')); R['afstand'] = R['afstand'].astype(str)
R = R.merge(df, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
code = R['categorie'].fillna('?').str[1:]
EXACT = {'YF': 7, 'YE': 8, 'YD': 9, 'YC': 10, 'YB': 11, 'YA': 12, 'C1': 13, 'C2': 14, 'B1': 15, 'B2': 16, 'A1': 17, 'A2': 18,
         'N1': 19, 'N2': 20, 'N3': 21, 'N4': 22}
lo = code.map({**EXACT, 'SA': 23, **{str(a): a for a in range(30, 95, 5)}}).astype(float)
hi = code.map({**EXACT, 'SA': 29, **{str(a): a + 4 for a in range(30, 95, 5)}}).astype(float)
s = R['seizoen'].astype(float)
R['b_lo'] = s - hi; R['b_hi'] = s - lo                               # geboortejaar ligt in [b_lo, b_hi]
ex = code.isin(list(EXACT))
b_ex = (s - lo)[ex].groupby(R.loc[ex, 'skater_id']).median()
b_rng = R.groupby('skater_id').agg(blo=('b_lo', 'max'), bhi=('b_hi', 'min'))      # doorsnede van alle banden
b_mid = ((b_rng['blo'] + b_rng['bhi']) / 2).where(b_rng['blo'] <= b_rng['bhi'], (b_rng['blo'] + b_rng['bhi']) / 2)
geb = b_ex.reindex(b_rng.index).fillna(b_mid)
R['geboren'] = R['skater_id'].map(geb)
R['leeftijd'] = R['seizoen'] - R['geboren']
R['leeftijd_exact'] = R['skater_id'].isin(b_ex.index)
R = R.drop(columns=['b_lo', 'b_hi'])
R.to_parquet(os.path.join(w, 'ritten_cat.parquet'))
print(len(R), 'ritten; categorie gekoppeld', round(100 * R['categorie'].notna().mean(), 2), '%; leeftijd bekend',
      round(100 * R['leeftijd'].notna().mean(), 2), '% (exact', round(100 * R['leeftijd_exact'].mean(), 1), '%)')
print(R.groupby(R['categorie'].str[1:].fillna('?'))['leeftijd'].median().to_dict())
