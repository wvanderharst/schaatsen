"""De grootste positieve verrassingen op de Olympische Spelen, alleen de rijderschok (ijs achteraf via de veldmediaan; rijderschok.py).
Kolommen als verrassingen_spelen_rijder.csv, plus de plaats op de vorige lijst. Gebruik: python verrassingen_spelen.py <werkmap> <uitkomst> [N]"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk, rijderschok as rs
from kwaliteit import ts
w, pad = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
V = rs.tabel(w, pad)
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan'])
R['afstand'] = R['afstand'].astype(str)
Tn = pk.toernooien(R); sp = set(Tn.loc[Tn['soort'] == 'Spelen', 'eid'])
V = V[(V['event_id'] % 1000000).isin(sp) & V['afstand'].isin(list(pk.ZES))].copy()
rk = ['event_id', 'dag', 'gender', 'afstand']
V = V.drop_duplicates(rk + ['skater_id'])
V['plaats'] = V.groupby(rk)['tijd'].rank(method='min').astype(int)
V = V[~V['k_val']].sort_values('kans').head(N).reset_index(drop=True)
oud = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verrassingen_spelen_rijder.csv')
vorig = {}
if os.path.exists(oud):
    O = pd.read_csv(oud, dtype=str); vorig = {(r['rijder'], r['Spelen'], r['afstand']): r['#'] for _, r in O.iterrows()}
rij = []
for i, r in V.iterrows():
    af = r['afstand'].replace('m', ' m'); naam = r['skater_name']; jaar = str(int(r['jaar']))
    tv = np.exp(r['pred']); ti = np.exp(r['pred'] + r['ijs_achteraf'])
    rij.append({'#': i + 1, 'rijder': naam, 'Spelen': jaar, 'afstand': af, 'tijd': ts(r['tijd']), 'verwacht vooraf': ts(tv),
                'ijs achteraf': f"{100 * (np.exp(r['ijs_achteraf']) - 1):+.1f}%".replace('.', ','), 'verwacht met ijs achteraf': ts(ti),
                'sneller (snelheid)': f"{100 * (ti / r['tijd'] - 1):+.1f}%".replace('.', ','), 'kans': f"1 op {1 / r['kans']:,.0f}".replace(',', '.'),
                'plaats': int(r['plaats']), 'vorige lijst': vorig.get((naam, jaar, af), '-')})
T = pd.DataFrame(rij)
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verrassingen_spelen_rijder_nieuw.csv'), index=False)
print('| ' + ' | '.join(T.columns) + ' |'); print('|' + '---|' * len(T.columns))
for _, r in T.iterrows():
    print('| ' + ' | '.join(str(v) for v in r.values) + ' |')
