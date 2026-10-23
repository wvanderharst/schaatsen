"""Kalibratie op WK's en Spelen: de top 5 vooraf (snelste verwachte tijd per race) tegenover de rest van het veld (6-10, 11+).
(1) de modelkans (pit, alle lagen): gem. z, sd z, staarten en log loss;  (2) alleen de rijder (ijs achteraf: residu min de veldmediaan van die
race, gedeeld door sd_rijder): gem. en sd.  Valpartijen tellen niet mee in de kalibratie (wel als percentage getoond).
Gebruik: python top5_piek.py <werkmap> <uitkomst>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk
w, pad = sys.argv[1], sys.argv[2]
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'gender', 'verdacht', 'jaar'])
R['afstand'] = R['afstand'].astype(str)
Tn = pk.toernooien(R); soort = dict(zip(Tn['eid'], Tn['soort']))
V = pq.read_table(pad, columns=['skater_id', 'dag', 'afstand', 'tijd', 'pred', 'sd_rijder', 'pit', 'nll', 'ok', 'k_val']).to_pandas(); V['afstand'] = V['afstand'].astype(str)
V = V.merge(R[~R['verdacht']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'])
V['eid'] = V['event_id'] % 1000000; V = V[V['eid'].isin(soort) & V['afstand'].isin(list(pk.ZES)) & V['ok'] & np.isfinite(V['pred']) & np.isfinite(V['pit'])].copy()
V['soort'] = V['eid'].map(soort)
rk = ['event_id', 'dag', 'gender', 'afstand']
V = V.drop_duplicates(rk + ['skater_id'])
V['n'] = V.groupby(rk)['pred'].transform('size'); V = V[V['n'] >= 10].copy()
V['vrang'] = V.groupby(rk)['pred'].rank(method='first')
V['groep'] = np.where(V['vrang'] <= 5, 'top 5', np.where(V['vrang'] <= 10, '6-10', '11+'))
V['res'] = np.log(V['tijd']) - V['pred']
V['ijs'] = V[~V['k_val']].groupby(rk)['res'].transform('median'); V['ijs'] = V.groupby(rk)['ijs'].transform('max')
V['zr'] = (V['res'] - V['ijs']) / V['sd_rijder']
V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9)).clip(-4, 4)
print(f"{V.groupby(rk).ngroups} races, {len(V)} ritten op WK's en Spelen")


def regel(x, nm):
    xv = x[~x['k_val']]; p = xv['pit']; z = xv['z']; zr = xv['zr'].clip(-4, 4)
    return (f"  {nm:8s} n {len(x):5d} | val {100 * x['k_val'].mean():4.1f}% | modelkans: gem z {z.mean():+.3f} (se {z.std() / np.sqrt(len(z)):.3f}) sd {z.std():.3f}"
            f" | snel 1/20 {(p <= .05).mean() / .05:.2f} 1/100 {(p <= .01).mean() / .01:.2f} | traag 1/20 {(p >= .95).mean() / .05:.2f} 1/100 {(p >= .99).mean() / .01:.2f}"
            f" | log loss {xv['nll'].mean():.3f} | alleen rijder: gem {zr.mean():+.3f} (se {zr.std() / np.sqrt(len(zr)):.3f}) sd {zr.std():.3f}")


for per, js in (('2008-2019', (2008, 2019)), ('2020+', (2020, 2030)), ('alle jaren', (2008, 2030))):
    X = V[V['jaar'].between(*js)]
    print(f'\n{per}:')
    for gr in ('top 5', '6-10', '11+'):
        print(regel(X[X['groep'] == gr], gr))
X = V[V['jaar'] >= 2008]
print('\nper soort toernooi (alle jaren), top 5 tegen 6+:')
for s, x in X.groupby('soort'):
    print(f' {s}'); print(regel(x[x['groep'] == 'top 5'], 'top 5')); print(regel(x[x['groep'] != 'top 5'], '6+'))
print('\ntop 5 per voorspelde plaats (alle jaren):')
for r_ in range(1, 6):
    print(regel(X[X['vrang'] == r_], f'nr {r_}'))
