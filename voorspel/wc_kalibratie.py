"""Kalibratie op internationale topwedstrijden (vooral World Cups): races met >= 10 landen en >= 2 van USA/CAN/JPN/CHN/KOR (dus geen EK),
november t/m maart, geen WK/Spelen (piek.py). Tegenover de overige ritten van dezelfde niveaus. Ook: het selectie-effect van 'rijders die
dat seizoen een WK of de Spelen reden' (kwalificatie hangt af van de World Cups ervoor).
Gebruik: python wc_kalibratie.py <werkmap> <uitkomst.parquet>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk

w, pad = sys.argv[1], sys.argv[2]
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'gender', 'afstand_idx'])
R['afstand'] = R['afstand'].astype(str)
T = pk.toernooien(R); piek_eids = set(T['eid']); wk_os = set(T.loc[T['soort'].isin(['Spelen', 'WK afstanden']), 'eid'])
x = R[R['afstand'].isin(pk.ZES)].copy(); x['eid'] = x['event_id'] % 1000000; x['maand'] = pd.to_datetime(x['dag'], unit='D').dt.month
E = x.groupby('eid').agg(landen=('land', 'nunique'), buiten=('land', lambda s: len(set(s) & pk.BUITEN_EU)), maand=('maand', 'first'))
int_eids = set(E[(E['landen'] >= 10) & (E['buiten'] >= 2) & E['maand'].isin([11, 12, 1, 2, 3])].index) - piek_eids
V = pq.read_table(pad, columns=['skater_id', 'dag', 'jaar', 'seizoen', 'afstand', 'tijd', 'pit', 'ok', 'k_val', 'k_c']).to_pandas(); V['afstand'] = V['afstand'].astype(str)
V = V.merge(R[['skater_id', 'dag', 'afstand', 'tijd', 'event_id']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
V = V[V['ok'] & np.isfinite(V['pit']) & V['afstand'].isin(list(pk.ZES)) & (V['jaar'] >= 2008)].copy()
V['eid'] = V['event_id'].fillna(-1).astype(np.int64) % 1000000
V['soort'] = np.where(V['eid'].isin(wk_os), 'WK/Spelen', np.where(V['eid'].isin(piek_eids), 'WK allround/sprint', np.where(V['eid'].isin(int_eids), 'internationaal (WC)', 'overig')))
V['z'] = norm.ppf(V['pit'].clip(1e-12, 1 - 1e-12)).clip(-4, 4)
dl = V.loc[V['soort'] == 'WK/Spelen', ['skater_id', 'seizoen']].drop_duplicates().assign(kw=True)
V = V.merge(dl, on=['skater_id', 'seizoen'], how='left'); V['kw'] = V['kw'].fillna(False).astype(bool)


def k(x):
    y = x[~x['k_val']]
    return pd.Series(dict(ritten=len(x), gem_z=y['z'].mean(), sd_z=y['z'].std(), snel20=(y['pit'] <= 0.05).mean() / 0.05, snel100=(y['pit'] <= 0.01).mean() / 0.01,
                          traag20=(y['pit'] >= 0.95).mean() / 0.05, traag100=(y['pit'] >= 0.99).mean() / 0.01))


pd.set_option('display.width', 220)
V['niv'] = pd.cut(V['k_c'], [-1, 0.02, 0.035, 0.06, 0.1, 9], labels=['top <2%', '2-3,5%', '3,5-6%', '6-10%', '>10%'])
for per, s in (('2008-2019', V['jaar'] <= 2019), ('2020+', V['jaar'] >= 2020)):
    print(f'\n== {per}: per soort wedstrijd en niveau vooraf (zonder valpartijen)')
    print(V[s].groupby(['soort', 'niv'], observed=True).apply(k).round(3).to_string())
    x = V[s & (V['soort'] == 'internationaal (WC)') & (V['k_c'] < 0.035)]
    print(f'\n   selectie-effect, internationaal, top < 3,5%:')
    print(x.groupby('kw').apply(k).rename(index={True: 'rijdt dat seizoen WK/Spelen', False: 'rijdt dat seizoen geen WK/Spelen'}).round(3).to_string())

# ontleding: ligt het aan de hele race (ijs / omstandigheden) of aan de rijders binnen de race?
rk = ['dag', 'event_id', 'afstand']
for per, s in (('2008-2019', V['jaar'] <= 2019), ('2020+', V['jaar'] >= 2020)):
    for soort in ('internationaal (WC)', 'overig'):
        y = V[s & (V['soort'] == soort) & ~V['k_val']]
        g = y.groupby(rk)['z']; n = g.transform('size'); y = y[n >= 5]; g = y.groupby(rk)['z']
        rm = g.mean(); rn = g.size(); binnen = (y['z'] - g.transform('mean')).std()
        tussen = np.var(rm) - np.mean(1 / rn)          # extra variantie van racegemiddelden boven wat toeval (1/n) geeft
        print(f'{per} {soort:20s} races {len(rm):6d} | gem racegemiddelde {rm.mean():+.3f} | extra variantie tussen races {tussen:.3f} (sd {np.sqrt(max(tussen, 0)):.3f}) | sd binnen race {binnen:.3f}')
