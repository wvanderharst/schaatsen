"""'Topper geweest': ruimere maat dan een titel. Alleen eerdere informatie.
 ooit_gat: per seizoen de beste kwaliteit (q = ln t - theta, ijs achteraf) van de rijder op de afstand tegenover de beste ter wereld dat
           seizoen (geslacht x afstand), in %; het kleinste verschil over alle EERDERE seizoenen (0 = ooit de snelste ter wereld).
 podia:    aantal top-3-plaatsen op internationale wedstrijden (World Cups, WK's, Spelen) op welke afstand dan ook, vóór deze dag.
Uitkomst: rijderschok (ijs eruit), top < 3,5%, WK/Spelen tegenover World Cups. Gebruik: python topper_geweest.py <werkmap> <uitkomst>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk, rijderschok as rs
w, pad = sys.argv[1], sys.argv[2]
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'gender', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'verdacht', 'seizoen'])
R['afstand'] = R['afstand'].astype(str)
Tn = pk.toernooien(R); pe = set(Tn['eid']); wkos = set(Tn.loc[Tn['soort'].isin(['Spelen', 'WK afstanden']), 'eid']); ie = pk.internationaal(R, pe)
U = pq.read_table(pad, columns=['skater_id', 'dag', 'afstand', 'tijd', 'theta', 'ok', 'k_val']).to_pandas(); U['afstand'] = U['afstand'].astype(str)
U = U.merge(R[~R['verdacht']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'])
U = U[U['ok'] & ~U['k_val'] & np.isfinite(U['theta']) & U['afstand'].isin(list(pk.ZES))]
U['q'] = np.log(U['tijd']) - U['theta']
sb = U.groupby(['skater_id', 'gender', 'afstand', 'seizoen'])['q'].min().reset_index()
sb['gat'] = 100 * (np.exp(sb['q'] - sb.groupby(['gender', 'afstand', 'seizoen'])['q'].transform('min')) - 1)
sb = sb.sort_values('seizoen'); sb['ooit_gat'] = sb.groupby(['skater_id', 'afstand'])['gat'].transform(lambda s: s.cummin().shift(1))
# podia op internationale wedstrijden
I = R[~R['verdacht'] & np.isfinite(R['tijd']) & (R['event_id'] % 1000000).isin(wkos | pe | ie) & R['afstand'].isin(list(pk.ZES))].copy()
I['plaats'] = I.groupby(['event_id', 'dag', 'gender', 'afstand'])['tijd'].rank(method='min')
P = I[I['plaats'] <= 3][['skater_id', 'dag']].sort_values('dag')
V = rs.tabel(w, pad); V['eid'] = V['event_id'] % 1000000
V['soort'] = np.where(V['eid'].isin(wkos), 'WK/Spelen', np.where(V['eid'].isin(ie), 'World Cups', 'overig'))
V = V[~V['k_val'] & (V['k_c'] < 0.035) & V['jaar'].between(2008, 2026) & V['afstand'].isin(list(pk.ZES))].copy()
V = V.merge(R[['skater_id', 'dag', 'afstand', 'tijd', 'seizoen']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left', suffixes=('', '_r'))
V = V.merge(sb[['skater_id', 'afstand', 'seizoen', 'ooit_gat']], on=['skater_id', 'afstand', 'seizoen'], how='left')
pd_ = P.groupby('skater_id')['dag'].apply(np.sort).to_dict()
V['podia'] = [np.searchsorted(pd_.get(s, np.array([])), d) for s, d in zip(V['skater_id'], V['dag'])]
pd.set_option('display.width', 200)
def tab(kol, bins, labels):
    V['b'] = pd.cut(V[kol], bins, labels=labels)
    t = V[V['soort'] != 'overig'].groupby(['b', 'soort'], observed=True).agg(ritten=('zr', 'size'), gem_z=('zr', lambda s: s.clip(-4, 4).mean()),
                                                                              traag_1op20=('kans', lambda p: (p >= 0.95).mean() / 0.05)).round(3).unstack('soort')
    m = t['gem_z']; t[('piekeffect', '')] = (m['WK/Spelen'] - m['World Cups']).round(3)
    print(f'\n{kol}:'); print(t.to_string())
tab('ooit_gat', [-1, 0.001, 0.5, 1.0, 2.0, 100], ['ooit wereldbeste', '< 0,5%', '0,5-1%', '1-2%', '> 2%'])
tab('podia', [-1, 0, 2, 9, 29, 1e9], ['0', '1-2', '3-9', '10-29', '30+'])
V['geen_seizoen'] = V['ooit_gat'].isna(); print('\nzonder eerder seizoen op de afstand:', int(V['geen_seizoen'].sum()), 'ritten')
