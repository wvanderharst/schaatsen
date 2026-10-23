"""Pieken rijders met een eerder hoger Kalman-niveau makkelijker? Per rit: gat = huidig Kalman-niveau (ex ante, zh) min het beste (snelste)
Kalman-niveau van de rijder op die afstand in de 3 jaar ervoor (alleen eerdere dagen), in %. Groter gat = was eerder duidelijk beter.
Uitkomst: de rijderschok (rijderschok.py, ijs achteraf eruit) van het huidige model, apart voor WK/Spelen, World Cups en overig.
Gebruik: python eerder_niveau.py <werkmap> <uitkomst.parquet>"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalman as kb, kalman_mv2 as km2, piek as pk, rijderschok as rs, plafond as pf

w, pad = sys.argv[1], sys.argv[2]
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'))
df, inv = kb.voorbereid(R); del R
o = km2.pass_(inv, json.load(open(os.path.join(w, 'kalman_par_mv2b.json'))))
D = df[['skater_id', 'dag', 'afstand', 'tijd', 'event_id', 'land', 'baan', 'jaar']].copy(); D['afstand'] = D['afstand'].astype(str)
D['zh'] = o['zh']
# beste eerdere niveau: _tweede met k=1 op q = zh (laagste = snelste); via plafond.gat met tijd = exp(zh), theta = 0, verwacht_q = zh
D['gat'] = -pf.gat(D['skater_id'].to_numpy(), D['afstand'].to_numpy(), D['dag'].to_numpy(), np.exp(D['zh'].to_numpy()), np.zeros(len(D)),
                   D['zh'].to_numpy(), np.isfinite(D['zh'].to_numpy()), W=1095.0, k=1)       # > 0: eerder sneller dan nu (in %)
pe = set(pk.toernooien(df)['eid']); Tn = pk.toernooien(df); wkos = set(Tn.loc[Tn['soort'].isin(['Spelen', 'WK afstanden']), 'eid'])
ie = pk.internationaal(df, pe)
V = rs.tabel(w, pad)
V = V.merge(D[['skater_id', 'dag', 'afstand', 'tijd', 'gat']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'])
V['eid'] = V['event_id'] % 1000000
V['soort'] = np.where(V['eid'].isin(wkos), 'WK/Spelen', np.where(V['eid'].isin(ie), 'World Cups', 'overig'))
V = V[~V['k_val'] & (V['k_c'] < 0.035) & np.isfinite(V['gat']) & V['jaar'].between(2008, 2026)]
V['gb'] = pd.cut(V['gat'], [-50, 0.25, 0.75, 1.5, 3, 50], labels=['< 0,25%', '0,25-0,75%', '0,75-1,5%', '1,5-3%', '> 3%'])
pd.set_option('display.width', 200)
for soort in ('WK/Spelen', 'World Cups', 'overig'):
    x = V[V['soort'] == soort]
    t = x.groupby('gb', observed=True).agg(ritten=('zr', 'size'), gem_schok_pct=('schok_pct', 'mean'), gem_z=('zr', lambda s: s.clip(-4, 4).mean()),
                                           snel_1op20=('kans', lambda p: (p <= 0.05).mean() / 0.05)).round(3)
    b = np.polyfit(x['gat'].clip(-1, 6), x['zr'].clip(-4, 4), 1)[0]; se = x['zr'].clip(-4, 4).std() / np.sqrt(len(x)) / x['gat'].clip(-1, 6).std()
    print(f'\n{soort} (top < 3,5%, 2008+): helling gem z per % gat {b:+.3f} (se {se:.3f})'); print(t.to_string())
