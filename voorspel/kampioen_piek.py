"""Rijden eerdere kampioenen (WK/Spelen-titel vóór deze dag) en eerdere wereldrecordhouders (een rit die op dat moment de snelste ooit
in de data was, geslacht x afstand; data vanaf 2006) beter dan verwacht op WK's en Spelen? Rijderschok (ijs achteraf eruit), top < 3,5%.
Gebruik: python kampioen_piek.py <werkmap> <uitkomst.parquet>"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk, rijderschok as rs, kampioenen as kp
w, pad = sys.argv[1], sys.argv[2]
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'skater_name', 'gender', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'verdacht'])
for c in ('afstand', 'skater_name', 'gender', 'baan'):
    R[c] = R[c].astype(str)
Tn = pk.toernooien(R); Ti = kp.titels(R, Tn)
wkos = set(Tn.loc[Tn['soort'].isin(['Spelen', 'WK afstanden']), 'eid']); ie = pk.internationaal(R, set(Tn['eid']))
# wereldrecords (benadering): rit die op dat moment de snelste ooit was (zonder verdachte ritten), vanaf 2008 zodat de beginjaren niet alles zijn
x = R[~R['verdacht'] & np.isfinite(R['tijd']) & R['afstand'].isin(list(pk.ZES))].sort_values('dag')
x['best_voor'] = x.groupby(['gender', 'afstand'])['tijd'].transform(lambda s: s.cummin().shift(1))
wr = x[(x['tijd'] < x['best_voor']) & (x['dag'] >= 13880)][['skater_id', 'dag', 'afstand', 'tijd', 'skater_name', 'baan']]
eerste_titel = Ti.groupby('skater_id')['dag'].min(); eerste_wr = wr.groupby('skater_id')['dag'].min()
V = rs.tabel(w, pad); V['eid'] = V['event_id'] % 1000000
V['soort'] = np.where(V['eid'].isin(wkos), 'WK/Spelen', np.where(V['eid'].isin(ie), 'World Cups', 'overig'))
V = V[~V['k_val'] & (V['k_c'] < 0.035) & V['jaar'].between(2008, 2026) & V['afstand'].isin(list(pk.ZES))]
V['kampioen'] = V['dag'] > V['skater_id'].map(eerste_titel).fillna(1e9)
V['wr_houder'] = V['dag'] > V['skater_id'].map(eerste_wr).fillna(1e9)
print(f'{len(Ti)} titels; {len(wr)} wereldrecords (benadering, 2008+) door {wr["skater_id"].nunique()} rijders, bijv.:',
      ', '.join(wr.sort_values('dag').tail(6).apply(lambda r: f"{r['skater_name']} {r['afstand']} {r['tijd']}", axis=1)))
pd.set_option('display.width', 200)
for kol in ('kampioen', 'wr_houder'):
    t = V.groupby(['soort', kol]).agg(ritten=('zr', 'size'), gem_schok_pct=('schok_pct', 'mean'), gem_z=('zr', lambda s: s.clip(-4, 4).mean()),
                                      se=('zr', lambda s: s.clip(-4, 4).std() / np.sqrt(len(s))), snel_1op20=('kans', lambda p: (p <= 0.05).mean() / 0.05),
                                      traag_1op20=('kans', lambda p: (p >= 0.95).mean() / 0.05)).round(3)
    print(f'\n{kol} (eerder dan deze rit):'); print(t.to_string())
    # verschil-in-verschil: (piek - World Cups) voor eerdere kampioenen tegenover de rest
    m = V.groupby(['soort', kol])['zr'].apply(lambda s: s.clip(-4, 4).mean())
    print(f'   piek-effect (WK/Spelen min World Cups): {kol} ja {m[("WK/Spelen", True)] - m[("World Cups", True)]:+.3f} | nee {m[("WK/Spelen", False)] - m[("World Cups", False)]:+.3f}')
