"""Snelle toets van kalman_mv3 (universele baanfactor): mediane |fout| vooraf op de ritten van het oude model (2020+), t.o.v. oud en MV2-basis."""
import sys, os, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, kalman_mv3 as k3
par = json.load(open(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1] != '-' else {}
par.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
df = pd.read_parquet('ritten_cat.parquet'); df, inv = k3.voorbereid(df)
t = time.time(); o = k3.pass_(inv, par); print('pass', round(time.time() - t, 1), 's', flush=True)
D = pd.DataFrame({'skater_id': df['skater_id'], 'dag': df['dag'], 'afstand': df['afstand'].astype(str), 'tijd': df['tijd'], 'jaar': df['jaar'],
                  'ev': o['ev'], 'e': o['e'], 'thr': o['thr'], 'th0': o['th0'], 'm': o['m'], 'baan': df['baan'].astype(str), 'theta_oud': df['theta']})
O = pd.read_parquet('oud_vooraf.parquet'); O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']); O['afstand'] = O['afstand'].astype(str)
V = pd.read_parquet('voorspel2_uitkomst.parquet')[['skater_id', 'dag', 'afstand', 'tijd', 'pred_basis', 'pred']]; V['afstand'] = V['afstand'].astype(str)
M = D.merge(O[['skater_id', 'dag', 'afstand', 'tijd', 'err', 'niveau_pct']], on=['skater_id', 'dag', 'afstand', 'tijd']).merge(V, on=['skater_id', 'dag', 'afstand', 'tijd'])
M = M[np.isfinite(M['ev']) & np.isfinite(M['pred_basis'])]
lt = np.log(M['tijd'])
for nm, s in (('alle', M['niveau_pct'] > -1), ('toppers', M['niveau_pct'] >= 90), ('elite', M['niveau_pct'] >= 97)):
    x = M[s]; l = lt[s]
    print(f"{nm:8s} n={len(x):7d} mediaan |fout| vooraf: oud {100*np.median(np.abs(x['err'])):.3f}  MV2-basis {100*np.median(np.abs(l - x['pred_basis'])):.3f}"
          f"  MV2-volledig {100*np.median(np.abs(l - x['pred'])):.3f}  MV3-basis {100*np.median(np.abs(x['ev'])):.3f} | dag (LOO) {100*np.median(np.abs(x['e'])):.3f}")
r = D[np.isfinite(D['thr'])].drop_duplicates(['dag', 'baan', 'afstand', 'jaar'])
print('racepeil per baan (mediaan %, 2020+): ', {b: round(100 * v, 2) for b, v in r[r['jaar'] >= 2020].groupby('baan')['thr'].median().reindex(['Heerenveen (NED)', 'Salt Lake City (USA)', 'Calgary (CAN)', 'Inzell (GER)', 'Hamar (NOR)', 'Kemerovo']).items()})
