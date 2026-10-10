import warnings; warnings.filterwarnings('ignore')
import gc, dataclasses as dc, numpy as np, pandas as pd, do_analysis as da
m = da.laad_volledig('model_full_tuned.pkl')
import sys
DAG = 'dag' in sys.argv
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag'):
    setattr(m, a, None)
m.obs = m.obs[m.obs['jaar'] >= 2010].copy()
gc.collect()
P0 = m.params
m.params = dc.replace(P0, vorm=False, vol_asym_bf=0.0)
_, err, var, mm = da.voorspelscore(m, 2010, dag=DAG)
if DAG:
    pd.DataFrame({'ud': (err / np.sqrt(var)).astype(np.float32), 'lsdd': (0.5 * np.log(var)).astype(np.float32)}, index=mm.index).to_pickle('vorm_lab_dag.pkl')
    print('dag klaar'); sys.exit()
o = mm
t_lim = da._tl_vector(m.t_lims, o['gender'].to_numpy(), o['afstand_idx'].to_numpy()); dl = np.exp(o['x_prior'].to_numpy())
vo = o['v_prior'].to_numpy() * (dl / (t_lim + dl)) ** 2; vt = o['v_theta'].to_numpy(); vb = o['var_baan'].to_numpy()
rel = 0.5 * np.log(var) - m.vol_ref[o['afstand_idx'].to_numpy(), np.digitize(da._niv_pre(o), da.VOL_BANDEN)]
zmax = np.maximum(o['z'].to_numpy(), err / np.sqrt(var))
val = (err > 0) & (zmax > P0.dip_val_z)
pf = m.piek_f.reindex(o.index)[['c0', 'c1', 'c2']].fillna(0.0).to_numpy()
V = m.vorm_f.reindex(o.index).fillna(0.0)
D = pd.DataFrame({'u': err / np.sqrt(var), 'lsd': 0.5 * np.log(var),
                  'r': (vo + vt) / (vo + vb), 'rel': rel, 'val': val, 'jr': o['jaar'].to_numpy(), 'af': o['afstand'].to_numpy(),
                  'niv': da._niv_pre(o), 'c0': pf[:, 0], 'c1': pf[:, 1], 'c2': pf[:, 2],
                  'Zk': V['Zk'].to_numpy(), 'F': V['F'].to_numpy(), 'S': V['S'].to_numpy(), 'Zl': V['Zl'].to_numpy()}, index=o.index)
for c in D.columns:
    if D[c].dtype == np.float64: D[c] = D[c].astype(np.float32)
D.to_pickle('vorm_lab_d.pkl')
import json
json.dump({'piek_drempel': [list(map(float, x)) for x in m.piek_drempel], 'piek_q': list(P0.piek_q), 'piek_m': P0.piek_m, 'bf': P0.vol_asym_bf,
           'vorm_par': list(P0.vorm_par)}, open('vorm_lab_meta.json', 'w'))
print('klaar', len(D), D[['u', 'r']].describe().round(3).to_string())
