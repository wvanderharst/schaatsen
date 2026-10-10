import warnings; warnings.filterwarnings('ignore')
import sys, gc, dataclasses as dc, numpy as np, pandas as pd, do_analysis as da
m = da.laad_volledig('model_full_tuned.pkl')
DAG = 'dag' in sys.argv
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag'):
    setattr(m, a, None)
m.obs = m.obs[m.obs['jaar'] >= 2010].copy(); gc.collect()
m.params = dc.replace(m.params, vorm=False, vol_asym_bf=0.0)
_, err, var, o = da.voorspelscore(m, 2010, dag=DAG)
anker = m.anker
if DAG:
    pd.DataFrame({'ud': err / np.sqrt(var), 'lsdd': 0.5 * np.log(var), 'dd': o['theta'].to_numpy() - anker}, index=o.index).astype(np.float32).to_pickle('ijs_lab_dag.pkl')
    sys.exit()
zmax = np.maximum(o['z'].to_numpy(), err / np.sqrt(var))
D = pd.DataFrame({'u': err / np.sqrt(var), 'lsd': 0.5 * np.log(var), 'dp': o['mu_baan'].to_numpy() - anker,
                  'val': (err > 0) & (zmax > m.params.dip_val_z), 'jr': o['jaar'].to_numpy(), 'af': o['afstand'].to_numpy(),
                  'niv': da._niv_pre(o), 'hoog': o['baan'].str.contains('Salt Lake|Calgary').to_numpy(),
                  'sk': o['skater_id'].to_numpy(), 'dag': o['dag'].to_numpy()}, index=o.index)
pf = m.piek_f.reindex(o.index)[['c0', 'c1', 'c2']].fillna(0.0).to_numpy()
D['c0'], D['c1'], D['c2'] = pf[:, 0], pf[:, 1], pf[:, 2]
for c in D.columns:
    if D[c].dtype == np.float64: D[c] = D[c].astype(np.float32)
D.to_pickle('ijs_lab_d.pkl'); print(len(D), 'anker', anker, D.dp.describe().round(4).to_dict())
