"""Baseline: de voorspelling vooraf van het oude model (model_full_tuned.pkl) per rit, met sleutels om te koppelen.
Schrijft oud_vooraf.parquet: skater_id, dag, afstand, tijd, err (ln-tijd min voorspelling), var, ll (eerlijke log loss vooraf, 2020+).
Draaien in de map met het model: python oud_vooraf.py"""
import sys, gc, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/schaatsen/schaatsmodel')
import numpy as np, pandas as pd, do_analysis as da
m = da.laad_volledig('model_full_tuned.pkl')
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag', 'diepte_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect()
_, err, var, o = da.voorspelscore(m, 2010, dag=False)
uit = pd.DataFrame({'skater_id': o['skater_id'].to_numpy(), 'dag': o['dag'].to_numpy(), 'afstand': o['afstand'].astype(str).to_numpy(),
                    'tijd': o['tijd'].to_numpy(), 'err': err, 'var': var, 'niveau_pct': o['niveau_pct'].to_numpy()}, index=o.index)
ll = pd.read_pickle('ll_vooraf.pkl')
uit['ll'] = ll.reindex(uit.index).to_numpy()
uit.reset_index(names='obs').to_parquet('oud_vooraf.parquet')
print(len(uit), 'rijen;', int(np.isfinite(uit['ll']).sum()), 'met log loss; MAE % 2020+:', round(100 * np.nanmedian(np.abs(err[o['jaar'].to_numpy() >= 2020])), 3))
