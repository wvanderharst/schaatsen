"""Eerlijke log loss (vooraf en dag) van model_full_tuned.pkl, totaal en per niveau (toppers >=90, elite >=97). Bewaart per rij."""
import warnings; warnings.filterwarnings('ignore')
import sys, gc, time, numpy as np, do_analysis as da
t = time.time()
pad = sys.argv[1] if len(sys.argv) > 1 else 'model_full_tuned.pkl'
m = da.laad_volledig(pad)
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag', 'diepte_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect()
niv = m.obs['niveau_pct']
for dag in (False, True):
    tt, _, _, rr = da.logloss_eerlijk(m, dag=dag, per_rij=True)
    lab = 'dag' if dag else 'vooraf'
    rr.astype(np.float32).to_pickle(f'll_{lab}.pkl')
    nv = niv.reindex(rr.index)
    print(f"{lab:6s} alle {tt:.5f} (n={len(rr)})  toppers {rr[nv >= 90].mean():.5f} (n={int((nv >= 90).sum())})  "
          f"elite {rr[nv >= 97].mean():.5f} (n={int((nv >= 97).sum())})  {time.time() - t:.0f} s", flush=True)
    gc.collect()
