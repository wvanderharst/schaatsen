import warnings; warnings.filterwarnings('ignore')
import sys, json, gc, time, dataclasses as dc, numpy as np, pandas as pd, do_analysis as da
t = time.time()
naam = sys.argv[1]; kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
p = da.Params(**json.load(open('params_getuned.json'))); p = dc.replace(p, **kw)
df = da.laad_dataset(da.data_csv(), verbose=False)
m = da.fit_volledig(df, p, n_iter=3, verbose=False); del df; gc.collect()
print(naam, 'fit klaar', round(time.time() - t), flush=True)
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag', 'diepte_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect()
qh = np.array([getattr(s, 'qh', 0.0) for s in m.staten.values()])
print(naam, 'qh percentielen', np.percentile(qh, [1, 10, 50, 90, 99]).round(3).tolist(), flush=True)
r = {}
for dag in (False, True):
    tt, _, _, rr = da.logloss_eerlijk(m, dag=dag, per_rij=True); r[dag] = tt
    rr.astype(np.float32).to_pickle(f'qv_{naam}_{"dag" if dag else "vooraf"}.pkl'); gc.collect()
print(naam, 'eerlijke log loss vooraf', round(r[False], 5), 'dag', round(r[True], 5), round(time.time() - t), flush=True)
if '--bewaar' in sys.argv:
    da.bewaar_veilig(m, f'model_{naam}.pkl'); print('opgeslagen', flush=True)
