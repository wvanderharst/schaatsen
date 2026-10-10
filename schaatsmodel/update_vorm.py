import warnings; warnings.filterwarnings('ignore')
import sys, json, gc, time, numpy as np, pandas as pd, do_analysis as da
t = time.time()
m = da.laad_volledig('model_full_tuned.pkl')
m.params = da.Params(**json.load(open('params_getuned.json')))
for a in ('_z_emp', '_z_emp_sorted', 'historie_ruw', 'vorm_f', 'vorm_dag', 'gas_vorm_dag', 'diepte_f', 'diepte_dag', 'verm_f', 'verm_reeks', 'verm_coef', 'staart_f', 'staart_dag', 'schaal_f', 'schaal_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect()
import dataclasses as _dc
verm = m.params.verm
m.params = _dc.replace(m.params, vorm=False, piek_diepte=False, verm=False)
if '--vergelijk' in sys.argv:
    nb = da.logloss_eerlijk(m)[0]; print('eerlijke log loss zonder vorm', round(nb, 5), round(time.time() - t), flush=True)
if verm:            # vermoeidheid vóór de vorm: de vorm leest de dag-z, en die hoort voor vermoeidheid gecorrigeerd te zijn
    m.params = _dc.replace(m.params, verm=True); da.schat_verm(m); print('verm geschat', round(time.time() - t), flush=True)
    if '--vergelijk' in sys.argv:
        nm_ = da.logloss_eerlijk(m)[0]; print('eerlijke log loss met verm, zonder vorm', round(nm_, 5), 'verschil', round(nm_ - nb, 5), flush=True)
m.params = _dc.replace(m.params, vorm=True, piek_diepte=True)
da.schat_vorm(m); print('vorm geschat', round(time.time() - t), flush=True)
if '--vergelijk' in sys.argv:
    nv = da.logloss_eerlijk(m)[0]; print('eerlijke log loss met vorm', round(nv, 5), 'verschil', round(nv - nb, 5), round(time.time() - t), flush=True)
da.bewaar_veilig(m, 'model_full_tuned.pkl'); print('opgeslagen', round(time.time() - t), flush=True)
