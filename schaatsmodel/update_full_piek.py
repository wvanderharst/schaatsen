import warnings; warnings.filterwarnings('ignore')
import pickle, json, time, numpy as np, do_analysis as da
t=time.time()
m=da.laad_volledig('model_full_tuned.pkl')
m.params=da.Params(**json.load(open('params_getuned.json')))
import gc
for a in ('_z_emp','_z_emp_sorted','_niv_ref','piek_f','piek_reeks','trend_f','trend_reeks','historie_ruw','gas_f','gas_dag','gas_vorm_dag', 'verm_f', 'verm_reeks', 'verm_coef'):
    if hasattr(m,a): delattr(m,a)
m.loc_schaal_tab=None; gc.collect()
da.schat_trend(m); print('trend',round(time.time()-t),flush=True)
da.schat_loc_schaal(m); print('loc',round(time.time()-t),flush=True)
da.schat_piek_kansen(m); print('piek',round(time.time()-t),m.piek_drempel,flush=True)
gc.collect(); da.bewaar_veilig(m,'model_full_tuned.pkl'); print('opgeslagen',round(time.time()-t))
