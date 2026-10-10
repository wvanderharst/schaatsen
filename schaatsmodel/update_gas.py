import warnings; warnings.filterwarnings('ignore')
import pickle, json, gc, time, numpy as np, pandas as pd, do_analysis as da
t=time.time()
m=da.laad_volledig('model_full_tuned.pkl')
m.params=da.Params(**json.load(open('params_getuned.json')))
for a in ('_z_emp','_z_emp_sorted','historie_ruw','gas_f','gas_dag','vol_ref','gas_vorm_dag', 'verm_f', 'verm_reeks', 'verm_coef'):
    if hasattr(m,a): setattr(m,a,None)
gc.collect()
V=da.schat_gas(m); gc.collect(); da.schat_vol_ref(m); print('gas+vol klaar',round(time.time()-t),flush=True)
da.bewaar_veilig(m,'model_full_tuned.pkl'); print('opgeslagen',round(time.time()-t),flush=True)
