import warnings; warnings.filterwarnings('ignore')
import json, pickle, sys, time
import do_analysis as da
t=time.time()
p = da.Params(**json.load(open(sys.argv[1] if len(sys.argv)>1 else 'params_getuned.json')))
out = sys.argv[2] if len(sys.argv)>2 else 'model_full_tuned.pkl'
df = da.laad_dataset(da.data_csv(), verbose=False)
m = da.fit_volledig(df, p, n_iter=3, verbose=False)
print('score', da.voorspelscore(m)[0], '%.0f s'%(time.time()-t))
import gc
for a in ('piek_f','piek_reeks','trend_f','trend_reeks','historie_ruw','_z_emp','_z_emp_sorted','_niv_ref','gas_f','gas_dag','vol_ref','gas_vorm_dag','vorm_f','vorm_dag','diepte_f','diepte_dag', 'verm_f', 'verm_reeks', 'verm_coef'):
    if hasattr(m,a): setattr(m,a,None)
gc.collect()
da.bewaar_veilig(m,out)
print('opgeslagen', flush=True)
