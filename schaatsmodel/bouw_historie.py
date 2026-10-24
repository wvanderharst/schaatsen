import warnings; warnings.filterwarnings('ignore')
import pickle, time, numpy as np, do_analysis as da
t=time.time()
m=da.laad_volledig('model_full_tuned.pkl')
import gc
for a in ('trend_f','piek_f','gas_f','gas_dag','trend_reeks','piek_reeks','_z_emp','_z_emp_sorted','historie_ruw'):
    if hasattr(m,a): setattr(m,a,None)
gc.collect()
df=da.laad_dataset(da.data_csv(), verbose=False)
obs_oud=m.obs[['x_prior','x_post']].to_numpy().copy()
m.bewaar_historie=True
m.fit(df, verbose=False)
print('fit klaar',round(time.time()-t),'max |verschil| x_prior', float(np.nanmax(np.abs(m.obs[['x_prior','x_post']].to_numpy()-obs_oud))),flush=True)
np.savez('hist_nieuw.npz', **m.historie_ruw); print('opgeslagen',round(time.time()-t))
