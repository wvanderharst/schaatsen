import warnings; warnings.filterwarnings('ignore')
import pickle, gc, numpy as np, pandas as pd, do_analysis as da
m=da.laad_volledig('model_full_tuned.pkl')
for a in ('piek_reeks','trend_reeks','gas_dag','historie_ruw','_z_emp','_z_emp_sorted'):
    if hasattr(m,a): setattr(m,a,None)
gc.collect()
z=da.normaal_z(m,0); z.astype(np.float32).to_pickle('normaal_z.pkl')
o=m.obs.loc[z.index]; ok=~(o.z>6).to_numpy()&o.jaar.ge(2020).to_numpy()
for lab,s in [('alle 2020+',ok),('vaste rijders',ok&(o.n_prior>=20).to_numpy()),('toppers',ok&(o.niveau_pct>=90).to_numpy())]:
    zz=z[s]; print(f"{lab:14s} |z|>3 {100*(zz.abs()>3).mean():.3f}%  |z|>4 {100*(zz.abs()>4).mean():.4f}%  |z|>6 {100*(zz.abs()>6).mean():.4f}%",flush=True)
