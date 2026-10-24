import warnings; warnings.filterwarnings('ignore')
import pickle, gc, numpy as np, pandas as pd, do_analysis as da
m=da.laad_volledig('model_full_tuned.pkl')
for a in ('piek_reeks','trend_reeks','gas_dag','historie_ruw','_z_emp','_z_emp_sorted'):
    if hasattr(m,a): setattr(m,a,None)
gc.collect()
o=m.obs; ok=~(o.z>6).to_numpy()&o.jaar.ge(2020).to_numpy()
zd=da.normaal_z_dag(m,0); zd.astype(np.float32).to_pickle('normaal_z_dag.pkl')
for lab,s in [('DAG alle 2020+',ok),('DAG n_prior>=20',ok&(o.n_prior>=20).to_numpy()),('DAG toppers niveau>=90',ok&(o.niveau_pct>=90).to_numpy())]:
    zz=zd.reindex(o.index)[s]; print(f"{lab:24s} |z|>3 {100*(zz.abs()>3).mean():.3f}%  |z|>4 {100*(zz.abs()>4).mean():.4f}%  |z|>6 {100*(zz.abs()>6).mean():.4f}%",flush=True)
x=o[(o.skater_id==31536)&(o.dag>=20400)].index; print('Leerdam dag-z:',zd.reindex(x).round(2).tolist(),flush=True)
