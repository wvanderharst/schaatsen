import warnings; warnings.filterwarnings('ignore')
import gc, dataclasses as dc, numpy as np, pandas as pd, do_analysis as da
m = da.laad_volledig('model_full_tuned.pkl')
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag'): setattr(m, a, None)
gc.collect()
m.params = dc.replace(m.params, vorm=False)
z = da.normaal_z_dag(m, 0).astype(np.float32)
o = m.obs.loc[z.index, ['skater_id', 'dag', 'event_id', 'afstand_idx', 'jaar']]
o['z'] = z; o.to_pickle('zbase.pkl'); print(len(o), flush=True)
