import warnings; warnings.filterwarnings('ignore')
import gc, numpy as np, pandas as pd, do_analysis as da
from scipy.special import ndtri
m = da.laad_volledig('model_full_tuned.pkl')
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'vorm_dag'): setattr(m, a, None)
gc.collect()
print(m.vorm_f.columns.tolist(), m.vorm_f.N.describe().round(3).to_dict(), flush=True)
_, err, var, mm = da.voorspelscore(m, 0, dag=True)
u = err / np.sqrt(var)
val = ((err > 0) & (mm['z'].to_numpy() > m.params.dip_val_z)) | ~np.isfinite(u)
g = da.GAS_GROEP[mm['afstand_idx'].to_numpy()] * 2 + (da._niv_pre(mm) >= 90).astype(int)
N = m.vorm_f.reindex(mm.index)['N'].fillna(0).to_numpy(); jr = mm['jaar'].to_numpy(); V = m.gas_vorm_dag
top = da._niv_pre(mm) >= 90
del mm, err, var; gc.collect()
ok = ~val & (jr >= 2010)
bins = np.digitize(N, [0.05, 0.5, 1.2])
def z_van(gam):
    uu = np.where(u < 0, u * np.exp(-gam * np.sqrt(N)), u); z = np.full(len(u), np.nan)
    for gg in range(8):
        s = (g == gg) & ok; z[s] = ndtri(np.clip(da._tp_cdf_v(V[gg], uu[s]), 1e-300, 1 - 1e-16))
    return z
for gam in (0.0, 0.15, 0.2, 0.25, 0.3):
    z = z_van(gam); out = []
    for per, sel in (('<=2019', ok & (jr <= 2019)), ('2020+', ok & (jr >= 2020))):
        r = [f"N{b}: {100*np.mean(z[sel & (bins == b)] < -2.5):.2f}%/{100*np.mean(z[sel & (bins == b)] < -3):.2f}%" for b in range(4)]
        out.append(per + ' ' + ' '.join(r))
    print(f'gamma {gam}: ' + ' | '.join(out), flush=True)
print('aantallen per bin (2020+):', [int((ok & (jr >= 2020) & (bins == b)).sum()) for b in range(4)])
