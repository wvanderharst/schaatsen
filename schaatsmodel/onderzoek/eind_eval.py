import sys, os; sys.path.insert(0, os.getcwd())
"""Basis A tegen de eindkeuze (verm in de breedte + korte vorm aK0/aK1 herschat), per rij; splitst het verschil naar rijen met en zonder vermoeidheidsverschuiving."""
import warnings; warnings.filterwarnings('ignore')
import sys, gc, time, dataclasses as dc, numpy as np, pandas as pd, do_analysis as da
t = time.time()
m = da.laad_volledig('model_full_tuned.pkl')
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'diepte_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect(); niv = m.obs['niveau_pct']
def run():
    r = {}
    for dag in (False, True):
        tt, ff, _, rr = da.logloss_eerlijk(m, dag=dag, per_rij=True); r[dag] = (rr, ff); gc.collect()
    return r
A = run()
q = list(m.params.vorm_par); q[4], q[5] = float(sys.argv[1]), float(sys.argv[2])
m.params = dc.replace(m.params, verm=True, vorm_par=tuple(q)); da.schat_verm(m); da.schat_vorm(m)
B = run()
V = m.verm_f[da.VERM_KOL].abs().sum(axis=1) > 0
for dag in (False, True):
    a, fa = A[dag]; b, fb = B[dag]; d = (b - a).dropna(); nv = niv.reindex(d.index).to_numpy(); v = V.reindex(d.index).to_numpy()
    print(f"{'dag' if dag else 'vooraf'}: A {a.mean():.5f} -> {b.mean():.5f} (fitjaren {fa:.5f} -> {fb:.5f})", flush=True)
    for lab, s in (('alle', nv >= 0), ('toppers', nv >= 90), ('elite', nv >= 97)):
        print(f"   {lab:8s} {d[s].mean():+.5f} (se {d[s].std() / np.sqrt(s.sum()):.5f}) | zonder verm-kenmerk {d[s & ~v].mean():+.5f} (n={int((s & ~v).sum())}) | met {d[s & v].mean():+.5f} (n={int((s & v).sum())})", flush=True)
