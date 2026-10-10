import sys, os; sys.path.insert(0, os.getcwd())
"""Volledig model A: vermoeidheid (vooraf) en schaal van de korte vorm (aK0, aK1), gekozen op de fitjaren. Bewaart niets."""
import warnings; warnings.filterwarnings('ignore')
import sys, gc, time, dataclasses as dc, numpy as np, do_analysis as da
t = time.time()
m = da.laad_volledig('model_full_tuned.pkl')
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'diepte_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect()
niv = m.obs['niveau_pct']
print('geladen', round(time.time() - t), flush=True)


def ll(lab):
    r = {}; fit = {}
    for dag in (False, True):
        tt, ff, _, rr = da.logloss_eerlijk(m, dag=dag, per_rij=True); r[dag] = rr; fit[dag] = ff; gc.collect()
    f = lambda x, lo: x[niv.reindex(x.index).to_numpy() >= lo].mean()
    print(f"{lab:26s} vooraf {r[False].mean():.5f} dag {r[True].mean():.5f} | toppers {f(r[False], 90):.5f} / {f(r[True], 90):.5f}"
          f" | elite {f(r[False], 97):.5f} / {f(r[True], 97):.5f} | fitjaren {fit[False]:.5f} / {fit[True]:.5f}  ({time.time() - t:.0f} s)", flush=True)
    return r


def verschil(a, b, lab):
    out = []
    for dag in (False, True):
        d = (b[dag] - a[dag]).dropna(); nv = niv.reindex(d.index).to_numpy()
        out.append(f"{'dag' if dag else 'vooraf'} {d.mean():+.5f} (se {d.std() / np.sqrt(len(d)):.5f}; toppers {d[nv >= 90].mean():+.5f}, elite {d[nv >= 97].mean():+.5f})")
    print(f"  verschil {lab}: " + ' | '.join(out), flush=True)


basis = ll('basis (bouw A)')
m.params = dc.replace(m.params, verm=True, verm_dag=False); da.schat_verm(m); da.schat_vorm(m)
verm = ll('verm (vooraf)'); verschil(basis, verm, 'verm t.o.v. basis')
q0 = tuple(m.params.vorm_par)
for paar in (sys.argv[1] if len(sys.argv) > 1 else '1:0.25,0.5:0.25').split(','):
    s0, s1 = (float(x) for x in paar.split(':'))
    q = list(q0); q[4] *= s0; q[5] *= s1
    m.params = dc.replace(m.params, vorm_par=tuple(q)); da.schat_vorm(m); gc.collect()
    r = ll(f'verm + aK0 x{s0:g} aK1 x{s1:g}'); verschil(verm, r, f'aK0 x{s0:g} aK1 x{s1:g} t.o.v. verm'); verschil(basis, r, f'aK0 x{s0:g} aK1 x{s1:g} t.o.v. basis')
