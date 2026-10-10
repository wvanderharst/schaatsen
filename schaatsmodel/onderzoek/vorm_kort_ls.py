"""Herschat de korte vorm (aK0, aK1) met kleinste kwadraten op jaren <= split_jaar: y = fout vóór de vorm (sd van eigen + veld)
minus de lange vormtermen, op [Zk, Zk*w]. Daarna eerlijke log loss. Gebruik (vanuit de hoofdmap): python onderzoek/vorm_kort_ls.py model_full_tuned.pkl [verm]"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, gc, time, dataclasses as dc, numpy as np
sys.path.insert(0, os.getcwd())
import do_analysis as da
t = time.time()
m = da.laad_volledig(sys.argv[1])
for a in ('trend_reeks', 'piek_reeks', 'gas_dag', 'historie_ruw', 'diepte_dag'):
    if hasattr(m, a): setattr(m, a, None)
gc.collect(); niv = m.obs['niveau_pct']
if 'verm' in sys.argv:
    m.params = dc.replace(m.params, verm=True); da.schat_verm(m)
da.schat_vorm(m)


def ll(lab):
    r = {}; fit = {}
    for dag in (False, True):
        tt, ff, _, rr = da.logloss_eerlijk(m, dag=dag, per_rij=True); r[dag] = rr; fit[dag] = ff; gc.collect()
    f = lambda x, lo: x[niv.reindex(x.index).to_numpy() >= lo].mean()
    print(f"{lab:30s} vooraf {r[False].mean():.5f} dag {r[True].mean():.5f} | toppers {f(r[False], 90):.5f} / {f(r[True], 90):.5f}"
          f" | elite {f(r[False], 97):.5f} / {f(r[True], 97):.5f} | fitjaren {fit[False]:.5f} / {fit[True]:.5f}  ({time.time() - t:.0f} s)", flush=True)
    return r


def verschil(a, b, lab):
    out = []
    for dag in (False, True):
        d = (b[dag] - a[dag]).dropna(); nv = niv.reindex(d.index).to_numpy()
        out.append(f"{'dag' if dag else 'vooraf'} {d.mean():+.5f} (se {d.std() / np.sqrt(len(d)):.5f}; toppers {d[nv >= 90].mean():+.5f}, elite {d[nv >= 97].mean():+.5f})")
    print(f"  verschil {lab}: " + ' | '.join(out), flush=True)


basis = ll('start' + (' (met verm)' if 'verm' in sys.argv else ''))
q = list(m.params.vorm_par); p = m.params
res = {}
for dag in (False, True):
    m._vorm_vang = True
    _, err, var, mm = da.voorspelscore(m, 0, dag=dag)
    u, V, w = m._vorm_u; m._vorm_vang = False; m._vorm_u = None
    Zk, F, S, Zl = (V[c].to_numpy(dtype=float) for c in ('Zk', 'F', 'S', 'Zl'))
    rest = (q[6] + q[7] * w) * Zl + q[8] * F + q[9] * S
    val = (err > 0) & (mm['z'].to_numpy() > p.dip_val_z)
    ok = np.isfinite(u) & ~val & (mm['jaar'].to_numpy() <= p.split_jaar) & (mm['jaar'].to_numpy() >= 2010)
    y = np.clip(u, -4, 4) - rest
    X = np.c_[np.ones(len(y)), Zk, Zk * w]
    b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    res[dag] = b[1:]; print('dag' if dag else 'vooraf', 'LS aK0, aK1 =', b[1:].round(5), '(oud', q[4], q[5], ')', flush=True)
for lab, (k0, k1) in (('LS vooraf', res[False]), ('LS dag', res[True]), ('LS gemiddeld', (res[False] + res[True]) / 2)):
    q2 = list(q); q2[4], q2[5] = float(k0), float(k1)
    m.params = dc.replace(p, vorm_par=tuple(q2)); da.schat_vorm(m); gc.collect()
    r = ll(f'{lab} ({k0:.4f}, {k1:.4f})'); verschil(basis, r, f'{lab} t.o.v. start')
