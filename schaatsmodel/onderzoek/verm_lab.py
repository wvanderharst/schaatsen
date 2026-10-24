"""Klein: vermoeidheidslaag en vorm-plateau op de testdata. Gebruik (vanuit de hoofdmap): python onderzoek/verm_lab.py [csv]; env PLATEAUS, SCHALEN, DAG0."""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json, gc, time, dataclasses as dc, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import do_analysis as da
t = time.time()
csv = sys.argv[1] if len(sys.argv) > 1 else 'speedskating_test.csv'
pk = 'klein_' + os.path.basename(csv).replace('.csv', '') + '.pkl'
if os.path.exists(pk):
    m = da.laad_volledig(pk)
else:
    p = da.Params(**json.load(open('params_getuned.json')))
    df = da.laad_dataset(csv, verbose=True)
    m = da.fit_volledig(df, p, n_iter=3, verbose=False); del df; gc.collect()
    da.bewaar_veilig(m, pk)
print('model klaar', round(time.time() - t), flush=True)
niv = m.obs['niveau_pct']


def ll(lab):
    r = {}
    for dag in (False, True):
        tt, _, _, rr = da.logloss_eerlijk(m, dag=dag, per_rij=True); r[dag] = rr
    f = lambda x, lo: x[niv.reindex(x.index).to_numpy() >= lo].mean()
    print(f"{lab:28s} vooraf {r[False].mean():.5f}  dag {r[True].mean():.5f}  | toppers vooraf {f(r[False], 90):.5f} dag {f(r[True], 90):.5f}"
          f"  | elite vooraf {f(r[False], 97):.5f} dag {f(r[True], 97):.5f}  ({time.time() - t:.0f} s)", flush=True)
    return r


def verschil(a, b, lab):
    out = []
    for dag in (False, True):
        d = (b[dag] - a[dag]).dropna(); nv = niv.reindex(d.index).to_numpy()
        se = d.std() / np.sqrt(len(d))
        out.append(f"{'dag' if dag else 'vooraf'} {d.mean():+.5f} (se {se:.5f}; toppers {d[nv >= 90].mean():+.5f}, elite {d[nv >= 97].mean():+.5f})")
    print(f"  verschil {lab}: " + ' | '.join(out), flush=True)


m.params = dc.replace(m.params, verm=False, vorm_plateau=0.0); da.schat_vorm(m)
basis = ll('basis (geen verm)')
m.params = dc.replace(m.params, verm=True); da.schat_verm(m); da.schat_vorm(m)
verm = ll('verm')
verschil(basis, verm, 'verm t.o.v. basis')
if os.environ.get('DAG0'):
    cd_ = m.verm_coef['dag']; m.verm_coef['dag'] = np.zeros_like(cd_); da.schat_vorm(m)
    v0 = ll('verm alleen vooraf'); verschil(basis, v0, 'verm alleen vooraf t.o.v. basis')
    verm = v0
q0 = tuple(m.params.vorm_par)
res = {}
for P in [float(x) for x in (os.environ.get('PLATEAUS') or '3,5').split(',')]:
    for sc in [float(x) for x in (os.environ.get('SCHALEN') or '1,0.7,0.5').split(',')]:
        q = list(q0); q[4] *= sc; q[5] *= sc
        m.params = dc.replace(m.params, vorm_plateau=P, vorm_par=tuple(q)); da.schat_vorm(m)
        _, fit, _ = da.logloss_eerlijk(m)
        r = ll(f'verm + plateau {P:g} d, aK x{sc:g}'); print(f'    fit-jaren nll {fit:.5f}', flush=True)
        verschil(verm, r, f'plateau {P:g}/x{sc:g} t.o.v. verm')
