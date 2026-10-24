"""Achteraf: correctie per baan als functie van het niveau g, geschat op ALLE rijders, alleen eerdere dagen:
  corr_b(g) = c_b . phi(g),  phi(g) = [1, exp(-g/0.04), exp(-g/0.15), g]; c_b uit een uitdovende (halfwaardetijd hl dagen) gewogen
  kleinste-kwadratenschatting op de fouten van eerdere dagen op baan b, met ridge k0 (krimp naar 0). Kiezen op 2012-2019."""
import numpy as np, pandas as pd
V = pd.read_parquet('voorspel2_uitkomst.parquet', columns=['dag', 'jaar', 'baan', 'tijd', 'pred', 'k_c', 'ok', 'k_val'])
V = V[V['ok'] & np.isfinite(V['pred'])].reset_index(drop=True)
r = np.clip((np.log(V['tijd']) - V['pred']).to_numpy(), -0.03, 0.03); g = V['k_c'].clip(0, 0.5).fillna(0.3).to_numpy()
PHI = np.c_[np.ones(len(V)), np.exp(-g / 0.04), np.exp(-g / 0.15), g]; P = PHI.shape[1]
okf = (~V['k_val']).to_numpy()
def corr(hl, k0):
    out = np.zeros(len(V))
    for b, idx in V.groupby('baan').indices.items():
        dag = V['dag'].to_numpy()[idx]; o = np.argsort(dag, kind='stable'); idx = idx[o]; dag = dag[o]
        X = PHI[idx]; y = r[idx]; m = okf[idx]
        ud, start = np.unique(dag, return_index=True); eind = np.r_[start[1:], len(dag)]
        A = np.zeros((P, P)); bv = np.zeros(P); prev = None
        for d, s_, e_ in zip(ud, start, eind):
            if prev is not None:
                f = 0.5 ** ((d - prev) / hl); A *= f; bv *= f
            c = np.linalg.solve(A + k0 * np.eye(P), bv)
            out[idx[s_:e_]] = X[s_:e_] @ c
            mm = m[s_:e_]; Xs = X[s_:e_][mm]; ys = y[s_:e_][mm]
            A += Xs.T @ Xs; bv += Xs.T @ ys; prev = d
    return out
def maat(rr, jaren):
    s = V['jaar'].between(*jaren).to_numpy()
    d = pd.DataFrame({'b': V['baan'].to_numpy()[s], 'n': pd.cut(g[s], [-1, .02, .04, .07, .12, .2, .3, 1]), 'r': 100 * rr[s]})
    t = d.groupby(['b', 'n'], observed=True)['r'].agg(['median', 'size']); t = t[t['size'] >= 300]
    return float(np.sqrt(np.average(t['median'] ** 2, weights=np.sqrt(t['size'])))), float(np.median(np.abs(rr[s])) * 100), d
b0, a0, _ = maat(r, (2012, 2019)); t0, at0, d0 = maat(r, (2020, 2026))
print(f'huidig: bias baan x niveau fit {b0:.3f} toets {t0:.3f} | mediane |fout| fit {a0:.3f} toets {at0:.3f}')
res = []
for hl in (180, 365, 730):
    for k0 in (50.0, 500.0, 5000.0):
        rr = r - corr(hl, k0); bf, af, _ = maat(rr, (2012, 2019)); bt, at, d = maat(rr, (2020, 2026))
        res.append((bf, bt, af, at, hl, k0, d)); print(f'hl {hl} k0 {k0:.0f}: bias fit {bf:.3f} toets {bt:.3f} | |fout| fit {af:.3f} toets {at:.3f}', flush=True)
best = min(res, key=lambda x: x[0]); d = best[6]
B = ['Salt Lake City (USA)', 'Calgary (CAN)', 'Inzell (GER)', 'Heerenveen (NED)']
print('beste', best[4], best[5]); print(d[d.b.isin(B)].pivot_table(index='b', columns='n', values='r', aggfunc='median', observed=True).round(2).to_string())
