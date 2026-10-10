"""Verrassings-volatiliteit in DAG-eenheden: doel = dag-z (t.o.v. het veld), features uit dag-z van eerdere dagen."""
import warnings; warnings.filterwarnings('ignore')
import time, numpy as np, pandas as pd
from numba import njit
from scipy import optimize
t0 = time.time()
d = pd.read_pickle('gas_d.pkl')[['sk', 'dag', 'ev', 'ai', 'jr', 'abs', 'np', 'val']]
ZD = pd.read_pickle('normaal_z_dag.pkl').reindex(d.index).astype(float).to_numpy()
ok = np.isfinite(ZD); d = d[ok]; ZD = ZD[ok]
o = np.lexsort((d.ai.to_numpy(), d.ev.to_numpy(), d.dag.to_numpy(), d.sk.to_numpy()))
SK = d.sk.to_numpy()[o].astype(np.int64); DG = d.dag.to_numpy()[o].astype(np.int64); ZO = ZD[o]; VAL = (d.val.to_numpy() | (ZD > 6))[o]


@njit(cache=True)
def kenmerk(sk, dg, z, val, c, H, cs, uf, us, uz):
    n = len(sk); vf = 0.0; vs = 0.0; vz = 0.0; ld = -1; i = 0
    while i < n:
        j = i
        while j < n and sk[j] == sk[i] and dg[j] == dg[i]:
            j += 1
        if i == 0 or sk[i] != sk[i - 1]:
            vf = 0.0; vs = 0.0; vz = 0.0; ld = dg[i]
        else:
            f = np.exp(-(dg[i] - ld) / H); vf *= f; vs *= f; vz *= f; ld = dg[i]
        for k in range(i, j):
            uf[k] = vf; us[k] = vs; uz[k] = vz
        af = 0.0; asl = 0.0; az = 0.0; nn = 0
        for k in range(i, j):
            if val[k]:
                continue
            if -z[k] - c > af:
                af = -z[k] - c
            if z[k] - cs > asl:
                asl = z[k] - cs
            az += min(max(z[k], -4.0), 4.0); nn += 1
        vf += af; vs += asl
        if nn > 0:
            vz += az / nn
        i = j
    return 0


def kenm(c, H, cs=2.5):
    a = [np.zeros(len(SK)) for _ in range(3)]
    kenmerk(SK, DG, ZO, VAL, c, H, cs, *a)
    out = []
    for x in a:
        y = np.zeros(len(SK)); y[o] = x; out.append(y)
    return out


JR = d.jr.to_numpy(); V = (d.val.to_numpy() | (ZD > 6))
FIT = (JR >= 2010) & (JR <= 2019) & ~V; TEST = (JR >= 2020) & ~V; TOP = (d['abs'] >= 90).to_numpy()
LOG2PI = 0.5 * np.log(2 * np.pi)


def ll(th, F, S, Z, sel):
    a, aZ, aS, bf, bs, bsS = th
    mu = a * F[sel] + aZ * Z[sel] + aS * S[sel]
    lf = bf * F[sel]; ls = bs * F[sel] + bsS * S[sel]
    x = ZD[sel] - mu
    # twee-delige normaal met gelijke dichtheid in 0 (sf, ss)
    sf = np.exp(lf); ss = np.exp(ls); s = np.where(x < 0, sf, ss)
    return -LOG2PI - np.log((sf + ss) / 2) - 0.5 * (x / s) ** 2


def fit(c, H, cs=2.5, act=(1, 1, 1, 1, 1, 1)):
    F, S, Z = kenm(c, H, cs); act = np.array(act, bool)
    f = lambda t: -ll(np.where(act, np.r_[t, 0][np.cumsum(act) - 1] if True else 0, 0), F, S, Z, FIT).mean()
    def f(t):
        th = np.zeros(6); th[act] = t
        return -ll(th, F, S, Z, FIT).mean()
    k = act.sum(); x0 = np.zeros(k); simp = np.vstack([x0] + [x0 + 0.05 * np.eye(k)[i] for i in range(k)])
    r = optimize.minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 3000, 'initial_simplex': simp})
    th = np.zeros(6); th[act] = r.x
    base = ll(np.zeros(6), F, S, Z, TEST); new = ll(th, F, S, Z, TEST); dd = -(new - base)
    sel = TEST & (F > 0)
    return th, dd.mean(), dd[TOP[TEST]].mean(), (-(ll(th, F, S, Z, sel) - ll(np.zeros(6), F, S, Z, sel))).mean(), (F, S, Z)


if __name__ == '__main__':
    print('n', len(ZD), 'fit', FIT.sum(), 'test', TEST.sum(), flush=True)
    rows = []
    for c in (1.5, 2.0, 2.5, 3.0):
        for H in (15, 30, 60, 120, 240):
            th, g, gt, gF, _ = fit(c, H)
            rows.append((c, H, *np.round(th, 4), round(g, 5), round(gt, 5), round(gF, 4)))
            print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
    pd.DataFrame(rows, columns=['c', 'H', 'a', 'aZ', 'aS', 'bf', 'bs', 'bsS', 'test', 'top', 'na_uitschieter']).to_csv('verr_lab2_grid.csv', index=False)
