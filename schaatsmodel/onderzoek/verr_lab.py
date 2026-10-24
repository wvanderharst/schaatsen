"""Verrassings-volatiliteit: na snelle uitschieters (dag-z van EERDERE dagen) schuift de verwachting en groeit de spreiding."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, time, numpy as np, pandas as pd
from numba import njit
from scipy import optimize, stats
from scipy.special import ndtri
import gas_run as G
t0 = time.time()
d = G.d
R = pd.read_pickle('gas3_res_S.pkl').reindex(d.index)
ZD = pd.read_pickle('normaal_z_dag.pkl').reindex(d.index).astype(float).fillna(0).to_numpy()
o = np.lexsort((d.ai.to_numpy(), d.ev.to_numpy(), d.dag.to_numpy(), d.sk.to_numpy()))
SK = d.sk.to_numpy()[o].astype(np.int64); DG = d.dag.to_numpy()[o].astype(np.int64); ZO = ZD[o]; VAL = d.val.to_numpy()[o]


@njit(cache=True)
def kenmerk(sk, dg, z, val, c, H, cs, uit_f, uit_s):
    """uit_f: som over eerdere dagen van exp(-dt/H)*max(0,-z-c); uit_s idem traag (z>cs, cs<=0: uit)."""
    n = len(sk); vf = 0.0; vs = 0.0; ld = -1; i = 0
    while i < n:
        j = i
        while j < n and sk[j] == sk[i] and dg[j] == dg[i]:
            j += 1
        if i == 0 or sk[i] != sk[i - 1]:
            vf = 0.0; vs = 0.0; ld = dg[i]
        else:
            f = np.exp(-(dg[i] - ld) / H); vf *= f; vs *= f; ld = dg[i]
        for k in range(i, j):
            uit_f[k] = vf; uit_s[k] = vs
        af = 0.0; asl = 0.0
        for k in range(i, j):
            if val[k]:
                continue
            if -z[k] - c > af:
                af = -z[k] - c
            zz = min(max(z[k], -4.0), 4.0)
            asl += zz / (j - i)
        vf += af; vs += asl
        i = j
    return 0


def kenm(c, H, cs=0.0):
    f = np.zeros(len(SK)); s = np.zeros(len(SK))
    kenmerk(SK, DG, ZO, VAL, c, H, cs, f, s)
    F = np.zeros(len(SK)); S_ = np.zeros(len(SK)); F[o] = f; S_[o] = s
    return F, S_


U = d.u.to_numpy(); M = R.m.to_numpy(); L = R.lf.to_numpy(); GR = d.grp.to_numpy(); PW = d[['pw0', 'pw1', 'pw2']].to_numpy()
FIT = ((d.jr <= 2019) & (d.jr >= 2010) & ~d.val).to_numpy(); TEST = ((d.jr >= 2020) & ~d.val).to_numpy()
TOP = (d['abs'] >= 90).to_numpy()


def logp(m, lf, ls, sel):
    u = U[sel]; m = m[sel]; lf = lf[sel]; ls = ls[sel]; g = GR[sel]; pw = PW[sel]
    x = u - m; lam = np.where(x < 0, lf, ls); eps = x / np.exp(lam)
    lp = np.zeros(len(u)); pit = np.zeros(len(u))
    for gg in range(4):
        s = g == gg; v = G.VORM[gg]
        th = [np.log(v[0] / (1 - v[0])), np.log(v[1]), np.log((v[2] - 1.5) / (30 - v[2])), np.log(v[3]), np.log((v[4] - 1.5) / (30 - v[4]))]
        lp[s] = G.tp_logpdf(th, eps[s]) - lam[s]
        F = G.tp_cdf(v, eps[s]); lo, hi = G.drempel_lo[gg], G.drempel_hi[gg]; Flo, Fhi = G.tp_cdf(v, lo), G.tp_cdf(v, hi); p = pw[s]
        reg = np.where(eps[s] < lo, 0, np.where(eps[s] > hi, 2, 1))
        lp[s] += np.log(p[np.arange(s.sum()), reg] / G.MASSA[gg][reg])
        pit[s] = np.where(reg == 0, p[:, 0] * F / Flo, np.where(reg == 1, p[:, 0] + p[:, 1] * (F - Flo) / (Fhi - Flo), p[:, 0] + p[:, 1] + p[:, 2] * (F - Fhi) / (1 - Fhi)))
    return lp, pit


def model(th, F, S_):
    a, bf, bs, a2, bs2 = th
    m = M + a * F + a2 * S_
    lf = L + bf * F
    ls = L + bs * F
    return m, lf, ls


rng_ = np.random.default_rng(1); FIT = FIT & (rng_.random(len(FIT)) < 0.5)        # halve steekproef (geheugen/snelheid)
BASE_FIT = logp(M, L, L, FIT)[0].mean(); BASE_TEST, _ = logp(M, L, L, TEST)


def fit(c, H, cs=0.0):
    F, S_ = kenm(c, H, cs)
    act = np.array([1, 1, 1, MET_Z, 0], bool)
    def f(t):
        th = np.zeros(5); th[act] = t
        return -logp(*model(th, F, S_), FIT)[0].mean()
    k = act.sum(); x0 = np.zeros(k); simp = np.vstack([x0] + [x0 + 0.05 * np.eye(k)[i] for i in range(k)])
    r = optimize.minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-7, 'maxiter': 800, 'initial_simplex': simp})
    th = np.zeros(5); th[act] = r.x
    lt, pit = logp(*model(th, F, S_), TEST)
    dd = -(lt - BASE_TEST)
    return th, -r.fun - BASE_FIT, dd.mean(), dd[TOP[TEST]].mean(), F, S_, pit


MET_Z = True
if __name__ == '__main__':
    res = []
    for c in (2.0, 2.5, 3.0):
        for H in (30, 90, 180):
            th, gfit, gt, gtop, *_ = fit(c, H)
            res.append((c, H, *np.round(th[:4], 4), round(gfit, 5), round(gt, 5), round(gtop, 5)))
            print(res[-1], f'{time.time()-t0:.0f}s', flush=True)
    pd.DataFrame(res, columns=['c', 'H', 'a', 'bf', 'bs', 'aZ', 'fit', 'test', 'top']).to_csv('verr_lab_grid.csv', index=False)
