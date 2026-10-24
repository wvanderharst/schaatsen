"""Stap 1: ijs-effect afhankelijk van niveau. shift (ln-tijd) = (ijs - HV) * k(niveau) [+ hoogland * h(niveau)]."""
import warnings; warnings.filterwarnings('ignore')
import json, time, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
t0 = time.time()
D = pd.read_pickle('ijs_lab_d.pkl'); Dd = pd.read_pickle('ijs_lab_dag.pkl').reindex(D.index)
M = json.load(open('vorm_lab_meta.json')); PD = {int(k): v for k, v in M['piek_drempel'].items()}
ANKER, HV = 0.005663383286446333, -0.01757989612499151
GR = {'kort': ['100m', '300m'], 'sprint': ['500m', '1000m'], 'midden': ['1500m', '3000m'], 'lang': ['5000m', '10000m']}
GIDX = {'kort': 0, 'sprint': 0, 'midden': 1, 'lang': 2}
g = np.full(len(D), -1); af = D.af.to_numpy()
for i, (k, a) in enumerate(GR.items()): g[np.isin(af, a)] = i
jr = D.jr.to_numpy()
U = {False: D.u.to_numpy().astype(float), True: Dd.ud.to_numpy().astype(float)}
SD = {False: np.exp(D.lsd.to_numpy().astype(float)), True: np.exp(Dd.lsdd.to_numpy().astype(float))}
LSD = {k: np.log(v) for k, v in SD.items()}
DEL = {False: D.dp.to_numpy().astype(float) + ANKER - HV, True: Dd.dd.to_numpy().astype(float) + ANKER - HV}
ok = ~D.val.to_numpy() & (g >= 0) & np.isfinite(U[False]) & np.isfinite(U[True])
p0v = np.array([M['piek_q'][0], M['piek_q'][1] - M['piek_q'][0], 1 - M['piek_q'][1]])
C = D[['c0', 'c1', 'c2']].to_numpy().astype(float); PW = (C + M['piek_m'] * p0v) / (C.sum(1, keepdims=True) + M['piek_m'])
NIV = D.niv.to_numpy().astype(float); HOOG = D.hoog.to_numpy().astype(float)
KN = np.array([0, 50, 75, 90, 97, 100.0])
B = np.zeros((len(D), len(KN)))                      # lineaire interpolatie-basis
for i in range(len(KN)):
    e = np.zeros(len(KN)); e[i] = 1; B[:, i] = np.interp(NIV, KN, e)


def unpack(p): return 1 / (1 + np.exp(-p[0])), np.exp(p[1]), 1.5 + 28.5 / (1 + np.exp(-p[2])), np.exp(p[3]), 1.5 + 28.5 / (1 + np.exp(-p[4]))
def _tl(x, nu): return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(x * x / nu)
def nll_u(p, uu):
    p0, sL, nL, sU, nU = unpack(p)
    return -np.where(uu < 0, np.log(2 * p0) + _tl(uu / sL, nL) - np.log(sL), np.log(2 * (1 - p0)) + _tl(uu / sU, nU) - np.log(sU))
from scipy.stats import t as T
def cdf(p, x):
    p0, sL, nL, sU, nU = unpack(p)
    return p0 * 2 * T.cdf(x / sL, nL) if x < 0 else p0 + (1 - p0) * (2 * T.cdf(x / sU, nU) - 1)
def piekw(p, uu, ww, gi):
    lo, hi = PD[gi]; P = np.array([cdf(p, lo), cdf(p, hi) - cdf(p, lo), 1 - cdf(p, hi)])
    rr = np.where(uu < lo, 0, np.where(uu > hi, 2, 1)); return np.log(ww[np.arange(len(uu)), rr] / P[rr])
rng = np.random.default_rng(0)
SUB = {gi: rng.choice(np.where(ok & (g == gi) & (jr <= 2019))[0], min(150000, (ok & (g == gi) & (jr <= 2019)).sum()), replace=False) for gi in range(4)}
def vorm_fit(uu, gi):
    obj = lambda p: np.mean(nll_u(p, uu[SUB[gi]]))
    r = minimize(obj, [0, 0, 1, 0, 1], method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-9})
    for _ in range(3):
        r2 = minimize(obj, r.x, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-9})
        if r2.fun >= r.fun - 1e-9: break
        r = r2
    return r.x


def shift(q, dag, sel=slice(None)):
    k = q[:6]; h = q[6:12]
    return (DEL[dag][sel] * (B[sel] @ k) + 0.01 * HOOG[sel] * (B[sel] @ h)) / SD[dag][sel]


def eerlijk(q, dag):
    uu = U[dag] - shift(q, dag); tot = np.full(len(uu), np.nan)
    for gi, gname in enumerate(GR):
        p = vorm_fit(uu, gi); te = np.where(ok & (g == gi) & (jr > 2019))[0]
        tot[te] = nll_u(p, uu[te]) + LSD[dag][te] - piekw(p, uu[te], PW[te], GIDX[gname])
    return tot, uu


TE = ok & (jr > 2019); EL = NIV >= 97; TOP = NIV >= 90
def rapport(nm, q):
    out = []
    for dag in (False, True):
        b, _ = eerlijk(np.zeros(12), dag); n, uu = eerlijk(q, dag); d = (n - b)[TE]
        h = TE & EL & (HOOG > 0); nh = TE & EL & (HOOG == 0)
        out.append(f"{'dag' if dag else 'vooraf'}: {np.nanmean(d):+.5f} top {np.nanmean(d[TOP[TE]]):+.5f} elite {np.nanmean(d[EL[TE]]):+.5f} "
                   f"| elite gem. u hoogland {np.mean(U[dag][h]):+.2f}->{np.mean(uu[h]):+.2f} overig {np.mean(U[dag][nh]):+.2f}->{np.mean(uu[nh]):+.2f}")
    print(nm, '\n   ' + '\n   '.join(out), f'{time.time()-t0:.0f}s', flush=True)


def fit(act, dag=False):
    uu0 = U[dag]; vormen = {gi: vorm_fit(uu0, gi) for gi in range(4)}
    idx = np.concatenate([rng.choice(np.where(ok & (g == gi) & (jr <= 2019))[0], min(150000, (ok & (g == gi) & (jr <= 2019)).sum()), replace=False) for gi in range(4)])
    gg = g[idx]; uu_s = uu0[idx]; pw = PW[idx]; ls = LSD[dag][idx]
    act = np.array(act, bool)
    def f(t):
        q = np.zeros(12); q[act] = t; uu = uu_s - shift(q, dag, idx); s = 0.0
        for gi, gname in enumerate(GR):
            ii = gg == gi; p = vormen[gi]
            s += np.sum(nll_u(p, uu[ii]) - piekw(p, uu[ii], pw[ii], GIDX[gname]))
        return s / len(idx)
    k = act.sum(); x0 = np.zeros(k); simp = np.vstack([x0] + [x0 + 0.1 * np.eye(k)[i] for i in range(k)])
    r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-9, 'maxiter': 4000, 'initial_simplex': simp})
    r = minimize(f, r.x, method='Powell', options={'xtol': 1e-4, 'ftol': 1e-9})
    q = np.zeros(12); q[act] = r.x; return q


if __name__ == '__main__':
    res = {}
    for nm, act in (('A: ijs x niveau', [1] * 6 + [0] * 6), ('B: hoogland x niveau', [0] * 6 + [1] * 6), ('C: beide', [1] * 12)):
        q = fit(act, dag=True); res[nm] = q.tolist()
        rapport(f'{nm} q={np.round(q, 3).tolist()}', q)
    json.dump(res, open('ijs_lab_dagfit.json', 'w'))
