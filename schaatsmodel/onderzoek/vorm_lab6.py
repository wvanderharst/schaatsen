"""Herfit van de vormlaag op exact de grootheden van het model (eerlijke log loss, vooraf en t.o.v. de dag)."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, time, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as T
t0 = time.time()
D = pd.read_pickle('vorm_lab_d.pkl'); Dd = pd.read_pickle('vorm_lab_dag.pkl').reindex(D.index)
M = json.load(open('vorm_lab_meta.json')); PD = {int(k): v for k, v in M['piek_drempel'].items()}
GR = {'kort': ['100m', '300m'], 'sprint': ['500m', '1000m'], 'midden': ['1500m', '3000m'], 'lang': ['5000m', '10000m']}
GIDX = {'kort': 0, 'sprint': 0, 'midden': 1, 'lang': 2}
g = np.full(len(D), -1); af = D.af.to_numpy()
for i, (k, a) in enumerate(GR.items()): g[np.isin(af, a)] = i
jr = D.jr.to_numpy(); ok = ~D.val.to_numpy() & (g >= 0) & np.isfinite(D.u.to_numpy()) & np.isfinite(Dd.ud.to_numpy())
p0v = np.array([M['piek_q'][0], M['piek_q'][1] - M['piek_q'][0], 1 - M['piek_q'][1]])
C = D[['c0', 'c1', 'c2']].to_numpy().astype(float); PW = (C + M['piek_m'] * p0v) / (C.sum(1, keepdims=True) + M['piek_m'])
W = np.clip((D.niv.to_numpy() - 80) / 20, 0, 1)
X = {k: D[k].to_numpy().astype(float) for k in ('Zk', 'F', 'S', 'Zl', 'u', 'lsd', 'r', 'rel')}
X['ud'] = Dd.ud.to_numpy().astype(float); X['lsdd'] = Dd.lsdd.to_numpy().astype(float)
BF = M['bf']; TOPN = D.niv.to_numpy()


def eff(q):
    aK0, aK1, aL0, aL1, a, aS, bf, bs, bsS, bf1 = q
    mu = (aK0 + aK1 * W) * X['Zk'] + (aL0 + aL1 * W) * X['Zl'] + a * X['F'] + aS * X['S']
    return mu, (bf + bf1 * W) * X['F'], bs * X['F'] + bsS * X['S']


def transform(q, dag):
    mu, lf, ls = eff(q)
    if dag:
        u1 = X['ud'] - mu; lam = np.where(u1 < 0, lf, ls); return u1 * np.exp(-lam), X['lsdd'] + lam
    r = np.minimum(X['r'], 1.0); u1 = X['u'] - mu * np.sqrt(r)
    lamf = 0.5 * np.log(np.maximum(1 + r * np.expm1(2 * lf), 1e-4)); lams = 0.5 * np.log(np.maximum(1 + r * np.expm1(2 * ls), 1e-4))
    lam = np.where(u1 < 0, lamf + BF * X['rel'], lams)
    return u1 * np.exp(-lam), X['lsd'] + lam


def unpack(p):
    return 1 / (1 + np.exp(-p[0])), np.exp(p[1]), 1.5 + 28.5 / (1 + np.exp(-p[2])), np.exp(p[3]), 1.5 + 28.5 / (1 + np.exp(-p[4]))


from scipy.special import gammaln
def _tl(x, nu):
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(x * x / nu)


def nll_u(p, uu):
    p0, sL, nL, sU, nU = unpack(p)
    return -np.where(uu < 0, np.log(2 * p0) + _tl(uu / sL, nL) - np.log(sL), np.log(2 * (1 - p0)) + _tl(uu / sU, nU) - np.log(sU))


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


def eerlijk(q, dag, per_rij=False, vormen=None):
    uu, ls = transform(q, dag); tot = np.full(len(uu), np.nan)
    for gi, gname in enumerate(GR):
        p = vormen[gi] if vormen is not None else vorm_fit(uu, gi)
        te = np.where(ok & (g == gi) & (jr > 2019))[0]
        tot[te] = nll_u(p, uu[te]) + ls[te] - piekw(p, uu[te], PW[te], GIDX[gname])
    return tot


def fitll(q, dag, vormen, idx):
    """log-lik op fit-rijen met vaste vormen (voor de optimalisatie)."""
    uu, ls = transform_sub(q, dag); s = 0.0
    for gi, gname in enumerate(GR):
        ii = idx[gi]; p = vormen[gi]
        s += np.sum(nll_u(p, uu[ii]) + ls[ii] - piekw(p, uu[ii], PW[ii], GIDX[gname]))
    return s / sum(len(i) for i in idx.values())


ALL = None
def transform_sub(q, dag):
    global X, W
    Xf, Wf = X, W
    X = XS; W = WS
    try:
        return transform(q, dag)
    finally:
        X, W = Xf, Wf


Q0 = np.array(M['vorm_par'][4:14]); Z = np.zeros(10)
TE = ok & (jr > 2019); top = TOPN >= 90; eli = TOPN >= 97
def rapport(nm, q):
    out = []
    for dag in (False, True):
        b = eerlijk(Z, dag); n = eerlijk(q, dag); d = (n - b)[TE]
        out.append(f"{'dag' if dag else 'vooraf'}: {np.nanmean(n[TE]):.5f} ({np.nanmean(d):+.5f}, top {np.nanmean(d[top[TE]]):+.5f}, elite {np.nanmean(d[eli[TE]]):+.5f})")
    print(nm, ' | '.join(out), f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    res = {}
    for dag in (False, True):
        uu, _ = transform(Z, dag); vormen = {gi: vorm_fit(uu, gi) for gi in range(4)}
        idx = {gi: rng.choice(np.where(ok & (g == gi) & (jr <= 2019) & (jr >= 2010))[0], min(120000, (ok & (g == gi) & (jr <= 2019)).sum()), replace=False) for gi in range(4)}
        alle = np.concatenate([idx[gi] for gi in range(4)]); pos = {}; k = 0
        for gi in range(4):
            pos[gi] = np.arange(k, k + len(idx[gi])); k += len(idx[gi])
        global XS, WS, PW_FULL
        XS = {kk: v[alle] for kk, v in X.items()}; WS = W[alle]
        PWs = PW[alle]
        def f(q):
            uu, ls = transform_sub(q, dag); s_ = 0.0
            for gi, gname in enumerate(GR):
                ii = pos[gi]; p = vormen[gi]
                s_ += np.sum(nll_u(p, uu[ii]) + ls[ii] - piekw(p, uu[ii], PWs[ii], GIDX[gname]))
            return s_ / len(alle)
        x0 = np.zeros(10); simp = np.vstack([x0] + [x0 + 0.03 * np.eye(10)[i] for i in range(10)])
        r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-9, 'maxiter': 4000, 'initial_simplex': simp})
        r = minimize(f, r.x, method='Powell', options={'xtol': 1e-5, 'ftol': 1e-9})
        res['dag' if dag else 'vooraf'] = r.x.tolist()
        print('geschat op', 'dag' if dag else 'vooraf', np.round(r.x, 4).tolist(), flush=True)
        rapport('  ->', r.x)
    json.dump(res, open('vorm_lab6.json', 'w'))
