"""Stap 2: piekgrootte per rijder met lang geheugen (jaren): gekrompen tempo van snelle overschrijdingen per wedstrijddag."""
import warnings; warnings.filterwarnings('ignore')
import sys, json, time, numpy as np, pandas as pd
from numba import njit
from scipy.optimize import minimize
import vorm_lab6 as L
from gas_lab import vorm_pass
t0 = time.time()
ZB = pd.read_pickle('zbase.pkl')
o = np.lexsort((ZB.afstand_idx.to_numpy(), ZB.event_id.to_numpy(), ZB.dag.to_numpy(), ZB.skater_id.to_numpy()))
SK = ZB.skater_id.to_numpy()[o].astype(np.int64); DG = ZB.dag.to_numpy()[o].astype(np.int64)
zz = ZB.z.to_numpy().astype(float); VAL = (~np.isfinite(zz) | (zz > 6))[o]; ZO = np.where(np.isfinite(zz), zz, 0)[o]
IDX = ZB.index[o]


@njit(cache=True)
def piek_pass(sk, dg, z, val, c, H, uit):
    """uit[:,0] = som exp(-dt/H) * max(0, -min_dag z - c); uit[:,1] = som exp(-dt/H) * 1 (wedstrijddagen); toestand VOOR de dag."""
    n = len(sk); e = 0.0; k = 0.0; ld = -1; i = 0
    while i < n:
        j = i
        while j < n and sk[j] == sk[i] and dg[j] == dg[i]:
            j += 1
        if i == 0 or sk[i] != sk[i - 1]:
            e = 0.0; k = 0.0
        else:
            f = np.exp(-(dg[i] - ld) / H); e *= f; k *= f
        ld = dg[i]
        for q in range(i, j):
            uit[q, 0] = e; uit[q, 1] = k
        a = 0.0; ok = False
        for q in range(i, j):
            if val[q]:
                continue
            ok = True
            if -z[q] - c > a:
                a = -z[q] - c
        if ok:
            e += a; k += 1.0
        i = j
    return 0


def zet_vorm():
    uit = np.zeros((len(o), 5)); a = np.zeros(len(o), np.int64); b = np.zeros(len(o), np.int64); s = np.zeros((len(o), 5))
    vorm_pass(SK, DG, ZO, VAL, 7.0, 2.5, 60.0, 2.5, uit, a, b, s)
    F = pd.DataFrame(uit, index=IDX, columns=['Zk', 'F', 'S', 'Zl', 'N']).reindex(L.D.index)
    for k in F.columns: L.X[k] = F[k].to_numpy()


def piek_feat(c, H):
    uit = np.zeros((len(o), 2)); piek_pass(SK, DG, ZO, VAL, c, H, uit)
    F = pd.DataFrame(uit, index=IDX, columns=['E', 'K']).reindex(L.D.index)
    return F.E.to_numpy(), F.K.to_numpy()


zet_vorm()
Q = np.array(json.load(open('params_getuned.json'))['vorm_par'], float)
r0_cache = {}


def transform(th, dag, E, K, m, r0, mode, sel=slice(None)):
    """mode 'erbij': huidige vormlaag + P; 'vervang': snelle-kant termen F en N eruit, P erin."""
    X = {k: v[sel] for k, v in L.X.items()}; W = L.W[sel]
    aK0, aK1, aL0, aL1, a, aS, bf, bs, bsS, bf1 = Q[4:14]; gN = Q[14]
    if mode == 'vervang':
        bf = bf1 = gN = 0.0
    gp, ap, gp1 = th
    P = (E[sel] + m * r0) / (K[sel] + m) / r0 - 1.0
    mu = (aK0 + aK1 * W) * X['Zk'] + (aL0 + aL1 * W) * X['Zl'] + a * X['F'] + aS * X['S'] + ap * P
    lf = (bf + bf1 * W) * X['F'] + gN * np.sqrt(X['N']) + (gp + gp1 * W) * P
    ls = bs * X['F'] + bsS * X['S']
    if dag:
        u1 = X['ud'] - mu; lam = np.where(u1 < 0, lf, ls); return u1 * np.exp(-lam), X['lsdd'] + lam
    r = np.minimum(X['r'], 1.0); u1 = X['u'] - mu * np.sqrt(r)
    lamf = 0.5 * np.log(np.maximum(1 + r * np.expm1(2 * lf), 1e-4)); lams = 0.5 * np.log(np.maximum(1 + r * np.expm1(2 * ls), 1e-4))
    lam = np.where(u1 < 0, lamf + L.BF * X['rel'], lams)
    return u1 * np.exp(-lam), X['lsd'] + lam


def eerlijk(th, dag, *a):
    uu, ls = transform(th, dag, *a); tot = np.full(len(uu), np.nan)
    for gi, gname in enumerate(L.GR):
        p = L.vorm_fit(uu, gi); te = np.where(L.ok & (L.g == gi) & (L.jr > 2019))[0]
        tot[te] = L.nll_u(p, uu[te]) + ls[te] - L.piekw(p, uu[te], L.PW[te], L.GIDX[gname])
    return tot


rng = np.random.default_rng(0)
IDXF = np.concatenate([rng.choice(np.where(L.ok & (L.g == gi) & (L.jr <= 2019))[0], min(150000, (L.ok & (L.g == gi) & (L.jr <= 2019)).sum()), replace=False) for gi in range(4)])


def fit(dag, E, K, m, r0, mode):
    uu0, _ = transform(np.zeros(3), dag, E, K, m, r0, mode); vormen = {gi: L.vorm_fit(uu0, gi) for gi in range(4)}
    gg = L.g[IDXF]; pw = L.PW[IDXF]
    def f(th):
        uu, ls = transform(th, dag, E, K, m, r0, mode, IDXF); s = 0.0
        for gi, gname in enumerate(L.GR):
            ii = gg == gi; p = vormen[gi]
            s += np.sum(L.nll_u(p, uu[ii]) + ls[ii] - L.piekw(p, uu[ii], pw[ii], L.GIDX[gname]))
        return s / len(IDXF)
    x0 = np.zeros(3); simp = np.vstack([x0] + [x0 + 0.05 * np.eye(3)[i] for i in range(3)])
    r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-9, 'maxiter': 600, 'initial_simplex': simp})
    return r.x


TE = L.TE; top = L.TOPN >= 90; eli = L.TOPN >= 97
BASE = {dag: eerlijk(np.zeros(3), dag, np.zeros(len(L.D)), np.zeros(len(L.D)), 1.0, 1.0, 'erbij') for dag in (False, True)}
print('basis (huidige vormlaag) klaar', f'{time.time()-t0:.0f}s', flush=True)
res = []
for mode in ('erbij', 'vervang'):
    for c, H in ((2.0, 730), (2.0, 1460), (1.5, 1460), (2.5, 1460)):
        E, K = piek_feat(c, H); okk = L.ok & (L.jr <= 2019) & (K > 0); r0 = E[okk].sum() / K[okk].sum()
        for m in (20.0,):
            th = fit(True, E, K, m, r0, mode)
            out = []
            for dag in (False, True):
                d = (eerlijk(th, dag, E, K, m, r0, mode) - BASE[dag])[TE]
                out.append(f"{'dag' if dag else 'vooraf'} {np.nanmean(d):+.5f} top {np.nanmean(d[top[TE]]):+.5f} elite {np.nanmean(d[eli[TE]]):+.5f}")
            res.append(dict(mode=mode, c=c, H=H, m=m, r0=r0, th=th.tolist()))
            print(mode, c, H, m, 'r0', round(r0, 4), 'th', np.round(th, 3).tolist(), ' | '.join(out), f'{time.time()-t0:.0f}s', flush=True)
json.dump(res, open('piek_lab.json', 'w'))
