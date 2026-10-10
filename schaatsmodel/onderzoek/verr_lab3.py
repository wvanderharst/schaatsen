"""Vormtoestand op dag-z: kort (niveau, sterker bij toppers) + lang (volatiliteit na uitschieters)."""
import warnings; warnings.filterwarnings('ignore')
import sys, time, json, numpy as np, pandas as pd
from scipy import optimize
import verr_lab2 as L
t0 = time.time()
ZD, FIT, TEST, TOP = L.ZD, L.FIT, L.TEST, L.TOP
AB = L.d['abs'].to_numpy()
W = np.clip((AB - 80) / 20, 0, 1)                      # 0 onder niveau 80, 1 bij de top
LOG2PI = L.LOG2PI


def kenm_all(Hk, c, Hl, cs):
    _, _, Zk = L.kenm(c, Hk, cs)                         # korte gemiddelde dag-z
    F, S, Zl = L.kenm(c, Hl, cs)                         # lange uitschieter-sommen
    return Zk, F, S, Zl


def ll(th, X, sel):
    aK0, aK1, aL0, aL1, a, aS, bf, bs, bsS, bf1 = th
    Zk, F, S, Zl = (x[sel] for x in X); w = W[sel]
    mu = (aK0 + aK1 * w) * Zk + (aL0 + aL1 * w) * Zl + a * F + aS * S
    lf = (bf + bf1 * w) * F; ls = bs * F + bsS * S
    x = ZD[sel] - mu; sf = np.exp(lf); ss = np.exp(ls); s = np.where(x < 0, sf, ss)
    return -LOG2PI - np.log((sf + ss) / 2) - 0.5 * (x / s) ** 2


def fit(X):
    f = lambda t: -ll(t, X, FIT).mean()
    x0 = np.zeros(10); simp = np.vstack([x0] + [x0 + 0.05 * np.eye(10)[i] for i in range(10)])
    r = optimize.minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-9, 'maxiter': 6000, 'initial_simplex': simp})
    r = optimize.minimize(f, r.x, method='Powell', options={'xtol': 1e-5, 'ftol': 1e-9})
    th = r.x; b = ll(np.zeros(10), X, TEST); n = ll(th, X, TEST); dd = -(n - b)
    return th, dd.mean(), dd[TOP[TEST]].mean(), dd[(AB[TEST] >= 97)].mean()


if __name__ == '__main__':
    rows = []
    for Hk in (3, 7, 14):
        for c, Hl in ((2.0, 240), (2.5, 240), (2.5, 480)):
            X = kenm_all(Hk, c, Hl, 2.5)
            th, g, gt, ge = fit(X)
            rows.append(dict(Hk=Hk, c=c, Hl=Hl, test=round(g, 5), top=round(gt, 5), elite=round(ge, 5), th=np.round(th, 4).tolist()))
            print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
    json.dump(rows, open('verr_lab3.json', 'w'))
