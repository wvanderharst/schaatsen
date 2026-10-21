"""Vormlaag op de kansschaal: vangt vormpieken en onrustige periodes die het filter en de correcties missen.

z = Phi^-1(PIT) van het model. De vormlaag maakt z' = (z - mu) / sigma, met mu en log sigma lineair in kenmerken uit EERDERE dagen:
- rijder: uitdovend gemiddelde van z (tijdconstante 7 / 30 / 90 dagen, gekrompen), snelle en trage schokken (het deel van z voorbij 2),
  extra variantie (z^2 - 1) en het aantal ritten;
- veld: het gemiddelde van het rijdergeheugen over de startlijst (een circuit dat recent sneller of trager rijdt dan voorspeld).
Alles ook x exp(-niveau / 4%), zodat de top eigen coefficienten krijgt.

Dit is een geldige verdeling voor de tijd (Gauss-copula op de modelverdeling): de log loss verandert exact met
log phi(z) - log phi(z') + log sigma. Eigen valpartijen (filtervlag) tellen niet mee in het geheugen en niet in de fit."""
import numpy as np, pandas as pd
from numba import njit
from scipy.stats import norm
from scipy.optimize import minimize

TS = (7.0, 30.0, 90.0)
LOC = ['m7', 'm30', 'm90', 'neg7', 'neg30', 'neg90', 'pos7', 'pos30', 'pos90', 'veld_m7', 'veld_m30', 'veld_m90']
SCH = ['v7', 'v30', 'v90', 'n30']


@njit(cache=True)
def _geheugen(rid, dag, X, T, out, cnt):
    """Rijen gesorteerd per rijder en dag. out[i] = som over EERDERE dagen van de rijder van exp(-dt / T) * X[j] (rijen met NaN tellen niet)."""
    n, p = X.shape
    s = np.zeros(p); c = 0.0; td = -1e9; ds = np.zeros(p); dc = 0.0; dd = -1e9; r0 = -1
    for i in range(n):
        if rid[i] != r0:
            s[:] = 0.0; c = 0.0; td = -1e9; ds[:] = 0.0; dc = 0.0; dd = -1e9; r0 = rid[i]
        if dag[i] != dd:
            if dd > -1e8:                                  # dag dd afsluiten
                f = np.exp(-(dd - td) / T) if td > -1e8 else 0.0
                for k in range(p):
                    s[k] = s[k] * f + ds[k]
                c = c * f + dc; td = dd
            ds[:] = 0.0; dc = 0.0; dd = dag[i]
        f = np.exp(-(dag[i] - td) / T) if td > -1e8 else 0.0
        for k in range(p):
            out[i, k] = s[k] * f
        cnt[i] = c * f
        ok = True
        for k in range(p):
            if X[i, k] != X[i, k]:
                ok = False
        if ok:
            for k in range(p):
                ds[k] += X[i, k]
            dc += 1.0


def kenmerken(rijder, dag, z, val, race, k0=2.0):
    """Vormkenmerken per rit (willekeurige rijvolgorde). z NaN = geen uitslag (telt niet mee, krijgt wel kenmerken)."""
    rijder = np.asarray(rijder); dag = np.asarray(dag, float); z = np.asarray(z, float); val = np.asarray(val, bool)
    rid = pd.factorize(rijder)[0].astype(np.int64)
    o = np.lexsort((dag, rid))
    zz = np.where(val, np.nan, z)[o]
    X = np.column_stack([zz, np.minimum(zz + 2, 0), np.maximum(zz - 2, 0), np.minimum(zz ** 2, 25) - 1])
    F = {}
    for T in TS:
        out = np.zeros_like(X); cnt = np.zeros(len(o))
        _geheugen(rid[o], dag[o], X, T, out, cnt)
        t = int(T)
        for nm, v in ((f'm{t}', out[:, 0] / (cnt + k0)), (f'neg{t}', out[:, 1]), (f'pos{t}', out[:, 2]), (f'v{t}', out[:, 3] / (cnt + k0)), (f'n{t}', np.log1p(cnt))):
            a = np.empty(len(o)); a[o] = v; F[nm] = a
    F = pd.DataFrame(F)
    r = pd.Series(np.asarray(race))
    for t in (int(T) for T in TS):                       # veld: gemiddelde rijdergeheugen over de startlijst
        F[f'veld_m{t}'] = F[f'm{t}'].groupby(r.to_numpy()).transform('mean').to_numpy()
    return F


def ontwerp(F, wg, loc=LOC, sch=SCH):
    Xm = np.column_stack([np.ones(len(F))] + [F[c].to_numpy() for c in loc] + [wg] + [F[c].to_numpy() * wg for c in loc])
    Xs = np.column_stack([np.ones(len(F))] + [F[c].to_numpy() for c in sch] + [wg] + [F[c].to_numpy() * wg for c in sch])
    nm_m = ['1'] + list(loc) + ['w'] + [c + '_w' for c in loc]; nm_s = ['1'] + list(sch) + ['w'] + [c + '_w' for c in sch]
    return np.nan_to_num(Xm), np.nan_to_num(Xs), nm_m, nm_s


def delta_nll(p, z, Xm, Xs):
    km = Xm.shape[1]; ls = Xs @ p[km:]; zp = (z - Xm @ p[:km]) * np.exp(-ls)
    return 0.5 * zp ** 2 + ls - 0.5 * z ** 2              # verandering in de log loss per rit (negatief = beter)


def fit(z, Xm, Xs, w=None, ridge=1e-3):
    w = np.ones(len(z)) if w is None else w; w = w / w.mean(); km = Xm.shape[1]
    def f(p):
        ls = Xs @ p[km:]; e = np.exp(-ls); zp = (z - Xm @ p[:km]) * e
        v = np.mean(w * (0.5 * zp ** 2 + ls - 0.5 * z ** 2)) + ridge * np.sum(p ** 2)
        g = np.r_[-(w * zp * e) @ Xm, (w * (1 - zp ** 2)) @ Xs] / len(z) + 2 * ridge * p
        return v, g
    return minimize(f, np.zeros(km + Xs.shape[1]), jac=True, method='L-BFGS-B').x


def pas_toe(p, z, Xm, Xs):
    """Geeft mu, sigma, de nieuwe PIT en de verandering in de log loss."""
    km = Xm.shape[1]; mu = Xm @ p[:km]; sg = np.exp(Xs @ p[km:])
    zp = (z - mu) / sg
    return mu, sg, norm.cdf(zp), 0.5 * zp ** 2 + np.log(sg) - 0.5 * z ** 2
