"""Score-gedreven (GAS) persoonlijke kalibratie op de kansschaal: per rijder een eigen verschuiving m en log-spreiding h van z = Phi^-1(p),
robuust bijgewerkt na elke rit (Student-t-score: gewicht w = (nu + 1) / (nu + u^2), dus uitschieters tellen minder):
  e = z - m, u = e / exp(h);  m <- m + a w e;  h <- h + b (w u^2 - 1);  tussen ritten uitdovend naar 0 met halfwaardetijd hl dagen.
Voorspellen: z' = (z - m) / exp(h); de log loss verandert exact met h + (z'^2 - z^2) / 2. Alleen eerdere dagen; eigen valpartijen tellen
niet mee in de update (wel in de beoordeling)."""
import numpy as np
from numba import njit
from scipy.stats import norm


@njit(cache=True)
def _gas(dag, rid, z, gebruik, a, b, nu, hl, nr, mo, ho):
    n = len(z); m = np.zeros(nr); h = np.zeros(nr); tl = np.full(nr, -1e9); i = 0
    while i < n:                                       # rijen gesorteerd op dag
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        for q in range(i, j):
            r = rid[q]
            f = 0.5 ** ((dag[q] - tl[r]) / hl) if tl[r] > -1e8 else 0.0
            mo[q] = m[r] * f; ho[q] = h[r] * f
        for q in range(i, j):
            if gebruik[q]:
                r = rid[q]
                f = 0.5 ** ((dag[q] - tl[r]) / hl) if tl[r] > -1e8 else 0.0
                m[r] *= f; h[r] *= f
                e = min(max(z[q], -4.0), 4.0) - m[r]; s = np.exp(h[r]); u2 = (e / s) ** 2
                w = (nu + 1.0) / (nu + u2)
                m[r] += a * w * e
                h[r] = min(max(h[r] + b * (min(w * u2, 16.0) - 1.0), -1.0), 1.0)
                tl[r] = dag[q]
        i = j


def laag(dag, rijder_id, p, gebruik, a=0.03, b=0.02, nu=5.0, hl=365.0, rij=None):
    """Rijen in willekeurige volgorde; geeft p', verandering in de log loss, m en h (0 buiten rij)."""
    ok = np.isfinite(p)
    rij = ok if rij is None else rij
    idx = np.flatnonzero(rij); o = idx[np.argsort(dag[idx], kind='stable')]
    _, rid = np.unique(np.asarray(rijder_id)[o], return_inverse=True)
    z = norm.ppf(np.clip(p, 1e-15, 1 - 1e-15))
    mo = np.zeros(len(o)); ho = np.zeros(len(o))
    _gas(np.asarray(dag, float)[o], rid.astype(np.int64), np.nan_to_num(z[o]), (gebruik & ok)[o], a, b, nu, hl, int(rid.max()) + 1, mo, ho)
    m = np.zeros(len(p)); h = np.zeros(len(p)); m[o] = mo; h[o] = ho
    z2 = (z - m) / np.exp(h)
    return np.where(ok, norm.cdf(z2), p), np.where(ok, h + 0.5 * (z2 ** 2 - z ** 2), 0.0), m, h
