"""Lopende ijking van de snelle staart (laatste laag van het model).

De vorm van de snelle staart hangt niet-lineair af van het niveau en verschuift in de tijd (toppers 90-97: in de fitjaren goed, daarna
steeds minder snelle uitschieters dan voorspeld). Twee geneste stappen, elk p' = Q (p/Q)^b voor p < Q (Q = 10%, daarna 1%):
b(g) is een gladde functie van het niveau g (Gauss-kernel, bandbreedte bw, geen groepen), geschat uit EERDERE dagen. Als p' uniform
is, is x = -ln(p/Q) exponentieel met rate b, dus b(g0) = (sum K w + k0) / (sum K w x + k0), w = 0,5^(leeftijd / hl), krimp naar 1.
De log loss verandert exact met -ln b - (b - 1) ln(p/Q). Eigen valpartijen tellen niet mee in de schatting."""
import numpy as np
from numba import njit

G = np.linspace(-0.03, 0.45, 49)


@njit(cache=True)
def _ijk(dag, g, x, gebruik, G, bw, hl, k0, out):
    n = len(dag); m = len(G); S1 = np.zeros(m); Sx = np.zeros(m); laatst = -1e9; i = 0
    while i < n:
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        if laatst > -1e8:
            f = 0.5 ** ((dag[i] - laatst) / hl)
            for k in range(m):
                S1[k] *= f; Sx[k] *= f
        for q in range(i, j):                        # eerst voorspellen met eerdere dagen, dan bijwerken
            pos = (g[q] - G[0]) / (G[1] - G[0])
            k = int(min(max(pos, 0.0), m - 1.001)); t = min(max(pos - k, 0.0), 1.0)
            out[q] = (1 - t) * (S1[k] + k0) / (Sx[k] + k0) + t * (S1[k + 1] + k0) / (Sx[k + 1] + k0)
        for q in range(i, j):
            if gebruik[q]:
                for k in range(m):
                    kk = np.exp(-0.5 * ((g[q] - G[k]) / bw) ** 2)
                    S1[k] += kk; Sx[k] += kk * x[q]
        laatst = dag[i]; i = j


def stap(dag, g, p, gebruik, Q, hl=365.0, bw=0.02, k0=200.0):
    """Een stap op rijen in willekeurige volgorde; geeft p', verandering in de log loss en b."""
    o = np.argsort(dag, kind='stable')
    pc = np.clip(np.nan_to_num(p, nan=1.0), 1e-300, 1.0)
    staart = (pc < Q) & np.isfinite(p); x = np.where(staart, -np.log(pc / Q), 0.0)
    b = np.ones(len(p)); bo = np.ones(len(p))
    _ijk(dag[o].astype(np.float64), np.clip(np.nan_to_num(g[o], nan=0.2), -0.03, 0.45), x[o], (staart & gebruik)[o], G, bw, hl, k0, bo)
    b[o] = bo
    p2 = np.where(staart, Q * (pc / Q) ** b, p)
    d = np.where(staart, -np.log(b) - (b - 1) * np.log(pc / Q), 0.0)
    return p2, d, b


def ijk(dag, g, p, gebruik, Qs=(0.1, 0.01), **kw):
    """Geneste stappen; geeft de nieuwe PIT, de verandering in de log loss en de b per stap (voor de kwantielen)."""
    d_tot = np.zeros(len(p)); bs = []
    for Q in Qs:
        p, d, b = stap(dag, g, p, gebruik, Q, **kw); d_tot += d; bs.append(b)
    return p, d_tot, bs


def terug(q, bs, Qs=(0.1, 0.01)):
    """Van een gewenste eind-PIT q terug naar de PIT voor de ijking (omgekeerde volgorde)."""
    q = np.asarray(q, float).copy()
    for Q, b in zip(Qs[::-1], bs[::-1]):
        q = np.where(q < Q, Q * (np.clip(q, 1e-300, 1) / Q) ** (1.0 / b), q)
    return q
