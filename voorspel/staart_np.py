"""Niet-parametrische lopende ijking van de kansen (PIT), per niveau (Gauss-kernel) en per afstand (gekrompen naar alle afstanden samen),
en de staartmassa per rijder. Doel: het aantal ritten sneller (en trager) dan 1/10, 1/20, 1/100, 1/1000 klopt op elk niveau, op elke afstand
en per rijder.

ijk: p' = G(p), G monotoon, vastgelegd op knopen x = 0,5 / 0,3 / 0,2 / 0,1 / 0,05 / 0,02 / 0,01 / ... / 1e-4 aan beide kanten (snel: p,
traag: 1 - p). Uit eerdere dagen (gewicht 0,5^(dagen / hl) x kernel in niveau): de massa m = P(p <= 0,5), en per knoop de voorwaardelijke
kans P(p <= x_j | p <= x_j-1), gekrompen naar de geijkte verhouding x_j / x_j-1 met k0 pseudo-ritten (alle afstanden samen); per afstand
gekrompen naar die schatting met k1 pseudo-ritten. Tussen de knopen een machtsfunctie (recht in log p, log G), voorbij de laatste knoop de
laatste helling. De log loss verandert exact met -ln G'(p).

rijder: lambda = (waargenomen + k) / (verwacht + k) snelle ritten (p < Q) in eerdere ritten van de rijder (uitdovend); p' = lambda p voor
p < Q, daarna lineair tot 1. Vangt de clustering: jonge, snel stijgende rijders (Wennemars, Stolz, Daleman) rijden jarenlang vaker snel
dan het niveau zegt. Eigen valpartijen tellen in beide niet mee in de schatting."""
import numpy as np
from numba import njit

GN = np.linspace(-0.03, 0.45, 49)
X = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 3e-4, 1e-4])
AFSTANDEN = ['500m', '1000m', '1500m', '3000m', '5000m', '10000m']
QS = np.array([0.1, 0.5, 0.9])                         # kwantielen waarvoor de omkering per rit wordt bewaard


@njit(cache=True)
def _keten(c, cp, X, k0, k1, G, G0):
    """Waarden op de knopen: G[0] = G0, daarna voorwaardelijke verhoudingen (eerst alle afstanden, dan deze afstand)."""
    G[0] = G0
    for j in range(1, len(X)):
        rp = (cp[j] + k0 * X[j] / X[j - 1]) / (cp[j - 1] + k0)
        rd = (c[j] + k1 * rp) / (c[j - 1] + k1)
        G[j] = G[j - 1] * rd


@njit(cache=True)
def _seg(x, X, G):
    """G(x) en helling s (d ln G / d ln x) voor x <= 0,5."""
    m = len(X); j = 0
    while j < m - 1 and x < X[j + 1]:
        j += 1
    if j < m - 1:
        s = np.log(G[j + 1] / G[j]) / np.log(X[j + 1] / X[j])
    else:
        s = np.log(G[m - 1] / G[m - 2]) / np.log(X[m - 1] / X[m - 2])
    return G[j] * (x / X[j]) ** s, s


@njit(cache=True)
def _seg_inv(y, X, G):
    """Omgekeerde van _seg: de x <= 0,5 met G(x) = y."""
    m = len(X); j = 0
    while j < m - 1 and y < G[j + 1]:
        j += 1
    if j < m - 1:
        s = np.log(G[j + 1] / G[j]) / np.log(X[j + 1] / X[j])
    else:
        s = np.log(G[m - 1] / G[m - 2]) / np.log(X[m - 1] / X[m - 2])
    return X[j] * (y / G[j]) ** (1.0 / s)


@njit(cache=True)
def _ijk(dag, g, d, p, gebruik, GN, X, QS, bw, hl, k0, k1, nd, p_o, d_o, inv_o):
    n = len(p); a_ = len(GN); m = len(X)
    CL = np.zeros((a_, nd, m)); CU = np.zeros((a_, nd, m)); N = np.zeros((a_, nd))
    cl = np.zeros(m); cu = np.zeros(m); clp = np.zeros(m); cup = np.zeros(m); GL = np.zeros(m); GU = np.zeros(m)
    laatst = -1e9; i = 0
    while i < n:                                       # rijen gesorteerd op dag
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        if laatst > -1e8:
            f = 0.5 ** ((dag[i] - laatst) / hl)
            CL *= f; CU *= f; N *= f
        for q in range(i, j):                          # eerst voorspellen met eerdere dagen
            pos = (g[q] - GN[0]) / (GN[1] - GN[0]); k = int(min(max(pos, 0.0), a_ - 1.001)); t = min(max(pos - k, 0.0), 1.0)
            nn = 0.0; npool = 0.0
            for jj in range(m):
                cl[jj] = 0.0; cu[jj] = 0.0; clp[jj] = 0.0; cup[jj] = 0.0
            for h in range(2):
                aa = k + h; wa = (1.0 - t) if h == 0 else t
                nn += wa * N[aa, d[q]]
                for dd in range(nd):
                    npool += wa * N[aa, dd]
                for jj in range(m):
                    cl[jj] += wa * CL[aa, d[q], jj]; cu[jj] += wa * CU[aa, d[q], jj]
                    for dd in range(nd):
                        clp[jj] += wa * CL[aa, dd, jj]; cup[jj] += wa * CU[aa, dd, jj]
            gp = (clp[0] + k0 * X[0]) / (npool + k0); mm = (cl[0] + k1 * gp) / (nn + k1)
            _keten(cl, clp, X, k0, k1, GL, mm)
            _keten(cu, cup, X, k0, k1, GU, 1.0 - mm)
            for c in range(len(QS)):                   # omkering: welke p geeft p' = QS[c]
                if QS[c] <= mm:
                    inv_o[q, c] = _seg_inv(QS[c], X, GL) if QS[c] < mm else 0.5
                else:
                    inv_o[q, c] = 1.0 - _seg_inv(1.0 - QS[c], X, GU)
            if np.isfinite(p[q]):
                pq = min(max(p[q], 1e-300), 1.0 - 1e-16)
                if pq <= X[0]:
                    gq, s = _seg(pq, X, GL); p_o[q] = gq; d_o[q] = -np.log(s) - np.log(gq / pq)
                else:
                    gq, s = _seg(1.0 - pq, X, GU); p_o[q] = 1.0 - gq; d_o[q] = -np.log(s) - np.log(gq / (1.0 - pq))
            else:
                p_o[q] = np.nan; d_o[q] = 0.0
        for q in range(i, j):                          # dan bijwerken
            if gebruik[q]:
                for a in range(a_):
                    kk = np.exp(-0.5 * ((g[q] - GN[a]) / bw) ** 2)
                    if kk < 1e-4:
                        continue
                    N[a, d[q]] += kk
                    for jj in range(m):
                        if p[q] <= X[jj]:
                            CL[a, d[q], jj] += kk
                        if 1.0 - p[q] <= X[jj]:
                            CU[a, d[q], jj] += kk
        laatst = dag[i]; i = j


def pd_codes(afstand):
    a = np.asarray(afstand).astype(str)
    d = np.full(len(a), -1, dtype=np.int64)
    for i, nm in enumerate(AFSTANDEN):
        d[a == nm] = i
    return d


def ijk(dag, g, afstand, p, gebruik, bw=0.02, hl=730.0, k0=30.0, k1=100.0, rij=None):
    """Rijen in willekeurige volgorde. rij: welke rijen een ijking krijgen (standaard: eindige p); gebruik: welke de schatting bijwerken.
    Geeft p', de verandering in de log loss en per rit de p die naar de kwantielen QS gaat (NaN buiten rij en andere afstanden)."""
    d = pd_codes(afstand)
    rij = np.isfinite(p) if rij is None else rij
    sel = rij & (d >= 0)
    idx = np.flatnonzero(sel); o = idx[np.argsort(dag[idx], kind='stable')]
    po = np.zeros(len(o)); do = np.zeros(len(o)); io = np.zeros((len(o), len(QS)))
    _ijk(dag[o].astype(np.float64), np.clip(np.nan_to_num(g[o], nan=0.2), -0.03, 0.45), d[o].astype(np.int64), np.asarray(p, float)[o],
         (gebruik & np.isfinite(p))[o], GN, X, QS, bw, hl, k0, k1, len(AFSTANDEN), po, do, io)
    p2 = np.array(p, dtype=float, copy=True); dl = np.zeros(len(p)); inv = np.full((len(p), len(QS)), np.nan)
    p2[o] = po; dl[o] = do; inv[o] = io
    return p2, dl, inv


@njit(cache=True)
def _rijder(dag, rid, p, gebruik, Q, hl, k, nr, lam_o):
    O = np.zeros(nr); E = np.zeros(nr); L = np.full(nr, -1e9); n = len(p); i = 0
    while i < n:                                       # rijen gesorteerd op dag
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        for q in range(i, j):                          # eerst voorspellen met eerdere dagen van de rijder
            r = rid[q]; f = 0.5 ** ((dag[q] - L[r]) / hl) if L[r] > -1e8 else 0.0
            lam_o[q] = (O[r] * f + k) / (E[r] * f + k)
        for q in range(i, j):
            if gebruik[q]:
                r = rid[q]
                if L[r] > -1e8:
                    f = 0.5 ** ((dag[q] - L[r]) / hl); O[r] *= f; E[r] *= f
                L[r] = dag[q]; E[r] += Q
                if p[q] < Q:
                    O[r] += 1.0
        i = j


def rijder(dag, rijder_id, p, gebruik, Q=0.05, hl=730.0, k=10.0, rij=None):
    """Staartmassa per rijder. Geeft p', de verandering in de log loss en lambda (1 buiten rij)."""
    rij = np.isfinite(p) if rij is None else rij
    idx = np.flatnonzero(rij); o = idx[np.argsort(dag[idx], kind='stable')]
    codes, rid = np.unique(rijder_id[o], return_inverse=True)
    lo = np.ones(len(o))
    _rijder(dag[o].astype(np.float64), rid.astype(np.int64), np.asarray(p, float)[o], (gebruik & np.isfinite(p))[o], Q, hl, k, len(codes), lo)
    lam = np.ones(len(p)); lam[o] = np.clip(lo, 0.25, min(4.0, 0.5 / Q))
    p2 = np.where(p < Q, lam * p, lam * Q + (p - Q) * (1 - lam * Q) / (1 - Q))
    d = np.where(p < Q, -np.log(lam), -np.log((1 - lam * Q) / (1 - Q)))
    ok = np.isfinite(p)
    return np.where(ok, p2, p), np.where(ok, d, 0.0), lam


def rijder_terug(q, lam, Q=0.05):
    """Omgekeerde van de rijdermap: de p met p' = q."""
    return np.where(q < lam * Q, q / lam, Q + (q - lam * Q) * (1 - Q) / (1 - lam * Q))
