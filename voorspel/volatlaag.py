"""Eigen volatiliteit per rijder als laag op de kansschaal. Een wisselvallige rijder is vooral vaker traag (slechte dagen) en nauwelijks vaker
uitzonderlijk snel, dus een scheve laag: aparte schaal voor de snelle (z < 0) en de trage kant (z > 0), gladde functie van het niveau en
x = ln(eigen spreiding).
Eigen spreiding (alleen EERDERE dagen, valpartijen niet): u = rijder-only residu / sd_rijder (residu min de veldmediaan van die race: ijs
achteraf), afgekapt op +-4; sqrt((som w u^2 + k0) / (som w + k0)), gewicht w halveert per hlr ritten.
Laag: z' = z / s_snel (z < 0) of z / s_traag (z > 0); s^2 = (som kernel z^2 aan die kant + k0) / (som kernel aan die kant + k0) uit eerdere
dagen (uitdovend, halfwaardetijd hl dagen); de mediaan blijft gelijk. De log loss verandert exact met ln s + (z'^2 - z^2) / 2."""
import numpy as np
from numba import njit
from scipy.stats import norm
from rustlaag import GN

GX = np.linspace(-0.5, 0.8, 27)


@njit(cache=True)
def _eigen(rid, dag, u, geldig, lam, k0, uit):
    n = len(u); i = 0
    while i < n:
        S = 0.0; W = 0.0; j = i
        while j < n and rid[j] == rid[i]:
            j += 1
        a = i
        while a < j:
            b = a
            while b < j and dag[b] == dag[a]:
                b += 1
            v = np.sqrt((S + k0) / (W + k0))
            for q in range(a, b):
                uit[q] = v
            for q in range(a, b):
                if geldig[q]:
                    S = lam * S + min(u[q] * u[q], 16.0); W = lam * W + 1.0
            a = b
        i = j


def eigen(skater_id, dag, rk, res, sd_rijder, geldig, hlr=14.0, k0=10.0):
    """ln(eigen spreiding) per rij uit eerdere dagen; res = ln t - pred."""
    res = np.asarray(res, float); geldig = np.asarray(geldig, bool) & np.isfinite(res)
    rk = np.asarray(rk); _, ri = np.unique(rk, return_inverse=True)
    med = np.full(ri.max() + 1, np.nan)
    import pandas as pd
    m_ = pd.Series(res[geldig]).groupby(ri[geldig]).median(); med[m_.index.to_numpy()] = m_.to_numpy()
    u = (res - med[ri]) / np.asarray(sd_rijder, float)
    geldig &= np.isfinite(u)
    o = np.lexsort((np.asarray(dag, float), np.asarray(skater_id)))
    _, rid = np.unique(np.asarray(skater_id)[o], return_inverse=True)
    uo = np.zeros(len(o))
    _eigen(rid.astype(np.int64), np.asarray(dag, float)[o], np.nan_to_num(u[o]), geldig[o], 0.5 ** (1.0 / hlr), k0, uo)
    x = np.empty(len(o)); x[o] = np.log(uo)
    return x


@njit(cache=True)
def _scheef(dag, g, x, z, gebruik, GN, GX, bw, bw_x, hl, k0, snel_o, traag_o, alleen_snel):
    n = len(z); a_ = len(GN); b_ = len(GX)
    Sn = np.zeros((a_, b_)); Qn = np.zeros((a_, b_)); Sp = np.zeros((a_, b_)); Qp = np.zeros((a_, b_)); laatst = -1e9; i = 0
    while i < n:
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        if laatst > -1e8:
            f = 0.5 ** ((dag[i] - laatst) / hl)
            Sn *= f; Qn *= f; Sp *= f; Qp *= f
        for q in range(i, j):
            pa = (g[q] - GN[0]) / (GN[1] - GN[0]); ka = int(min(max(pa, 0.0), a_ - 1.001)); ta = min(max(pa - ka, 0.0), 1.0)
            pb = (x[q] - GX[0]) / (GX[1] - GX[0]); kb = int(min(max(pb, 0.0), b_ - 1.001)); tb = min(max(pb - kb, 0.0), 1.0)
            sn = 0.0; qn = 0.0; sp = 0.0; qp = 0.0
            for ha in range(2):
                wa = (1.0 - ta) if ha == 0 else ta
                for hb in range(2):
                    w = wa * ((1.0 - tb) if hb == 0 else tb)
                    sn += w * Sn[ka + ha, kb + hb]; qn += w * Qn[ka + ha, kb + hb]; sp += w * Sp[ka + ha, kb + hb]; qp += w * Qp[ka + ha, kb + hb]
            snel_o[q] = np.sqrt(max((qn + k0) / (sn + k0), 0.25))
            traag_o[q] = 1.0 if alleen_snel else np.sqrt(max((qp + k0) / (sp + k0), 0.25))
        for q in range(i, j):
            if gebruik[q]:
                zz = min(max(z[q], -4.0), 4.0)
                for a in range(a_):
                    ka_ = np.exp(-0.5 * ((g[q] - GN[a]) / bw) ** 2)
                    if ka_ < 1e-4:
                        continue
                    for b in range(b_):
                        kk = ka_ * np.exp(-0.5 * ((x[q] - GX[b]) / bw_x) ** 2)
                        if kk < 1e-5:
                            continue
                        if zz < 0:
                            Sn[a, b] += kk; Qn[a, b] += kk * zz * zz
                        else:
                            Sp[a, b] += kk; Qp[a, b] += kk * zz * zz
        laatst = dag[i]; i = j


def laag(dag, g, x, p, gebruik, GX=GX, bw_x=0.1, bw=0.02, hl=1095.0, k0=30.0, alleen_snel=False, rij=None):
    """Geeft p', verandering in de log loss, s_snel en s_traag (1 buiten rij of zonder x)."""
    ok = np.isfinite(p)
    rij = ok if rij is None else rij
    idx = np.flatnonzero(rij & np.isfinite(x)); o = idx[np.argsort(dag[idx], kind='stable')]
    z = norm.ppf(np.clip(p, 1e-15, 1 - 1e-15))
    so = np.ones(len(o)); to = np.ones(len(o))
    _scheef(np.asarray(dag, float)[o], np.clip(np.nan_to_num(g[o], nan=0.2), GN[0], GN[-1]), np.clip(x[o], GX[0], GX[-1]), np.nan_to_num(z[o]),
            (gebruik & ok)[o], GN, GX, bw, bw_x, hl, k0, so, to, alleen_snel)
    s1 = np.ones(len(p)); s2 = np.ones(len(p)); s1[o] = so; s2[o] = to
    s = np.where(z < 0, s1, s2); z2 = z / s
    return np.where(ok, norm.cdf(z2), p), np.where(ok, np.log(s) + 0.5 * (z2 ** 2 - z ** 2), 0.0), s1, s2


def terug(q, s1, s2):
    """Omgekeerde: de p met p' = q."""
    z2 = norm.ppf(q)
    return norm.cdf(z2 * np.where(z2 < 0, s1, s2))


@njit(cache=True)
def _staart(dag, g, x, p, gebruik, GN, GX, bw, bw_x, hl, k0, k1, Q, m_o, b_o):
    n = len(p); a_ = len(GN); b_ = len(GX)
    S = np.zeros((a_, b_)); C = np.zeros((a_, b_)); L = np.zeros((a_, b_)); laatst = -1e9; i = 0
    while i < n:
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        if laatst > -1e8:
            f = 0.5 ** ((dag[i] - laatst) / hl)
            S *= f; C *= f; L *= f
        for q in range(i, j):
            pa = (g[q] - GN[0]) / (GN[1] - GN[0]); ka = int(min(max(pa, 0.0), a_ - 1.001)); ta = min(max(pa - ka, 0.0), 1.0)
            pb = (x[q] - GX[0]) / (GX[1] - GX[0]); kb = int(min(max(pb, 0.0), b_ - 1.001)); tb = min(max(pb - kb, 0.0), 1.0)
            s = 0.0; c = 0.0; l = 0.0
            for ha in range(2):
                wa = (1.0 - ta) if ha == 0 else ta
                for hb in range(2):
                    w = wa * ((1.0 - tb) if hb == 0 else tb)
                    s += w * S[ka + ha, kb + hb]; c += w * C[ka + ha, kb + hb]; l += w * L[ka + ha, kb + hb]
            m_o[q] = min(max((c + k0 * Q) / (s + k0), 0.25 * Q), 2.5 * Q)
            b_o[q] = min(max((c + k1) / (l + k1), 0.6), 1.6)
        for q in range(i, j):
            if gebruik[q]:
                st = p[q] < Q; lu = -np.log(max(p[q], 1e-12) / Q) if st else 0.0
                for a in range(a_):
                    ka_ = np.exp(-0.5 * ((g[q] - GN[a]) / bw) ** 2)
                    if ka_ < 1e-4:
                        continue
                    for b in range(b_):
                        kk = ka_ * np.exp(-0.5 * ((x[q] - GX[b]) / bw_x) ** 2)
                        if kk < 1e-5:
                            continue
                        S[a, b] += kk
                        if st:
                            C[a, b] += kk; L[a, b] += kk * lu
        laatst = dag[i]; i = j


def staart(dag, g, x, p, gebruik, GX=GX, bw_x=0.1, bw=0.02, hl=1095.0, k0=300.0, k1=30.0, Q=0.05, rij=None):
    """Snelle staart (p < Q) naar niveau x eigen spreiding: massa m en vorm b, p' = m (p / Q)^b; daarboven lineair naar 1.
    Geeft p', verandering in de log loss, m en b (Q en 1 buiten rij of zonder x)."""
    ok = np.isfinite(p)
    rij = ok if rij is None else rij
    idx = np.flatnonzero(rij & np.isfinite(x)); o = idx[np.argsort(dag[idx], kind='stable')]
    mo = np.full(len(o), Q); bo = np.ones(len(o))
    _staart(np.asarray(dag, float)[o], np.clip(np.nan_to_num(g[o], nan=0.2), GN[0], GN[-1]), np.clip(x[o], GX[0], GX[-1]), np.nan_to_num(np.asarray(p, float)[o], nan=0.5),
            (gebruik & ok)[o], GN, GX, bw, bw_x, hl, k0, k1, Q, mo, bo)
    m = np.full(len(p), Q); b = np.ones(len(p)); m[o] = mo; b[o] = bo
    pc = np.clip(p, 1e-15, 1.0)
    st = pc < Q
    p2 = np.where(st, m * (pc / Q) ** b, m + (pc - Q) * (1 - m) / (1 - Q))
    d = np.where(st, -np.log(m * b / Q) - (b - 1) * np.log(pc / Q), -np.log((1 - m) / (1 - Q)))
    return np.where(ok, p2, p), np.where(ok, d, 0.0), m, b


def staart_terug(q, m, b, Q=0.05):
    return np.where(q < m, Q * (np.clip(q, 1e-300, None) / m) ** (1 / b), Q + (q - m) * (1 - Q) / (1 - m))
