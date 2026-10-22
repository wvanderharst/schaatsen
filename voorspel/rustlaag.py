"""Rust en taper: lopende verschuiving op de kansschaal naar het aantal dagen sinds de vorige race (r = ln(1 + dagen)) en het niveau.
Bij de top: na 4-10 dagen rust vaker sneller dan voorspeld (taper); op dezelfde dag, de tweede dag van een toernooi en na 1-5 maanden
zonder race trager. Bij zwakkere rijders is het patroon anders, dus een gladde functie van beide (geen groepen).

z = Phi^-1(p), z' = (z - mu) / sigma; mu en sigma uit eerdere dagen: uitdovende sommen (0,5^(dagen / hl)) met een Gauss-kernel in niveau
(bw) en in r (bw_r) op een rooster, krimp met k0 pseudo-ritten naar mu = 0, sigma = 1 (z afgekapt op +-4). Zonder schaal: sigma = 1.
De log loss verandert exact met ln sigma + (z'^2 - z^2) / 2. Eigen valpartijen tellen niet mee in de schatting."""
import numpy as np
from numba import njit
from scipy.stats import norm

GN = np.linspace(-0.03, 0.45, 49)
GR = np.linspace(0.0, 6.0, 25)


@njit(cache=True)
def _rust(dag, g, r, z, gebruik, GN, GR, bw, bw_r, hl, k0, schaal, mu_o, sg_o):
    n = len(z); a_ = len(GN); b_ = len(GR)
    S1 = np.zeros((a_, b_)); Sz = np.zeros((a_, b_)); Sz2 = np.zeros((a_, b_)); laatst = -1e9; i = 0
    while i < n:
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        if laatst > -1e8:
            f = 0.5 ** ((dag[i] - laatst) / hl)
            S1 *= f; Sz *= f; Sz2 *= f
        for q in range(i, j):
            pa = (g[q] - GN[0]) / (GN[1] - GN[0]); ka = int(min(max(pa, 0.0), a_ - 1.001)); ta = min(max(pa - ka, 0.0), 1.0)
            pb = (r[q] - GR[0]) / (GR[1] - GR[0]); kb = int(min(max(pb, 0.0), b_ - 1.001)); tb = min(max(pb - kb, 0.0), 1.0)
            s1 = 0.0; sz = 0.0; sz2 = 0.0
            for ha in range(2):
                wa = (1.0 - ta) if ha == 0 else ta
                for hb in range(2):
                    w = wa * ((1.0 - tb) if hb == 0 else tb)
                    s1 += w * S1[ka + ha, kb + hb]; sz += w * Sz[ka + ha, kb + hb]; sz2 += w * Sz2[ka + ha, kb + hb]
            mu = sz / (s1 + k0); mu_o[q] = mu
            if schaal:
                v = (sz2 + k0) / (s1 + k0) - mu * mu
                sg_o[q] = np.sqrt(max(v, 0.25))
            else:
                sg_o[q] = 1.0
        for q in range(i, j):
            if gebruik[q]:
                zz = min(max(z[q], -4.0), 4.0)
                for a in range(a_):
                    ka_ = np.exp(-0.5 * ((g[q] - GN[a]) / bw) ** 2)
                    if ka_ < 1e-4:
                        continue
                    for b in range(b_):
                        kk = ka_ * np.exp(-0.5 * ((r[q] - GR[b]) / bw_r) ** 2)
                        if kk < 1e-5:
                            continue
                        S1[a, b] += kk; Sz[a, b] += kk * zz; Sz2[a, b] += kk * zz * zz
        laatst = dag[i]; i = j


def rust(dag, g, gap, p, gebruik, bw=0.02, bw_r=0.3, hl=1095.0, k0=100.0, schaal=True, rij=None):
    """Rijen in willekeurige volgorde. rij: welke rijen mu en sigma krijgen (standaard: eindige p); gebruik: welke de schatting bijwerken.
    Geeft p', de verandering in de log loss, mu en sigma."""
    ok = np.isfinite(p)
    rij = ok if rij is None else rij
    idx = np.flatnonzero(rij); o = idx[np.argsort(dag[idx], kind='stable')]
    z = norm.ppf(np.clip(p, 1e-15, 1 - 1e-15))
    r = np.log1p(np.clip(np.nan_to_num(gap, nan=400.0), 0, 400.0))
    mo = np.zeros(len(o)); so = np.ones(len(o))
    _rust(dag[o].astype(np.float64), np.clip(np.nan_to_num(g[o], nan=0.2), -0.03, 0.45), r[o], np.nan_to_num(z[o]), (gebruik & ok)[o],
          GN, GR, bw, bw_r, hl, k0, schaal, mo, so)
    mu = np.zeros(len(p)); sg = np.ones(len(p)); mu[o] = mo; sg[o] = so
    z2 = (z - mu) / sg
    p2 = np.where(ok, norm.cdf(z2), p)
    d = np.where(ok, np.log(sg) + 0.5 * (z2 ** 2 - z ** 2), 0.0)
    return p2, d, mu, sg


def rust_terug(q, mu, sg):
    """Omgekeerde: de p met p' = q (z = mu + sigma z')."""
    return norm.cdf(mu + sg * norm.ppf(q))
