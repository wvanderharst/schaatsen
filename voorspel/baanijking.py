"""Lopende ijking per baan op de kansschaal: hoeveel sneller (mu) en hoe wisselvalliger (sigma) valt het op deze baan uit, als gladde
functie van het niveau van de rijder (Gauss-kernel, bandbreedte bw) en van de dag in het toernooi (kernel op dag1 = exp(-dagnummer)).
Voorbeeld: in SLC was het ijs op de eerste dag met een elite-veld bij elk bezoek sinds 2020 sneller dan voorspeld (-0,06 tot -0,67%),
en op hoogland op dag 1 was de spreiding 1,5x zo groot als het model rekende. Een algemeen model, geschat op 2008-2019, ziet dat niet.

Per baan houdt de laag uitdovende sommen bij over eerdere wedstrijddagen (gewicht 0,5^(bezoekdagen terug / hl_bezoek) x 0,5^(dagen / hl_dag)):
S1 = sum K, Sz = sum K z, Sz2 = sum K min(z^2, 16), op een rooster in niveau x dag1. Dan mu = Sz / (S1 + k0) en
sigma^2 = max((Sz2 + k0) / (S1 + k0) - mu^2, 0,25), en z' = (z - mu) / sigma. De log loss verandert exact met
ln sigma + (z'^2 - z^2) / 2. Alleen eerdere dagen; eigen valpartijen tellen niet mee."""
import numpy as np
from numba import njit

GN = np.linspace(-0.03, 0.45, 33)
GD = np.array([1.0, np.exp(-1.0), np.exp(-2.0), 0.0])


@njit(cache=True)
def _ijk(baan, dag, g, d1, z, gebruik, GN, GD, bw, bwd, hl_b, hl_d, k0, mu_o, sg_o):
    n = len(z); nb = baan.max() + 1; m = len(GN); md = len(GD)
    S1 = np.zeros((nb, m, md)); Sz = np.zeros((nb, m, md)); Sz2 = np.zeros((nb, m, md)); laatst = np.full(nb, -1e9)
    i = 0
    while i < n:                                       # rijen gesorteerd op baan, dag
        j = i
        while j < n and baan[j] == baan[i] and dag[j] == dag[i]:
            j += 1
        b = baan[i]
        if laatst[b] > -1e8:
            f = 0.5 ** (1.0 / hl_b) * 0.5 ** ((dag[i] - laatst[b]) / hl_d)
            for a in range(m):
                for c in range(md):
                    S1[b, a, c] *= f; Sz[b, a, c] *= f; Sz2[b, a, c] *= f
        for q in range(i, j):                          # voorspellen met eerdere dagen (bilineair op het rooster)
            pos = (g[q] - GN[0]) / (GN[1] - GN[0]); k = int(min(max(pos, 0.0), m - 1.001)); t = min(max(pos - k, 0.0), 1.0)
            # dag1 is aflopend in GD
            c = 0
            while c < md - 2 and d1[q] < GD[c + 1]:
                c += 1
            u = min(max((GD[c] - d1[q]) / (GD[c] - GD[c + 1]), 0.0), 1.0)
            s1 = sz = sz2 = 0.0
            for (aa, wa) in ((k, 1 - t), (k + 1, t)):
                for (cc, wc) in ((c, 1 - u), (c + 1, u)):
                    s1 += wa * wc * S1[b, aa, cc]; sz += wa * wc * Sz[b, aa, cc]; sz2 += wa * wc * Sz2[b, aa, cc]
            mu = sz / (s1 + k0); v = (sz2 + k0) / (s1 + k0) - mu * mu
            mu_o[q] = mu; sg_o[q] = np.sqrt(max(v, 0.25))
        for q in range(i, j):
            if gebruik[q]:
                zz = min(max(z[q], -4.0), 4.0)
                for a in range(m):
                    ka = np.exp(-0.5 * ((g[q] - GN[a]) / bw) ** 2)
                    if ka < 1e-4:
                        continue
                    for c in range(md):
                        kk = ka * np.exp(-0.5 * ((d1[q] - GD[c]) / bwd) ** 2)
                        S1[b, a, c] += kk; Sz[b, a, c] += kk * zz; Sz2[b, a, c] += kk * zz * zz
        laatst[b] = dag[i]; i = j


def ijk(baan_code, dag, g, dag1, z, gebruik, bw=0.02, bwd=0.25, hl_b=15.0, hl_d=1095.0, k0=30.0):
    """Rijen in willekeurige volgorde; geeft mu en sigma per rit (uit eerdere dagen op dezelfde baan)."""
    o = np.lexsort((dag, baan_code))
    mu = np.zeros(len(z)); sg = np.ones(len(z)); mo = np.zeros(len(z)); so = np.ones(len(z))
    zf = np.nan_to_num(z)
    _ijk(baan_code[o].astype(np.int64), dag[o].astype(np.float64), np.clip(np.nan_to_num(g[o], nan=0.2), -0.03, 0.45),
         np.nan_to_num(dag1[o], nan=1.0), zf[o], (gebruik & np.isfinite(z))[o], GN, GD, bw, bwd, hl_b, hl_d, k0, mo, so)
    mu[o] = mo; sg[o] = so
    return mu, sg
