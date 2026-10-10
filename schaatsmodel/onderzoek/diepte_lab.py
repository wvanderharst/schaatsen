"""Piekdiepte per rijder: in de piekzone (z < zlo) wordt de afstand tot de drempel geschaald met tau_i (lang geheugen, gekrompen)."""
import numpy as np, pandas as pd, json, time
from numba import njit
from scipy.stats import norm
t0 = time.time()
ZB = pd.read_pickle('zbase.pkl')
zc = pd.read_pickle('normaal_z_dag.pkl').reindex(ZB.index).astype(float).to_numpy()   # huidige dag-z (met vormlaag)
niv = pd.read_pickle('gas_d.pkl')['abs'].reindex(ZB.index).to_numpy()
o = np.lexsort((ZB.afstand_idx.to_numpy(), ZB.event_id.to_numpy(), ZB.dag.to_numpy(), ZB.skater_id.to_numpy()))
SK = ZB.skater_id.to_numpy()[o].astype(np.int64); DG = ZB.dag.to_numpy()[o].astype(np.int64)
zb = ZB.z.to_numpy().astype(float); VAL = (~np.isfinite(zb) | (zb > 6))[o]; ZO = np.where(np.isfinite(zb), zb, 0)[o]
ZLO = norm.ppf(0.07)


@njit(cache=True)
def diepte_pass(sk, dg, z, val, zlo, H, uit):
    n = len(sk); se = 0.0; sn = 0.0; ld = -1; i = 0
    while i < n:
        j = i
        while j < n and sk[j] == sk[i] and dg[j] == dg[i]:
            j += 1
        if i == 0 or sk[i] != sk[i - 1]:
            se = 0.0; sn = 0.0
        else:
            f = np.exp(-(dg[i] - ld) / H); se *= f; sn *= f
        ld = dg[i]
        for q in range(i, j):
            uit[q, 0] = se; uit[q, 1] = sn
        for q in range(i, j):
            if (not val[q]) and z[q] < zlo:
                se += zlo - z[q]; sn += 1.0
        i = j
    return 0


def feat(H):
    u = np.zeros((len(o), 2)); diepte_pass(SK, DG, ZO, VAL, ZLO, H, u)
    U = np.zeros_like(u); U[o] = u; return U[:, 0], U[:, 1]


dt = pd.to_datetime(ZB.dag.to_numpy(), unit='D'); seiz = np.where(dt.month >= 7, dt.year, dt.year - 1)
ok = np.isfinite(zc) & (zc < 6)
jr = ZB.jaar.to_numpy()
FIT = ok & (jr >= 2010) & (jr <= 2019); TEST = ok & (jr >= 2020)
e0 = np.mean(ZLO - zb[FIT & (zb < ZLO)])
print('populatie gemiddelde overschrijding in de piekzone (fit):', round(e0, 3), ' (normaal zou', round(norm.pdf(ZLO) / 0.07 + ZLO, 3), 'zijn)', flush=True)
SB = pd.DataFrame({'sk': ZB.skater_id.to_numpy(), 's': seiz, 'niv': niv})


def transform(tau):
    z = zc.copy(); s = z < ZLO
    z[s] = ZLO + (z[s] - ZLO) / tau[s]
    return z, np.where(s, np.log(tau), 0.0)


def seizoen(z, sel):
    d = SB[sel].copy(); d['z'] = z[sel]
    S = d.groupby(['sk', 's']).agg(n=('z', 'size'), zmin=('z', 'min'), niv=('niv', 'median')).reset_index()
    S = S[S.n >= 8]; S['pit'] = 1 - (1 - norm.cdf(S.zmin)) ** S.n
    S = S.sort_values(['sk', 's']); nx = S.groupby('sk').shift(-1); okk = nx.s == S.s + 1
    T = S[okk & (S.niv >= 90)]; v = nx.pit[okk & (S.niv >= 90)]
    r = lambda x: f"{100*(x<.05).mean():.1f}/{100*(x<.10).mean():.1f}"
    return (f"alle {r(S.pit)} top {r(S.pit[S.niv>=90])} elite {r(S.pit[S.niv>=97])} | top na extreem seizoen {r(v[T.pit<.05])} "
            f"na gewoon seizoen {r(v[T.pit>.5])}")


def score(z, lj, sel):
    return np.mean(-norm.logpdf(z[sel]) + lj[sel])


base_z = zc
print('huidig        ', seizoen(base_z, TEST), flush=True)
res = []
for H in (730.0, 1460.0, 1e9):
    E, N = feat(H)
    for m in (3.0, 8.0, 20.0):
        R = (E + m * e0) / (N + m) / e0
        for kap in (0.5, 1.0, 1.5):
            tau = R ** kap
            z, lj = transform(tau)
            d_all = score(z, lj, TEST) - score(base_z, np.zeros(len(z)), TEST)
            top = TEST & (niv >= 90); eli = TEST & (niv >= 97)
            d_top = score(z, lj, top) - score(base_z, np.zeros(len(z)), top); d_el = score(z, lj, eli) - score(base_z, np.zeros(len(z)), eli)
            print(f'H {H:.0f} m {m:.0f} kappa {kap}: logscore {d_all:+.5f} top {d_top:+.5f} elite {d_el:+.5f} | {seizoen(z, TEST)}', flush=True)
