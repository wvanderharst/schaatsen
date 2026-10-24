import warnings; warnings.filterwarnings('ignore')
import sys, json, time, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtri
import vorm_lab6 as L
from gas_lab import vorm_pass
t0 = time.time()
G = pd.read_pickle('gas_d.pkl')[['sk', 'dag', 'ev', 'ai']]
ZB = pd.read_pickle('normaal_z_dag.pkl').reindex(G.index).astype(float).to_numpy()
o = np.lexsort((G.ai.to_numpy(), G.ev.to_numpy(), G.dag.to_numpy(), G.sk.to_numpy()))
SK = G.sk.to_numpy()[o].astype(np.int64); DG = G.dag.to_numpy()[o].astype(np.int64)
VAL = (~np.isfinite(ZB) | (ZB > 6))[o]; ZO = np.where(np.isfinite(ZB), ZB, 0)[o]


def feats(Hk, c, Hl, cs=2.5):
    uit = np.zeros((len(o), 4)); a = np.zeros(len(o), np.int64); b = np.zeros(len(o), np.int64); s = np.zeros((len(o), 4))
    vorm_pass(SK, DG, ZO, VAL, float(Hk), float(c), float(Hl), float(cs), uit, a, b, s)
    U = np.zeros((len(o), 4)); U[o] = uit
    F = pd.DataFrame(U, index=G.index, columns=['Zk', 'F', 'S', 'Zl']).reindex(L.D.index)
    for k in F.columns: L.X[k] = F[k].to_numpy()


# aantal snelle uitschieters (z<-3) in de 90 dagen ervoor (staartcontrole)
feats(7, 3.0, 90 / 1e9 if False else 1e9)        # placeholder, overschreven hieronder
cnt = np.zeros(len(o)); dd = pd.DataFrame({'sk': SK, 'dag': DG, 'f': ((ZO < -3) & ~VAL).astype(float)})
g = dd.groupby(['sk', 'dag']).f.sum().reset_index(); g['t'] = pd.to_datetime(g.dag, unit='D')
r = g.groupby('sk').rolling('90D', on='t', closed='left').f.sum().reset_index(level=0, drop=True)
g['f90'] = r.fillna(0).to_numpy()
F90 = pd.DataFrame({'sk': SK, 'dag': DG}, index=G.index[o]).merge(g[['sk', 'dag', 'f90']], on=['sk', 'dag'], how='left').f90.to_numpy()
F90s = pd.Series(F90, index=G.index[o]).reindex(L.D.index).to_numpy()


def staart(q, dag=True):
    uu, _ = L.transform(q, dag); z = np.full(len(uu), np.nan)
    for gi in range(4):
        p = L.vorm_fit(uu, gi); s = L.ok & (L.g == gi)
        z[s] = ndtri(np.clip(np.array([L.cdf(p, x) for x in []]) if False else _cdfv(p, uu[s]), 1e-300, 1 - 1e-16))
    te = L.TE; out = []
    for k in (0, 1, 2):
        sel = te & (np.minimum(F90s, 2) == k)
        out.append(f"f90={k}: sd {np.nanstd(z[sel]):.2f} z<-3 {100*np.nanmean(z[sel] < -3):.2f}%")
    return ' | '.join(out)


from scipy.stats import t as T
def _cdfv(p, x):
    p0, sL, nL, sU, nU = L.unpack(p)
    return np.where(x < 0, p0 * 2 * T.cdf(x / sL, nL), p0 + (1 - p0) * (2 * T.cdf(x / sU, nU) - 1))


def fit_dag(rng=np.random.default_rng(0)):
    dag = True; uu, _ = L.transform(L.Z, dag); vormen = {gi: L.vorm_fit(uu, gi) for gi in range(4)}
    idx = {gi: rng.choice(np.where(L.ok & (L.g == gi) & (L.jr <= 2019) & (L.jr >= 2010))[0], min(120000, (L.ok & (L.g == gi) & (L.jr <= 2019)).sum()), replace=False) for gi in range(4)}
    # overweeg rijen na uitschieters (zeldzaam maar waar het om gaat): neem ze allemaal mee
    for gi in range(4):
        extra = np.where(L.ok & (L.g == gi) & (L.jr <= 2019) & (L.jr >= 2010) & (F90s >= 1))[0]
        idx[gi] = np.unique(np.r_[idx[gi], extra])
    alle = np.concatenate([idx[gi] for gi in range(4)]); pos = {}; k = 0
    nb = {gi: (L.ok & (L.g == gi) & (L.jr <= 2019) & (L.jr >= 2010)).sum() for gi in range(4)}
    wts = np.zeros(len(alle))
    for gi in range(4):
        pos[gi] = np.arange(k, k + len(idx[gi])); k += len(idx[gi])
        # gewichten: rij-kans-correctie zodat het een zuivere schatting van het gemiddelde blijft
        sel = idx[gi]; p_in = np.where(F90s[sel] >= 1, 1.0, 120000 / nb[gi]); wts[pos[gi]] = 1 / p_in
    wts /= wts.sum()
    L.XS = {kk: v[alle] for kk, v in L.X.items()}; L.WS = L.W[alle]; PWs = L.PW[alle]
    def f(q):
        uu, ls = L.transform_sub(q, dag); tot = np.zeros(len(alle))
        for gi, gname in enumerate(L.GR):
            ii = pos[gi]; p = vormen[gi]
            tot[ii] = L.nll_u(p, uu[ii]) + ls[ii] - L.piekw(p, uu[ii], PWs[ii], L.GIDX[gname])
        return np.sum(tot * wts)
    x0 = np.zeros(10); simp = np.vstack([x0] + [x0 + 0.03 * np.eye(10)[i] for i in range(10)])
    r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-10, 'maxiter': 4000, 'initial_simplex': simp})
    r = minimize(f, r.x, method='Powell', options={'xtol': 1e-5, 'ftol': 1e-10})
    return r.x


if __name__ == '__main__':
    feats(7, 2.5, 1000)
    print('geen vormlaag:', staart(L.Z), f'{time.time()-t0:.0f}s', flush=True)
    res = {}
    for Hl in (60, 180, 1000):
        feats(7, 2.5, Hl); q = fit_dag()
        L.rapport(f'Hl={Hl} q={np.round(q, 4).tolist()}', q)
        print('   staart:', staart(q), flush=True); res[Hl] = q.tolist()
    json.dump(res, open('vorm_lab7.json', 'w'))
