import warnings; warnings.filterwarnings('ignore')
import sys, json, time, numpy as np, pandas as pd
from scipy import optimize
from scipy.special import expit, ndtri
import gas_run as G
from gas_lab import gas_pass2, gas_pass3

t0 = time.time()
x = G.d.sort_values(['sk', 'dag', 'ev', 'ai'], kind='stable')
A = dict(sk=x.sk.to_numpy().astype(np.int64), dag=x.dag.to_numpy().astype(np.int64), ai=x.ai.to_numpy().astype(np.int64), u=x.u.to_numpy(),
         grp=x.grp.to_numpy().astype(np.int64), val=x.val.to_numpy(), pw=x[['pw0', 'pw1', 'pw2']].to_numpy(),
         fit=((x.jr <= 2019) & (x.jr >= 2010) & ~x.val).to_numpy(), test=((x.jr >= 2020) & ~x.val).to_numpy())
M = np.zeros(len(x)); LF = np.zeros(len(x)); LS = np.zeros(len(x))


def run(par, asym, scoor):
    if asym:
        return gas_pass3(A['sk'], A['dag'], A['ai'], A['u'], A['grp'], A['val'], scoor, G.VORM, np.asarray(par, float), -1.0, 2.5,
                         A['pw'], G.drempel_lo, G.drempel_hi, G.MASSA, M, LF, LS)
    r = gas_pass2(A['sk'], A['dag'], A['ai'], A['u'], A['grp'], A['val'], scoor, G.VORM, np.asarray(par, float), True, -1.0, 2.5,
                  A['pw'], G.drempel_lo, G.drempel_hi, G.MASSA, M, LF)
    LS[:] = LF
    return r


def par_van(th):
    p = np.zeros(12)
    p[0], p[1] = expit(th[0]), expit(th[1]); p[2:6] = np.log1p(np.exp(th[2:6]))
    p[6], p[7] = expit(th[6]), expit(th[7]); p[8:12] = np.log1p(np.exp(th[8:12]))
    return p


def schat(asym, th0):
    n = A['fit'].sum()
    f = lambda th: -run(par_van(th), asym, A['fit']) / n
    r = optimize.minimize(f, th0, method='Nelder-Mead', options={'maxiter': 1200, 'xatol': 1e-3, 'fatol': 1e-6})
    r = optimize.minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 1200, 'xatol': 1e-4, 'fatol': 1e-7})
    return r.x, par_van(r.x), -r.fun


def bewaar(par, asym, naam):
    run(par, asym, A['test'])
    g = A['grp']; xx = A['u'] - M
    lam = np.where(xx < 0, LF, LS); eps = xx / np.exp(lam)
    lp = np.zeros(len(eps)); pit = np.zeros(len(eps))
    for gg in range(4):
        s = g == gg; v = G.VORM[gg]
        th = [np.log(v[0] / (1 - v[0])), np.log(v[1]), np.log((v[2] - 1.5) / (30 - v[2])), np.log(v[3]), np.log((v[4] - 1.5) / (30 - v[4]))]
        lp[s] = G.tp_logpdf(th, eps[s]) - lam[s]
        F = G.tp_cdf(v, eps[s]); lo, hi = G.drempel_lo[gg], G.drempel_hi[gg]; Flo, Fhi = G.tp_cdf(v, lo), G.tp_cdf(v, hi); pw = A['pw'][s]
        reg = np.where(eps[s] < lo, 0, np.where(eps[s] > hi, 2, 1))
        lp[s] += np.log(pw[np.arange(s.sum()), reg] / G.MASSA[gg][reg])
        pit[s] = np.where(reg == 0, pw[:, 0] * F / Flo, np.where(reg == 1, pw[:, 0] + pw[:, 1] * (F - Flo) / (Fhi - Flo),
                                                                 pw[:, 0] + pw[:, 1] + pw[:, 2] * (F - Fhi) / (1 - Fhi)))
    pd.DataFrame({'lp': lp, 'zN': ndtri(np.clip(pit, 1e-300, 1 - 1e-16)), 'm': M, 'lf': LF, 'ls': LS, 'pit': pit},
                 index=x.index).to_pickle(f'gas3_res_{naam}.pkl')


start = np.r_[np.log(0.2275 / 0.7725), np.log(0.24 / 0.76), np.log(np.expm1([0.05, 0.002, 0.016, 0.022])),
              np.log(0.977 / 0.023), np.log(0.57 / 0.43), np.log(np.expm1([0.02, 0.022, 0.0011, 0.0037]))]
uit = {}
for k in sys.argv[1:]:
    asym = (k == 'J')
    if k == 'basis':
        par = np.zeros(12); bewaar(par, False, 'basis'); print('basis klaar', flush=True); continue
    th, par, ll = schat(asym, start.copy())
    bewaar(par, asym, k); uit[k] = par.tolist()
    print(f'{k} {"asymmetrisch" if asym else "symmetrisch"} par={np.round(par, 4).tolist()} fit-ll={ll:.5f} ({time.time()-t0:.0f}s)', flush=True)
    json.dump(uit, open('gas3_par.json', 'w'))
