import warnings; warnings.filterwarnings('ignore')
import json, sys, time, numpy as np, pandas as pd
from scipy import optimize, stats
from scipy.special import expit, ndtri
from gas_lab import gas_pass

t0 = time.time()
d = pd.read_pickle('gas_d.pkl')
d = d[np.isfinite(d.u)]
DR = json.load(open('gas_drempel.json'))
LSG = np.array([0, 0, 1, 2])                       # kort/sprint -> 0, midden -> 1, lang -> 2 (piek-drempels)
drempel_lo = np.array([DR[str(LSG[g])][0] for g in range(4)]); drempel_hi = np.array([DR[str(LSG[g])][1] for g in range(4)])


def bereid(d, per):
    k = ['sk', 'dag', 'ev', 'ai'] if per == 'rijder' else ['sk', 'ai', 'dag', 'ev']
    x = d.sort_values(k, kind='stable')
    key = (x.sk.to_numpy() * 10 + (0 if per == 'rijder' else x.ai.to_numpy())).astype(np.int64)
    dag = x.dag.to_numpy().astype(np.int64)
    gap = np.r_[-1, np.diff(dag)]; gap[np.r_[True, key[1:] != key[:-1]]] = -1
    return x, key, gap.astype(np.float64)


def tp_logpdf(th, u):
    p0 = expit(th[0]); sL, nL, sU, nU = np.exp(th[1]), 1.5 + 28.5 * expit(th[2]), np.exp(th[3]), 1.5 + 28.5 * expit(th[4])
    return np.where(u < 0, np.log(2 * p0) + stats.t.logpdf(u / sL, nL) - np.log(sL), np.log(2 * (1 - p0)) + stats.t.logpdf(u / sU, nU) - np.log(sU))


def vorm_uit(th):
    return np.array([expit(th[0]), np.exp(th[1]), 1.5 + 28.5 * expit(th[2]), np.exp(th[3]), 1.5 + 28.5 * expit(th[4])])


def tp_cdf(v, x):
    p0, sL, nL, sU, nU = v
    return np.where(x < 0, p0 * 2 * stats.t.cdf(x / sL, nL), p0 + (1 - p0) * (2 * stats.t.cdf(x / sU, nU) - 1))


fit_rij = (d.jr <= 2019) & (d.jr >= 2010) & ~d.val
rng = np.random.default_rng(0)
VORM = np.zeros((4, 5)); MASSA = np.zeros((4, 3))
for g in range(4):
    uu = d.u[fit_rij & (d.grp == g)].to_numpy(); s = rng.choice(uu, min(200000, len(uu)), replace=False)
    r = optimize.minimize(lambda th: -np.mean(tp_logpdf(th, s)), [0, 0, 1, 0, 1], method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-9})
    r = optimize.minimize(lambda th: -np.mean(tp_logpdf(th, s)), r.x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-9})
    VORM[g] = vorm_uit(r.x)
    lo, hi = tp_cdf(VORM[g], drempel_lo[g]), tp_cdf(VORM[g], drempel_hi[g])
    MASSA[g] = [lo, hi - lo, 1 - hi]
print('vorm per groep (p0, sL, nuL, sU, nuU):'); print(np.round(VORM, 3)); print('massa', np.round(MASSA, 4), flush=True)

PER = {}
for per in ('rijder', 'rijder_afstand'):
    x, key, gap = bereid(d, per)
    PER[per] = dict(x=x, key=key, gap=gap, u=x.u.to_numpy(), grp=x.grp.to_numpy().astype(np.int64), val=x.val.to_numpy(),
                    pw=x[['pw0', 'pw1', 'pw2']].to_numpy(), fit=((x.jr <= 2019) & (x.jr >= 2010) & ~x.val).to_numpy(),
                    test=((x.jr >= 2020) & ~x.val).to_numpy(), m=np.zeros(len(x)), l=np.zeros(len(x)))


def run(per, par, modus, kap, scoor):
    P = PER[per]
    return gas_pass(P['key'], P['gap'], P['u'], P['grp'], P['val'], scoor, VORM, np.asarray(par, float), modus, kap, -1.0, 2.5,
                    P['pw'], drempel_lo, drempel_hi, MASSA, P['m'], P['l'])


def par_van(th, met_m):
    phi_m, phi_l = expit(th[0]), expit(th[1])
    a = np.log1p(np.exp(th[2:]))                         # >= 0
    if met_m:
        return [phi_m, phi_l, a[0], a[1], a[2], a[3]]
    return [0.0, phi_l, 0.0, 0.0, a[0], a[1]]


def schat(per, met_m, modus, kap):
    P = PER[per]; nfit = P['fit'].sum()
    th0 = np.r_[2.0, 2.0, np.full(4 if met_m else 2, -3.5)]
    f = lambda th: -run(per, par_van(th, met_m), modus, kap, P['fit']) / nfit
    r = optimize.minimize(f, th0, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-3, 'fatol': 1e-6})
    r = optimize.minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-7})
    return par_van(r.x, met_m), -r.fun


def rij_scores(per, par, modus, kap):
    """log-score per rij (test) en PIT -> normaal-equivalente z."""
    P = PER[per]
    run(per, par, modus, kap, P['test'])
    x = P['x']; m, l = P['m'].copy(), P['l'].copy()
    g = P['grp']; eps = (P['u'] - m) / np.exp(l)
    lp = np.zeros(len(eps)); pit = np.zeros(len(eps))
    for gg in range(4):
        s = g == gg; v = VORM[gg]
        th = [np.log(v[0] / (1 - v[0])), np.log(v[1]), np.log((v[2] - 1.5) / (30 - v[2])), np.log(v[3]), np.log((v[4] - 1.5) / (30 - v[4]))]
        lp[s] = tp_logpdf(th, eps[s]) - l[s]
        F = tp_cdf(v, eps[s]); lo, hi = drempel_lo[gg], drempel_hi[gg]
        Flo, Fhi = tp_cdf(v, lo), tp_cdf(v, hi); pw = P['pw'][s]
        reg = np.where(eps[s] < lo, 0, np.where(eps[s] > hi, 2, 1))
        lp[s] += np.log(pw[np.arange(s.sum()), reg] / MASSA[gg][reg])
        pit[s] = np.where(reg == 0, pw[:, 0] * F / Flo, np.where(reg == 1, pw[:, 0] + pw[:, 1] * (F - Flo) / (Fhi - Flo),
                                                                   pw[:, 0] + pw[:, 1] + pw[:, 2] * (F - Fhi) / (1 - Fhi)))
    zN = ndtri(np.clip(pit, 1e-300, 1 - 1e-16))
    return pd.DataFrame({'lp': lp, 'zN': zN, 'm': m, 'l': l, 'pit': pit}, index=x.index)


if __name__ == '__main__':
    varianten = [
        ('basis (geen GAS)', 'rijder', None, 0, 0.0),
        ('A volatiliteit, t-score', 'rijder', False, 0, 0.0),
        ('B volatiliteit, snel onbegrensd', 'rijder', False, 1, 3.0),
        ('C niveau+vol, t-score', 'rijder', True, 0, 0.0),
        ('D niveau+vol, snel onbegrensd', 'rijder', True, 1, 3.0),
        ('E = D per rijder x afstand', 'rijder_afstand', True, 1, 3.0),
        ('F = B per rijder x afstand', 'rijder_afstand', False, 1, 3.0),
    ]
    sel = sys.argv[1:] or [v[0][0] for v in varianten]
    uitk = {}
    for nm, per, met_m, modus, kap in varianten:
        if nm[0] not in sel and not nm.startswith('basis'):
            continue
        if met_m is None:
            par = [0, 0, 0, 0, 0, 0]; ll = None
        else:
            par, ll = schat(per, met_m, modus, kap)
        R = rij_scores(per, par, modus, kap)
        R.to_pickle(f'gas_res_{nm[0]}.pkl'); uitk[nm] = par
        print(f'{nm:34s} par(phi_m, phi_l, am_snel, am_traag, al_snel, al_traag)={np.round(par, 4).tolist()}  fit-ll={ll}  ({time.time()-t0:.0f}s)', flush=True)
    json.dump({k: list(map(float, v)) for k, v in uitk.items()}, open('gas_par.json', 'w'))
