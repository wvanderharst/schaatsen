"""Proef: dikkere staarten. Verschillende verdelingen voor de gestandaardiseerde fout u = (ln t - voorspelling) / sd, per afstandsgroep,
parameters lineair in Z = [1, exp(-niveau/4%), niveau, jong]; geschat op 2008-2019 (toppers dubbel), getoetst op 2020+.
  tpt   : twee-delige Student-t (zoals het model): per kant schaal en vrijheidsgraden, plus de massa links van 0;
  tpgt  : twee-delige gegeneraliseerde t (McDonald-Newey): per kant schaal, piek p en staart q (p = 2: Student-t met nu = 2q);
  mix   : twee-delige Student-t met een brede bijmenging: (1 - pi) f(u) + pi f(u / c) / c, pi lineair in Z (logit), c geschat.
Log loss in ln-tijd-eenheden (-ln f(u) + ln sd).
Gebruik: python verdeling_lab.py <uitkomst.parquet> <familie>   (per familie, parallel; schrijft verdeling_<familie>.npz)
         python verdeling_lab.py <uitkomst.parquet> samen      (vergelijking)"""
import os, sys, importlib.util
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.special import gammaln, betaln, betainc
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location('em', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite', 'elite_model.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staartijking as si
GROEP = {'500m': 'sprint', '1000m': 'sprint', '1500m': 'midden', '3000m': 'midden', '5000m': 'lang', '10000m': 'lang'}
JONG = ['jeugd', 'junior_C', 'junior_B', 'junior_A', 'neo']


def gt_logpdf(x, s, p, q):
    """Symmetrische gegeneraliseerde t: ln f(x) = ln p - ln(2 s q^(1/p) B(1/p, q)) - (q + 1/p) ln(1 + |x/s|^p / q)."""
    a = np.abs(x / s)
    return np.log(p) - np.log(2 * s) - np.log(q) / p - betaln(1 / p, q) - (q + 1 / p) * np.log1p(a ** p / q)


def gt_tail(x, s, p, q):
    """P(|X| <= |x|) = I_z(1/p, q), z = w / (1 + w), w = |x/s|^p / q."""
    w = np.abs(x / s) ** p / q
    return betainc(1 / p, q, w / (1 + w))


def t_logpdf(x, s, nu):
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s) - (nu + 1) / 2 * np.log1p((x / s) ** 2 / nu)


NU_L = 30.0                                            # bovengrens vrijheidsgraden aan de snelle kant (familie tptL<grens>: geforceerd dikker)


def nu_(x, hi=30.0):
    return 1.5 + (hi - 1.5) / (1 + np.exp(-x))          # zoals in het model: vrijheidsgraden tussen 1,5 en 30


def params(theta, Z, n):
    return [Z @ theta[i * Z.shape[1]:(i + 1) * Z.shape[1]] for i in range(n)]


def logpdf(fam, theta, u, Z):
    if fam.startswith('tpt'):
        a, lsL, lnL, lsU, lnU = params(theta, Z, 5)
        p0 = 1 / (1 + np.exp(-a)); sL, sU = np.exp(lsL), np.exp(lsU); nL, nU = nu_(lnL, NU_L), nu_(lnU)
        lo = u < 0
        return np.where(lo, np.log(2 * p0) + t_logpdf(u, sL, nL), np.log(2 * (1 - p0)) + t_logpdf(u, sU, nU))
    if fam == 'tpgt':
        a, lsL, lpL, lqL, lsU, lpU, lqU = params(theta, Z, 7)
        p0 = 1 / (1 + np.exp(-a)); sL, sU = np.exp(lsL), np.exp(lsU)
        pL, pU = 0.5 + np.exp(lpL), 0.5 + np.exp(lpU); qL, qU = 0.3 + np.exp(lqL), 0.3 + np.exp(lqU)
        lo = u < 0
        return np.where(lo, np.log(2 * p0) + gt_logpdf(u, sL, pL, qL), np.log(2 * (1 - p0)) + gt_logpdf(u, sU, pU, qU))
    if fam == 'mix':
        k = Z.shape[1]; base = logpdf('tpt', theta[:5 * k], u, Z)
        pi = 1 / (1 + np.exp(-(Z @ theta[5 * k:6 * k]))); c = 1.5 + np.exp(theta[6 * k])
        breed = logpdf('tpt', theta[:5 * k], u / c, Z) - np.log(c)
        return np.logaddexp(np.log1p(-pi) + base, np.log(pi) + breed)


def cdf(fam, theta, u, Z):
    if fam.startswith('tpt'):
        a, lsL, lnL, lsU, lnU = params(theta, Z, 5)
        from scipy.stats import t as T
        p0 = 1 / (1 + np.exp(-a)); sL, sU = np.exp(lsL), np.exp(lsU); nL, nU = nu_(lnL, NU_L), nu_(lnU)
        return np.where(u < 0, p0 * 2 * T.cdf(u / sL, nL), p0 + (1 - p0) * (2 * T.cdf(u / sU, nU) - 1))
    if fam == 'tpgt':
        a, lsL, lpL, lqL, lsU, lpU, lqU = params(theta, Z, 7)
        p0 = 1 / (1 + np.exp(-a)); sL, sU = np.exp(lsL), np.exp(lsU)
        pL, pU = 0.5 + np.exp(lpL), 0.5 + np.exp(lpU); qL, qU = 0.3 + np.exp(lqL), 0.3 + np.exp(lqU)
        return np.where(u < 0, p0 * (1 - gt_tail(u, sL, pL, qL)), p0 + (1 - p0) * gt_tail(u, sU, pU, qU))
    if fam == 'mix':
        k = Z.shape[1]; pi = 1 / (1 + np.exp(-(Z @ theta[5 * k:6 * k]))); c = 1.5 + np.exp(theta[6 * k])
        return (1 - pi) * cdf('tpt', theta[:5 * k], u, Z) + pi * cdf('tpt', theta[:5 * k], u / c, Z)


def start(fam, k):
    z = lambda v: [v] + [0.0] * (k - 1)
    if fam.startswith('tpt'):
        return np.array(z(0.0) + z(np.log(0.8)) + z(-1.5) + z(np.log(1.0)) + z(-2.0))
    if fam == 'tpgt':
        return np.array(z(0.0) + z(np.log(0.8)) + z(np.log(1.5)) + z(np.log(2.0)) + z(np.log(1.0)) + z(np.log(1.5)) + z(np.log(1.5)))
    if fam == 'mix':
        return np.r_[start('tpt', k), z(-3.0), [np.log(1.5)]]


def fit(fam, u, Z):
    f = lambda th: -np.mean(logpdf(fam, th, u, Z))
    th = minimize(f, start(fam, Z.shape[1]), method='L-BFGS-B', options={'maxiter': 3000}).x
    return minimize(f, th, method='Powell', options={'maxiter': 20000, 'xtol': 1e-4, 'ftol': 1e-9}).x


def kort(r):
    if 'rijders' in r:
        return r.split('met ')[1].split(' ritten')[0] + ':' + r.split(':')[1].split(',')[0].strip() + '/' + r.split('verwacht')[1].split()[0]
    return r.split('kans<=')[1].split(':')[0].strip() + ' ' + r.split('(')[1].split(')')[0]


if __name__ == '__main__':
    pad = sys.argv[1]
    V = pq.read_table(pad, columns=['skater_id', 'dag', 'jaar', 'seizoen', 'afstand', 'tijd', 'groep', 'u', 'sd', 'ok', 'k_val', 'k_c']).to_pandas()
    V['afstand'] = V['afstand'].astype(str); V['groep'] = V['groep'].astype(str)
    V = V[V['ok'] & np.isfinite(V['u']) & V['afstand'].isin(list(GROEP))].reset_index(drop=True)
    O = pd.read_parquet(os.path.join(os.path.dirname(pad) or '.', 'oud_vooraf.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'll', 'niveau_pct'])
    O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']); O['afstand'] = O['afstand'].astype(str)
    V = V.merge(O, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    gq = V['k_c'].clip(0, 0.4).fillna(0.2).to_numpy()
    Z = np.column_stack([np.ones(len(V)), np.exp(-gq / 0.04), gq, V['groep'].isin(JONG).to_numpy().astype(float)])
    u = V['u'].to_numpy(); ag = V['afstand'].map(GROEP).to_numpy(); jr = V['jaar'].to_numpy()
    fitm = (jr >= 2008) & (jr <= 2019) & ~V['k_val'].to_numpy(); toets = (jr >= 2020) & np.isfinite(V['ll'].to_numpy())
    res = {}
    wat = sys.argv[2]
    fams = [] if wat == 'samen' else [wat]
    if wat == 'samen':
        for f in sorted(os.listdir(os.path.dirname(pad) or '.')):
            if f.startswith('verdeling_') and f.endswith('.npz'):
                d = np.load(os.path.join(os.path.dirname(pad) or '.', f))
                for nm in d.files:
                    if nm.endswith('_ll'):
                        res[nm[:-3]] = (d[nm], d[nm[:-3] + '_pit'])
    for fam in fams:
        if fam.startswith('tptL'):
            NU_L = float(fam[4:])
        ll = np.full(len(V), np.nan); pit = np.full(len(V), np.nan)
        for g in ('sprint', 'midden', 'lang'):
            idx = np.flatnonzero(fitm & (ag == g)); idx = np.random.default_rng(0).choice(idx, min(len(idx), 150000), replace=False)
            top = Z[idx, 1] > 0.3; ii = np.r_[idx, idx[top]]
            th = fit(fam, u[ii], Z[ii])
            s = ag == g; ll[s] = -logpdf(fam, th, u[s], Z[s]) + np.log(V['sd'].to_numpy()[s]); pit[s] = cdf(fam, th, u[s], Z[s])
            k = Z.shape[1]; v = lambda i: th[i * k:(i + 1) * k]
            if fam.startswith('tpt') or fam == 'mix':
                print(f'  {g}: {fam} elite (Z1=1, niveau 0): nuL {nu_(v(2)[0] + v(2)[1], NU_L):.2f} nuU {nu_(v(4)[0] + v(4)[1]):.2f} | breed (niveau 20%): nuL {nu_(v(2)[0] + 0.2 * v(2)[2], NU_L):.2f} nuU {nu_(v(4)[0] + 0.2 * v(4)[2]):.2f}'
                      + (f' | pi elite {1 / (1 + np.exp(-(v(5)[0] + v(5)[1]))):.4f} c {1.5 + np.exp(th[6 * k]):.2f}' if fam == 'mix' else ''), flush=True)
            if fam == 'tpgt':
                print(f'  {g}: tpgt elite (Z1=1, niveau 0): pL {0.5 + np.exp(v(2)[0] + v(2)[1]):.2f} qL {0.3 + np.exp(v(3)[0] + v(3)[1]):.2f} | pU {0.5 + np.exp(v(5)[0] + v(5)[1]):.2f} qU {0.3 + np.exp(v(6)[0] + v(6)[1]):.2f}', flush=True)
        # snelle staart voorbij 2% (zoals model2): p' = Q (p/Q)^b, b = b0 + b1 w, geschat op de fitjaren
        Q = 0.02; ft = fitm & (pit < Q) & np.isfinite(pit); xs = -np.log(pit[ft] / Q); ws = Z[ft, 1]
        bb = minimize(lambda c: -np.mean(np.log(np.maximum(c[0] + c[1] * ws, 1e-3)) - np.maximum(c[0] + c[1] * ws, 1e-3) * xs), np.array([1.0, 0.0]), method='Nelder-Mead').x
        br = np.maximum(bb[0] + bb[1] * Z[:, 1], 0.2); st = (pit < Q) & np.isfinite(pit); pc = np.clip(pit, 1e-300, 1)
        ll = np.where(st, ll - np.log(br) - (br - 1) * np.log(pc / Q), ll); pit = np.where(st, Q * (pc / Q) ** br, pit)
        res[fam] = (ll.copy(), pit.copy())
        okp = np.isfinite(pit)
        p2, d, _ = si.ijk(V['dag'].to_numpy(), V['k_c'].to_numpy(), np.where(okp, pit, np.nan), okp & ~V['k_val'].to_numpy())
        res[fam + '+ijk'] = (np.where(okp, ll + d, ll), np.where(okp, p2, pit))
        print(fam, 'klaar | staart b0 %.3f b1 %.3f' % tuple(bb), flush=True)
        np.savez(os.path.join(os.path.dirname(pad) or '.', f'verdeling_{fam}.npz'), **{f'{nm}_{x}': v for nm in (fam, fam + '+ijk') for x, v in zip(('ll', 'pit'), res[nm])})
    if wat != 'samen':
        sys.exit(0)
    T = toets
    print('\nlog loss 2020+ (basisverdeling, zonder vormlaag en ijkingen):')
    for fam, (ll, pit) in res.items():
        print(f"  {fam:9s} | alle {np.nanmean(ll[T]):.4f} | toppers {np.nanmean(ll[T & (V['niveau_pct'] >= 90).to_numpy()]):.4f} | elite {np.nanmean(ll[T & (V['niveau_pct'] >= 97).to_numpy()]):.4f}")
    print('\nkalibratie 2020+:')
    for nm, s in (('toppers', T & (V['niveau_pct'] >= 90).to_numpy()), ('elite', T & (V['niveau_pct'] >= 97).to_numpy())):
        for fam, (ll, pit) in res.items():
            x = V.loc[s, ['skater_id', 'seizoen']].copy(); x['p'] = pit[s]
            fout, rg, _, _ = em.toets(x, vanaf_seizoen=2019)
            print(f'  {nm:8s} {fam:9s} fouten {fout:2d} | snel ' + ' '.join(kort(r) for r in rg[:8]) + ' | traag ' + ' '.join(kort(r) for r in rg[8:11]))
