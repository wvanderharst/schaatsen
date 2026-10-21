"""Proef: vormlaag op de kansschaal. z = Phi^-1(PIT) van het model; de vormlaag maakt z' = (z - mu) / sigma met mu en log sigma lineair
in kenmerken uit EERDERE ritten (strikt eerdere dagen) van dezelfde rijder en van het veld. Dit is een geldige verdeling voor de tijd
(Gauss-copula op de modelverdeling), en de log loss verandert exact met log phi(z') - log sigma - log phi(z).

Kenmerken (uitdovend met tijdconstante T dagen, 7 / 30 / 90):
- rijder: gemiddelde z (gekrompen), snelle en trage schokken (deel voorbij 2), extra variantie (z^2 - 1), aantal;
  apart het racedeel van z (gemiddelde z van de race: ijs / veldpeil) en het eigen deel (z - racegemiddelde);
- veld: het gemiddelde over de startlijst van het rijdergeheugen (het circuit is recent sneller of trager dan voorspeld)."""
import os, sys
import numpy as np, pandas as pd
from numba import njit
from scipy.stats import norm
from scipy.optimize import minimize

TS = (7.0, 30.0, 90.0)


@njit(cache=True)
def _geheugen(rid, dag, X, T, out, cnt):
    """rid/dag gesorteerd per rijder en dag. out[i] = som over eerdere dagen van de rijder van exp(-dt/T) * X[j] (NaN overgeslagen)."""
    n, p = X.shape
    s = np.zeros(p); c = 0.0; td = -1e9; ds = np.zeros(p); dc = 0.0; dd = -1e9; r0 = -1
    for i in range(n):
        if rid[i] != r0:
            s[:] = 0.0; c = 0.0; td = -1e9; ds[:] = 0.0; dc = 0.0; dd = -1e9; r0 = rid[i]
        if dag[i] != dd:
            if dd > -1e8:                                  # dag dd afsluiten
                f = np.exp(-(dd - td) / T) if td > -1e8 else 0.0
                for k in range(p):
                    s[k] = s[k] * f + ds[k]
                c = c * f + dc; td = dd
            ds[:] = 0.0; dc = 0.0; dd = dag[i]
        f = np.exp(-(dag[i] - td) / T) if td > -1e8 else 0.0
        for k in range(p):
            out[i, k] = s[k] * f
        cnt[i] = c * f
        ok = True
        for k in range(p):
            if X[i, k] != X[i, k]:
                ok = False
        if ok:
            for k in range(p):
                ds[k] += X[i, k]
            dc += 1.0


def laad(pad):
    V = pd.read_parquet(pad)
    V = V[V['ok'] & np.isfinite(V['pit'])].copy()
    if 'rk_' not in V:
        V['rk_'] = V.groupby(['dag', 'baan', 'gender', 'afstand'], sort=False).ngroup()
    V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9)).clip(-6, 6)
    return V.sort_values(['skater_id', 'dag'], kind='stable').reset_index(drop=True)


def kenmerken(V, k0=2.0):
    z = V['z'].to_numpy(); val = V['k_val'].to_numpy()
    zm = V.groupby('rk_')['z'].transform('mean').to_numpy(); nr = V.groupby('rk_')['z'].transform('size').to_numpy()
    zr = np.where(nr >= 4, zm, np.nan)                           # racedeel (ijs / veldpeil), alleen bij >= 4 rijders
    zi = np.where(np.isfinite(zr), z - zr, z)
    zz = np.where(val, np.nan, z)                                # eigen valpartijen tellen niet mee
    X = np.column_stack([zz, np.minimum(zz + 2, 0), np.maximum(zz - 2, 0), np.minimum(zz ** 2, 25) - 1,
                         np.where(val, np.nan, np.nan_to_num(zr)), np.where(val, np.nan, zi)])
    rid = V['skater_id'].astype('category').cat.codes.to_numpy().astype(np.int64); dag = V['dag'].to_numpy(float)
    F = {}
    for T in TS:
        out = np.zeros_like(X); cnt = np.zeros(len(V))
        _geheugen(rid, dag, X, T, out, cnt)
        t = int(T)
        F[f'm{t}'] = out[:, 0] / (cnt + k0); F[f'neg{t}'] = out[:, 1]; F[f'pos{t}'] = out[:, 2]
        F[f'v{t}'] = out[:, 3] / (cnt + k0); F[f'n{t}'] = np.log1p(cnt)
        F[f'mr{t}'] = out[:, 4] / (cnt + k0); F[f'mi{t}'] = out[:, 5] / (cnt + k0)
    F = pd.DataFrame(F, index=V.index)
    for t in (int(T) for T in TS):                                # veld: gemiddelde rijdergeheugen over de startlijst
        F[f'veld_m{t}'] = F[f'm{t}'].groupby(V['rk_']).transform('mean')
        F[f'veld_mr{t}'] = F[f'mr{t}'].groupby(V['rk_']).transform('mean')
    return F


def nll(p, z, Xm, Xs):
    km = Xm.shape[1]; mu = Xm @ p[:km]; ls = Xs @ p[km:]
    zp = (z - mu) * np.exp(-ls)
    return np.mean(0.5 * zp ** 2 + ls - 0.5 * z ** 2)            # verschil in log loss (negatief = beter)


def fit(z, Xm, Xs, ridge=1e-3):
    p0 = np.zeros(Xm.shape[1] + Xs.shape[1])
    f = lambda p: nll(p, z, Xm, Xs) + ridge * np.sum(p ** 2)
    def g(p):
        km = Xm.shape[1]; mu = Xm @ p[:km]; ls = Xs @ p[km:]; e = np.exp(-ls); zp = (z - mu) * e
        gm = -(zp * e) @ Xm / len(z); gs = (1 - zp ** 2) @ Xs / len(z)
        return np.r_[gm, gs] + 2 * ridge * p
    return minimize(f, p0, jac=g, method='L-BFGS-B').x


def ontwerp(V, F, loc, sch, top_inter=True):
    wg = np.exp(-V['k_c'].clip(0, 0.5).fillna(0.2).to_numpy() / 0.04)
    cols_m = [np.ones(len(V))] + [F[c].to_numpy() for c in loc]; nm_m = ['1'] + list(loc)
    cols_s = [np.ones(len(V))] + [F[c].to_numpy() for c in sch]; nm_s = ['1'] + list(sch)
    if top_inter:
        cols_m += [wg] + [F[c].to_numpy() * wg for c in loc]; nm_m += ['w'] + [c + '_w' for c in loc]
        cols_s += [wg] + [F[c].to_numpy() * wg for c in sch]; nm_s += ['w'] + [c + '_w' for c in sch]
    return np.nan_to_num(np.column_stack(cols_m)), np.nan_to_num(np.column_stack(cols_s)), nm_m, nm_s


def staarten(pit):
    return {k: (f(pit)).mean() / q for k, q, f in (('s1000', 1e-3, lambda p: p <= 1e-3), ('s100', 1e-2, lambda p: p <= 1e-2), ('s20', .05, lambda p: p <= .05),
                                                    ('t20', .05, lambda p: p >= .95), ('t100', 1e-2, lambda p: p >= .99), ('t1000', 1e-3, lambda p: p >= .999))}


KOLOM = ['skater_id', 'dag', 'jaar', 'seizoen', 'baan', 'gender', 'afstand', 'tijd', 'pit', 'nll', 'ok', 'k_val', 'k_c']


def laad_toets(pad, oud='oud_vooraf.parquet'):
    import pyarrow.parquet as pq
    kol = KOLOM + (['rk_'] if 'rk_' in pq.read_schema(pad).names else [])
    V = pd.read_parquet(pad, columns=kol)
    V = V[V['ok'] & np.isfinite(V['pit'])].copy(); V['afstand'] = V['afstand'].astype(str)
    if 'rk_' not in V:
        V['rk_'] = V.groupby(['dag', 'baan', 'gender', 'afstand'], sort=False).ngroup()
    V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9)).clip(-6, 6)
    O = pd.read_parquet(oud, columns=['skater_id', 'dag', 'afstand', 'tijd', 'll', 'niveau_pct'])
    O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']); O['afstand'] = O['afstand'].astype(str)
    V = V.merge(O, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    return V.sort_values(['skater_id', 'dag'], kind='stable').reset_index(drop=True)


def verslag(V, pit2, d, m, label):
    import importlib.util
    spec = importlib.util.spec_from_file_location('em', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite', 'elite_model.py'))
    em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
    uit = []
    for nm, s in (('alle', m), ('toppers', m & (V['niveau_pct'] >= 90).to_numpy()), ('elite', m & (V['niveau_pct'] >= 97).to_numpy())):
        uit.append(f"{nm} {V['nll'].to_numpy()[s].mean():.4f} -> {V['nll'].to_numpy()[s].mean() + d[s].mean():.4f}")
    print(label, '| log loss', ' | '.join(uit))
    for nm, s in (('toppers', m & (V['niveau_pct'] >= 90).to_numpy()), ('elite', m & (V['niveau_pct'] >= 97).to_numpy())):
        for k, p in (('voor', V['pit'].to_numpy()), ('na', pit2)):
            x = V[s][['skater_id', 'seizoen']].copy(); x['p'] = p[s]
            fout, rg, _, _ = em.toets(x, vanaf_seizoen=2019)
            kort = [r.split(':')[0].replace('kans<=', '').strip() + ' ' + (r.split('(')[1].split(')')[0] if '(' in r else r.split(':')[1].split(',')[0].strip() + '/' + r.split('verwacht')[1].split()[0]) for r in rg]
            print(f'   {nm:8s} {k:4s} fouten {fout:2d} | ' + ' '.join(kort))


if __name__ == '__main__':
    V = laad_toets(sys.argv[1] if len(sys.argv) > 1 else 'voorspel2_uitkomst.parquet')
    F = kenmerken(V)
    jr = V['jaar'].to_numpy(); z = V['z'].to_numpy()
    fitm = (jr >= 2012) & (jr <= 2019) & ~V['k_val'].to_numpy()
    toets = (jr >= 2020) & np.isfinite(V['ll'].to_numpy())
    M = ['m7', 'm30', 'm90', 'neg7', 'neg30', 'neg90', 'pos7', 'pos30', 'pos90']
    varianten = {
        'rijder z': (M, ['v7', 'v30', 'v90', 'n30']),
        '+ veld': (M + ['veld_m7', 'veld_m30', 'veld_m90', 'veld_mr7', 'veld_mr30', 'veld_mr90'], ['v7', 'v30', 'v90', 'n30']),
        'alleen loc + veld': (M + ['veld_m7', 'veld_m30', 'veld_m90', 'veld_mr7', 'veld_mr30', 'veld_mr90'], []),
    }
    for nm, (loc, sch) in varianten.items():
        Xm, Xs, nm_m, nm_s = ontwerp(V, F, loc, sch)
        p = fit(z[fitm], Xm[fitm], Xs[fitm])
        km = Xm.shape[1]; mu = Xm @ p[:km]; ls = Xs @ p[km:]; zp = (z - mu) * np.exp(-ls)
        d = 0.5 * zp ** 2 + ls - 0.5 * z ** 2; pit2 = norm.cdf(zp)
        verslag(V, pit2, d, toets, nm)
        print('   loc', ' '.join(f'{a} {b:+.3f}' for a, b in zip(nm_m, p[:km])))
        print('   sch', ' '.join(f'{a} {b:+.3f}' for a, b in zip(nm_s, p[km:])), flush=True)
