"""Diagnose: zetten pieken in vorm door? Voor elke rit met kans <= 1/50 snel (of traag): hoe zijn de volgende ritten van dezelfde rijder
(1-7, 8-30, 31-90 dagen later) verdeeld t.o.v. de voorspelling? Bij goede vorm-tracking: z van de volgende ritten gemiddeld 0 en
1/20 snel / traag in 5%. Ook: kalibratie per niveau in de fitjaren en de toetsjaren (vormfout of verschuiving in de tijd)."""
import sys
import numpy as np, pandas as pd
from scipy.stats import norm
V = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'voorspel2_uitkomst.parquet', columns=['skater_id', 'dag', 'jaar', 'baan', 'tijd', 'pred', 'pit', 'ok', 'k_val', 'k_c'])
V = V[V['ok'] & np.isfinite(V['pit']) & ~V['k_val']].sort_values(['skater_id', 'dag'], kind='stable').reset_index(drop=True)
V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9))
V['niv'] = pd.cut(V['k_c'], [-1, 0.02, 0.04, 0.07, 0.12, 0.2, 1], labels=['<2%', '2-4%', '4-7%', '7-12%', '12-20%', '>20%'])
f = lambda x: pd.Series({'n': len(x), 'snel1/1000': (x.pit <= 1e-3).mean() / 1e-3, 'snel1/100': (x.pit <= .01).mean() / .01, 'snel1/20': (x.pit <= .05).mean() / .05,
                         'traag1/20': (x.pit >= .95).mean() / .05, 'traag1/100': (x.pit >= .99).mean() / .01, 'traag1/1000': (x.pit >= .999).mean() / 1e-3})
for nm, s in (('fit 2012-2019', V['jaar'].between(2012, 2019)), ('toets 2020+', V['jaar'] >= 2020)):
    print(nm, '(zonder eigen valpartijen)'); print(V[s].groupby('niv', observed=True).apply(f).round(2).to_string())
# vorm na een schok: volgende ritten van dezelfde rijder
d = V[['skater_id', 'dag', 'z', 'pit', 'k_c']].to_numpy()
sid = V['skater_id'].to_numpy(); dag = V['dag'].to_numpy(); z = V['z'].to_numpy(); pit = V['pit'].to_numpy(); kc = V['k_c'].to_numpy(); jr = V['jaar'].to_numpy()
start = np.r_[0, np.flatnonzero(sid[1:] != sid[:-1]) + 1]; eind = np.r_[start[1:], len(V)]
rijen = []
for s_, e_ in zip(start, eind):
    for i in range(s_, e_):
        schok = 'snel' if pit[i] <= 0.02 else ('traag' if pit[i] >= 0.98 else None)
        if schok is None:
            continue
        for j in range(i + 1, e_):
            dt = dag[j] - dag[i]
            if dt <= 0: continue
            if dt > 90: break
            rijen.append((schok, '1-7' if dt <= 7 else ('8-30' if dt <= 30 else '31-90'), z[j], pit[j], kc[i] < 0.06, jr[j] >= 2020))
R = pd.DataFrame(rijen, columns=['schok', 'later', 'z', 'pit', 'top', 'toets'])
R = R[R.toets]                                   # alleen volgende ritten in de toetsjaren
g = lambda x: pd.Series({'n': len(x), 'gem z': x.z.mean(), 'snel1/20': (x.pit <= .05).mean() / .05, 'traag1/20': (x.pit >= .95).mean() / .05})
for top in (True, False):
    print('\nna een schok (kans <= 1/50), volgende ritten van dezelfde rijder,', 'toppers (<6%)' if top else 'overige rijders')
    print(R[R.top == top].groupby(['schok', 'later']).apply(g).round(3).to_string())

# aantallen per rijder x seizoen (toetsjaren): clusteren de uitschieters bij dezelfde rijders meer dan de kansen zeggen?
T = V[V['jaar'] >= 2020].copy(); T['top'] = T['k_c'] < 0.06
for q, nm in ((0.05, 'snel1/20'), (0.01, 'snel1/100')):
    T['x'] = (T['pit'] <= q).astype(float)
    A = T.groupby(['skater_id', 'jaar', 'top']).agg(n=('x', 'size'), k=('x', 'sum')).reset_index()
    A = A[A.n >= 5]
    for top in (True, False):
        a = A[A.top == top]; ev = a.n * q; vv = a.n * q * (1 - q)
        print(f"{nm} per rijder x seizoen ({'toppers' if top else 'overigen'}, >=5 ritten): {len(a)} seizoenen, "
              f"waargenomen/verwacht {a.k.sum() / ev.sum():.2f}, spreiding (var/binomiaal) {((a.k - ev) ** 2).sum() / vv.sum():.2f}, "
              f"seizoenen met >= 3x verwacht en >= 2: {((a.k >= 3 * ev) & (a.k >= 2)).sum()} (onder binomiaal ~"
              f"{sum(1 - __import__('scipy.stats', fromlist=['binom']).binom.cdf(max(np.ceil(3 * n * q), 2) - 1, n, q) for n in a.n):.0f})")
