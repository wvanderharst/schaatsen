"""Diagnose: is er te weinig lange-termijngeheugen? Alleen EERDERE ritten van de rijder op dezelfde afstand.
1. persistentie: gemiddelde z (na het hele model) in vensters 90-365 en 365-730 dagen terug, tegen z nu;
2. bewezen niveau: beste en op een na beste kwaliteit q = ln t - theta (achteraf, gecorrigeerd voor alles) in de laatste 730 dagen,
   min de verwachting vooraf (pred - th_hat_veld). Negatief = ooit duidelijk sneller dan nu verwacht.
Gebruik: python geheugen_diag.py <werkmap>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import norm
from numba import njit

w = sys.argv[1]
V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'dag', 'jaar', 'afstand', 'tijd', 'pred', 'pit', 'theta',
                                                                      'th_hat_veld', 'k_c', 'ok', 'k_val', 'baan']).to_pandas()
V['afstand'] = V['afstand'].astype(str); V['baan'] = V['baan'].astype(str)
V = V[V['ok'] & ~V['k_val'] & np.isfinite(V['pit']) & np.isfinite(V['theta']) & V['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m', '10000m'])].copy()
V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9)).clip(-4, 4)
V['e'] = np.log(V['tijd']) - V['pred']
V['q'] = np.log(V['tijd']) - V['theta']; V['qx'] = V['pred'] - V['th_hat_veld']
V = V.sort_values(['skater_id', 'afstand', 'dag']).reset_index(drop=True)
gid = V.groupby(['skater_id', 'afstand'], sort=False).ngroup().to_numpy().astype(np.int64)


@njit(cache=True)
def _venster(gid, dag, x, W, uit_s, uit_n, uit_m1, uit_m2):
    """Per rit: som, aantal, kleinste en op een na kleinste x over eerdere dagen (dag - W <= d < dag) in dezelfde groep."""
    n = len(x)
    for i in range(n):
        s = 0.0; c = 0.0; m1 = 1e9; m2 = 1e9; j = i - 1
        while j >= 0 and gid[j] == gid[i] and dag[j] >= dag[i] - W:
            if dag[j] < dag[i] and x[j] == x[j]:
                s += x[j]; c += 1.0
                if x[j] < m1:
                    m2 = m1; m1 = x[j]
                elif x[j] < m2:
                    m2 = x[j]
            j -= 1
        uit_s[i] = s; uit_n[i] = c; uit_m1[i] = m1 if m1 < 1e8 else np.nan; uit_m2[i] = m2 if m2 < 1e8 else np.nan


def venster(kol, W):
    o = [np.zeros(len(V)) for _ in range(4)]
    _venster(gid, V['dag'].to_numpy().astype(np.float64), V[kol].to_numpy().astype(np.float64), float(W), *o)
    return o


for W in (90, 365, 730):
    sm, n_, _, _ = venster('z', W); V[f's{W}'] = sm; V[f'n{W}'] = n_
V['zm90'] = V['s90'] / V['n90'].where(V['n90'] > 0)
V['n_a'] = V['n365'] - V['n90']; V['z_a'] = (V['s365'] - V['s90']) / V['n_a'].where(V['n_a'] > 0)
V['n_b'] = V['n730'] - V['n365']; V['z_b'] = (V['s730'] - V['s365']) / V['n_b'].where(V['n_b'] > 0)
_, _, V['qbest'], V['q2'] = venster('q', 730)
V['gat1'] = V['qbest'] - V['qx']; V['gat2'] = V['q2'] - V['qx']
T = V[(V['jaar'] >= 2020)]
for nm, s in (('top < 3,5%', T['k_c'] < 0.035), ('3,5-10%', (T['k_c'] >= 0.035) & (T['k_c'] < 0.1)), ('alle', T['k_c'] > -9)):
    x = T[s]
    print(f'\n== {nm} (n = {len(x)}) ==')
    for k, n in (('zm90', 'n90'), ('z_a', 'n_a'), ('z_b', 'n_b')):
        y = x[(x[n] >= 3) & np.isfinite(x[k])]
        b = np.polyfit(y[k], y['z'], 1)[0]
        print(f'  persistentie {k:5s} (>=3 ritten): r = {np.corrcoef(y[k], y["z"])[0, 1]:+.3f}, helling {b:+.3f}, n = {len(y)}')
    y = x[np.isfinite(x['gat2'])].copy()
    y['bin'] = pd.cut(100 * (np.exp(y['gat2']) - 1), [-50, -4, -3, -2, -1, 0, 1, 50])
    t = y.groupby('bin', observed=True).agg(n=('z', 'size'), z=('z', 'mean'), e_pct=('e', lambda s: 100 * s.mean()), snel20=('pit', lambda s: (s < 0.05).mean() / 0.05))
    print('  op een na beste kwaliteit (730 d) t.o.v. verwachting vooraf, in %  ->  gem. z, gem. fout %, snel 1/20 (x verwacht):')
    print('  ' + t.round(3).to_string().replace('\n', '\n  '))
