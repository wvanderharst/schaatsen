"""Junioren in het ritmodel: groeien ze sneller (of trager) dan voorspeld, en verandert dat over de seizoenen? Gemiddelde z = Phi^-1(pit)
(negatief = sneller dan voorspeld) en de staarten, naar leeftijd x niveau vooraf (k_c) en naar seizoen; zonder valpartijen. Ook per
rijder: de gemiddelde z over een seizoen (persistent onderschat = groei gemist). Gebruik: python junior_trend.py <werkmap> [uitkomst]"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import norm
w = sys.argv[1]; pad = sys.argv[2] if len(sys.argv) > 2 else os.path.join(w, 'voorspel2_uitkomst.parquet')
V = pq.read_table(pad, columns=['skater_id', 'gender', 'groep', 'dag', 'jaar', 'seizoen', 'afstand', 'pit', 'ok', 'k_val', 'k_c', 'lf']).to_pandas()
V = V[V['ok'] & np.isfinite(V['pit']) & ~V['k_val'] & np.isfinite(V['lf'])].copy()
V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9)).clip(-4, 4)
V['lft'] = np.floor(V['lf']).astype(int)
V['niveau'] = pd.cut(V['k_c'], [-1, 0.03, 0.06, 0.10, 0.18, 9], labels=['<3%', '3-6%', '6-10%', '10-18%', '>18%'])


def blok(x):
    return f"{x['z'].mean():+.3f} ({len(x)})"


print('gem. z naar leeftijd x niveau vooraf (alle seizoenen):')
A = V[V['lft'].between(12, 30)]
t = A.groupby(['lft', 'niveau'], observed=True)['z'].agg(['mean', 'size']).unstack()
print(t['mean'].round(3).to_string()); print('aantal'); print(t['size'].to_string())
print('\nsnel 1/20 en 1/100 (waargenomen / verwacht) naar leeftijd, niveau < 10%:')
for l, x in A[A['k_c'] < 0.10].groupby('lft'):
    print(f"  {l}: {(x['pit'] < .05).mean() / .05:.2f} {(x['pit'] < .01).mean() / .01:.2f} | traag {(x['pit'] > .95).mean() / .05:.2f} {(x['pit'] > .99).mean() / .01:.2f} (n {len(x)})")
print('\ngem. z per seizoen voor junioren (16-19 jaar) naar niveau:')
J = V[V['lft'].between(16, 19)]
print(J.groupby(['jaar', 'niveau'], observed=True)['z'].mean().unstack().round(3).to_string())
print('\nper rijder-seizoen (16-19 jaar, >= 8 ritten): verdeling van de gemiddelde z (persistent onderschat = groei gemist)')
P = J.groupby(['skater_id', 'seizoen']).agg(z=('z', 'mean'), n=('z', 'size'), k=('k_c', 'median'))
P = P[P['n'] >= 8]
P['nv'] = pd.cut(P['k'], [-1, 0.06, 0.10, 0.18, 9], labels=['<6%', '6-10%', '10-18%', '>18%'])
for nv, x in P.groupby('nv', observed=True):
    sdv = np.sqrt((x['z'] - x['z'].mean()).pow(2).mean() - (1.0 / x['n']).mean())
    print(f"  {nv:6s}: gem {x['z'].mean():+.3f}, sd tussen rijder-seizoenen {x['z'].std():.3f} (ruis alleen {np.sqrt((1 / x['n']).mean()):.3f}; echte spreiding ~{sdv:.3f}), n {len(x)}")
