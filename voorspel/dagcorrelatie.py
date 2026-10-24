"""Samenhang tussen ritten op een dag (voor meerkampen): (1) de eigen prestatie van een rijder (rijderschok, ijs achteraf eruit) tussen
twee ritten op dezelfde dag en op twee opeenvolgende dagen; (2) het ijs (racepeil t.o.v. het gewone peil van die baan, geslacht en afstand)
tussen races op dezelfde baan op dezelfde dag en op de volgende dag. Robuust: correlatie van rangen (Spearman) en van afgekapte waarden.
Gebruik: python dagcorrelatie.py <werkmap> <sitemap met r/>"""
import gzip, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

w, S = sys.argv[1], sys.argv[2]
rij = []
for b in range(200):
    d = json.load(gzip.open(os.path.join(S, 'r', f'{b}.json.gz')))
    for sid, x in d.items():
        for v in x['r']:
            if v[8] is not None and not v[10]:
                rij.append((int(sid), v[0], v[1], np.log1p(v[8] / 100)))
X = pd.DataFrame(rij, columns=['s', 'dag', 'a', 'z']); X['z'] = X['z'].clip(-0.05, 0.05)


def paren(X, verschuiving, zelfde_afstand):
    A = X.rename(columns={'z': 'z1', 'a': 'a1'}); B = X.rename(columns={'z': 'z2', 'a': 'a2'}); B['dag'] = B['dag'] - verschuiving
    M = A.merge(B, on=['s', 'dag'])
    M = M[(M['a1'] < M['a2']) if verschuiving == 0 and not zelfde_afstand else (M['a1'] == M['a2']) if zelfde_afstand else M['a1'] >= 0]
    return M


for nm, sh, za in (('zelfde dag, andere afstand', 0, False), ('volgende dag (alle afstanden)', 1, False)):
    M = paren(X, sh, za)
    M = M.sample(min(len(M), 400000), random_state=1)
    print(f"rijder {nm}: {len(M)} paren, Pearson {np.corrcoef(M['z1'], M['z2'])[0, 1]:.3f}, Spearman {spearmanr(M['z1'], M['z2'])[0]:.3f}")
# zelfde afstand twee keer op een dag (sprint: 2 x 500 m) -> zelfde dag, zelfde afstand, verschillende ritten
G = X.groupby(['s', 'dag', 'a'])['z'].apply(list); G = G[G.apply(len) == 2]
z1 = np.array([g[0] for g in G]); z2 = np.array([g[1] for g in G])
print(f"rijder zelfde dag, zelfde afstand twee keer: {len(G)} paren, Pearson {np.corrcoef(z1, z2)[0, 1]:.3f}, Spearman {spearmanr(z1, z2)[0]:.3f}")
# ijs
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['gender', 'afstand', 'dag', 'seizoen', 'event_id', 'baan', 'theta', 'verdacht'])
for c in ('gender', 'afstand', 'baan'):
    R[c] = R[c].astype(str)
R = R[~R['verdacht']]
Q = R.groupby(['event_id', 'dag', 'gender', 'afstand']).agg(theta=('theta', 'median'), n=('theta', 'size'), baan=('baan', 'first'), seizoen=('seizoen', 'first')).reset_index()
Q = Q[Q['n'] >= 5]
Q['r'] = Q['theta'] - Q.groupby(['baan', 'gender', 'afstand', 'seizoen'])['theta'].transform('median')
Q['r'] = Q['r'].clip(-0.05, 0.05)
for nm, sh in (('zelfde dag, andere afstand', 0), ('volgende dag', 1)):
    A = Q[['baan', 'dag', 'gender', 'afstand', 'r']]; B = A.copy(); B['dag'] = B['dag'] - sh
    M = A.merge(B, on=['baan', 'dag', 'gender'], suffixes=('1', '2'))
    M = M[M['afstand1'] < M['afstand2']] if sh == 0 else M
    print(f"ijs {nm}: {len(M)} paren, Pearson {np.corrcoef(M['r1'], M['r2'])[0, 1]:.3f}, Spearman {spearmanr(M['r1'], M['r2'])[0]:.3f}")
