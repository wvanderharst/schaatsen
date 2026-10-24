"""Eigen volatiliteit per rijder: vangt de voorspelde spreiding (sd_rijder) het verschil tussen een constante en een wisselvallige rijder?
Rijder-only residu r = ln t - pred - mediaan van het veld in die race (ijs achteraf). Per rij de eigen spreiding uit EERDERE ritten
(robuust: sqrt(mean(min(r^2, (4 sd)^2))) over de laatste N ritten, alle afstanden, in eenheden van de eigen sd_rijder). Toets op 2020+, top
vooraf (k_c < 6%), zonder valpartijen: (1) realiseert de voorspelde sd_rijder zich per kwintiel?  (2) voorspelt de eigen spreiding uit het
verleden |r| / sd_rijder nog (helling in log|r|)?  Gebruik: python volatiliteit.py <werkmap>"""
import os, sys
import numpy as np, pandas as pd

w = sys.argv[1]; N = int(os.environ.get('N', '20'))
M = pd.read_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'pred', 'sd', 'sd_rijder', 'k_c', 'k_val', 'rk_', 'ok', 'pit'])
M = M[M['ok'] & np.isfinite(M['tijd']) & np.isfinite(M['pred'])].copy()
M['res'] = np.log(M['tijd']) - M['pred']
M['res'] -= M[~M['k_val']].groupby('rk_')['res'].median().reindex(M['rk_']).to_numpy()       # ijs achteraf: veldmediaan
M['u'] = M['res'] / M['sd_rijder']
M = M[np.isfinite(M['u'])].sort_values(['skater_id', 'dag'], kind='stable')
v = M['u'].where(~M['k_val']).clip(-4, 4) ** 2                                                 # valpartijen tellen niet mee
g = v.groupby(M['skater_id'])
som = g.transform(lambda s: s.fillna(0).cumsum().shift(1)); n = g.transform(lambda s: s.notna().cumsum().shift(1))
somN = g.transform(lambda s: s.fillna(0).rolling(N, min_periods=1).sum().shift(1)); nN = g.transform(lambda s: s.notna().rolling(N, min_periods=1).sum().shift(1))
K0 = 10.0                                                                                       # krimp naar 1 met 10 pseudo-ritten
M['eigen'] = np.sqrt((somN + K0) / (nN + K0)); M['n_eerder'] = n
T = M[(pd.to_datetime(M['dag'], unit='D').dt.year >= 2020) & (M['k_c'] < 0.06) & ~M['k_val'] & (M['n_eerder'] >= 20)].copy()
print(f'toetsritten {len(T)}, rijders {T.skater_id.nunique()}')
T['srel'] = T['sd_rijder'] / T.groupby(['afstand'])['sd_rijder'].transform('median')
for kol, nm in (('srel', 'voorspelde sd_rijder (t.o.v. mediaan afstand)'), ('eigen', f'eigen spreiding laatste {N} ritten (in sd_rijder)')):
    T['b'] = pd.qcut(T[kol], 5, labels=False)
    print(f'\nkwintielen van {nm}:')
    print('  kwintiel | gem. waarde | voorspelde sd_rijder (rel) | gerealiseerde sd r (rel, robuust 1,4826 MAD) | sd u | snel 1/20 | traag 1/20')
    med = 1.4826 * np.median(np.abs(T['res']))
    for b, d in T.groupby('b'):
        mad = 1.4826 * np.median(np.abs(d['res'] - d['res'].median()))
        print(f'  {b + 1}        | {d[kol].mean():.3f}       | {d["sd_rijder"].mean() / T["sd_rijder"].mean():.3f}                      | {mad / med:.3f}'
              f'                                    | {1.4826 * np.median(np.abs(d["u"] - d["u"].median())):.3f} | {(d["u"] < -1.645).mean() / 0.05:.2f}      | {(d["u"] > 1.645).mean() / 0.05:.2f}')
la = np.log(np.abs(T['u']) + 1e-4); X = np.column_stack([np.ones(len(T)), np.log(T['eigen'])])
b = np.linalg.lstsq(X, la, rcond=None)[0]
res = la - X @ b; se = np.sqrt(np.sum(res ** 2) / (len(T) - 2) * np.linalg.inv(X.T @ X)[1, 1])
print(f'\nhelling log|u| op log(eigen spreiding): {b[1]:.3f} (se {se:.3f}); 0 = sd_rijder vangt de eigen volatiliteit al, 1 = helemaal niet')

print('\neindkans (pit, alle lagen) per kwintiel eigen spreiding: waargenomen / verwacht')
T['b'] = pd.qcut(T['eigen'], 5, labels=False)
print('  kwintiel | snel 1/20 | snel 1/100 | traag 1/20 | traag 1/100 | n')
for b_, d in T.groupby('b'):
    q = d['pit']
    print(f'  {b_ + 1}        | {(q < .05).mean() / .05:.2f}      | {(q < .01).mean() / .01:.2f}       | {(q > .95).mean() / .05:.2f}       | {(q > .99).mean() / .01:.2f}        | {len(d)}')
