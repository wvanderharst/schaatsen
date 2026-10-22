"""Dagpeil zonder krimp. Het dagpeil theta uit het oude model is een schatting die naar de verwachting voor de baan wordt getrokken (en
waarin extreme afwijkingen van rijders zijn afgekapt). Daardoor komt maar 40-60% van een gezamenlijke afwijking van het veld (iedereen
sneller of trager dan voorspeld) als ijs in theta; de rest wordt de rijders toegerekend.

Nieuw dagpeil per race: theta_nieuw = w (verwachting vooraf + mediaan residu van het veld t.o.v. onze voorspellingen) + (1 - w) theta_oud,
w = n / (n + n0): kleine races blijven dicht bij de oude meting. Meetonzekerheid v_theta = w * 1,57 gem(sd_rijder^2) / n + (1 - w) v_oud.
Eigen valpartijen tellen niet mee. Races zonder uitkomst van het model (andere afstanden) houden de oude theta.
Gebruik: python dagpeil_veld.py <werkmap> [n0]   -> <werkmap>/ritten_cat_basis_veld.parquet (daarna maak_baanfactor met BASIS=...)"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

w = sys.argv[1]; n0 = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'),
                  columns=['skater_id', 'dag', 'afstand', 'tijd', 'pred', 'th_hat_veld', 'theta', 'sd_rijder', 'ok', 'k_val']).to_pandas()
V['afstand'] = V['afstand'].astype(str)
B = pd.read_parquet(os.path.join(w, 'ritten_cat_basis.parquet')); B['afstand'] = B['afstand'].astype(str)
rk = ['dag', 'event_id', 'gender', 'afstand_idx']
key = ['skater_id', 'dag', 'afstand', 'tijd']
X = B[key + rk[1:]].reset_index().merge(V.drop_duplicates(key), on=key, how='inner')
X = X[X['ok'] & ~X['k_val'] & np.isfinite(X['tijd']) & np.isfinite(X['pred'])]
X['res'] = np.log(X['tijd']) - X['pred']
R = X.groupby(rk).agg(c=('res', 'median'), n=('res', 'size'), thv=('th_hat_veld', 'first'), v_r=('sd_rijder', lambda s: np.mean(s ** 2))).reset_index()
B = B.merge(R, on=rk, how='left')
heeft = np.isfinite(B['c']) & np.isfinite(B['thv']) & np.isfinite(B['theta'])
wt = np.where(heeft, B['n'] / (B['n'] + n0), 0.0)
nieuw = np.where(heeft, wt * (B['thv'] + np.clip(B['c'], -0.05, 0.05)) + (1 - wt) * B['theta'], B['theta'])
v_nieuw = np.where(heeft, wt * 1.57 * B['v_r'] / np.maximum(B['n'], 1) + (1 - wt) * B['v_theta'], B['v_theta'])
print(f"races met nieuw dagpeil: {B.loc[heeft].drop_duplicates(rk).shape[0]} | verschil nieuw - oud: sd {100 * np.nanstd((nieuw - B['theta'])[heeft]):.3f}%, "
      f"mediaan {100 * np.nanmedian((nieuw - B['theta'])[heeft]):+.3f}%")
B['theta_oud'] = B['theta']; B['theta'] = nieuw; B['v_theta'] = v_nieuw
B.drop(columns=['c', 'n', 'thv', 'v_r']).to_parquet(os.path.join(w, 'ritten_cat_basis_veld.parquet'))
