import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
K = pd.read_parquet('kenmerken.parquet', columns=['dag','event_id','gender','afstand_idx','baan','theta','v_theta','jaar','ijs_vm','ijs_vn','ijs_ba','ijs_bn','ijs_ev','ijs_evn','ijs_gl','hoog','skater_id'])
r = K.groupby(['dag','event_id','gender','afstand_idx']).agg(**{c: (c, 'first') for c in ['baan','theta','v_theta','jaar','ijs_vm','ijs_vn','ijs_ba','ijs_bn','ijs_ev','ijs_evn','ijs_gl','hoog']}, k=('skater_id','size')).reset_index()
r = r[np.isfinite(r.theta) & (r.afstand_idx >= 2)]
glob = r[r.jaar <= 2019].groupby(['gender','afstand_idx']).theta.median().rename('gm')
r = r.join(glob, on=['gender','afstand_idx'])
for k in (3, 10):
    off = r.ijs_vm.where(r.ijs_vn > 0, r.ijs_ba)
    r[f'base{k}'] = np.where(r.ijs_vn > 0, (r.ijs_vn * r.ijs_vm + k * r.ijs_ba.where(r.ijs_bn > 0, r.gm)) / (r.ijs_vn + k), r.ijs_ba.where(r.ijs_bn > 0, r.gm))
tr = r[r.jaar <= 2019]; te = r[r.jaar >= 2020]
def fit(cols, d):
    X = np.c_[np.ones(len(d)), d[cols].to_numpy()]; w = np.sqrt(d.k.clip(1, 30).to_numpy())
    return np.linalg.lstsq(X * w[:, None], d.theta.to_numpy() * w, rcond=None)[0]
def sd(c, d, cols):
    X = np.c_[np.ones(len(d)), d[cols].to_numpy()]; e = d.theta.to_numpy() - X @ c
    w = d.k.clip(1, 30); return np.sqrt(np.average(e**2, weights=w)), np.average(np.abs(e), weights=w)
r['evw'] = r.ijs_ev * r.ijs_evn / (r.ijs_evn + 2)
for k in (3, 10):
    r[f'b{k}_ev'] = r[f'base{k}']
tr = r[r.jaar <= 2019]; te = r[r.jaar >= 2020]
for cols in (['base3','evw','ijs_gl'], ['base3','evw','ijs_gl','recw']):
    d_tr = tr.dropna(subset=cols); c = fit(cols, d_tr); s = sd(c, te.dropna(subset=cols), cols)
    print(cols, 'coef', np.round(c, 3), 'test sd %.4f mae %.4f' % s)
print('aandeel races met eerdere toernooidag', (te.ijs_evn > 0).mean().round(3))
print('theta sd test (gewogen)', np.sqrt(np.average((te.theta - np.average(te.theta, weights=te.k.clip(1,30)))**2, weights=te.k.clip(1,30))).round(4))
