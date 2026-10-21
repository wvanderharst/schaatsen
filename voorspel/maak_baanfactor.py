"""Berekent de universele baanfactor per rit (theta_rijder, zie baanfactor.py) en zet die in ritten_cat.parquet (origineel: ritten_cat_basis.parquet).
Niveau van de rijder vooraf en van het veld uit een Kalman-pass met het veldpeil; A en TM geschat op de fitjaren (<= 2019).
Gebruik: python maak_baanfactor.py <map> <kalman_par.json>"""
import os, sys, json, shutil, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import kalman_mv2 as kf, baanfactor as bf
from kalman import voorbereid
w = sys.argv[1]; par = json.load(open(sys.argv[2]))
basis = os.path.join(w, 'ritten_cat_basis.parquet')
if not os.path.exists(basis):
    shutil.copy(os.path.join(w, 'ritten_cat.parquet'), basis)
R0 = pd.read_parquet(basis).drop(columns=['theta_rijder'], errors='ignore')
df, inv = voorbereid(R0.copy()); o = kf.pass_(inv, par)
g = o['c']; wv = 1 / np.maximum(o['vl'] + o['r'], 1e-6)
gbar = bf.veldniveau(inv['rk'], g, wv)
hv = bf.hv_peil(df); thv = np.array([hv.get((gg, a), np.nan) for gg, a in zip(df['gender'], df['afstand_idx'])])
d = df['theta'].to_numpy() - thv
e = o['e']; ok = np.isfinite(e) & ~o['val'] & (df['afstand_idx'].to_numpy() >= 2) & (df['jaar'].to_numpy() <= 2019)
# stap 1: snelheidsterm (toppers winnen meer op snel ijs) zonder veldterm; stap 2: veldterm met A en TM vast (samen schatten is niet
# identificeerbaar: A sloeg dan om naar negatief en snelle banen werden voor de elite te traag voorspeld)
A, TM = bf.schat_a(e[ok], d[ok], g[ok], gbar[ok])
B, TB, r2 = bf.schat_b(e[ok], d[ok], g[ok], gbar[ok], A, TM)
print('baanfactor: A %.3f TM %.3f B %.4f TB %.3f (verklaarde fractie %.5f)' % (A, TM, B, TB, r2))
gg = np.where(np.isfinite(g), g, gbar)
df['theta_rijder'] = np.where(np.isfinite(d), bf.factor(df['theta'].to_numpy(), d, gg, gbar, A, TM, B, TB), df['theta'].to_numpy())
json.dump({'A': float(A), 'TM': float(TM), 'B': float(B), 'TB': float(TB)}, open(os.path.join(w, 'baanfactor_par.json'), 'w'))
df.to_parquet(os.path.join(w, 'ritten_cat.parquet'))
x = df[np.isfinite(d)]
print('verschil baanfactor per rijder - veldpeil (%%): p1 %.3f  p50 %.3f  p99 %.3f' % tuple(100 * np.nanquantile(x['theta_rijder'] - x['theta'], [0.01, 0.5, 0.99])))
