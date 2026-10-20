"""Stemt de parameters van het Kalmanfilter af op de jaren 2012-2019 (Nelder-Mead): mediane |fout| van alle ritten (500 m en langer)
plus die van de toppers (niveau vooraf binnen 6% en binnen 3% van de wereldtop). Gebruik: python kalman_tune.py [max evaluaties] [start.json] [uit.json]"""
import sys, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/schaatsen/voorspel')
import os, importlib
import numpy as np, pandas as pd
kf = importlib.import_module(os.environ.get('KALMAN', 'kalman'))
from scipy.optimize import minimize
df = pd.read_parquet('ritten_cat.parquet'); df, inv = kf.voorbereid(df)
sen = df['afstand_idx'].to_numpy() >= 2; jr = df['jaar'].to_numpy(); sc = sen & (jr >= 2012) & (jr <= 2019)
import os
START = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
kf.PAR.update(START)
VRIJ = [k for k in ['G1', 'TG', 'G2', 'Q0', 'Q1', 'TQ', 'QA', 'RHO', 'RHO0', 'P0C', 'P0A', 'KR', 'SNEL', 'GAMMA', 'HF', 'QG', 'KRG', 'SNG', 'GAG', 'P0V', 'QV', 'LV'] if k in kf.PAR]
if os.environ.get('VRIJ'): VRIJ = os.environ['VRIJ'].split(',')
NEG = {'G1'}; LIN = {'G2', 'QG', 'KRG', 'SNG', 'GAG'}; LOGIT = {'RHO', 'RHO0'}
START.update({k: v for k, v in kf.PAR.items() if k not in START})
UIT = sys.argv[3] if len(sys.argv) > 3 else 'kalman_par.json'
def naar(x):
    p = {}
    for k, v in zip(VRIJ, x):
        p[k] = -np.exp(v) if k in NEG else (v if k in LIN else (1 / (1 + np.exp(-v)) if k in LOGIT else np.exp(v)))
    return p
x0 = [np.log(-kf.PAR[k]) if k in NEG else (kf.PAR[k] if k in LIN else (np.log(kf.PAR[k] / (1 - kf.PAR[k])) if k in LOGIT else np.log(kf.PAR[k]))) for k in VRIJ]
stap = np.array([0.003 if k == 'G2' else (0.3 if k in LIN else (0.6 if k in LOGIT else 0.25)) for k in VRIJ])
simplex = np.vstack([np.array(x0)] + [np.array(x0) + np.eye(len(VRIJ))[i] * stap[i] for i in range(len(VRIJ))])
best = [1e9, None]; n_ev = [0]; t0 = time.time()
def doel(x):
    p = naar(x); o = kf.pass_(inv, p); ok = sc & np.isfinite(o['e']) & ~o['val']
    if ok.sum() < 1e5: return 1e9
    e = np.abs(o['e']); v = 100 * (np.median(e[ok]) + np.median(e[ok & (o['zh'] < 0.06)]) + np.median(e[ok & (o['zh'] < 0.03)]))
    n_ev[0] += 1
    if v < best[0]:
        best[0] = v; best[1] = p
        json.dump(p, open(UIT, 'w'), indent=1)
    if n_ev[0] % 20 == 0:
        print(n_ev[0], 'beste', round(best[0], 4), round(time.time() - t0), 's', flush=True)
    return v
print('start', round(doel(np.array(x0)), 4), flush=True)
r = minimize(doel, np.array(x0), method='Nelder-Mead', options={'maxfev': int(sys.argv[1]) if len(sys.argv) > 1 else 400, 'xatol': 1e-3, 'fatol': 1e-4, 'adaptive': True, 'initial_simplex': simplex})
print('klaar', round(best[0], 4), {k: round(v, 6) for k, v in best[1].items()}, flush=True)
