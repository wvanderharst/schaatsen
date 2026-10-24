"""Hoeveel WK-afstanden- en olympische medailles komen uit een uitzonderlijke rit, en is dat wat het model verwacht?
Simulatie per race met de modelverdeling: ln t_i = pred_i + sd_ijs * eta (gedeeld, het ijs van die race) + sd_rijder_i * eps_i,
eta en eps Student-t (nu 5, geschaald tot sd 1). Per gesimuleerde race: wie wint de medailles en hoe uitzonderlijk was hun rit
(kans p = F((ln t - pred) / sd)). Waargenomen met dezelfde p. Gebruik: python medaille_verrassing.py <werkmap> <uitkomst.parquet>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import t as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk
w, pad = sys.argv[1], sys.argv[2]; NU = 5.0; c = np.sqrt((NU - 2) / NU)
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'gender', 'verdacht'])
R['afstand'] = R['afstand'].astype(str)
Tn = pk.toernooien(R); Tn = Tn[Tn['soort'].isin(['Spelen', 'WK afstanden'])]
V = pq.read_table(pad, columns=['skater_id', 'dag', 'afstand', 'tijd', 'pred', 'sd', 'sd_rijder', 'sd_ijs', 'pit', 'ok', 'k_val']).to_pandas(); V['afstand'] = V['afstand'].astype(str)
V = V.merge(R[~R['verdacht']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'])
V['eid'] = V['event_id'] % 1000000; V = V[V['eid'].isin(Tn['eid']) & V['afstand'].isin(list(pk.ZES)) & V['ok'] & np.isfinite(V['pred'])]
rng = np.random.default_rng(0); S = 2000
GR = [0.1, 0.05, 0.01]
obs = {q: 0 for q in GR}; obs_mod = {q: 0 for q in GR}; sim = {q: 0.0 for q in GR}; nmed = 0; races = 0
for (eid, g, a), x in V.groupby(['eid', 'gender', 'afstand']):
    x = x.sort_values('tijd').drop_duplicates('skater_id')        # per rijder de snelste rit (500 m tot 2012 twee keer)
    if len(x) < 8:
        continue
    races += 1
    pr, sr, si = x['pred'].to_numpy(), x['sd_rijder'].to_numpy(), x['sd_ijs'].to_numpy()
    sd = np.sqrt(sr ** 2 + si ** 2)
    pz = T.cdf((np.log(x['tijd'].to_numpy()) - pr) / sd / c, NU)
    med = np.argsort(np.log(x['tijd'].to_numpy()))[:3]; nmed += 3
    for q in GR:
        obs[q] += int((pz[med] <= q).sum()); obs_mod[q] += int((x['pit'].to_numpy()[med] <= q).sum())
    eta = T.rvs(NU, size=(S, 1), random_state=rng) * c; eps = T.rvs(NU, size=(S, len(x)), random_state=rng) * c
    lt = pr + si * eta + sr * eps
    ps = T.cdf((lt - pr) / sd / c, NU)
    top = np.argsort(lt, axis=1)[:, :3]
    pm = np.take_along_axis(ps, top, axis=1)
    for q in GR:
        sim[q] += (pm <= q).sum(1).mean()
print(f'{races} races (WK afstanden + Spelen), {nmed} medailles')
for q in GR:
    print(f'medailles uit een rit met kans <= 1/{int(1 / q)}: waargenomen {obs[q]} ({100 * obs[q] / nmed:.0f}%), volgens modelkans {obs_mod[q]}, verwacht (simulatie) {sim[q]:.0f} ({100 * sim[q] / nmed:.0f}%)')
