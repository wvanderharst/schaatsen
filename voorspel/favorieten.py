"""Kalibratie van de zes favorieten per race (snelste verwachte tijd vooraf) op WK afstanden + Spelen en op World Cups:
(1) hun kansen (gem. z, staarten), (2) de rangorde: aantal favorieten in de top 6 / op het podium / winnaar, waargenomen tegenover een
simulatie met de modelverdeling (gedeelde ijsschok per race + eigen ruis, Student-t nu 5). Gebruik: python favorieten.py <werkmap> <uitkomst>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import t as T, norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk
w, pad = sys.argv[1], sys.argv[2]; NU = 5.0; c = np.sqrt((NU - 2) / NU); rng = np.random.default_rng(1); S = 2000
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'gender', 'verdacht', 'jaar'])
R['afstand'] = R['afstand'].astype(str)
Tn = pk.toernooien(R); pe = set(Tn['eid']); wkos = set(Tn.loc[Tn['soort'].isin(['Spelen', 'WK afstanden']), 'eid']); wc = pk.internationaal(R, pe)
V = pq.read_table(pad, columns=['skater_id', 'dag', 'afstand', 'tijd', 'pred', 'sd_rijder', 'sd_ijs', 'pit', 'ok', 'k_val', 'theta', 'th_hat_veld', 'k_c']).to_pandas(); V['afstand'] = V['afstand'].astype(str)
RIJDER = len(sys.argv) > 3 and sys.argv[3] == 'rijder'

V = V.merge(R[~R['verdacht']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'])
V['eid'] = V['event_id'] % 1000000; V = V[V['afstand'].isin(list(pk.ZES)) & V['ok'] & np.isfinite(V['pred']) & np.isfinite(V['pit'])]
if RIJDER:   # prestatie van de rijder: ijs achteraf gecorrigeerd met het veld zelf (mediaan van ln t - verwachting over de andere rijders)
    V = V[~V['k_val']].copy(); V['res'] = np.log(V['tijd']) - V['pred']
    rk = ['event_id', 'dag', 'gender', 'afstand']
    V['n_r'] = V.groupby(rk)['res'].transform('size'); V = V[V['n_r'] >= 5].copy()
    med = V.groupby(rk)['res'].transform('median')
    V['ijs'] = med                                      # (mediaan van een veld van >= 5 rijders; de eigen rit telt nauwelijks mee)
    V['zr'] = (V['res'] - V['ijs']) / V['sd_rijder']
    V = V[np.isfinite(V['zr'])]
    ref = np.sort(V.loc[(V['jaar'] <= 2019) & (V['jaar'] >= 2008) & (V['k_c'] < 0.06) & ~V['k_val'], 'zr'].to_numpy())   # vorm: alle toprijders 2008-2019
    V['pit'] = (np.searchsorted(ref, V['zr'].to_numpy(), side='right') + 0.5) / (len(ref) + 1)
    print(f'referentie: {len(ref)} toprritten, mediaan {np.median(ref):+.3f}, robuuste sd {(np.percentile(ref, 75) - np.percentile(ref, 25)) / 1.349:.3f}')
for nm, eset in (('WK afstanden + Spelen', wkos), ('World Cups', wc)):
    for per, js in (('2008-2019', (2008, 2019)), ('2020+', (2020, 2030))):
        X = V[V['eid'].isin(eset) & V['jaar'].between(*js)]
        fav = []; o = {'top6': 0, 'podium': 0, 'winst': 0}; s_ = {'top6': 0.0, 'podium': 0.0, 'winst': 0.0}; nr = 0
        for _, x in X.groupby(['event_id', 'dag', 'gender', 'afstand']):
            x = x.drop_duplicates('skater_id')
            if len(x) < 10:
                continue
            nr += 1; x = x.sort_values('pred'); f = np.arange(len(x)) < 6
            fav.append(x[f])
            rang = np.argsort(np.argsort(np.log(x['tijd'].to_numpy())))
            o['top6'] += int((rang[f] < 6).sum()); o['podium'] += int((rang[f] < 3).sum()); o['winst'] += int((rang[f] < 1).sum())
            pr, sr, si = x['pred'].to_numpy(), x['sd_rijder'].to_numpy(), x['sd_ijs'].to_numpy()
            if RIJDER:     # geen ijsschok: alleen de eigen prestatie, getrokken uit de referentieverdeling
                lt = pr + sr * ref[rng.integers(0, len(ref), size=(S, len(x)))]
            else:
                lt = pr + si * T.rvs(NU, size=(S, 1), random_state=rng) * c + sr * T.rvs(NU, size=(S, len(x)), random_state=rng) * c
            rs = np.argsort(np.argsort(lt, axis=1), axis=1)[:, f]
            s_['top6'] += (rs < 6).sum(1).mean(); s_['podium'] += (rs < 3).sum(1).mean(); s_['winst'] += (rs < 1).sum(1).mean()
        F = pd.concat(fav); Fv = F[~F['k_val']]; z = norm.ppf(Fv['pit'].clip(1e-9, 1 - 1e-9)).clip(-4, 4)
        print(f"\n{nm} {per}: {nr} races, {len(F)} favorietenritten | gem z {z.mean():+.3f} sd {z.std():.3f} | snel 1/20 {(Fv['pit'] <= .05).mean() / .05:.2f} 1/100 {(Fv['pit'] <= .01).mean() / .01:.2f}"
              f" | traag 1/20 {(Fv['pit'] >= .95).mean() / .05:.2f} 1/100 {(Fv['pit'] >= .99).mean() / .01:.2f} | val {100 * F['k_val'].mean():.1f}%")
        print('   favorieten in de uitslag (waargenomen / verwacht): ' + ', '.join(f'{k} {o[k]} / {s_[k]:.0f}' for k in o))
