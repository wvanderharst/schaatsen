"""Proef: vormlaag op de kansschaal (zie vormlaag.py), achteraf op de uitvoer van model2. Fit op 2012-2019 (zonder eigen valpartijen),
toets op 2020+ op dezelfde ritten als de vergelijking met het oude model (log loss en kalibratie van toppers en elite)."""
import os, sys, importlib.util
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vormlaag as vl

KOLOM = ['skater_id', 'dag', 'jaar', 'seizoen', 'baan', 'gender', 'afstand', 'tijd', 'pit', 'nll', 'ok', 'k_val', 'k_c']
spec = importlib.util.spec_from_file_location('em', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite', 'elite_model.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)


def laad(pad, oud='oud_vooraf.parquet'):
    kol = KOLOM + (['rk_'] if 'rk_' in pq.read_schema(pad).names else [])
    V = pd.read_parquet(pad, columns=kol)
    V = V[V['ok'] & np.isfinite(V['pit'])].copy(); V['afstand'] = V['afstand'].astype(str)
    if 'rk_' not in V:
        V['rk_'] = V.groupby(['dag', 'baan', 'gender', 'afstand'], sort=False).ngroup()
    V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9)).clip(-6, 6)
    O = pd.read_parquet(oud, columns=['skater_id', 'dag', 'afstand', 'tijd', 'll', 'niveau_pct'])
    O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']); O['afstand'] = O['afstand'].astype(str)
    return V.merge(O, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left').reset_index(drop=True)


def kort(r):
    if 'rijders' in r:
        return r.split('met ')[1].split(' ritten')[0] + ':' + r.split(':')[1].split(',')[0].strip() + '/' + r.split('verwacht')[1].split()[0]
    return r.split('kans<=')[1].split(':')[0].strip() + ' ' + r.split('(')[1].split(')')[0]


def verslag(V, pit2, d, m, label):
    uit = []
    for nm, s in (('alle', m), ('toppers', m & (V['niveau_pct'] >= 90).to_numpy()), ('elite', m & (V['niveau_pct'] >= 97).to_numpy())):
        uit.append(f"{nm} {V['nll'].to_numpy()[s].mean():.4f} -> {V['nll'].to_numpy()[s].mean() + d[s].mean():.4f}")
    print(label, '| log loss', ' | '.join(uit))
    for nm, s in (('toppers', m & (V['niveau_pct'] >= 90).to_numpy()), ('elite', m & (V['niveau_pct'] >= 97).to_numpy())):
        for k, p in (('voor', V['pit'].to_numpy()), ('na', pit2)):
            x = V[s][['skater_id', 'seizoen']].copy(); x['p'] = p[s]
            fout, rg, _, _ = em.toets(x, vanaf_seizoen=2019)
            print(f'   {nm:8s} {k:4s} fouten {fout:2d} | snel ' + ' '.join(kort(r) for r in rg[:8]) + ' | traag ' + ' '.join(kort(r) for r in rg[8:]))


if __name__ == '__main__':
    V = laad(sys.argv[1] if len(sys.argv) > 1 else 'voorspel2_uitkomst.parquet')
    F = vl.kenmerken(V['skater_id'], V['dag'], V['z'], V['k_val'], V['rk_'])
    wg = np.exp(-V['k_c'].clip(0, 0.5).fillna(0.2).to_numpy() / 0.04)
    jr = V['jaar'].to_numpy(); z = V['z'].to_numpy()
    fitm = (jr >= 2012) & (jr <= 2019) & ~V['k_val'].to_numpy()
    toets = (jr >= 2020) & np.isfinite(V['ll'].to_numpy())
    M = ['m7', 'm30', 'm90', 'neg7', 'neg30', 'neg90', 'pos7', 'pos30', 'pos90']
    varianten = {
        'rijder': (M, vl.SCH, None),
        'rijder + veld': (vl.LOC, vl.SCH, None),
        'rijder + veld, toppers x3': (vl.LOC, vl.SCH, 1 + 2 * (wg > 0.3)),
        'alleen locatie': (vl.LOC, [], None),
    }
    for nm, (loc, sch, w) in varianten.items():
        Xm, Xs, nm_m, nm_s = vl.ontwerp(F, wg, loc, sch)
        p = vl.fit(z[fitm], Xm[fitm], Xs[fitm], None if w is None else w[fitm])
        mu, sg, pit2, d = vl.pas_toe(p, z, Xm, Xs)
        verslag(V, pit2, d, toets, nm)
        km = Xm.shape[1]
        print('   loc', ' '.join(f'{a} {b:+.3f}' for a, b in zip(nm_m, p[:km])))
        print('   sch', ' '.join(f'{a} {b:+.3f}' for a, b in zip(nm_s, p[km:])), flush=True)
