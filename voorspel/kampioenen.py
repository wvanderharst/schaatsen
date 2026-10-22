"""Persoonlijke kalibratie van alle wereldkampioenen en olympisch kampioenen (2007-2026, de toernooien uit piek.py).
Kampioen: WK afstanden en Spelen de snelste tijd per geslacht x afstand; WK allround en sprint de eindstand in punten (samalog: tijd per
500 m opgeteld) van de rijders die alle afstanden reden (allround vrouwen 500/1500/3000/5000, mannen 500/1500/5000/10000; sprint 2x500 en
2x1000). Kalibratie over alle ritten van die rijders, valpartijen INBEGREPEN (het model werkt daar niet mee bij, maar ze gebeuren wel);
ook zonder de titelraces zelf (die zijn op de uitkomst gekozen).
Gebruik: python kampioenen.py <werkmap> [vanaf_jaar]"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import norm, binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk

AR = {'v': ['500m', '1500m', '3000m', '5000m'], 'm': ['500m', '1500m', '5000m', '10000m']}


def titels(R, T):
    R = R[~R['verdacht'] & np.isfinite(R['tijd'])].copy()
    R['eid'] = R['event_id'] % 1000000
    uit = []
    for _, t in T[T['eid'] >= 0].iterrows():
        x = R[R['eid'] == t['eid']]
        if t['soort'] in ('Spelen', 'WK afstanden'):
            for (g, a), y in x.groupby(['gender', 'afstand']):
                if a in ('500m', '1000m', '1500m', '3000m', '5000m', '10000m') and len(y) >= 5:
                    r = y.loc[y['tijd'].idxmin()]
                    uit.append(dict(skater_id=r['skater_id'], naam=r['skater_name'], gender=g, titel=f"{t['soort']} {t['jaar']} {a}", eid=t['eid'],
                                    afstand=a, dag=r['dag']))
        else:
            for g, y in x.groupby('gender'):
                y = y.assign(p=y['tijd'] / (y['afstand'].str[:-1].astype(float) / 500))
                if t['soort'] == 'WK allround':
                    need = AR.get(g, AR['m'])
                    ok = y.groupby('skater_id')['afstand'].apply(lambda s: set(need) <= set(s))
                else:
                    ok = y.groupby('skater_id')['afstand'].apply(lambda s: (s == '500m').sum() >= 2 and (s == '1000m').sum() >= 2)
                ids = ok[ok].index
                if not len(ids):
                    continue
                if t['soort'] == 'WK allround':
                    pt = y[y['skater_id'].isin(ids) & y['afstand'].isin(need)].groupby('skater_id')['p'].sum()
                else:
                    pt = y[y['skater_id'].isin(ids) & y['afstand'].isin(['500m', '1000m'])].groupby('skater_id')['p'].sum()
                w = pt.idxmin(); r = y[y['skater_id'] == w].iloc[0]
                uit.append(dict(skater_id=w, naam=r['skater_name'], gender=g, titel=f"{t['soort']} {t['jaar']}", eid=t['eid'], afstand='alle', dag=r['dag']))
    return pd.DataFrame(uit)


def telling(p, s, q, kant):
    x = p[s]
    return int((x <= q).sum() if kant == 'snel' else (x >= 1 - q).sum()), len(x) * q


if __name__ == '__main__':
    w = sys.argv[1]; vanaf = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'skater_name', 'gender', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'verdacht'])
    for c in ('afstand', 'skater_name', 'gender', 'baan'):
        R[c] = R[c].astype(str)
    T = pk.toernooien(R); Ti = titels(R, T)
    print(f'{len(Ti)} titels, {Ti["skater_id"].nunique()} kampioenen')
    V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'skater_name', 'dag', 'jaar', 'afstand', 'tijd', 'pit', 'ok', 'k_val', 'k_c']).to_pandas()
    V['afstand'] = V['afstand'].astype(str); V['skater_name'] = V['skater_name'].astype(str)
    V = V.merge(R[['skater_id', 'dag', 'afstand', 'tijd', 'event_id']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    V['eid'] = V['event_id'].fillna(-1).astype(np.int64) % 1000000
    V = V[V['ok'] & np.isfinite(V['pit']) & V['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m', '10000m']) & (V['jaar'] >= vanaf)].copy()
    kamp = set(Ti['skater_id'])
    tk = set(zip(Ti['skater_id'], Ti['eid']))
    V['kampioen'] = V['skater_id'].isin(kamp)
    V['titelrace'] = [(s, e) in tk for s, e in zip(V['skater_id'], V['eid'])]
    V['z'] = norm.ppf(V['pit'].clip(1e-9, 1 - 1e-9))
    p = V['pit'].to_numpy()

    def regel(nm, s):
        x = V[s]; out = f'{nm:44s} ritten {len(x):6d} | val {int(x["k_val"].sum()):3d} | gem z {x["z"].clip(-4, 4).mean():+.3f} sd {x["z"].clip(-4, 4).std():.3f} |'
        for kant in ('snel', 'traag'):
            for q in (0.05, 0.01, 0.001):
                O, E = telling(p, s.to_numpy(), q, kant); out += f' {kant} 1/{int(1 / q)} {O}/{E:.1f}'
            out += ' |'
        return out
    print(f'\nGroepen ({vanaf}+, valpartijen inbegrepen):')
    print(regel('alle kampioenen, alle ritten', V['kampioen']))
    print(regel('alle kampioenen, zonder de titelraces', V['kampioen'] & ~V['titelrace']))
    print(regel('alle kampioenen, alleen titelraces', V['kampioen'] & V['titelrace']))
    print(regel('alle kampioenen, zonder titelraces en zonder val', V['kampioen'] & ~V['titelrace'] & ~V['k_val']))
    print(regel('overige top < 2% (geen kampioen)', ~V['kampioen'] & (V['k_c'] < 0.02)))
    # per rijder: aantallen per rijder met >= k ritten kans <= 1/50 (snel en traag), zonder titelraces
    x = V[V['kampioen'] & ~V['titelrace']]
    for kant, pp in (('snel', x['pit']), ('traag', 1 - x['pit'])):
        k = pd.Series(pp.to_numpy() <= 0.02).groupby(x['skater_id'].to_numpy()).agg(['sum', 'size'])
        print(f'  {kant}: rijders met >= k ritten kans <= 1/50 (zonder titelraces): ' +
              ', '.join(f'>={kk}: {int((k["sum"] >= kk).sum())}/{binom.sf(kk - 1, k["size"], 0.02).sum():.1f}' for kk in (2, 3, 4, 5, 6, 8)))
    # per kampioen
    rows = []
    for sid, y in V[V['kampioen']].groupby('skater_id'):
        yy = y[~y['titelrace']]; n = len(yy)
        if n < 10:
            continue
        zt = yy['z'].clip(-4, 4)
        rows.append(dict(naam=y['skater_name'].iloc[0], titels=int((Ti['skater_id'] == sid).sum()), ritten=n, val=int(yy['k_val'].sum()),
                         gem_z=round(zt.mean(), 2), sd_z=round(zt.std(), 2),
                         snel20=f"{int((yy['pit'] <= 0.05).sum())}/{n * 0.05:.1f}", snel100=f"{int((yy['pit'] <= 0.01).sum())}/{n * 0.01:.1f}",
                         traag20=f"{int((yy['pit'] >= 0.95).sum())}/{n * 0.05:.1f}", traag100=f"{int((yy['pit'] >= 0.99).sum())}/{n * 0.01:.1f}",
                         t_gem=round(zt.mean() / (zt.std() / np.sqrt(n)), 1)))
    P = pd.DataFrame(rows).sort_values('gem_z')
    pd.set_option('display.width', 250)
    print(f'\nPer kampioen ({vanaf}+, zonder de titelraces, valpartijen inbegrepen; t_gem = gem z / standaardfout):')
    print(P.to_string(index=False))
    Ti.to_csv(os.path.join(w, 'kampioenen_titels.csv'), index=False); P.to_csv(os.path.join(w, f'kampioenen_kalibratie_{vanaf}.csv'), index=False)
