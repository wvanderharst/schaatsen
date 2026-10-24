"""Winnen underdogs op de Olympische Spelen zo vaak als voorspeld? Per race (event, dag, geslacht, afstand; >= 10 rijders) de kans op winst en
podium per rijder uit een simulatie van alleen de rijderschok (het ijs is binnen een race voor iedereen gelijk en verandert de volgorde niet):
ln t = verwachting + sd_rijder x trekking uit de werkelijke gestandaardiseerde rijderschokken van toprijders 2008-2019 (rijderschok.py).
Daarna waargenomen tegen verwacht naar winkans vooraf en naar voorspelde plaats, met de spreiding van een som van onafhankelijke kansen.
Races waarin een rijder twee ritten heeft (500 m 2010/2014: twee omlopen op een dag) vallen af.
Gebruik: python spelen_winst.py <werkmap> <uitkomst> [soort: Spelen | WK afstanden | World Cups]"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk, rijderschok as rs
w, pad = sys.argv[1], sys.argv[2]; SOORT = sys.argv[3] if len(sys.argv) > 3 else 'Spelen'; S = 20000; C = float(os.environ.get('SCHAAL', '1')); rng = np.random.default_rng(7)
V = rs.tabel(w, pad)
ref = np.sort(V.loc[V['jaar'].between(2008, 2019) & (V['k_c'] < 0.06) & ~V['k_val'], 'zr'].to_numpy())
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan'])
R['afstand'] = R['afstand'].astype(str)
Tn = pk.toernooien(R); eids = pk.internationaal(R, set(Tn['eid'])) if SOORT == 'World Cups' else set(Tn.loc[Tn['soort'] == SOORT, 'eid'])
V = V[(V['event_id'] % 1000000).isin(eids) & V['afstand'].isin(list(pk.ZES))].copy()
rk = ['event_id', 'dag', 'gender', 'afstand']
uit = []; weg = 0
for key, x in V.groupby(rk):
    if x['skater_id'].duplicated().any():
        weg += 1; continue
    if len(x) < 10:
        continue
    pr = x['pred'].to_numpy(); sr = x['sd_rijder'].to_numpy()
    lt = pr + C * sr * ref[rng.integers(0, len(ref), size=(S, len(x)))]
    r_ = np.argsort(np.argsort(lt, axis=1), axis=1)
    pw = (r_ == 0).mean(0); pp = (r_ < 3).mean(0)
    echt = x['tijd'].rank(method='min').to_numpy()
    vr = pd.Series(pr).rank(method='first').to_numpy()
    for i in range(len(x)):
        uit.append(dict(race=key, jaar=int(x['jaar'].iloc[0]), gender=key[2], afstand=key[3], rijder=x['skater_name'].iloc[i], vrang=int(vr[i]), p_win=pw[i], p_pod=pp[i],
                        win=echt[i] == 1, pod=echt[i] <= 3, val=bool(x['k_val'].iloc[i])))
U = pd.DataFrame(uit)
print(f'schaal rijderschok x {C}'); U['race'] = U['race'].astype(str)
print(f'{SOORT}: {U["race"].nunique()} races, {len(U)} ritten (races met twee omlopen weggelaten: {weg})')


def tab(gr, lab):
    print(f'\n{lab}:\n  groep             | rijders | winst waargenomen / verwacht (sd) | podium waargenomen / verwacht (sd)')
    for g, x in U.groupby(gr, observed=True):
        print(f"  {str(g):17s} | {len(x):7d} | {int(x['win'].sum()):3d} / {x['p_win'].sum():5.1f} ({np.sqrt((x['p_win'] * (1 - x['p_win'])).sum()):.1f})"
              f"                | {int(x['pod'].sum()):3d} / {x['p_pod'].sum():5.1f} ({np.sqrt((x['p_pod'] * (1 - x['p_pod'])).sum()):.1f})")


U['winkans'] = pd.cut(U['p_win'], [-1, 0.01, 0.05, 0.10, 0.25, 0.50, 1.01], labels=['< 1%', '1-5%', '5-10%', '10-25%', '25-50%', '> 50%'])
U['plaats vooraf'] = pd.cut(U['vrang'], [0, 1, 3, 5, 10, 99], labels=['nr 1', '2-3', '4-5', '6-10', '11+'])
tab('winkans', 'naar winkans vooraf')
tab('plaats vooraf', 'naar voorspelde plaats')
W = U[U['win']]
lw = np.log(W['p_win'].clip(1e-4)); el = U.groupby('race').apply(lambda x: (x['p_win'] * np.log(x['p_win'].clip(1e-4))).sum())
print(f'\nwinnaars: gem. winkans vooraf {W["p_win"].mean():.3f} (verwacht {U.groupby("race").apply(lambda x: (x["p_win"] ** 2).sum()).mean():.3f});'
      f' gem. ln(winkans) {lw.mean():.3f} (verwacht {el.mean():.3f}); winnaars met winkans < 10%: {(W["p_win"] < 0.10).sum()} van {len(W)}')
print('\nper jaar: winst waargenomen / verwacht voor de nr 1 vooraf en voor rijders met winkans < 10%')
for j, x in U.groupby('jaar'):
    a = x[x['vrang'] == 1]; b = x[x['p_win'] < 0.10]
    print(f"  {j}: nr 1 {int(a['win'].sum())} / {a['p_win'].sum():.1f} | winkans < 10% {int(b['win'].sum())} / {b['p_win'].sum():.1f}")
print('\nwinnaars met de kleinste winkans vooraf:')
for _, r in W.sort_values('p_win').head(12).iterrows():
    print(f"  {r['jaar']} {r['gender']} {r['afstand']:6s} {r['rijder']:28s} winkans {100 * r['p_win']:.1f}% (plaats vooraf {r['vrang']})")
