"""Data voor de website met alle rijders: een index en per rijder (in NB blokken, gzip-JSON) de seizoenen en alle ritten.
Per seizoen (eind van het seizoen, Kalman-momentopname): per afstand het niveau als tijd op een normale dag in Heerenveen
(exp(wereldtop-referentie + niveau + peil Heerenveen)), de achterstand op de top-10 van DAT seizoen (stijgend niveau over de jaren),
hetzelfde niveau omgerekend naar het niveau van nu (achterstand op de top-10 + top-10 van het laatste seizoen), de wereldrang, aantal ritten;
titels (WK, Spelen) en World Cup-zeges; bij junioren (15-19) de groei t.o.v. vorig seizoen en de titelkansen (titels.py, buiten de steekproef).
Per rit: datum, afstand, baan, tijd, tijd omgerekend naar een normale dag in Heerenveen (exp(ln t - peil van de race + peil Heerenveen)),
plaats in de race, soort wedstrijd (Spelen, WK, World Cup) en de prestatie van alleen de rijder (rijderschok: ijs achteraf via de
veldmediaan; nooit de gecombineerde schok).
Gebruik: python site_data.py <werkmap> <uitmap>"""
import os, sys, json, gzip
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk, rijderschok as rs

AF = ['100m', '300m', '500m', '1000m', '1500m', '3000m', '5000m', '10000m']
NB = 200


def peil_hv(R, V):
    """Peil van een normale dag in Heerenveen per geslacht x afstand: mediaan peil van Heerenveen-races in de laatste drie seizoenen met een
    topveld (mediaan niveau vooraf < 5%), anders van alle Heerenveen-races (jeugdafstanden)."""
    rk = ['dag', 'baan', 'gender', 'afstand']
    hvR = R[R['baan'].str.startswith('Heerenveen') & (R['seizoen'] >= R['seizoen'].max() - 3)]
    alle = hvR.drop_duplicates(rk).groupby(['gender', 'afstand'])['theta'].median()
    V = V[V['baan'].str.startswith('Heerenveen') & (V['seizoen'] >= V['seizoen'].max() - 3)].copy()
    V['veld'] = V.groupby(rk)['k_c'].transform('median')
    top = V[V['veld'] < 0.05].drop_duplicates(rk).groupby(['gender', 'afstand'])['theta'].median()
    return top.combine_first(alle)


def main(w, uit):
    os.makedirs(os.path.join(uit, 'r'), exist_ok=True)
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'skater_name', 'gender', 'land', 'afstand', 'afstand_idx', 'dag', 'seizoen',
                                                                       'event_id', 'baan', 'tijd', 'theta', 'verdacht', 'geboren'])
    for c in ('afstand', 'baan', 'gender', 'skater_name', 'land'):
        R[c] = R[c].astype(str)
    V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'gender', 'dag', 'afstand', 'tijd', 'baan', 'seizoen', 'theta', 'k_c']).to_pandas()
    for c in ('afstand', 'baan', 'gender'):
        V[c] = V[c].astype(str)
    thv = peil_hv(R, V); del V
    R['tijd_hv'] = np.exp(np.log(R['tijd']) - R['theta'] + R.set_index(['gender', 'afstand']).index.map(thv).to_numpy(float))
    # soort wedstrijd
    Tn = pk.toernooien(R); soort = dict(zip(Tn['eid'], Tn['soort'])); wc = pk.internationaal(R, set(Tn['eid']))
    eid = R['event_id'] % 1000000
    R['soort'] = np.where(eid.isin([e for e, s in soort.items() if s == 'Spelen']), 'S',
                          np.where(eid.isin(list(soort)), 'W', np.where(eid.isin(wc), 'C', '')))
    rk = ['event_id', 'dag', 'gender', 'afstand']
    R['plaats'] = R.groupby(rk)['tijd'].rank(method='min'); R['veld'] = R.groupby(rk)['tijd'].transform('size')
    # rijderschok (alleen de rijder, ijs achteraf)
    S_ = rs.tabel(w, os.path.join(w, 'voorspel2_uitkomst.parquet'))[['skater_id', 'dag', 'afstand', 'tijd', 'schok_pct', 'kans']]
    R = R.merge(S_.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left'); del S_
    print('ritten klaar', len(R), flush=True)

    # seizoenen: Kalman-momentopnamen
    M = pd.read_parquet(os.path.join(w, 'talent_momentopnamen.parquet'))
    M = M[M['seizoen'] <= R['seizoen'].max()]
    ref = (R.assign(y=np.log(R['tijd']) - R['theta'])[(R['seizoen'] <= 2019) & ~R['verdacht']].groupby(['gender', 'afstand'])['y'].quantile(0.01))
    seiz = []
    laatste = int(M['seizoen'].max())
    for j, a in enumerate(AF):
        x = M[['skater_id', 'seizoen', 'gender', 'leeftijd', 'x_' + a, 'na_' + a]].rename(columns={'x_' + a: 'x', 'na_' + a: 'na'})
        x = x[x['na'] >= 2].copy(); x['afstand'] = a
        if a == '10000m':
            x = x[x['gender'] != 'v']                  # geen officiele afstand bij de vrouwen: de 'top-10' zijn marathonrijdsters
        sen = x[(x['leeftijd'] >= 21) & (x['na'] >= 3)]
        t10 = sen.groupby(['gender', 'seizoen'])['x'].apply(lambda v: np.sort(v.to_numpy())[9] if len(v) >= 10 else np.nan).rename('t10')
        t10 = t10.groupby(level='gender').transform(lambda v: v.rolling(3, center=True, min_periods=1).mean())
        x = x.join(t10, on=['gender', 'seizoen'])
        nu = t10.groupby(level='gender').last()
        x['rang'] = x[x['na'] >= 3].groupby(['gender', 'seizoen'])['x'].rank(method='min')
        r_ = x['gender'].map(lambda g: ref.get((g, a), np.nan)); h_ = x['gender'].map(lambda g: thv.get((g, a), np.nan))
        x['t_hv'] = np.exp(r_ + x['x'] + h_)
        x['achter_t10'] = 100 * (np.exp(x['x'] - x['t10']) - 1)
        x['t_nu'] = np.exp(r_ + x['x'] - x['t10'] + x['gender'].map(nu) + h_)
        seiz.append(x)
    Z = pd.concat(seiz, ignore_index=True)
    print('seizoensniveaus klaar', len(Z), flush=True)

    # titels en World Cup-zeges per rijder x seizoen; titelkansen
    import titels as tl
    per, tot, T = tl.uitkomsten(w)
    T = T.sort_values('dag')
    tnamen = T.groupby(['skater_id', 'seizoen'])['titel'].apply(list)
    kans = pd.read_parquet(os.path.join(w, 'titels_alle.parquet')).set_index(['skater_id', 'seizoen']) if os.path.exists(os.path.join(w, 'titels_alle.parquet')) else None
    groei = M.set_index(['skater_id', 'seizoen'])[['verbetering', 'L_jeugd', 'leeftijd']]

    # index
    info = R.sort_values('dag').groupby('skater_id').agg(n=('skater_name', 'last'), g=('gender', 'last'), l=('land', lambda s: s.mode().iloc[0]),
                                                         b=('geboren', 'last'), v=('seizoen', 'min'), t=('seizoen', 'max'), k=('tijd', 'size'))
    titels_rijder = per.groupby(level='skater_id')[['n_wk', 'n_os', 'n_wc']].sum()
    info = info.join(titels_rijder).fillna({'n_wk': 0, 'n_os': 0, 'n_wc': 0})
    beste = Z[(Z['na'] >= 3) & Z['afstand'].isin(AF[2:])].groupby('skater_id')['achter_t10'].min()
    info['best'] = beste
    info['blok'] = (info.index.to_numpy() % NB)
    banen = sorted(R['baan'].unique()); bidx = {b: i for i, b in enumerate(banen)}
    idx = [[int(i), r['n'], r['g'], r['l'], (int(r['b']) if np.isfinite(r['b']) else None), int(r['v']), int(r['t']), int(r['k']), int(r['n_wk']), int(r['n_os']), int(r['n_wc']),
            (round(float(r['best']), 2) if np.isfinite(r['best']) else None)] for i, r in info.iterrows()]
    import baan_materiaal as bm
    era = bm.era(w)                                  # baan en materiaal: peil van Heerenveen per seizoen t.o.v. nu (ln; > 0 = toen trager)
    meta = {'banen': banen, 'afstanden': AF, 'laatste': laatste, 'nb': NB, 'era': {str(k): round(float(v), 5) for k, v in era.items()},
            'kol': ['id', 'naam', 'geslacht', 'land', 'geboren', 'van', 'tot', 'ritten', 'wk', 'os', 'wc', 'best_achter_t10'],
            'peil_hv': {f'{g}|{a}': round(float(v), 5) for (g, a), v in thv.items()}}
    with gzip.open(os.path.join(uit, 'index.json.gz'), 'wt', encoding='utf-8') as f:
        json.dump({'meta': meta, 'rijders': idx}, f, ensure_ascii=False, separators=(',', ':'))
    print('index klaar', len(idx), flush=True)

    # per blok
    R['blok'] = R['skater_id'] % NB; Z['blok'] = Z['skater_id'] % NB
    R['d'] = R['dag'].astype(int); R['a'] = R['afstand_idx'].astype(int); R['bi'] = R['baan'].map(bidx)
    Rg = dict(tuple(R.groupby('blok'))); Zg = dict(tuple(Z.groupby('blok')))
    rnd = lambda v, k=2: (round(float(v), k) if v is not None and np.isfinite(v) else None)
    for b in range(NB):
        rr = Rg.get(b); zz = Zg.get(b); out = {}
        if rr is None:
            continue
        for sid, x in rr.sort_values(['dag', 'a']).groupby('skater_id'):
            ritten = [[int(d), int(a), int(bi), rnd(t, 2), rnd(th, 2), (int(p) if np.isfinite(p) else None), int(n), s, rnd(sp, 2), rnd(kk, 4), int(vd)]
                      for d, a, bi, t, th, p, n, s, sp, kk, vd in zip(x['d'], x['a'], x['bi'], x['tijd'], x['tijd_hv'], x['plaats'], x['veld'], x['soort'], x['schok_pct'], x['kans'], x['verdacht'])]
            seizoenen = {}
            if zz is not None:
                for _, z in zz[zz['skater_id'] == sid].iterrows():
                    s = int(z['seizoen'])
                    e = seizoenen.setdefault(s, {'lft': (int(z['leeftijd']) if np.isfinite(z['leeftijd']) else None), 'af': {}})
                    e['af'][z['afstand']] = [rnd(z['t_hv']), rnd(z['achter_t10']), rnd(z['t_nu']), (int(z['rang']) if np.isfinite(z['rang']) else None), int(z['na'])]
            for s, e in seizoenen.items():
                k_ = (sid, s)
                if k_ in per.index:
                    p = per.loc[k_]; e['wk'] = int(p['n_wk']); e['os'] = int(p['n_os']); e['wc'] = int(p['n_wc'])
                if k_ in tnamen.index:
                    e['titels'] = tnamen.loc[k_]
                if k_ in groei.index:
                    g_ = groei.loc[k_]
                    if isinstance(g_, pd.DataFrame):
                        g_ = g_.iloc[-1]
                    if np.isfinite(g_['verbetering']) and 13 <= g_['leeftijd'] <= 19:
                        e['groei'] = rnd(100 * (np.exp(g_['verbetering']) - 1))
                if kans is not None and k_ in kans.index:
                    kk = kans.loc[k_]
                    e['kans'] = {c: rnd(kk[c], 4) for c in ('p_wk', 'e_wk', 'p_os', 'e_os', 'p_wc', 'e_wc')}
            out[int(sid)] = {'s': {str(k): v for k, v in sorted(seizoenen.items())}, 'r': ritten}
        with gzip.open(os.path.join(uit, 'r', f'{b}.json.gz'), 'wt', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        if b % 20 == 0:
            print('blok', b, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
