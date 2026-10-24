"""Backtest van de talentvoorspellingen in de tijd: op een peilmoment S (eind van seizoen S) wordt het juniorenmodel (titels.py, variant
rel+) alleen gefit op wat toen bekend was (seizoenen t/m S), en voorspeld voor alle junioren van 15-19 jaar in seizoen S: verwacht aantal en
kans op minstens een WK-titel / olympische titel / World Cup-zege in de seizoenen S+1 t/m het laatste volledige seizoen (2025-26), met het
werkelijke aantal te verdelen titels per seizoen (de kalender). Vergeleken met wat er gebeurde. Geen ijkingsstap (die zou ook alleen op
gegevens tot S mogen). Gebruik: python titels_backtest.py <werkmap> [peilmomenten, bv. 2010,2012,2014]"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import titels as tl

w = sys.argv[1]; CUTS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '2010,2012,2014').split(',')]
per, tot, J = pickle.load(open(os.path.join(w, 'titels_invoer.pkl'), 'rb'))
tl.VARIANT = 'rel+'
NAAM = {'n_wk': 'WK-titel', 'n_os': 'olympische titel', 'n_wc': 'World Cup-zege'}
uit = []
for S in CUTS:
    Jf = J[J['seizoen'] < S]; Jn = J[J['seizoen'] == S]
    res = Jn[['skater_id', 'seizoen', 'naam', 'gender', 'land', 'geboren', 'leeftijd', 'L_jeugd', 'Lr', 'rang_leeftijd']].set_index(['skater_id', 'seizoen'])
    for doel in ('n_wk', 'n_os', 'n_wc'):
        for eerste in (False, True):
            if doel == 'n_os':
                a, b = tl.os_rijen(Jf, per, tot, eerste, tot_seizoen=S); Dfit = pd.concat([a, b], ignore_index=True)
                _, Dn = tl.os_rijen(Jn, per, tot, False)
            else:
                Dfit = tl.rijen(Jf, per, tot, doel, eerste, tot_seizoen=S)
                Dn = tl.rijen(Jn, per, tot, doel, False)
            Dn = Dn[Dn['s'] > S].copy()
            Dn['mu'] = tl.fit_voorspel(Dfit, Dn, extra=(doel == 'n_os'))
            g = Dn.groupby(['skater_id', 'seizoen']).agg(mu=('mu', 'sum'), y=('y', 'sum'))
            if eerste:
                res['p_' + doel] = 1 - np.exp(-g['mu'])
            else:
                res['e_' + doel] = g['mu']; res['y_' + doel] = g['y']
    res['peil'] = S
    uit.append(res.reset_index())
    print(f'peilmoment {S}: {len(res)} junioren, gefit op {len(Jf)} momentopnamen', flush=True)
U = pd.concat(uit, ignore_index=True).fillna({c: 0.0 for c in ('y_n_wk', 'y_n_os', 'y_n_wc')})
U.to_parquet(os.path.join(w, 'titels_backtest.parquet'))
# eerder gewonnen (voor het peilmoment) telt niet als voorspelling
eerder = {d: per[per[d] > 0].reset_index().groupby('skater_id')['seizoen'].min() for d in ('n_wk', 'n_os', 'n_wc')}


def auc(y, s):
    return tl.auc(np.asarray(y, float), np.asarray(s, float))


for doel in ('n_wk', 'n_os', 'n_wc'):
    al = U['skater_id'].map(eerder[doel]).fillna(9999) <= U['seizoen']
    X = U[~al].copy(); X['ooit'] = (X['y_' + doel] > 0).astype(float); p = X['p_' + doel]
    print(f"\n=== {NAAM[doel]} (peilmomenten {', '.join(map(str, CUTS))}; uitkomst t/m 2025-26) ===")
    print(f"junioren {len(X)}, met minstens een daarna {int(X['ooit'].sum())}; som van de kansen {p.sum():.1f}; verwacht aantal {X['e_' + doel].sum():.1f}, werkelijk {X['y_' + doel].sum():.0f}")
    print(f"AUC model {auc(X['ooit'], p):.3f} | alleen niveau (t.o.v. de top van dat seizoen) {auc(X['ooit'], -X['Lr']):.3f} | rang in eigen leeftijd {auc(X['ooit'], -X['rang_leeftijd']):.3f}")
    t = X.groupby(pd.cut(p, [0, 0.005, 0.02, 0.05, 0.15, 0.4, 1.0]), observed=True).agg(junioren=('ooit', 'size'), voorspeld=('p_' + doel, 'mean'), werkelijk=('ooit', 'mean'))
    print('kalibratie:'); print(t.round(3).to_string())
    for n in (10, 25, 50):
        top = X.groupby(['peil', 'gender'], group_keys=False).apply(lambda x: x.nlargest(n, 'p_' + doel))
        niv = X.groupby(['peil', 'gender'], group_keys=False).apply(lambda x: x.nsmallest(n, 'Lr'))
        print(f"  in de top-{n} per peilmoment x geslacht: model {int(top['ooit'].sum())} van {int(X['ooit'].sum())} latere winnaars (verwacht {top['p_' + doel].sum():.1f}),"
              f" alleen niveau {int(niv['ooit'].sum())}")
pd.set_option('display.width', 220)
for S in CUTS:
    for g, x in U[U['peil'] == S].groupby('gender'):
        x = x.sort_values('p_n_wk', ascending=False).head(15)
        print(f"\n{S}/{(S + 1) % 100:02d} {'vrouwen' if g == 'v' else 'mannen'}: top-15 naar kans op een WK-titel (en wat er gebeurde t/m 2025-26)")
        print(x[['naam', 'land', 'leeftijd', 'p_n_wk', 'y_n_wk', 'p_n_os', 'y_n_os', 'p_n_wc', 'y_n_wc']].rename(columns={
            'p_n_wk': 'kans WK', 'y_n_wk': 'WK-titels', 'p_n_os': 'kans OS', 'y_n_os': 'OS-titels', 'p_n_wc': 'kans WC', 'y_n_wc': 'WC-zeges'}).round(3).to_string(index=False))
# latere kampioenen: waar stonden ze?
K = U[U['y_n_wk'] + U['y_n_os'] > 0].copy()
K['rang_model'] = [int((U[(U['peil'] == r.peil) & (U['gender'] == r.gender)]['p_n_wk'] > r.p_n_wk).sum()) + 1 for r in K.itertuples()]
K['rang_niveau'] = [int((U[(U['peil'] == r.peil) & (U['gender'] == r.gender)]['Lr'] < r.Lr).sum()) + 1 for r in K.itertuples()]
print('\nlatere wereld- of olympisch kampioenen: plaats op de lijst van hun peilmoment (model tegen alleen niveau)')
print(K.sort_values(['peil', 'rang_model'])[['peil', 'naam', 'land', 'leeftijd', 'p_n_wk', 'rang_model', 'rang_niveau', 'y_n_wk', 'y_n_os']].round(3).to_string(index=False))
