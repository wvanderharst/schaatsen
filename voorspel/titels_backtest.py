"""Backtest van de talentvoorspellingen in de tijd: op een peilmoment S (eind van seizoen S) wordt het juniorenmodel (titels.py) alleen gefit
op wat toen bekend was (seizoenen t/m S), en voorspeld voor alle junioren van 15-19 jaar in seizoen S: verwacht aantal en kans op minstens
een WK-titel / olympische titel / World Cup-zege in de seizoenen S+1 t/m het laatste volledige seizoen (2025-26), met het werkelijke aantal te
verdelen titels per seizoen. Vergeleken met wat er gebeurde, per geslacht, en met rangschikken op alleen het niveau. Geen ijkingsstap.
Varianten (VARIANTEN=variant:apart,...): variant uit titels.ontwerp, apart = 1 om mannen en vrouwen apart te fitten.
Gebruik: python titels_backtest.py <werkmap> [peilmomenten, bv. 2010,2012,2014]"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import titels as tl

w = sys.argv[1]; CUTS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '2010,2012,2014').split(',')]
per, tot, J = pickle.load(open(os.path.join(w, 'titels_invoer.pkl'), 'rb'))
if 'verb2' not in J:
    J = tl.momenten(w); pickle.dump((per, tot, J), open(os.path.join(w, 'titels_invoer.pkl'), 'wb'))
NAAM = {'n_wk': 'WK-titel', 'n_os': 'olympische titel', 'n_wc': 'World Cup-zege'}
eerder = {d: per[per[d] > 0].reset_index().groupby('skater_id')['seizoen'].min() for d in ('n_wk', 'n_os', 'n_wc')}
auc = lambda y, s: tl.auc(np.asarray(y, float), np.asarray(s, float))


def backtest():
    uit = []
    for S in CUTS:
        Jf = J[J['seizoen'] < S]; Jn = J[J['seizoen'] == S]
        res = Jn[['skater_id', 'seizoen', 'naam', 'gender', 'land', 'geboren', 'leeftijd', 'L_jeugd', 'Lr', 'rang_leeftijd', 'verbetering']].set_index(['skater_id', 'seizoen'])
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
    return pd.concat(uit, ignore_index=True).fillna({c: 0.0 for c in ('y_n_wk', 'y_n_os', 'y_n_wc')})


def rapport(U, label):
    for doel in ('n_wk', 'n_os', 'n_wc'):
        al = U['skater_id'].map(eerder[doel]).fillna(9999) <= U['seizoen']
        X = U[~al].copy(); X['ooit'] = (X['y_' + doel] > 0).astype(float)
        for g, x in X.groupby('gender'):
            top = lambda n, k, asc: x.groupby('peil', group_keys=False).apply(lambda z: z.nsmallest(n, k) if asc else z.nlargest(n, k))['ooit'].sum()
            p = x['p_' + doel].clip(1e-9, 1 - 1e-9)
            ll = np.mean(x['ooit'] * np.log(p) + (1 - x['ooit']) * np.log(1 - p))
            print(f"  RES {label:9s} {NAAM[doel]:16s} {'vrouwen' if g == 'v' else 'mannen '} | winnaars {int(x['ooit'].sum()):3d} som kansen {p.sum():6.1f}"
                  f" | aantal {x['y_' + doel].sum():4.0f} verwacht {x['e_' + doel].sum():6.1f} | log-lik {ll:.5f} | AUC {auc(x['ooit'], p):.3f} (niveau {auc(x['ooit'], -x['Lr']):.3f})"
                  f" | top-10 {int(top(10, 'p_' + doel, False))} (niveau {int(top(10, 'Lr', True))}) top-25 {int(top(25, 'p_' + doel, False))} (niveau {int(top(25, 'Lr', True))})", flush=True)


def groei_belang():
    """Hoe zwaar weegt groei? Bij gelijk niveau (t.o.v. de top van het seizoen) en leeftijd: kans op een latere WK-titel / World Cup-zege
    naar groei van het laatste jaar, per geslacht (logistische regressie op de momentopnamen tot 2014, uitkomst t/m 2025-26)."""
    from scipy.special import expit
    for doel in ('n_wk', 'n_wc'):
        X = J[J['seizoen'].between(2008, 2014) & J['verbetering'].notna()].copy()
        later = per[per[doel] > 0].reset_index()
        win = later.groupby('skater_id')['seizoen'].apply(set)
        X['y'] = [float(any(s > z for s in win.get(i, ()))) for i, z in zip(X['skater_id'], X['seizoen'])]
        for g, x in X.groupby('gender'):
            Lr = x['Lr'].to_numpy(); lLr = np.log(np.clip(Lr, -0.03, None) + 0.05); t0 = x['t0'].to_numpy(); v = x['verb'].to_numpy()
            Z = np.c_[np.ones(len(x)), lLr, Lr, t0, lLr * t0, v / 0.05]
            b = np.zeros(Z.shape[1]); y = x['y'].to_numpy()
            for _ in range(50):
                p = expit(Z @ b); H = Z.T @ (Z * (p * (1 - p) + 1e-9)[:, None]) + np.eye(len(b)) * 1e-3; b += np.linalg.solve(H, Z.T @ (y - p))
            se = np.sqrt(np.diag(np.linalg.inv(H)))
            print(f"  GROEI {NAAM[doel]:16s} {'vrouwen' if g == 'v' else 'mannen '}: per 5% extra groei in het laatste jaar x {np.exp(b[-1]):.2f} op de odds"
                  f" (t = {b[-1] / se[-1]:.1f}; {int(y.sum())} met een latere zege van {len(x)})", flush=True)


if __name__ == '__main__':
    print('groei bij gelijk niveau en leeftijd:'); groei_belang()
    for vv in (os.environ.get('VARIANTEN') or 'rel+:0').split(','):
        var, ap = vv.split(':'); tl.VARIANT = var; tl.APART = ap == '1'
        U = backtest(); U.to_parquet(os.path.join(w, f'titels_backtest_{var}_{ap}.parquet'))
        print(f'\n## {var}, {"apart" if tl.APART else "samen"}', flush=True)
        rapport(U, f'{var}:{ap}')
