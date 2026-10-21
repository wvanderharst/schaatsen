"""Talentmodel: welke jonge rijders worden de beste profs?
Aan het eind van elk jeugdseizoen (leeftijd 13..19) wordt de stand van het Kalmanfilter (kalman_mv2, momentopname per rijder x seizoen)
vertaald naar de kans dat de rijder op 21-24 jaar het niveau van de wereldtop-50 (en top-10) haalt op zijn/haar beste seniorenafstand.
  niveau        : Kalman-niveau x_j = ln tijd - theta - wereldtop op afstand j (500 m .. 10 km), alleen afstanden met >= 3 ritten (senior)
                  of >= 2 ritten (jeugd).
  uitkomst      : beste seniorenniveau op 21..24 jaar <= drempel top-50 (of top-10): per seizoen het 50e (10e) beste niveau van de
                  rijders van 21+, mediaan over 2012..2019. Wie stopt of geen senior-data heeft, haalt het niet.
  kenmerken     : niveau nu (beste, sprint, duur), afstand tot de beste van de eigen leeftijd, rang in de eigen leeftijd (log),
                  groeisnelheid v, verbetering t.o.v. vorig seizoen, onzekerheid, aantal ritten, jaren actief, Nederland, geslacht;
                  alles ook x (leeftijd - 16) zodat de weging met de leeftijd verandert (functies, geen groepen).
  fit / toets   : logistische regressie op de geboortejaren 1992..1996, toets op 1997..2001 (uitkomst volledig waargenomen t/m 2025).
Gebruik: python talent.py <map met ritten_cat.parquet> <kalman_par.json> <uitmap> [nu]   (nu: ook de lijst voor de jeugd van nu)"""
import os, sys, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import kalman_mv2 as kf
from kalman import voorbereid

AF = ['100m', '300m', '500m', '1000m', '1500m', '3000m', '5000m', '10000m']
SEN = [2, 3, 4, 5, 6, 7]
FIT_JAREN = (1992, 1996); TOETS_JAREN = (1997, 2001)


def momentopnamen(w, par):
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'))
    R, inv = voorbereid(R); o = kf.pass_(inv, par, snap=True); s = o['snap']
    _, uniq = pd.factorize(R['skater_id'])
    S = pd.DataFrame({'skater_id': uniq[s['rid']], 'seizoen': s['seizoen'], 'n_seizoen': s['n'], 'lf': s['lf'], 'dag': s['dag']})
    for j, a in enumerate(AF):
        S['x_' + a] = s['x'][:, j]; S['p_' + a] = s['p'][:, j]; S['na_' + a] = s['na'][:, j]
    S['v'] = s['x'][:, 8]; S['pv'] = s['p'][:, 8]
    info = R.sort_values('dag').groupby('skater_id').agg(naam=('skater_name', 'last'), gender=('gender', 'last'), land=('land', lambda x: x.mode().iloc[0]),
                                                          geboren=('geboren', 'last'), exact=('leeftijd_exact', 'last'), eerste=('seizoen', 'min'))
    S = S.join(info, on='skater_id')
    S['leeftijd'] = S['seizoen'] - S['geboren']
    # piek: beste en op-een-na-beste echte rit van het seizoen (ln tijd - theta - wereldtop) op 500 m .. 3000 m
    z = pd.DataFrame({'skater_id': R['skater_id'].to_numpy(), 'seizoen': R['seizoen'].to_numpy(), 'z': inv['z'], 'a': R['afstand_idx'].to_numpy()})
    z = z[(z['a'] >= 2) & (z['a'] <= 5) & np.isfinite(z['z'])]
    pk = z.groupby(['skater_id', 'seizoen'])['z'].agg(z_best='min', z_2e=lambda v: np.sort(v.to_numpy())[1] if len(v) > 1 else np.nan)
    S = S.join(pk, on=['skater_id', 'seizoen'])
    return S, R


def niveaus(S):
    def beste(cols, nmin):
        X = np.column_stack([np.where(S['na_' + AF[j]] >= nmin, S['x_' + AF[j]], np.inf) for j in cols])
        b = X.min(1); arg = np.array(cols)[X.argmin(1)]
        return np.where(np.isfinite(b), b, np.nan), arg
    S['L_sen'], S['af_sen'] = beste(SEN, 3)
    S['L_jeugd'], S['af_jeugd'] = beste([2, 3, 4, 5], 2)
    S['L_sprint'], _ = beste([2, 3], 2); S['L_duur'], _ = beste([4, 5, 6], 2)
    return S


def uitkomst(S):
    sen = S[(S['leeftijd'] >= 21) & np.isfinite(S['L_sen'])]
    drempel = {}
    for g, x in sen[(sen['seizoen'] >= 2012) & (sen['seizoen'] <= 2019)].groupby('gender'):
        per = x.groupby('seizoen')['L_sen'].apply(lambda v: np.sort(v.to_numpy()))
        drempel[(g, 50)] = float(np.median([v[49] for v in per if len(v) >= 50])); drempel[(g, 10)] = float(np.median([v[9] for v in per if len(v) >= 10]))
    w = sen[(sen['leeftijd'] >= 21) & (sen['leeftijd'] <= 24)]
    best = w.groupby('skater_id')['L_sen'].min().rename('L_best_21_24')
    return best, drempel


def kenmerken(S):
    S = S.sort_values(['skater_id', 'seizoen'])
    S['L_vorig'] = S.groupby('skater_id')['L_jeugd'].shift(1); S['s_vorig'] = S.groupby('skater_id')['seizoen'].shift(1)
    S['verbetering'] = np.where(S['s_vorig'] == S['seizoen'] - 1, S['L_vorig'] - S['L_jeugd'], np.nan)
    S['L_vorig2'] = S.groupby('skater_id')['L_jeugd'].shift(2); S['s_vorig2'] = S.groupby('skater_id')['seizoen'].shift(2)
    S['verbetering2'] = np.where(S['s_vorig2'] == S['seizoen'] - 2, S['L_vorig2'] - S['L_jeugd'], np.nan)
    S['n_tot'] = S.groupby('skater_id')['n_seizoen'].cumsum()
    g = S.groupby(['gender', 'seizoen', 'leeftijd'])['L_jeugd']
    S['rang_leeftijd'] = g.rank(method='min'); S['beste_leeftijd'] = g.transform('min')
    F = pd.DataFrame(index=S.index)
    F['L'] = S['L_jeugd']; F['L2'] = S['L_jeugd'] ** 2
    F['achter_beste'] = S['L_jeugd'] - S['beste_leeftijd']
    F['log_rang'] = np.log(S['rang_leeftijd'])
    F['v'] = S['v'].clip(-0.15, 0.15)
    F['verbetering'] = S['verbetering'].clip(-0.1, 0.3).fillna(0.0); F['verb_onb'] = S['verbetering'].isna().astype(float)
    F['verbetering2'] = S['verbetering2'].clip(-0.1, 0.4).fillna(0.0); F['verb2_onb'] = S['verbetering2'].isna().astype(float)
    F['versnelling'] = (S['verbetering'] - (S['verbetering2'] - S['verbetering'])).clip(-0.3, 0.3).fillna(0.0)
    F['piek'] = (S['z_best'] - S['L_jeugd']).clip(-0.1, 0.1).fillna(0.0); F['piek2'] = (S['z_2e'] - S['L_jeugd']).clip(-0.1, 0.1).fillna(0.0)
    F['profiel'] = (S['L_sprint'] - S['L_duur']).clip(-0.1, 0.1).fillna(0.0)
    F['onz'] = np.log(np.sqrt(np.column_stack([S['p_' + a] for a in AF[2:6]]).min(1)))
    F['log_n'] = np.log1p(S['n_seizoen']); F['log_ntot'] = np.log1p(S['n_tot'])
    F['jaren_actief'] = (S['seizoen'] - S['eerste']).clip(0, 12)
    F['ned'] = (S['land'] == 'NED').astype(float); F['vrouw'] = (S['gender'] == 'v').astype(float)
    t = (S['leeftijd'] - 16.0) / 3.0
    for c in list(F.columns):
        if c != 'vrouw':
            F[c + '_t'] = F[c] * t
    F['t'] = t; F['t2'] = t ** 2
    return S, F


def logistisch(X, y, lam=1.0, n_iter=50):
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    b = np.zeros(Z.shape[1]); I = np.eye(Z.shape[1]) * lam; I[0, 0] = 0
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30))); W = p * (1 - p) + 1e-9
        H = Z.T @ (Z * W[:, None]) + I; g = Z.T @ (y - p) - I @ b
        d = np.linalg.solve(H, g); b += d
        if np.max(np.abs(d)) < 1e-8: break
    return lambda Xn: 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ b, -30, 30))), b


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def draai(w, par, uit=None):
    S, R = momentopnamen(w, par); S = niveaus(S)
    best, drempel = uitkomst(S); S = S.join(best, on='skater_id')
    S, F = kenmerken(S)
    jeugd = S['exact'] & (S['leeftijd'] >= 13) & (S['leeftijd'] <= 19) & np.isfinite(S['L_jeugd']) & np.isfinite(F).all(1)
    regels = [f"drempels (niveau t.o.v. wereldtop): " + ', '.join(f"{g} top-{k}: {100 * v:.2f}%" for (g, k), v in sorted(drempel.items()))]
    res = {}; res_y = {}
    for k in (50, 10):
        dr = S['gender'].map({g: drempel[(g, k)] for g in ('m', 'v')})
        y = (S['L_best_21_24'] <= dr).astype(float).to_numpy()
        fit = jeugd & S['geboren'].between(*FIT_JAREN); toets = jeugd & S['geboren'].between(*TOETS_JAREN)
        f, b = logistisch(F[fit].to_numpy(float), y[fit.to_numpy()])
        S[f'p_top{k}'] = f(np.nan_to_num(F.to_numpy(float)))
        res[k] = (y, f); res_y[k] = y
        for leeft in (14, 15, 16, 17, 18, 19):
            m_ = toets & (S['leeftijd'] == leeft)
            yy = y[m_.to_numpy()]; pp = S.loc[m_, f'p_top{k}'].to_numpy(); base = -S.loc[m_, 'L_jeugd'].to_numpy()
            if yy.sum() < 3: continue
            # precisie in de top-N per geboortejaar x geslacht (N = aantal dat het echt haalde): model vs alleen huidig niveau
            raak_m = raak_b = tot = 0
            for _, x in S[m_].assign(y=yy).groupby(['geboren', 'gender']):
                n_ = int(x['y'].sum())
                if n_ == 0: continue
                tot += n_; raak_m += int(x.nlargest(n_, f'p_top{k}')['y'].sum()); raak_b += int(x.nsmallest(n_, 'L_jeugd')['y'].sum())
            regels.append(f"top-{k} op leeftijd {leeft}: {int(m_.sum())} rijders, {int(yy.sum())} haalden het | AUC model {auc(yy, pp):.3f}, alleen niveau {auc(yy, base):.3f} |"
                          f" raak in de top-N: model {raak_m}/{tot}, alleen niveau {raak_b}/{tot}")
    # opgesplitst: kans dat de rijder doorgaat tot 21-24 (met seniorenniveau) x verwacht seniorenniveau als hij/zij doorgaat
    door = np.isfinite(S['L_best_21_24']).to_numpy().astype(float)
    fit = (jeugd & S['geboren'].between(*FIT_JAREN)).to_numpy(); toets = (jeugd & S['geboren'].between(*TOETS_JAREN)).to_numpy()
    Xa = np.nan_to_num(F.to_numpy(float))
    f_door, _ = logistisch(Xa[fit], door[fit]); S['p_door'] = f_door(Xa)
    from model import huber
    fd = fit & (door == 1)
    mu, sd = Xa[fd].mean(0), Xa[fd].std(0) + 1e-9; Zx = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]
    bL = huber(Zx[fd], S['L_best_21_24'].to_numpy()[fd]); S['L_verwacht'] = Zx @ bL
    res = S['L_best_21_24'].to_numpy()[fd] - S['L_verwacht'].to_numpy()[fd]
    # spreiding van de rest als functie van de leeftijd (jonger = onzekerder): log|res| lineair in de leeftijd
    A_ = lambda m: np.c_[np.ones(m.sum()), S['leeftijd'].to_numpy()[m], np.exp(-np.clip(S['L_verwacht'].to_numpy()[m], -0.1, 0.4) / 0.04)]
    alle = np.ones(len(S), bool)
    ga = np.linalg.lstsq(A_(fd), np.log(np.abs(res) + 1e-5), rcond=None)[0]       # spreiding: functie van leeftijd en verwacht niveau
    sc = np.exp(A_(alle) @ ga); sc *= np.median(np.abs(res) / sc[fd]) / 0.6745
    S['L_sd'] = sc
    rs = np.sort(res / sc[fd])                                  # empirische verdeling van de gestandaardiseerde rest
    for k in (50, 10):
        dr = S['gender'].map({g: drempel[(g, k)] for g in ('m', 'v')}).to_numpy()
        q = np.searchsorted(rs, (dr - S['L_verwacht'].to_numpy()) / sc, side='right') / len(rs)
        S[f'p2_top{k}'] = S['p_door'] * q
        y = res_y[k]
        for leeft in (14, 15, 16, 17, 18, 19):
            m_ = toets & (S['leeftijd'] == leeft).to_numpy(); yy = y[m_]
            if yy.sum() < 3: continue
            raak = tot = 0
            for _, x in S[m_].assign(y=yy).groupby(['geboren', 'gender']):
                n_ = int(x['y'].sum())
                if n_: tot += n_; raak += int(x.nlargest(n_, f'p2_top{k}')['y'].sum())
            regels.append(f"opgesplitst top-{k} op leeftijd {leeft}: AUC {auc(yy, S.loc[m_, f'p2_top{k}'].to_numpy()):.3f} | raak in de top-N {raak}/{tot}")
    if uit:
        S.to_parquet(os.path.join(uit, 'talent_momentopnamen.parquet'))
    return S, F, drempel, regels


if __name__ == '__main__':
    w = sys.argv[1]; par = json.load(open(sys.argv[2])); uit = sys.argv[3] if len(sys.argv) > 3 else w
    S, F, drempel, regels = draai(w, par, uit)
    print('\n'.join(regels))


def huidige_lijst(S, F, drempel, R, uit, n=40):
    """De jeugd van nu (laatste momentopname sinds seizoen 2025, leeftijd 13..19): kans op wereldtop-50 / top-10 als senior (21-24 jaar),
    met het model gefit op alle volledig waargenomen geboortejaren (1992..2001), en het verwachte seniorenniveau als tijd in Heerenveen."""
    jeugd = S['exact'] & (S['leeftijd'] >= 13) & (S['leeftijd'] <= 19) & np.isfinite(S['L_jeugd'])
    Xa = np.nan_to_num(F.to_numpy(float)); fit = (jeugd & S['geboren'].between(1992, 2001)).to_numpy()
    for k in (50, 10):
        dr = S['gender'].map({g: drempel[(g, k)] for g in ('m', 'v')})
        y = (S['L_best_21_24'] <= dr).astype(float).to_numpy()
        f, _ = logistisch(Xa[fit], y[fit]); S[f'kans_top{k}'] = f(Xa)
    door = np.isfinite(S['L_best_21_24']).to_numpy().astype(float)
    f, _ = logistisch(Xa[fit], door[fit]); S['kans_door'] = f(Xa)
    from model import huber
    fd = fit & (door == 1); mu, sd = Xa[fd].mean(0), Xa[fd].std(0) + 1e-9; Zx = np.c_[np.ones(len(Xa)), (Xa - mu) / sd]
    S['L_senior'] = Zx @ huber(Zx[fd], S['L_best_21_24'].to_numpy()[fd])
    nu = S[jeugd & (S['seizoen'] >= 2025)].sort_values('seizoen').groupby('skater_id').tail(1)
    # tijd in Heerenveen: exp(wereldtop + niveau + racepeil Heerenveen) op de beste afstand nu
    lt = np.log(R['tijd'].to_numpy(float))
    ref = (R.assign(y=lt - R['theta'])[(R['jaar'] <= 2019) & R['theta'].notna() & ~R['verdacht']]
           .groupby(['gender', 'afstand_idx'])['y'].quantile(0.01))
    hv = R[(R['baan'] == 'Heerenveen (NED)') & (R['seizoen'] >= 2022)].groupby(['gender', 'afstand_idx'])['theta'].median()
    def tijd(g, j, L):
        t = np.exp(ref.get((g, j), np.nan) + L + hv.get((g, j), 0.0))
        m, s = divmod(t, 60)
        return (f"{int(m)}:{s:05.2f}" if m >= 1 else f"{s:.2f}").replace('.', ',') if np.isfinite(t) else ''
    nu['beste_afstand'] = [AF[int(j)] for j in nu['af_jeugd']]
    nu['tijd_nu_hv'] = [tijd(g, int(j), L) for g, j, L in zip(nu['gender'], nu['af_jeugd'], nu['L_jeugd'])]
    nu['tijd_senior_hv'] = [tijd(g, int(j), L) for g, j, L in zip(nu['gender'], nu['af_jeugd'], nu['L_senior'])]
    kol = ['gender', 'naam', 'land', 'geboren', 'leeftijd', 'seizoen', 'beste_afstand', 'tijd_nu_hv', 'tijd_senior_hv', 'kans_top50', 'kans_top10', 'kans_door']
    lijst = (nu.sort_values('kans_top50', ascending=False).groupby('gender').head(n)[kol]
             .assign(geboren=lambda d: d['geboren'].astype(int), leeftijd=lambda d: d['leeftijd'].astype(int),
                     kans_top50=lambda d: (100 * d['kans_top50']).round(1), kans_top10=lambda d: (100 * d['kans_top10']).round(1),
                     kans_door=lambda d: (100 * d['kans_door']).round(0)))
    lijst.to_csv(os.path.join(uit, 'talent_nu.csv'), index=False)
    return lijst


if __name__ == '__main__' and len(sys.argv) > 4 and sys.argv[4] == 'nu':
    R_ = pd.read_parquet(os.path.join(sys.argv[1], 'ritten_cat.parquet'))
    L_ = huidige_lijst(S, F, drempel, R_, uit)
    print(L_.groupby('gender').head(15).to_string(index=False))
