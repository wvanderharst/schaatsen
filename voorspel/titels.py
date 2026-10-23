"""Welke junioren worden wereldkampioen (senioren)? Per junior-momentopname (eind van het seizoen, leeftijd 15..19, talent.py) voor elk later
seizoen tot en met 36 jaar het aantal
  WK   : seniorentitels op het WK (afstanden, allround, sprint; kampioenen.py / kampioenen_titels.csv),
  OS   : olympische titels (apart),
  WC   : overwinningen in World Cup-races (internationale wedstrijden buiten WK/Spelen, veld >= 10 rijders, mediaanleeftijd >= 20,
         >= 5 rijders met niveau vooraf < 3%: de A-groep).
Model (geen groepen, alles functies): verwacht aantal in seizoen s = (te verdelen titels/races in s voor dat geslacht) x exp(eta),
eta = glad in de leeftijd in s (derdegraads) + juniorkenmerken (niveau t.o.v. de wereldtop, achterstand op de beste van de eigen leeftijd,
log rang in de eigen leeftijd, verbetering, leeftijd van de momentopname, geslacht, Nederland) + niveau x leeftijd in s en x leeftijd nu.
Poisson voor het aantal; voor 'minstens een' hetzelfde model op de seizoenen tot en met de eerste (kans = 1 - exp(-som)).
Alle kalenderseizoenen tellen mee, ook als de rijder gestopt is (stoppen hoort bij de uitkomst).
Spelen: zelfde kenmerken, gezamenlijk gefit met de WK-afstandstitels plus een eigen constante (weinig Spelen om alleen van te leren).
Toets buiten de steekproef: kruisfit even/oneven geboortejaren, vergeleken binnen de waargenomen seizoenen.
Gebruik: python titels.py <werkmap>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk

LAATSTE = 2025                 # laatste volledige seizoen (2025-26: Spelen en WK 2026)
MAX_LFT = 36
HIER = os.path.dirname(os.path.abspath(__file__))


def seizoen_van(dag):
    d = pd.to_datetime(np.asarray(dag), unit='D')
    return np.where(d.month >= 7, d.year, d.year - 1)


def uitkomsten(w):
    """Per rijder x seizoen: n_wk, n_os, n_wkaf (alleen WK afstanden), n_wc; en per geslacht x seizoen de totalen."""
    T = pd.read_csv(os.path.join(HIER, 'kampioenen_titels.csv'))
    T['seizoen'] = seizoen_van(T['dag']); T['os'] = T['titel'].str.startswith('Spelen'); T['af'] = T['titel'].str.startswith('WK afstanden')
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'gender', 'verdacht'])
    R['afstand'] = R['afstand'].astype(str)
    Tn = pk.toernooien(R); wc = pk.internationaal(R, set(Tn['eid']))
    V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'k_c', 'lf']).to_pandas()
    V['afstand'] = V['afstand'].astype(str)
    X = R[(R['event_id'] % 1000000).isin(wc) & ~R['verdacht'] & np.isfinite(R['tijd']) & R['afstand'].isin(list(pk.ZES))]
    X = X.merge(V.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    rk = ['event_id', 'dag', 'gender', 'afstand']
    G = X.groupby(rk).agg(n=('skater_id', 'nunique'), lft=('lf', 'median'), top=('k_c', lambda s: (s < 0.03).sum()))
    A = G[(G['n'] >= 10) & (G['lft'] >= 20) & (G['top'] >= 5)].index
    Xa = X.set_index(rk).loc[A].reset_index()
    W = Xa.loc[Xa.groupby(rk)['tijd'].idxmin(), ['skater_id', 'gender', 'dag']]
    W['seizoen'] = seizoen_van(W['dag'])
    Y = pd.concat([T.assign(n_wk=~T['os'], n_os=T['os'], n_wkaf=T['af'], n_wc=False)[['skater_id', 'gender', 'seizoen', 'n_wk', 'n_os', 'n_wkaf', 'n_wc']],
                   W.assign(n_wk=False, n_os=False, n_wkaf=False, n_wc=True)[['skater_id', 'gender', 'seizoen', 'n_wk', 'n_os', 'n_wkaf', 'n_wc']]])
    for c in ('n_wk', 'n_os', 'n_wkaf', 'n_wc'):
        Y[c] = Y[c].astype(float)
    per = Y.groupby(['skater_id', 'seizoen'])[['n_wk', 'n_os', 'n_wkaf', 'n_wc']].sum()
    tot = Y.groupby(['gender', 'seizoen'])[['n_wk', 'n_os', 'n_wkaf', 'n_wc']].sum()
    return per, tot, T


def toekomst_totalen(seizoenen):
    """Te verdelen per geslacht in een toekomstig seizoen s (kampioenschappen in kalenderjaar s + 1): Spelen als (s + 1) mod 4 == 2,
    WK afstanden (5 titels) als er geen Spelen zijn, allround en sprint (2) in even jaren, World Cups ~ het gemiddelde van 2022-2025."""
    j = np.asarray(seizoenen) + 1; os_ = (j % 4 == 2)
    return pd.DataFrame({'n_wk': 5.0 * ~os_ + 2.0 * (j % 2 == 0), 'n_os': 5.0 * os_, 'n_wkaf': 5.0 * ~os_, 'n_wc': np.nan}, index=seizoenen)


KEN = ['L', 'achter', 'lrang', 'verb', 'verb_onb', 't0', 'vrouw', 'ned']


def momenten(w):
    S = pd.read_parquet(os.path.join(w, 'talent_momentopnamen.parquet'),
                        columns=['skater_id', 'seizoen', 'naam', 'gender', 'land', 'geboren', 'exact', 'leeftijd', 'L_jeugd', 'beste_leeftijd',
                                 'rang_leeftijd', 'verbetering', 'L_sen', 'af_jeugd'])
    # stijgend niveau: de seniorentop van hetzelfde seizoen (10e beste seniorenniveau, 21+, per geslacht), glad over de seizoenen
    sen = S[(S['leeftijd'] >= 21) & np.isfinite(S['L_sen'])]
    t10 = sen.groupby(['gender', 'seizoen'])['L_sen'].apply(lambda v: np.sort(v.to_numpy())[9] if len(v) >= 10 else np.nan).rename('top10')
    t10 = t10.groupby(level='gender').transform(lambda v: v.rolling(3, center=True, min_periods=1).mean())
    J = S[S['exact'] & S['leeftijd'].between(15, 19) & np.isfinite(S['L_jeugd'])].copy()
    J = J.join(t10, on=['gender', 'seizoen'])
    J['Lr'] = (J['L_jeugd'] - J['top10']).clip(-0.05, 0.40)
    J['L'] = J['L_jeugd'].clip(-0.05, 0.40); J['achter'] = (J['L_jeugd'] - J['beste_leeftijd']).clip(0, 0.4)
    J['lrang'] = np.log(J['rang_leeftijd']); J['verb_onb'] = J['verbetering'].isna().astype(float)
    J['verb'] = J['verbetering'].clip(-0.1, 0.3).fillna(0.0); J['t0'] = (J['leeftijd'] - 17.0) / 2.0
    J['vrouw'] = (J['gender'] == 'v').astype(float); J['ned'] = (J['land'] == 'NED').astype(float)
    return J.reset_index(drop=True)


VARIANT = os.environ.get('TVAR', 'glad')


def ontwerp(D):
    """Kenmerken x leeftijd in het doelseizoen: tau = (leeftijd - 26) / 6, glad (derdegraads). Niveau als gladde functie:
    ln(L + 0,02) (verzadigt bij de allerbesten) en L, met interacties met tau en met de leeftijd van de momentopname."""
    tau = (D['a'].to_numpy(float) - 26.0) / 6.0
    L = D['L'].to_numpy(float); t0 = D['t0'].to_numpy(float); lL = np.log(np.clip(L, -0.015, None) + 0.02)
    basis = [tau, tau ** 2, tau ** 3, D['vrouw'].to_numpy(float), t0]
    if VARIANT == 'oud':
        cols = [D[k].to_numpy(float) for k in KEN] + [tau, tau ** 2, tau ** 3, L * tau, L * t0]
    elif VARIANT == 'L':
        cols = basis + [L, L * tau, L * t0]
    elif VARIANT == 'glad':
        cols = basis + [lL, L, lL * tau, lL * t0]
    elif VARIANT == 'mix':
        Lr = D['Lr'].to_numpy(float); lLr = np.log(np.clip(Lr, -0.03, None) + 0.05)
        cols = basis + [lL, L, lL * tau, lL * t0, lLr, D['verb'].to_numpy(float), D['verb_onb'].to_numpy(float), D['ned'].to_numpy(float), D['achter'].to_numpy(float)]
    elif VARIANT in ('rel', 'rel+'):
        Lr = D['Lr'].to_numpy(float); lLr = np.log(np.clip(Lr, -0.03, None) + 0.05)
        cols = basis + [lLr, Lr, lLr * tau, lLr * t0]
        if VARIANT == 'rel+':
            cols += [D['verb'].to_numpy(float), D['verb_onb'].to_numpy(float), D['ned'].to_numpy(float), D['achter'].to_numpy(float)]
    elif VARIANT == 'glad+':
        cols = basis + [lL, L, lL * tau, lL * t0, D['verb'].to_numpy(float), D['verb_onb'].to_numpy(float), D['ned'].to_numpy(float), D['achter'].to_numpy(float)]
    return np.column_stack(cols)


def poisson(X, y, off, lam=5.0, w=None, n_iter=60, extra=None):
    """Poisson-regressie met offset (IRLS, kleine ridge op de gestandaardiseerde kenmerken). Geeft een voorspelfunctie voor eta."""
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9; Z = np.c_[np.ones(len(X)), (X - mu_) / sd_]
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(Z.shape[1]); b[0] = np.log(max((w * y).sum(), 1e-3) / (w * np.exp(off)).sum())
    I = np.eye(Z.shape[1]) * lam; I[0, 0] = 0
    for _ in range(n_iter):
        eta = np.clip(Z @ b + off, -30, 10); m = np.exp(eta)
        H = Z.T @ (Z * (w * m)[:, None]) + I; g = Z.T @ (w * (y - m)) - I @ b
        d = np.linalg.solve(H, g); b += d
        if np.max(np.abs(d)) < 1e-7:
            break
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu_) / sd_] @ b


def rijen(J, per, tot, doel, eerste=False, tot_toekomst=None, tot_seizoen=LAATSTE):
    """Per momentopname alle latere seizoenen (tot en met tot_seizoen en leeftijd MAX_LFT): kenmerken, leeftijd a, offset, uitkomst."""
    uit = []
    for s in range(int(J['seizoen'].min()) + 1, tot_seizoen + 1):
        m = (J['seizoen'] < s) & (s - J['geboren'] <= MAX_LFT) & (s - J['geboren'] >= 17)
        if not m.any():
            continue
        D = J[m].copy(); D['s'] = s; D['a'] = s - D['geboren']
        if tot_toekomst is not None and s > LAATSTE:
            D['tot'] = float(tot_toekomst.loc[s, doel]) if np.isfinite(tot_toekomst.loc[s, doel]) else np.nan
        else:
            D['tot'] = D['gender'].map({g: tot[doel].get((g, s), 0.0) for g in ('m', 'v')}).astype(float)
        uit.append(D)
    D = pd.concat(uit, ignore_index=True)
    D = D[D['tot'] > 0].copy()
    D['y'] = per[doel].reindex(pd.MultiIndex.from_arrays([D['skater_id'], D['s']])).fillna(0.0).to_numpy()
    if eerste:
        # alleen seizoenen tot en met de eerste (na de momentopname; wie al eerder won valt af)
        eerder = per[per[doel] > 0].reset_index().groupby('skater_id')['seizoen'].min()
        f = D['skater_id'].map(eerder).fillna(9999).to_numpy()
        D = D[(D['s'] <= f) & (D['seizoen'] < f)].copy(); D['y'] = (D['y'] > 0).astype(float)
    return D


def fit_voorspel(Dfit, Dnu, extra=False):
    X = ontwerp(Dfit); Xn = ontwerp(Dnu)
    if extra:            # Spelen gezamenlijk met de WK-afstandstitels: eigen constante voor de Spelen
        X = np.c_[X, Dfit['is_os'].to_numpy(float)]; Xn = np.c_[Xn, Dnu['is_os'].to_numpy(float)]
    f = poisson(X, Dfit['y'].to_numpy(float), np.log(Dfit['tot'].to_numpy(float)))
    return np.exp(np.clip(f(Xn) + np.log(Dnu['tot'].to_numpy(float)), -30, 5))


def os_rijen(J, per, tot, eerste, tot_toekomst=None, tot_seizoen=LAATSTE):
    a = rijen(J, per, tot, 'n_wkaf', eerste, tot_toekomst, tot_seizoen).assign(is_os=0.0)
    b = rijen(J, per, tot, 'n_os', eerste, tot_toekomst, tot_seizoen).assign(is_os=1.0)
    return a, b


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def toets(J, per, tot):
    """Kruisfit even/oneven geboortejaren; per momentopname de som over de waargenomen seizoenen (verwacht aantal, kans op minstens een)."""
    regels = []
    res = {}
    for doel in ('n_wk', 'n_os', 'n_wc'):
        oos = {}
        for eerste in (False, True):
            if doel == 'n_os':
                a, b = os_rijen(J, per, tot, eerste); D = pd.concat([a, b], ignore_index=True)
            else:
                D = rijen(J, per, tot, doel, eerste)
            D['mu'] = np.nan
            for par in (0, 1):
                fit = (D['geboren'] % 2 != par); doelr = ~fit
                if doel == 'n_os':
                    fit &= True
                D.loc[doelr, 'mu'] = fit_voorspel(D[fit], D[doelr], extra=(doel == 'n_os'))
            if doel == 'n_os':
                D = D[D['is_os'] == 1]
            oos[eerste] = D.groupby(['skater_id', 'seizoen']).agg(mu=('mu', 'sum'), y=('y', 'sum'))
        M = oos[False].rename(columns={'mu': 'verwacht', 'y': 'aantal'}).join(oos[True].rename(columns={'mu': 'mu1', 'y': 'y1'}), how='left')
        M['kans'] = 1 - np.exp(-M['mu1']); M['ooit'] = (M['aantal'] > 0).astype(float)
        M['al'] = M['mu1'].isna()                       # al gewonnen voor de momentopname: valt af in de toets van 'minstens een'
        M = M.join(J.set_index(['skater_id', 'seizoen'])[['naam', 'gender', 'geboren', 'leeftijd', 'L_jeugd', 'rang_leeftijd']])
        res[doel] = M
        regels.append(f'\n== {doel} (buiten de steekproef, binnen de waargenomen seizoenen)')
        Ma = M
        M = M[~M['al']]
        regels.append(f"  (al gewonnen voor de momentopname, weggelaten: {int(Ma['al'].sum())})")
        for lf, x in M.groupby('leeftijd'):
            if x['ooit'].sum() < 2:
                continue
            regels.append(f"  leeftijd {int(lf)}: {len(x)} momentopnamen, {int(x['ooit'].sum())} met minstens een | verwacht aantal {x['verwacht'].sum():.1f} tegen {x['aantal'].sum():.0f}"
                          f" | kans: som {x['kans'].sum():.1f} tegen {int(x['ooit'].sum())}, AUC {auc(x['ooit'].to_numpy(), x['kans'].to_numpy()):.3f}"
                          f" (alleen niveau {auc(x['ooit'].to_numpy(), -x['L_jeugd'].to_numpy()):.3f})")
        x = M[M['leeftijd'] >= 17]
        t = x.groupby(pd.cut(x['kans'], [0, 0.002, 0.01, 0.03, 0.1, 0.3, 1.0]), observed=True).agg(n=('ooit', 'size'), voorspeld=('kans', 'mean'), echt=('ooit', 'mean'),
                                                                                                  verwacht_aantal=('verwacht', 'sum'), aantal=('aantal', 'sum'))
        regels.append('  kalibratie (leeftijd 17-19): \n' + t.round(4).to_string())
    return res, regels


def nu(J, per, tot):
    """Gefit op alle momentopnamen; voorspeld voor de jeugd van nu (laatste momentopname in seizoen >= 2024, leeftijd 15..19) over de hele
    carriere (tot en met 36 jaar)."""
    N = J[(J['seizoen'] >= LAATSTE - 1)].sort_values('seizoen').groupby('skater_id').tail(1).copy()
    N = N[N['leeftijd'] <= 19]
    tt = toekomst_totalen(np.arange(LAATSTE + 1, LAATSTE + 25))
    wcgem = tot['n_wc'].xs(slice(None)).groupby(level='seizoen').sum()
    tt['n_wc'] = float(np.mean([tot['n_wc'].get((g, s), np.nan) for g in ('m', 'v') for s in range(2022, LAATSTE + 1)]))
    eind = int((N['geboren'] + MAX_LFT).max())
    tt = toekomst_totalen(np.arange(LAATSTE + 1, eind + 1)); tt['n_wc'] = float(np.mean([tot['n_wc'].get((g, s), np.nan) for g in ('m', 'v') for s in range(2022, LAATSTE + 1)]))
    uit = N.set_index('skater_id')[['naam', 'gender', 'land', 'geboren', 'leeftijd', 'seizoen', 'L_jeugd', 'rang_leeftijd']].copy()
    for doel in ('n_wk', 'n_os', 'n_wc'):
        for eerste in (False, True):
            if doel == 'n_os':
                a, b = os_rijen(J, per, tot, eerste); Dfit = pd.concat([a, b], ignore_index=True)
                a2, b2 = os_rijen(N, per, tot, False, tt, eind); Dn = b2[b2['s'] > LAATSTE]
            else:
                Dfit = rijen(J, per, tot, doel, eerste)
                Dn = rijen(N, per, tot, doel, False, tt, eind); Dn = Dn[Dn['s'] > LAATSTE]
            Dn = Dn.copy(); Dn['mu'] = fit_voorspel(Dfit, Dn, extra=(doel == 'n_os'))
            s_ = Dn.groupby('skater_id')['mu'].sum()
            uit[doel + ('_kans' if eerste else '_verwacht')] = (1 - np.exp(-s_)) if eerste else s_
    return uit.fillna(0.0)


if __name__ == '__main__' and os.environ.get('ALLEEN_TOETS'):
    w = sys.argv[1]; per, tot, T = uitkomsten(w); J = momenten(w)
    import pickle; pickle.dump((per, tot, J), open(os.path.join(w, 'titels_invoer.pkl'), 'wb'))
    sys.exit(0)

if __name__ == '__main__':
    w = sys.argv[1]
    per, tot, T = uitkomsten(w)
    print('titels per geslacht x seizoen (laatste jaren):'); print(tot.unstack('gender').tail(8).to_string())
    J = momenten(w)
    print(f'{len(J)} junior-momentopnamen, geboren {int(J.geboren.min())}-{int(J.geboren.max())}')
    res, regels = toets(J, per, tot)
    print('\n'.join(regels))
    for doel, M in res.items():
        M.reset_index().to_parquet(os.path.join(w, f'titels_toets_{doel}.parquet'))
    U = nu(J, per, tot)
    U.reset_index().to_parquet(os.path.join(w, 'titels_nu.parquet'))
    print(U.sort_values('n_wk_kans', ascending=False).groupby('gender').head(15).to_string())
