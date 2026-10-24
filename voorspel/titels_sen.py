"""Titelkansen voor elk seizoen vanaf 18 jaar (doorgetrokken van titels.py, dat 15-19 jaar doet). Per momentopname (eind van het seizoen,
leeftijd 18-39) voor elk later seizoen tot en met 40 jaar het aantal WK-titels, olympische titels en World Cup-zeges; voorspeld vanaf het
volgende seizoen: verwacht aantal en kans op minstens een (nog een).
Kenmerken (glad, geen groepen): seniorenniveau (beste afstand met >= 3 ritten; anders het jeugdniveau) t.o.v. de seniorentop-10 van HETZELFDE
seizoen, als ln(L + 0,05) en lineair, met interacties met de leeftijd in het doelseizoen en de leeftijd nu; verandering t.o.v. vorig seizoen;
wat de rijder al won (ln(1 + titels/zeges tot en met dit seizoen), ook x leeftijd in het doelseizoen); geslacht; Nederland; leeftijd nu
(kwadratisch). Zelfde Poisson-opzet met offset (te verdelen titels/races per seizoen) als titels.py; de kans op minstens een via dezelfde
vorm op de seizoenen tot en met de eerste zege NA de momentopname. Spelen gezamenlijk met de WK-afstandstitels plus een eigen constante.
Toets buiten de steekproef: kruisfit even/oneven geboortejaren, binnen de waargenomen seizoenen; daarna een gladde ijking (titels.ijk_kans,
titels.ijk_aantal). Gebruik: python titels_sen.py <werkmap>"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import titels as tl

tl.MAX_LFT = 40
LAATSTE = tl.LAATSTE


def momenten_sen(w, per):
    S = pd.read_parquet(os.path.join(w, 'talent_momentopnamen.parquet'),
                        columns=['skater_id', 'seizoen', 'naam', 'gender', 'land', 'geboren', 'exact', 'leeftijd', 'L_jeugd', 'L_sen'])
    sen = S[(S['leeftijd'] >= 21) & np.isfinite(S['L_sen'])]
    t10 = sen.groupby(['gender', 'seizoen'])['L_sen'].apply(lambda v: np.sort(v.to_numpy())[9] if len(v) >= 10 else np.nan).rename('top10')
    t10 = t10.groupby(level='gender').transform(lambda v: v.rolling(3, center=True, min_periods=1).mean())
    S = S.join(t10, on=['gender', 'seizoen']).sort_values(['skater_id', 'seizoen'])
    S['Ls'] = S['L_sen'].where(np.isfinite(S['L_sen']), S['L_jeugd'])
    S['Lv'] = S.groupby('skater_id')['Ls'].shift(1); S['sv'] = S.groupby('skater_id')['seizoen'].shift(1)
    J = S[S['exact'] & S['leeftijd'].between(18, 39) & np.isfinite(S['Ls'])].copy()
    J['Lr'] = (J['Ls'] - J['top10']).clip(-0.05, 0.40)
    d = np.where(J['sv'] == J['seizoen'] - 1, J['Lv'] - J['Ls'], np.nan)
    J['dL'] = np.clip(np.nan_to_num(d), -0.1, 0.2); J['dL_onb'] = np.isnan(d).astype(float)
    cum = per.groupby(level='skater_id').cumsum()                      # tot en met het seizoen
    c = cum.reindex(pd.MultiIndex.from_arrays([J['skater_id'], J['seizoen']]))
    # seizoenen zonder zege: laatste stand ervoor
    allc = per.reset_index().sort_values('seizoen')
    for k in ('n_wk', 'n_os', 'n_wc'):
        allc['c_' + k] = allc.groupby('skater_id')[k].cumsum()
        m = pd.merge_asof(J[['skater_id', 'seizoen']].reset_index().sort_values('seizoen'), allc[['skater_id', 'seizoen', 'c_' + k]].sort_values('seizoen'),
                          on='seizoen', by='skater_id', direction='backward').set_index('index')
        J['h_' + k] = np.log1p(m['c_' + k].reindex(J.index).fillna(0.0))
    J['t0'] = (J['leeftijd'] - 26.0) / 5.0
    J['vrouw'] = (J['gender'] == 'v').astype(float); J['ned'] = (J['land'] == 'NED').astype(float)
    J['L'] = J['Ls']
    return J.reset_index(drop=True)


def ontwerp_sen(D):
    tau = (D['a'].to_numpy(float) - 28.0) / 6.0; t0 = D['t0'].to_numpy(float)
    Lr = D['Lr'].to_numpy(float); lLr = np.log(np.clip(Lr, -0.03, None) + 0.05)
    hw, hc, ho = D['h_n_wk'].to_numpy(float), D['h_n_wc'].to_numpy(float), D['h_n_os'].to_numpy(float)
    cols = [tau, tau ** 2, tau ** 3, D['vrouw'].to_numpy(float), t0, t0 ** 2, lLr, Lr, lLr * tau, lLr * t0,
            D['dL'].to_numpy(float), D['dL_onb'].to_numpy(float), hw, hc, ho, hw * tau, hc * tau, D['ned'].to_numpy(float)]
    return np.column_stack(cols)


tl.ontwerp = ontwerp_sen


def rijen(J, per, tot, doel, eerste, tt=None, eind=LAATSTE):
    if doel == 'n_os':
        a, b = tl.os_rijen(J, per, tot, False, tt, eind); D = pd.concat([a, b], ignore_index=True)
    else:
        D = tl.rijen(J, per, tot, doel, False, tt, eind)
    if eerste:                                   # tot en met de eerste zege NA de momentopname
        k = ['skater_id', 'seizoen'] + (['is_os'] if doel == 'n_os' else [])
        f = D[D['y'] > 0].groupby(k)['s'].min().rename('f')
        D = D.join(f, on=k); D = D[D['f'].isna() | (D['s'] <= D['f'])].drop(columns='f').copy(); D['y'] = (D['y'] > 0).astype(float)
    return D


def bereken(J, per, tot):
    eind = int((J['geboren'] + tl.MAX_LFT).max())
    tt = tl.toekomst_totalen(np.arange(LAATSTE + 1, eind + 1))
    tt['n_wc'] = float(np.mean([tot['n_wc'].get((g, s), np.nan) for g in ('m', 'v') for s in range(2022, LAATSTE + 1)]))
    U = J[['skater_id', 'seizoen']].copy(); toets = []
    for doel, kort in (('n_wk', 'wk'), ('n_os', 'os'), ('n_wc', 'wc')):
        som = {}
        for eerste in (False, True):
            Dfit = rijen(J, per, tot, doel, eerste)
            Dall = rijen(J, per, tot, doel, False, tt, eind)
            if doel == 'n_os':
                Dall = Dall[Dall['is_os'] == 1]
            Dall = Dall.copy(); Dall['mu'] = np.nan
            for par in (0, 1):
                m_ = (Dall['geboren'] % 2 == par).to_numpy()
                Dall.loc[m_, 'mu'] = tl.fit_voorspel(Dfit[Dfit['geboren'] % 2 != par], Dall[m_], extra=(doel == 'n_os'))
            nu_ = Dall['seizoen'].to_numpy() >= LAATSTE - 1
            Dall['mu_alles'] = Dall['mu']
            Dall.loc[nu_, 'mu_alles'] = tl.fit_voorspel(Dfit, Dall[nu_], extra=(doel == 'n_os'))
            k = ['skater_id', 'seizoen']
            obs = Dall[Dall['s'] <= LAATSTE]
            if eerste:   # y in het venster: minstens een zege na de momentopname
                yv = obs.groupby(k)['y'].sum().gt(0).astype(float)
            else:
                yv = obs.groupby(k)['y'].sum()
            som[eerste] = (Dall.groupby(k)['mu_alles'].sum().rename('carriere'), obs.groupby(k)['mu'].sum().rename('venster'), yv.rename('y'))
        V0 = pd.concat([som[False][1], som[False][2]], axis=1); V1 = pd.concat([som[True][1], som[True][2]], axis=1)
        ca, da = tl.ijk_aantal(V0['venster'].to_numpy(), V0['y'].to_numpy())
        ck, dk = tl.ijk_kans(V1['venster'].to_numpy(), V1['y'].to_numpy())
        V1 = V1.join(J.set_index(k)[['leeftijd', 'Lr']])
        V1['p'] = 1 - np.exp(-ck * np.clip(V1['venster'], 1e-12, None) ** dk)
        for nm, m in (('18-21', V1['leeftijd'] <= 21), ('22-26', V1['leeftijd'].between(22, 26)), ('27+', V1['leeftijd'] >= 27)):
            x = V1[m]
            if x['y'].sum() < 3:
                continue
            toets.append(f"{doel} leeftijd {nm}: {len(x)} momentopnamen, {int(x['y'].sum())} met een zege daarna | kans som {x['p'].sum():.1f}"
                         f" | AUC {tl.auc(x['y'].to_numpy(), x['p'].to_numpy()):.3f} (alleen niveau t.o.v. top-10 {tl.auc(x['y'].to_numpy(), -x['Lr'].to_numpy()):.3f})"
                         f" | top-100: voorspeld {x.nlargest(100, 'p')['p'].sum():.1f} echt {x.nlargest(100, 'p')['y'].sum():.0f}")
        print(f'{doel}: ijking aantal c {ca:.3f} d {da:.3f} | kans c {ck:.3f} d {dk:.3f}', flush=True)
        U = U.join((ca * np.clip(som[False][0], 1e-12, None) ** da).rename('e_' + kort), on=k)
        U = U.join((1 - np.exp(-ck * np.clip(som[True][0], 1e-12, None) ** dk)).rename('p_' + kort), on=k)
    print('\n'.join(toets))
    return U.fillna(0.0)


if __name__ == '__main__':
    w = sys.argv[1]
    per, tot, J0 = pickle.load(open(os.path.join(w, 'titels_invoer.pkl'), 'rb'))
    J = momenten_sen(w, per)
    print(f'{len(J)} momentopnamen 18-39 jaar', flush=True)
    U = bereken(J, per, tot)
    U.to_parquet(os.path.join(w, 'titels_alle_sen.parquet'))
    N = U.merge(J[['skater_id', 'seizoen', 'naam', 'gender', 'land', 'geboren', 'leeftijd']], on=['skater_id', 'seizoen'])
    N = N[N['seizoen'] >= LAATSTE - 1].sort_values('seizoen').groupby('skater_id').tail(1)
    N.to_csv(os.path.join(tl.HIER, f'titelkansen_senioren_{LAATSTE + 1}.csv'), index=False)
    for g, x in N.groupby('gender'):
        print(f'\n{g}:'); print(x.sort_values('e_wk', ascending=False).head(20)[['naam', 'land', 'leeftijd', 'p_wk', 'e_wk', 'p_os', 'e_os', 'p_wc', 'e_wc']].round(3).to_string(index=False))
