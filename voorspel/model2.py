"""Nieuwe voorspeller, versie 2: basis = ijs vooraf (th_hat) + wereldtop (ref) + Kalman-niveau (gemeenschappelijk + per afstand, met
groei per leeftijd) + vorm; daarbovenop correctiefuncties (bewezen PR en seizoensbeste t.o.v. het niveau, laatste fouten,
leeftijd x tijd sinds de vorige rit, niveau x baan, onzekerheid van het niveau) en een schaal per rit (Kalman-onzekerheid, eigen
spreiding, ijs). Foutverdeling: twee-delige Student-t per afstandsgroep op de fitjaren (zoals de eerlijke log loss van het oude model).
Gebruik: python model2.py <map> [kalman_par.json]"""
import sys, os, json, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import importlib
kf = importlib.import_module(os.environ.get('KALMAN', 'kalman_mv2'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite'))
from model import huber, twee_t, t_nll, t_cdf, ijs_voorspelling, vergelijk, AF, GROEP, FIT_TOT, KOL

JONG = ['jeugd', 'junior_C', 'junior_B', 'junior_A', 'neo']
NIVEAU_INTERACTIE = True


def met_kalman(w, par):
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'))
    R, inv = kf.voorbereid(R); o = kf.pass_(inv, par)
    K = pd.read_parquet(os.path.join(w, 'kenmerken.parquet'), columns=KOL)
    assert len(K) == len(R) and (K['skater_id'].to_numpy() == R['skater_id'].to_numpy()).all() and (K['tijd'].to_numpy() == R['tijd'].to_numpy()).all()
    for k in ('zh', 'vl', 'r', 'e', 'na', 'c', 'd', 'f', 'pc'):
        K['k_' + k] = o[k]
    K['k_val'] = o['val']; K['ref'] = inv['refq']; K['lf'] = inv['lf']
    K = K[K['afstand'].astype(str).isin(AF)].copy(); K['afstand'] = K['afstand'].astype(str)
    return K


def ontwerp(K):
    F = pd.DataFrame(index=K.index)
    lvl = K['ref'] + K['k_c'] + K['k_d']                     # Kalman-niveau in ln tijd - theta
    g_ = (K['k_c'] + K['k_d']).clip(0, 0.5); F['g'] = g_; F['g2'] = g_ ** 2
    F['f'] = K['k_f']
    F['ela'] = (K['ela'] * np.exp(-K['gapa'] / 60.0)).fillna(0); F['elany'] = (K['elany'] * np.exp(-K['elage'] / 30.0)).fillna(0)
    hasp = K['pr'] < 9; prd = np.where(hasp, (K['pr'] - lvl).clip(-0.15, 0.05), 0.0)
    F['pr'] = prd; F['pr_n'] = prd / np.sqrt(K['k_na'] + 1); F['pr_geen'] = (~hasp).astype(float)
    F['prraw'] = np.where(K['prraw'] < 90, (K['prraw'] - K['th_hat'] - lvl).clip(-0.15, 0.15), 0.0)     # bewezen PR t.o.v. de verwachte tijd hier
    has3 = K['sb3'] < 9; F['sb3'] = np.where(has3, (K['sb3'] - lvl).clip(-0.15, 0.1), 0.0); F['sb3_geen'] = (~has3).astype(float)
    hasc = K['sbc'] < 9; F['sbc'] = np.where(hasc, (K['sbc'] - lvl).clip(-0.15, 0.1), 0.0); F['sbc_geen'] = (~hasc).astype(float)
    lf = K['lf']; bek = np.isfinite(lf); lf = lf.fillna(25.0)
    dta = (K['gapa'].clip(0, 548) / 365.0).where(K['k_na'] > 0, 0.0)
    for k, v in {'a1': np.exp(-(lf - 12) / 3.0), 'a2': np.exp(-(lf - 12) / 8.0), 'a3': (lf - 27).clip(0) / 10.0, 'a4': (lf - 45).clip(0) / 10.0}.items():
        F[k] = np.where(bek, v, 0.0); F[k + '_dt'] = np.where(bek, v * dta, 0.0)
    F['leeftijd_onb'] = (~bek).astype(float); F['dt'] = dta
    F['inv_n'] = 1.0 / (K['k_na'] + 1.0); F['logn'] = np.log1p(K['k_na']); F['nieuw_af'] = (K['k_na'] == 0).astype(float)
    F['onz'] = np.log(K['k_vl'] / K['k_r'])                   # onzekerheid van het niveau t.o.v. de ritruis
    F['seizoenstart'] = (K['gap'] > 150).astype(float); F['lgap'] = np.log1p(K['gap'].clip(0, 2000).fillna(2000))
    dth = (K['th_hat'] - K['gm']).clip(-0.06, 0.06)
    F['hoog_g'] = K['hoog'] * g_; F['dth_g'] = dth * g_; F['hoog'] = K['hoog']; F['dth'] = dth
    F['vrouw'] = (K['gender'] == 'v').astype(float)
    for a in AF[1:]:
        F['af_' + a] = (K['afstand'] == a).astype(float)
    wg = np.exp(-g_ / 0.04)                                   # 1 bij de wereldtop, ~0 ver erachter: correcties als functie van het niveau
    if NIVEAU_INTERACTIE:
        for c in list(F.columns):
            F[c + '_w'] = F[c] * wg
        F['w'] = wg
    S = pd.DataFrame(index=K.index)
    S['w'] = wg
    S['lv'] = np.log(K['k_vl'] + K['k_r']); S['lvr'] = F['onz']; S['lv_w'] = S['lv'] * wg; S['inv_n'] = F['inv_n']; S['lgapa'] = np.log1p(K['gapa'].clip(0, 2000).fillna(2000))
    for c in ('seizoenstart', 'a1', 'a2', 'a3', 'leeftijd_onb', 'g', 'hoog', 'vrouw', 'nieuw_af') + tuple('af_' + a for a in AF[1:]):
        S[c] = F[c]
    return F, S


def draai(w, par, verbose=True, segment=True):
    t0 = time.time()
    K = met_kalman(w, par); K, c_ijs = ijs_voorspelling(K)
    F, S = ontwerp(K)
    zt = np.log(K['tijd']) - K['theta'] - K['ref']                  # z met het echte racepeil
    e = (zt - K['k_zh']).to_numpy()
    X = np.c_[np.ones(len(F)), F.to_numpy(float)]; Sx = np.c_[np.ones(len(S)), S.to_numpy(float)]
    ok = (np.isfinite(K["k_zh"]) & np.isfinite(K["y"]) & ~K["k_val"] & ~K["verdacht"] & np.isfinite(K["th_hat"])).to_numpy().copy()
    ok &= np.isfinite(X).all(1) & np.isfinite(Sx).all(1)
    jr = K['jaar'].to_numpy(); fit = ok & (jr <= FIT_TOT) & (jr >= 2008) & np.isfinite(e)
    # aparte correctiefuncties voor jonge rijders (t/m neo) en de rest: groei en ervaring werken anders
    seg = K['groep'].isin(JONG).to_numpy() if segment else np.zeros(len(K), bool)
    sc_r = np.sqrt(K['k_vl'] + K['k_r']).to_numpy()            # eerste schaal: de Kalman-onzekerheid + ritruis
    for ronde in range(3):                                      # locatie (gewogen met 1 / schaal^2) en schaal om en om
        loc = np.zeros(len(K)); coefs = {}
        for sg in (True, False):
            m_ = fit & (seg == sg)
            if m_.sum() < 1000: continue
            b = huber(X[m_], e[m_], w=1.0 / sc_r[m_] ** 2); coefs['jong' if sg else 'ouder'] = pd.Series(b, index=['const'] + list(F.columns))
            loc[seg == sg] = X[seg == sg] @ b
        rr = e - loc; lr = np.log(np.abs(rr) + 1e-5)
        ff = fit & np.isfinite(lr)
        gs = np.linalg.lstsq(Sx[ff], lr[ff], rcond=None)[0]; sc_r = np.exp(Sx @ gs)
        sc_r *= np.median(np.abs(rr[ff]) / sc_r[ff]) / 0.6745
    K['pred'] = K['th_hat'] + K['ref'] + K['k_zh'] + loc
    K['pred_basis'] = K['th_hat'] + K['ref'] + K['k_zh']
    rc = K[fit].drop_duplicates(['dag', 'event_id', 'gender', 'afstand_idx'])
    Ix = lambda d: np.c_[np.ones(len(d)), np.log1p(d['ijs_vn']), np.log1p(d['ijs_evn']), d['hoog'], np.log(d['v_theta'].clip(1e-7))]
    rci = (rc['theta'] - rc['th_hat']).to_numpy()
    gi = np.linalg.lstsq(Ix(rc), np.log(np.abs(rci) + 1e-5), rcond=None)[0]
    sc_i = np.exp(Ix(K) @ gi); sc_i *= np.median(np.abs(rci) / np.exp(Ix(rc) @ gi)) / 0.6745
    K['sd'] = np.sqrt(sc_r ** 2 + sc_i ** 2)
    K['err'] = np.log(K['tijd']) - K['pred']; K['u'] = K['err'] / K['sd']
    K['ag'] = K['afstand'].map(GROEP); K['nll'] = np.nan; K['pit'] = np.nan; par_t = {}
    for gname in ('sprint', 'midden', 'lang'):
        sf = fit & (K['ag'] == gname).to_numpy(); u_f = K['u'].to_numpy()[sf]
        u_f = np.random.default_rng(0).choice(u_f, min(len(u_f), 200000), replace=False)
        p = twee_t(u_f); par_t[gname] = p
        sa = ok & (K['ag'] == gname).to_numpy()
        K.loc[sa, 'nll'] = t_nll(p, K.loc[sa, 'u'].to_numpy()) + np.log(K.loc[sa, 'sd'].to_numpy())
        K.loc[sa, 'pit'] = t_cdf(p, K.loc[sa, 'u'].to_numpy())
    K['ok'] = ok
    if verbose:
        print('ijs-coef', np.round(c_ijs, 3), '| tijd', round(time.time() - t0), 's', flush=True)
        print(pd.DataFrame(coefs).round(4).to_string())
    return K, coefs, gs, gi, par_t


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else '.'
    par = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    K, coefs, gs, gi, par_t = draai(w, par)
    K['e'] = K['err']                                   # vergelijk() gebruikt err; e niet nodig
    M, regels = vergelijk(K, w)
    print('\n'.join(regels))
    import elite_model as em                            # kalibratietoets: kans per rit = PIT van de voorspelling vooraf
    M['p'] = M['pit']
    for nm, x in (('toppers >=90', M[M['niveau_pct'] >= 90]), ('elite >=97', M[M['niveau_pct'] >= 97])):
        fout, rg, nr_, nk = em.toets(x, vanaf_seizoen=2019)
        print(f'kalibratie {nm}: {nr_} ritten van {nk} rijders, {fout} fouten', *[r for r in rg if 'FOUT' in r or 'kans<=' in r], sep='\n  ')
    K[['skater_id', 'skater_name', 'gender', 'groep', 'dag', 'jaar', 'seizoen', 'baan', 'afstand', 'tijd', 'pred', 'pred_basis', 'sd', 'u', 'pit', 'nll', 'ok',
       'th_hat', 'theta']].to_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'))
