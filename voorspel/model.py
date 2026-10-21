"""Nieuwe voorspeller van rittijden: ln tijd = ijs vooraf + niveau + vorm + correcties (functies van PR, seizoensbeste, laatste fouten,
leeftijd x tijd sinds de vorige rit, niveau x baan, andere afstanden), met een eigen schaal per rit (eigen spreiding, ervaring, rustperiode,
leeftijd, onzekerheid van het ijs) en een geijkte foutverdeling (twee-delige Student-t per afstandsgroep, zoals de eerlijke log loss).
Fit op de jaren <= FIT_TOT, toets op de jaren erna. Gebruik: python model.py <map met kenmerken.parquet en oud_vooraf.parquet>"""
import sys, os, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as T

VELD_IJS = os.environ.get('VELD_IJS', '1') == '1'          # ijsbonus voor sterke velden (beter geprepareerd ijs)
FIT_TOT = int(os.environ.get('FIT_TOT', '2019'))      # fit op de jaren t/m FIT_TOT (2019: toets op 2020+; 2026: alles, voor voorspellingen nu)
AF = ['500m', '1000m', '1500m', '3000m', '5000m', '10000m']
GROEP = {'500m': 'sprint', '1000m': 'sprint', '1500m': 'midden', '3000m': 'midden', '5000m': 'lang', '10000m': 'lang'}
KOL = ['skater_id', 'skater_name', 'gender', 'groep', 'afstand', 'afstand_idx', 'dag', 'jaar', 'seizoen', 'event_id', 'baan', 'tijd', 'verdacht',
       'theta', 'v_theta', 'leeftijd', 'mu', 'n', 'gapa', 'gap', 'f', 'f2', 's', 'sa', 'pr', 'prraw', 'sb3', 'sbc', 'ela', 'elany', 'elage', 'e', 'val',
       'y', 'ijs_vm', 'ijs_vn', 'ijs_ba', 'ijs_bn', 'ijs_ev', 'ijs_evn', 'ijs_gl', 'ijs_rec', 'ijs_recn', 'hoog', 'theta_rijder'] + ['mu_' + a for a in AF] + ['n_' + a for a in AF]


def huber(X, y, w=None, c=1.345, n_iter=30):
    """Robuuste lineaire regressie (Huber, IRLS) met gewichten w = 1 / schaal^2 per rij; de Huber-grens geldt voor de gestandaardiseerde residuen."""
    w = np.ones(len(y)) if w is None else w
    b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    for _ in range(n_iter):
        r = (y - X @ b) * np.sqrt(w); s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9      # gestandaardiseerd (w = 1 / schaal^2)
        hw = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12)) * w
        b_new = np.linalg.lstsq(X * np.sqrt(hw)[:, None], y * np.sqrt(hw), rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-7:
            b = b_new; break
        b = b_new
    return b


def laad(w):
    K = pd.read_parquet(os.path.join(w, 'kenmerken.parquet'), columns=KOL)
    K = K[K['afstand'].astype(str).isin(AF)].copy(); K['afstand'] = K['afstand'].astype(str)
    return K


def ijs_voorspelling(K):
    """Racepeil vooraf: lopend baangemiddelde (afstand x geslacht, gekrompen naar de baan), eerdere toernooidagen, algemene stand,
    en (VELD_IJS) een ijsbonus als functie van het niveau van het veld t.o.v. het gewone veld op die baan."""
    fit = (K['jaar'] <= FIT_TOT) & np.isfinite(K['theta'])
    gm = K[fit].groupby(['gender', 'afstand_idx'])['theta'].median().rename('gm')
    K = K.join(gm, on=['gender', 'afstand_idx'])
    ba = K['ijs_ba'].where(K['ijs_bn'] > 0, K['gm'])
    K['ijs_basis'] = np.where(K['ijs_vn'] > 0, (K['ijs_vn'] * K['ijs_vm'] + 3 * ba) / (K['ijs_vn'] + 3), ba)
    K['ijs_evw'] = K['ijs_ev'] * K['ijs_evn'] / (K['ijs_evn'] + 2)
    K['ijs_recw'] = K['ijs_rec'] * (1 - np.exp(-K['ijs_recn'] / 3.0))
    IJS = ['ijs_basis', 'ijs_evw', 'ijs_gl', 'ijs_recw']
    if VELD_IJS and 'k_c' in K:
        # elite rijden vaak op beter geprepareerd ijs: verwachte ijsbonus = f(niveau van het veld van de race) - f(gewoon veldniveau op die baan),
        # f(g) = exp(-g / T) voor T = 5% en 15% (functies, geen groepen). Het veld is vooraf bekend (startlijst); de baan alleen uit eerdere dagen.
        rk = ['dag', 'event_id', 'gender', 'afstand_idx']
        gR = K.groupby(rk)['k_c'].transform('median').clip(-0.05, 0.5)
        ev = K.assign(gR=gR).drop_duplicates(rk)[rk + ['baan', 'gR']].sort_values(['baan', 'dag'], kind='stable')
        ev['gV'] = ev.groupby('baan')['gR'].transform(lambda x: x.shift(1).ewm(alpha=0.05, adjust=True).mean())
        K = K.join(ev.set_index(rk)['gV'], on=rk)
        gV = K['gV'].fillna(gR).fillna(0.2); gR = gR.fillna(gV)
        for T, nm in ((0.05, 'ijs_veld05'), (0.15, 'ijs_veld15')):
            K[nm] = np.exp(-gR / T) - np.exp(-gV / T)
        IJS += ['ijs_veld05', 'ijs_veld15']
    r = K[fit].drop_duplicates(['dag', 'event_id', 'gender', 'afstand_idx'])
    X = np.c_[np.ones(len(r)), r[IJS].to_numpy()]
    c = np.linalg.lstsq(X, r['theta'].to_numpy(), rcond=None)[0]
    K['th_hat'] = np.c_[np.ones(len(K)), K[IJS].to_numpy()] @ c
    return K, c


def kenmerken(K, D=None, ref=None):
    """Ontwerpmatrix voor de locatie (fout t.o.v. niveau + vorm) en voor de schaal. D: typisch verschil tussen de niveaus op twee
    afstanden (geslacht x afstand x afstand), ref: ln-tijd van de wereldtop (geslacht x afstand); beide uit de fitjaren."""
    if ref is None:
        x = K[(K['jaar'] <= FIT_TOT) & np.isfinite(K['y']) & ~K['val']]
        ref = x.groupby(['gender', 'afstand'])['y'].quantile(0.01)
    if D is None:
        x = K[(K['jaar'] <= FIT_TOT) & (K['n'] >= 5)]
        D = {}
        for g, xg in x.groupby('gender'):
            for a in AF:
                xa = xg[xg['afstand'] == a]
                for b in AF:
                    if b != a:
                        s = xa[xa['n_' + b] >= 5]
                        D[(g, a, b)] = float(np.median(s['mu'] - s['mu_' + b])) if len(s) > 200 else np.nan
    F = pd.DataFrame(index=K.index)
    g_ = K['mu'] - [ref.get((gg, a), np.nan) for gg, a in zip(K['gender'], K['afstand'])]
    g_ = g_.clip(0, 0.5); F['g'] = g_; F['g2'] = g_ ** 2
    F['f'] = K['f']; F['f2'] = K['f2']
    F['ela'] = K['ela'] * np.exp(-K['gapa'] / 60.0); F['elany'] = K['elany'] * np.exp(-K['elage'] / 30.0)
    prd = (K['pr'] - K['mu']).clip(-0.15, 0.0); F['pr'] = prd; F['pr_n'] = prd / np.sqrt(K['n'] + 1)
    raw = (K['prraw'] - K['th_hat'] - K['mu']).clip(-0.15, 0.15); F['prraw'] = raw          # bewezen PR t.o.v. de verwachte tijd hier
    has3 = K['sb3'] < 9; F['sb3'] = np.where(has3, (K['sb3'] - K['mu']).clip(-0.15, 0.1), 0.0); F['sb3_geen'] = (~has3).astype(float)
    hasc = K['sbc'] < 9; F['sbc'] = np.where(hasc, (K['sbc'] - K['mu']).clip(-0.15, 0.1), 0.0); F['sbc_geen'] = (~hasc).astype(float)
    # andere afstanden: gewogen gemiddelde van het niveau dat de andere afstanden aangeven
    num = np.zeros(len(K)); den = np.zeros(len(K)); ai = K['afstand'].map({a: i for i, a in enumerate(AF)}).to_numpy()
    for bi, b in enumerate(AF):
        d_ = np.array([D.get((gg, a, b), np.nan) for gg, a in zip(K['gender'], K['afstand'])])
        nb = K['n_' + b].to_numpy(); ok = (ai != bi) & (nb >= 3) & np.isfinite(d_)
        wgt = np.where(ok, nb / (nb + 5) / (1.0 + np.abs(ai - bi)), 0.0)
        num += wgt * np.nan_to_num(K['mu_' + b].to_numpy() + d_ - K['mu'].to_numpy()); den += wgt
    xd = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0); xd = np.clip(xd, -0.1, 0.1)
    F['xd'] = xd; F['xd_n'] = xd * 5.0 / (K['n'].to_numpy() + 5.0); F['xd_geen'] = (den == 0).astype(float)
    # leeftijd: groei van jonge rijders en achteruitgang van ouderen als exponentiele / lineaire functies, ook x tijd sinds de vorige rit
    lf = K['leeftijd'] + ((K['dag'] - (K['seizoen'] - 1970) * 365.25 - 182) / 365.25).clip(0, 1)     # leeftijd op de racedag (ongeveer)
    bek = np.isfinite(lf); lf = lf.fillna(25.0)
    dta = (K['gapa'].clip(0, 548) / 365.0)
    A = {'a1': np.exp(-(lf - 12) / 3.0), 'a2': np.exp(-(lf - 12) / 8.0), 'a3': (lf - 27).clip(0) / 10.0, 'a4': (lf - 45).clip(0) / 10.0}
    for k, v in A.items():
        F[k] = np.where(bek, v, 0.0); F[k + '_dt'] = np.where(bek, v * dta, 0.0)
    F['leeftijd_onb'] = (~bek).astype(float); F['dt'] = dta
    F['inv_n'] = 1.0 / (K['n'] + 1.0); F['logn'] = np.log1p(K['n'])
    F['seizoenstart'] = (K['gap'] > 150).astype(float); F['lgap'] = np.log1p(K['gap'].clip(0, 2000))
    # baan x niveau: hoogland en snel ijs helpen toppers anders dan het veld (theta is een veldgemiddelde)
    dth = (K['th_hat'] - K['gm']).clip(-0.06, 0.06)
    F['hoog_g'] = K['hoog'] * g_; F['dth_g'] = dth * g_; F['hoog'] = K['hoog']; F['dth'] = dth
    F['vrouw'] = (K['gender'] == 'v').astype(float)
    for a in AF[1:]:
        F['af_' + a] = (K['afstand'] == a).astype(float)
    return F, D, ref


SCHAAL = ['ls', 'lsa', 'inv_n', 'lgapa', 'seizoenstart', 'a1', 'a2', 'a3', 'leeftijd_onb', 'g', 'hoog', 'vrouw'] + ['af_' + a for a in AF[1:]]


def schaal_kenmerken(K, F):
    S = pd.DataFrame(index=K.index)
    S['ls'] = np.log(K['s']); S['lsa'] = np.log(K['sa']); S['inv_n'] = F['inv_n']; S['lgapa'] = np.log1p(K['gapa'].clip(0, 2000))
    for c in ('seizoenstart', 'a1', 'a2', 'a3', 'leeftijd_onb', 'g', 'hoog', 'vrouw') + tuple('af_' + a for a in AF[1:]):
        S[c] = F[c]
    return S


def twee_t(u):
    """Twee-delige Student-t (kans p0 aan de snelle kant, eigen schaal en vrijheidsgraden per kant): fit op u, geeft parameters."""
    def nll(p):
        p0 = 1 / (1 + np.exp(-p[0])); sL, nL, sU, nU = np.exp(p[1]), 1.5 + 28.5 / (1 + np.exp(-p[2])), np.exp(p[3]), 1.5 + 28.5 / (1 + np.exp(-p[4]))
        f = np.where(u < 0, p0 * 2 * T.pdf(u / sL, nL) / sL, (1 - p0) * 2 * T.pdf(u / sU, nU) / sU)
        return -np.mean(np.log(np.maximum(f, 1e-300)))
    return minimize(nll, np.zeros(5), method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-8}).x


def t_nll(p, u):
    p0 = 1 / (1 + np.exp(-p[0])); sL, nL, sU, nU = np.exp(p[1]), 1.5 + 28.5 / (1 + np.exp(-p[2])), np.exp(p[3]), 1.5 + 28.5 / (1 + np.exp(-p[4]))
    f = np.where(u < 0, p0 * 2 * T.pdf(u / sL, nL) / sL, (1 - p0) * 2 * T.pdf(u / sU, nU) / sU)
    return -np.log(np.maximum(f, 1e-300))


def t_cdf(p, u):
    p0 = 1 / (1 + np.exp(-p[0])); sL, nL, sU, nU = np.exp(p[1]), 1.5 + 28.5 / (1 + np.exp(-p[2])), np.exp(p[3]), 1.5 + 28.5 / (1 + np.exp(-p[4]))
    return np.where(u < 0, p0 * 2 * T.cdf(u / sL, nL), p0 + (1 - p0) * (2 * T.cdf(u / sU, nU) - 1))


def draai(w, verbose=True):
    t0 = time.time()
    K = laad(w); K, c_ijs = ijs_voorspelling(K)
    F, D, ref = kenmerken(K)
    ok = (K['n'] >= 1) & np.isfinite(K['y']) & ~K['val'] & ~K['verdacht'] & np.isfinite(K['th_hat'])
    fit = ok & (K['jaar'] <= FIT_TOT) & (K['jaar'] >= 2008)
    cols = list(F.columns)
    X = np.c_[np.ones(len(F)), F[cols].to_numpy(float)]
    e = K['e'].to_numpy()                               # ln tijd - theta - niveau - vorm (fout met het echte racepeil)
    fin = np.isfinite(X).all(1)
    if verbose:
        bad = {c: int((~np.isfinite(F[c].to_numpy(float)) & ok.to_numpy()).sum()) for c in cols}
        print('niet-eindige kenmerken (bruikbare ritten):', {k: v for k, v in bad.items() if v}, flush=True)
    ok = ok & fin; fit = fit & fin & np.isfinite(e)
    b = huber(X[fit.to_numpy()], e[fit.to_numpy()])
    loc = X @ b
    coef = pd.Series(b, index=['const'] + cols)
    # voorspelling vooraf: ijs vooraf + niveau + vorm + correctie
    K['pred'] = K['th_hat'] + K['mu'] + K['f'] + loc
    K['pred_basis'] = K['th_hat'] + K['mu'] + K['f']        # zonder de correctiefuncties
    K['res_r'] = e - loc                                    # rijdersdeel (met het echte ijs)
    K['res_i'] = K['theta'] - K['th_hat']                   # ijsdeel
    # schaal: rijdersdeel en ijsdeel apart, beide als log-lineaire functie
    S = schaal_kenmerken(K, F); Sx = np.c_[np.ones(len(S)), S.to_numpy(float)]
    fr = fit.to_numpy()
    lr = np.log(np.abs(K['res_r'].to_numpy()) + 1e-5)
    gs = np.linalg.lstsq(Sx[fr], lr[fr], rcond=None)[0]
    sc_r = np.exp(Sx @ gs); k_r = np.median(np.abs(K['res_r'].to_numpy()[fr]) / sc_r[fr]) / 0.6745; sc_r *= k_r
    ri = K[fit].drop_duplicates(['dag', 'event_id', 'gender', 'afstand_idx'])
    Ix = lambda d: np.c_[np.ones(len(d)), np.log1p(d['ijs_vn']), np.log1p(d['ijs_evn']), d['hoog'], np.log(d['v_theta'].clip(1e-7))]
    gi = np.linalg.lstsq(Ix(ri), np.log(np.abs(ri['res_i'].to_numpy()) + 1e-5), rcond=None)[0]
    sc_i = np.exp(Ix(K) @ gi); k_i = np.median(np.abs(ri['res_i'].to_numpy()) / np.exp(Ix(ri) @ gi)) / 0.6745; sc_i *= k_i
    K['sd'] = np.sqrt(sc_r ** 2 + sc_i ** 2)
    K['err'] = np.log(K['tijd']) - K['pred']; K['u'] = K['err'] / K['sd']
    # foutverdeling per afstandsgroep op de fitjaren, NLL op de toetsjaren
    K['ag'] = K['afstand'].map(GROEP); K['nll'] = np.nan; K['pit'] = np.nan; par = {}
    for gname in ('sprint', 'midden', 'lang'):
        sf = fit & (K['ag'] == gname); u_f = K.loc[sf, 'u'].to_numpy()
        rng = np.random.default_rng(0); u_f = rng.choice(u_f, min(len(u_f), 200000), replace=False)
        p = twee_t(u_f); par[gname] = p
        sa = ok & (K['ag'] == gname)
        K.loc[sa, 'nll'] = t_nll(p, K.loc[sa, 'u'].to_numpy()) + np.log(K.loc[sa, 'sd'].to_numpy())
        K.loc[sa, 'pit'] = t_cdf(p, K.loc[sa, 'u'].to_numpy())
    K['ok'] = ok
    if verbose:
        print('ijs-coef', np.round(c_ijs, 3), '| tijd', round(time.time() - t0), 's', flush=True)
        print(coef.round(4).to_string())
    return K, coef, gs, gi, par


def vergelijk(K, w):
    """Zelfde ritten als het oude model (2020+, geen valpartijen volgens beide, met log loss): MAE en log loss."""
    O = pd.read_parquet(os.path.join(w, 'oud_vooraf.parquet'))
    O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd'])
    O['afstand'] = O['afstand'].astype(str)
    M = K[K['ok'] & (K['jaar'] >= FIT_TOT + 1)].merge(O[['skater_id', 'dag', 'afstand', 'tijd', 'err', 'var', 'll', 'niveau_pct']],
                                                    on=['skater_id', 'dag', 'afstand', 'tijd'], suffixes=('', '_oud'))
    regels = []
    for nm, s in (('alle', M['niveau_pct'] > -1), ('toppers >=90', M['niveau_pct'] >= 90), ('elite >=97', M['niveau_pct'] >= 97),
                  ('jong (t/m neo)', M['groep'].isin(['jeugd', 'junior_C', 'junior_B', 'junior_A', 'neo'])), ('senior', M['groep'] == 'senior')):
        x = M[s]
        regels.append(f"{nm:15s} n={len(x):7d} | mediaan |fout| %: oud {100 * np.median(np.abs(x['err_oud'])):.3f}  nieuw {100 * np.median(np.abs(x['err'])):.3f}"
                      f"  basis {100 * np.median(np.abs(np.log(x['tijd']) - x['pred_basis'])):.3f} | gem |fout| %: oud {100 * np.mean(np.abs(x['err_oud'])):.3f} nieuw {100 * np.mean(np.abs(x['err'])):.3f}"
                      f" | log loss: oud {x['ll'].mean():.4f} nieuw {x['nll'].mean():.4f} (verschil {x['nll'].mean() - x['ll'].mean():+.4f})")
    return M, regels


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else '.'
    K, coef, gs, gi, par = draai(w)
    M, regels = vergelijk(K, w)
    print('\n'.join(regels))
    K[['skater_id', 'dag', 'afstand', 'tijd', 'pred', 'pred_basis', 'sd', 'u', 'pit', 'nll', 'ok', 'th_hat']].to_parquet(os.path.join(w, 'voorspel_uitkomst.parquet'))
