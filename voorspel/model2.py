"""Nieuwe voorspeller, versie 2: basis = ijs vooraf (th_hat) + wereldtop (ref) + Kalman-niveau (gemeenschappelijk + per afstand, met
groei per leeftijd) + vorm; daarbovenop correctiefuncties (bewezen PR en seizoensbeste t.o.v. het niveau, laatste fouten,
leeftijd x tijd sinds de vorige rit, niveau x baan, onzekerheid van het niveau) en een schaal per rit (Kalman-onzekerheid, eigen
spreiding, ijs). Foutverdeling: twee-delige Student-t per afstandsgroep op de fitjaren (zoals de eerlijke log loss van het oude model).
Gebruik: python model2.py <map> [kalman_par.json]"""
import sys, os, json, time, math, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from numba import njit
import importlib
kf = importlib.import_module(os.environ.get('KALMAN', 'kalman_mv2'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite'))
from model import huber, twee_t, t_nll, t_cdf, ijs_voorspelling, vergelijk, AF, GROEP, FIT_TOT, KOL, WAT_ALS_EVENT
from scipy.optimize import minimize
from scipy.stats import t as T
HUBER_C = float(os.environ.get('HUBER_C', '0.7'))
UITSCHIETERS = os.environ.get('UITSCHIETERS', '1') == '1'
PBASE = 0.5 * math.erfc(2.0 / math.sqrt(2.0)) * 1.5
import baanfactor as bf
BIAS_HL = float(os.environ.get('BIAS_HL', '45'))
BAAN_TOP_HL = float(os.environ.get('BAAN_TOP_HL', '365'))   # 0 = uit
BAAN_TOP_K = 30.0
SZ_TOP = os.environ.get('SZ_TOP', '1') == '1'
BAAN_FUNCTIE = os.environ.get('BAAN_FUNCTIE', '1') == '1'
HOOG_SCHAAL = os.environ.get('HOOG_SCHAAL', '0') == '1'
SCHOK = os.environ.get('SCHOK', '0') == '1'      # schokgeheugen uit het filter in correcties/schaal/vorm (niet geldig getoetst; vervangen door de vormlaag)
VORM = os.environ.get('VORM', '1') == '1'          # vormlaag op de kansschaal (vormlaag.py)
VORM_TOP = os.environ.get('VORM_TOP', '1') == '1'
SNEL_A = float(os.environ.get('SNEL_A', '0'))     # 0,2 haalt de baanbias weg (SLC -0,11 -> -0,04) maar maakt de log loss slechter (elite -3,149 -> -3,135)
SNEL_K = 20.0
Q_STAART = 0.02                                       # snelle staart voorbij deze kans krijgt een eigen (dikkere) vorm      # halfwaardetijd (dagen) van de online biascorrectie; 0 = uit


def seizoen_top(K, r, ok, w, bw=2.0, hl=2.0, k0=60.0):
    """Per rit: gewogen (w) gemiddelde fout op deze baan in deze week van het seizoen in eerdere seizoenen (kernel bw weken, halfwaardetijd
    hl seizoenen, krimp k0)."""
    r = np.clip(np.nan_to_num(r), -0.03, 0.03); ww = np.where(ok & np.isfinite(r), w, 0.0)
    wk = ((K['dag'] - ((K['seizoen'] - 1970) * 365.25 + 181)) // 7).clip(0, 51).astype(int).to_numpy()
    seiz = np.sort(K['seizoen'].unique()); W = np.arange(52); Kw = np.exp(-(W[:, None] - W[None, :]) ** 2 / (2 * bw * bw))
    out = np.zeros(len(K))
    for b, idx in K.groupby('baan').indices.items():
        if len(idx) < 200: continue
        si = np.searchsorted(seiz, K['seizoen'].to_numpy()[idx])
        S_ = np.zeros((len(seiz), 52)); C_ = np.zeros((len(seiz), 52))
        np.add.at(S_, (si, wk[idx]), ww[idx] * r[idx]); np.add.at(C_, (si, wk[idx]), ww[idx])
        for j in np.unique(si):
            if j == 0: continue
            a = 0.5 ** ((seiz[j] - 1 - seiz[:j]) / hl)
            sw = (a[:, None] * S_[:j]).sum(0) @ Kw.T; cw = (a[:, None] * C_[:j]).sum(0) @ Kw.T
            m = idx[si == j]; out[m] = (sw / (cw + k0))[wk[m]]
    return out


def baan_functie(K, r, ok, hl=730.0, k0=50.0):
    """Per baan: corr(g) = c . [1, exp(-g/0,04), exp(-g/0,15), g], c uit een uitdovende kleinste-kwadratenschatting op de fouten van
    eerdere dagen op die baan (alle rijders, halfwaardetijd hl dagen, ridge k0)."""
    r = np.clip(np.nan_to_num(r), -0.03, 0.03); g = K['k_c'].clip(0, 0.5).fillna(0.3).to_numpy()
    PHI = np.c_[np.ones(len(K)), np.exp(-g / 0.04), np.exp(-g / 0.15), g]; Pn = PHI.shape[1]
    out = np.zeros(len(K)); dagall = K['dag'].to_numpy()
    for b, idx in K.groupby('baan').indices.items():
        o = np.argsort(dagall[idx], kind='stable'); idx = idx[o]; dag = dagall[idx]
        X = PHI[idx]; y = r[idx]; m = ok[idx] & np.isfinite(r[idx])
        ud, start = np.unique(dag, return_index=True); eind = np.r_[start[1:], len(dag)]
        A = np.zeros((Pn, Pn)); bv = np.zeros(Pn); prev = None
        for d, s_, e_ in zip(ud, start, eind):
            if prev is not None:
                f = 0.5 ** ((d - prev) / hl); A *= f; bv *= f
            c = np.linalg.solve(A + k0 * np.eye(Pn), bv)
            out[idx[s_:e_]] = X[s_:e_] @ c
            mm = m[s_:e_]; Xs = X[s_:e_][mm]; A += Xs.T @ Xs; bv += Xs.T @ y[s_:e_][mm]; prev = d
    return out


@njit(cache=True)
def _schok(rid, dag, u, nr, T, o_neg, o_pos, o_v, o_n):
    # per rijder lopende sommen over EERDERE dagen, uitdovend met tijdconstante T dagen
    sn = np.zeros(nr); sp = np.zeros(nr); sv = np.zeros(nr); sc = np.zeros(nr); td = np.full(nr, -1e9)
    pn = np.zeros(nr); pp = np.zeros(nr); pv = np.zeros(nr); pc = np.zeros(nr); pd_ = np.full(nr, -1e9)    # bijdragen van de huidige dag
    for i in range(len(rid)):
        r_ = rid[i]; d = dag[i]
        if d != pd_[r_]:                          # nieuwe dag voor deze rijder: bijdragen van de vorige dag toevoegen
            if pd_[r_] > -1e8:
                f = np.exp(-(pd_[r_] - td[r_]) / T) if td[r_] > -1e8 else 0.0
                sn[r_] = sn[r_] * f + pn[r_]; sp[r_] = sp[r_] * f + pp[r_]; sv[r_] = sv[r_] * f + pv[r_]; sc[r_] = sc[r_] * f + pc[r_]
                td[r_] = pd_[r_]
            pn[r_] = 0.0; pp[r_] = 0.0; pv[r_] = 0.0; pc[r_] = 0.0; pd_[r_] = d
        f = np.exp(-(d - td[r_]) / T) if td[r_] > -1e8 else 0.0
        o_neg[i] = sn[r_] * f; o_pos[i] = sp[r_] * f; o_v[i] = sv[r_] * f; o_n[i] = sc[r_] * f
        if u[i] == u[i]:
            x = u[i]
            pn[r_] += min(x + 2.0, 0.0); pp[r_] += max(x - 2.0, 0.0)      # alleen het deel voorbij 2 sd: schokken
            pv[r_] += min(x * x, 25.0) - 1.0; pc[r_] += 1.0
    return 0


def schok_geheugen(rid, dag, u, Ts=(7.0, 30.0, 90.0)):
    """Per rit (rijen in tijdsvolgorde): uitdovende sommen over eerdere dagen van de rijder van de snelle schokken (deel van u voorbij -2),
    trage schokken (voorbij +2) en de extra variantie (u^2 - 1), en het uitdovende aantal ritten, voor tijdconstanten Ts."""
    out = {}; nr = int(rid.max()) + 1
    for T in Ts:
        a = {k: np.zeros(len(rid)) for k in ('neg', 'pos', 'v', 'n')}
        _schok(rid.astype(np.int64), dag.astype(np.float64), u.astype(np.float64), nr, float(T), a['neg'], a['pos'], a['v'], a['n'])
        t = int(T)
        out[f'neg{t}'] = a['neg']; out[f'pos{t}'] = a['pos']
        out[f'v{t}'] = a['v'] / (a['n'] + 2.0); out[f'n{t}'] = np.log1p(a['n'])
    return out


def per_wedstrijddag(dag, r, ok, w, a, k0):
    """Per rit: EW-gemiddelde (gewicht a per wedstrijddag op deze baan) van de gewogen fouten van EERDERE dagen, gekrompen met k0."""
    r = np.clip(np.nan_to_num(r), -0.03, 0.03); ww = np.where(ok & np.isfinite(r), w, 0.0)
    d = pd.DataFrame({'dag': dag, 'sw': ww * r, 'w': ww}).groupby('dag')[['sw', 'w']].sum()
    dagen = d.index.to_numpy(); S = d.to_numpy(); acc = np.zeros(2); st = np.zeros((len(dagen), 2))
    for i in range(len(dagen)):
        if i > 0:
            acc *= (1 - a)
        st[i] = acc; acc = acc + S[i]
    pos = np.clip(np.searchsorted(dagen, dag), 0, len(dagen) - 1); x = st[pos]
    laatste = dag > dagen[-1]; x[laatste] = acc
    return x[:, 0] / (x[:, 1] + k0)


def online_baan(dag, r, ok, w, hl):
    """Per dag het uitdovende gewogen gemiddelde (gewicht w) van de geknipte fouten van EERDERE dagen, en het effectieve gewicht."""
    r = np.clip(np.nan_to_num(r), -0.03, 0.03); m = ok & np.isfinite(r)
    d = pd.DataFrame({'dag': dag[m], 'rw': r[m] * w[m], 'w': w[m]}).groupby('dag').agg(sw=('rw', 'sum'), nw=('w', 'sum'))
    dagen = d.index.to_numpy(); S = d.to_numpy(); acc = np.zeros(2); vorige = None; uit = np.zeros((len(dagen), 2))
    if len(dagen) == 0:
        return np.zeros(len(dag)), np.zeros(len(dag))
    for i, dg in enumerate(dagen):
        if vorige is not None:
            acc *= 0.5 ** ((dg - vorige) / hl)
        uit[i] = [acc[0] / acc[1] if acc[1] > 0 else 0.0, acc[1]]
        acc += S[i]; vorige = dg
    pos = np.clip(np.searchsorted(dagen, dag), 0, len(dagen) - 1)
    laatste = dag > dagen[-1]                              # na de laatste dag met data: stand inclusief die dag
    b = uit[pos]; b[laatste] = [acc[0] / acc[1] if acc[1] > 0 else 0.0, acc[1]]
    return b[:, 0], b[:, 1]


def online_bias(dag, r, ok, w, hl):
    """Per dag het uitdovende gemiddelde van de geknipte fouten r (|r| <= 3%) van alle EERDERE dagen: algemeen en gewogen met w."""
    r = np.clip(np.nan_to_num(r), -0.03, 0.03); m = ok & np.isfinite(r)
    d = pd.DataFrame({'dag': dag[m], 'r': r[m], 'w': w[m], 'rw': r[m] * w[m]}).groupby('dag').agg(s=('r', 'sum'), n=('r', 'size'), sw=('rw', 'sum'), nw=('w', 'sum'))
    dagen = d.index.to_numpy(); S = d.to_numpy(); acc = np.zeros(4); vorige = None; uit = np.zeros((len(dagen), 2))
    for i, dg in enumerate(dagen):
        if vorige is not None:
            acc *= 0.5 ** ((dg - vorige) / hl)
        uit[i] = [acc[0] / acc[1] if acc[1] > 50 else 0.0, acc[2] / acc[3] if acc[3] > 20 else 0.0]
        acc += S[i]; vorige = dg
    pos = np.searchsorted(dagen, dag)                    # dagen zonder eigen ritten: stand van de eerstvolgende dag met ritten (alleen eerdere info)
    pos = np.clip(pos, 0, len(dagen) - 1)
    b = uit[pos]
    return b[:, 0], b[:, 1]


def _tpar(p, Z):
    """Parameters van de twee-delige t als lineaire functie van de kenmerken Z (kolommen 1, exp(-niveau / 4%), niveau, jong):
    elk van de vijf (kans snel, schaal en vrijheidsgraden per kant) is a + b w + c niveau + d jong."""
    Z = np.atleast_2d(Z); k = Z.shape[1]
    q = [Z @ np.asarray(p[k * i:k * (i + 1)]) for i in range(5)]
    return 1 / (1 + np.exp(-q[0])), np.exp(q[1]), 1.5 + 28.5 / (1 + np.exp(-q[2])), np.exp(q[3]), 1.5 + 28.5 / (1 + np.exp(-q[4]))


def _t_logpdf(x, nu):
    from scipy.special import gammaln
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(x * x / nu)


def tw_nll(p, u, w):
    p0, sL, nL, sU, nU = _tpar(p, w)
    lo = u < 0
    lf = np.where(lo, np.log(2 * p0) + _t_logpdf(u / sL, nL) - np.log(sL), np.log(2 * (1 - p0)) + _t_logpdf(u / sU, nU) - np.log(sU))
    return -lf


def tw_cdf(p, u, w):
    p0, sL, nL, sU, nU = _tpar(p, w)
    return np.where(u < 0, p0 * 2 * T.cdf(u / sL, nL), p0 + (1 - p0) * (2 * T.cdf(u / sU, nU) - 1))


def tw_ppf(p, q, w):
    """Inverse van tw_cdf (exact): de u met tw_cdf(p, u, w) = q."""
    p0, sL, nL, sU, nU = _tpar(p, w)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return np.where(q < p0, sL * T.ppf(np.clip(q / (2 * p0), 1e-15, 0.5), nL), sU * T.ppf(np.clip((q - p0) / (2 * (1 - p0)) + 0.5, 0.5, 1 - 1e-15), nU))


def kwantiel_u(K, par_t, Zq, q):
    """Kwantiel q (per rit) van de gestandaardiseerde fout u, inclusief de vormlaag en de staartcorrectie (in omgekeerde volgorde)."""
    q = np.broadcast_to(np.asarray(q, float), (len(K),))
    from scipy.stats import norm as _n
    if 'vorm_mu' in K:                                   # vormlaag: z' = (z - mu) / sigma, dus z = mu + sigma z'
        q = _n.cdf(K['vorm_mu'].to_numpy() + K['vorm_sg'].to_numpy() * _n.ppf(q))
    b_r = K['staart_br'].to_numpy() if 'staart_br' in K else np.ones(len(K))
    q = np.where(q < Q_STAART, Q_STAART * (q / Q_STAART) ** (1.0 / b_r), q)     # staartcorrectie terugdraaien
    uq = np.full(len(K), np.nan); ag = K['afstand'].map(GROEP).to_numpy()
    for g, p in par_t.items():
        s_ = ag == g
        if s_.any():
            uq[s_] = tw_ppf(p, q[s_], Zq[s_])
    return uq


def vorm_toepassen(K, fit, ok, verbose=True):
    """Vormlaag (vormlaag.py) op de kansschaal: geschat op de fitjaren (zonder eigen valpartijen), toegepast op alle ritten."""
    import vormlaag as vl
    from scipy.stats import norm as _n
    pit = K['pit'].to_numpy()
    z = np.where(ok & np.isfinite(pit), _n.ppf(np.clip(pit, 1e-9, 1 - 1e-9)), np.nan).clip(-6, 6)
    F = vl.kenmerken(K['skater_id'].to_numpy(), K['dag'].to_numpy(), z, K['k_val'].to_numpy(), K['rk_'].to_numpy())
    wg = np.exp(-K['k_c'].clip(0, 0.5).fillna(0.2).to_numpy() / 0.04)
    nep = K['event_id'].to_numpy() <= WAT_ALS_EVENT        # wat-als-ritten (wat_als_heerenveen.py): zonder veldaanpassing
    for c in [c for c in F.columns if c.startswith('veld_')]:
        F.loc[nep, c] = 0.0
    Xm, Xs, nm_m, nm_s = vl.ontwerp(F, wg)
    fm = fit & np.isfinite(z) & (K['jaar'].to_numpy() >= 2012)
    p = vl.fit(z[fm], Xm[fm], Xs[fm], (1 + 2 * (wg > 0.3))[fm] if VORM_TOP else None)
    mu, sg, pit2, d = vl.pas_toe(p, np.nan_to_num(z), Xm, Xs)
    K['pit_basis'] = pit; K['vorm_mu'] = mu; K['vorm_sg'] = sg
    K['pit'] = np.where(np.isfinite(z), pit2, pit); K['nll'] = np.where(np.isfinite(z), K['nll'] + d, K['nll'])
    K.attrs['vorm_par'] = [float(v) for v in p]
    if verbose:
        km = Xm.shape[1]
        print('vormlaag locatie', ' '.join(f'{a} {b:+.3f}' for a, b in zip(nm_m, p[:km])))
        print('vormlaag schaal ', ' '.join(f'{a} {b:+.3f}' for a, b in zip(nm_s, p[km:])))
    return K


def twee_t_w(u, Z):
    p_ = twee_t(u); k = Z.shape[1]
    x0 = np.concatenate([[p_[i]] + [0.0] * (k - 1) for i in range(5)])
    return minimize(lambda p: np.mean(tw_nll(p, u, Z)), x0, method='Powell', options={'maxiter': 40000, 'xtol': 1e-4, 'ftol': 1e-9}).x


def kenm_verdeling(K):
    """Kenmerken voor de vorm van de foutverdeling: 1, exp(-niveau / 4%), niveau (0..0,4), jong."""
    gq = K['k_c'].clip(lower=0, upper=0.4).fillna(0.2).to_numpy()
    Z = [np.ones(len(K)), np.exp(-gq / 0.04), gq, K['groep'].isin(JONG).to_numpy().astype(float)]
    if HOOG_SCHAAL:
        hz = K['hoog'].fillna(0.0).to_numpy()
        Z += [hz, hz * Z[1]]
    if SCHOK and 'sk_v30' in K:
        Z += [K['sk_v30'].to_numpy(), K['sk_n30'].to_numpy()]      # onrust van de laatste weken: dikkere staarten
    if UITSCHIETERS and 'k_pf' in K:                    # historische deviatie: eigen kans op snelle / trage uitschieters (log t.o.v. de basis)
        Z += [np.log(K['k_pf'].to_numpy() / PBASE), np.log(K['k_ps'].to_numpy() / PBASE)]
    return np.column_stack(Z)

JONG = ['jeugd', 'junior_C', 'junior_B', 'junior_A', 'neo']
NIVEAU_INTERACTIE = True


def met_kalman(w, par):
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'))
    R, inv = kf.voorbereid(R); o = kf.pass_(inv, par)
    K = pd.read_parquet(os.path.join(w, 'kenmerken.parquet'), columns=KOL)
    assert len(K) == len(R) and (K['skater_id'].to_numpy() == R['skater_id'].to_numpy()).all() and np.array_equal(K['tijd'].to_numpy(float), R['tijd'].to_numpy(float), equal_nan=True)
    for k in ('zh', 'vl', 'r', 'e', 'na', 'c', 'd', 'f', 'pc', 'pf', 'ps'):
        if k in o:
            K['k_' + k] = o[k]
    if 'theta_rijder' not in K:
        K['theta_rijder'] = K['theta']
    K['rk_'] = inv['rk']
    if SCHOK:
        u = o['e'] / np.sqrt(np.maximum(o['vl'] + o['r'], 1e-10))
        u = np.where(o['val'] | ~np.isfinite(u), np.nan, np.clip(u, -6, 6))
        uit = schok_geheugen(inv['rid'], inv['dag'], u)
        for k_, v_ in uit.items():
            K['sk_' + k_] = v_
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
    if UITSCHIETERS and 'k_pf' in K:                    # historische deviatie als uitschieters (snel / traag), niet week-op-week
        F['uit_snel'] = np.log(K['k_pf'] / PBASE); F['uit_traag'] = np.log(K['k_ps'] / PBASE)
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
    for c in ('seizoenstart', 'a1', 'a2', 'a3', 'leeftijd_onb', 'g', 'vrouw', 'nieuw_af') + tuple('af_' + a for a in AF[1:]) + (('uit_snel', 'uit_traag') if 'uit_snel' in F else ()):
        S[c] = F[c]
    if HOOG_SCHAAL:
        # baansnelheid in de schaal (maakte de log loss beter maar de kalibratie van de elite slechter; standaard uit)
        dth_ = (K['th_hat'] - K['gm']).clip(-0.06, 0.06).fillna(0.0)
        S['hoog'] = K['hoog']; S['hoog_w'] = K['hoog'] * wg; S['dth'] = dth_; S['dth_w'] = dth_ * wg
    if SCHOK:
        # schokgeheugen: na een snelle schok blijft de rijder weken sneller (vorm), na elke schok is het onrustiger (beide staarten)
        for c in [c for c in K.columns if c.startswith('sk_')]:
            F[c] = K[c]; F[c + '_w'] = K[c] * wg
            if c.startswith('sk_v') or c.startswith('sk_n'):
                S[c] = K[c]
    return F, S


def draai(w, par, verbose=True, segment=True):
    t0 = time.time()
    K = met_kalman(w, par); K, c_ijs = ijs_voorspelling(K)
    # universele baanfactor vooraf: zelfde functie als achteraf (baanfactor.py), op het voorspelde dagpeil
    bfp = json.load(open(os.path.join(w, 'baanfactor_par.json'))) if os.path.exists(os.path.join(w, 'baanfactor_par.json')) else None
    K['th_hat_veld'] = K['th_hat']
    if bfp:
        hv = K[(K['baan'] == 'Heerenveen (NED)') & (K['jaar'] <= 2019)].groupby(['gender', 'afstand_idx'])['theta'].median()
        thv = np.array([hv.get((g, a), np.nan) for g, a in zip(K['gender'], K['afstand_idx'])])
        gbar = bf.veldniveau(K['rk_'].to_numpy(), K['k_c'].to_numpy(), 1 / np.maximum((K['k_vl'] + K['k_r']).to_numpy(), 1e-6))
        gg = np.where(np.isfinite(K['k_c']), K['k_c'], gbar); dd = (K['th_hat'] - thv).to_numpy()
        K['th_hat'] = np.where(np.isfinite(dd), bf.factor(K['th_hat'].to_numpy(), dd, gg, gbar, bfp['A'], bfp['TM'], bfp.get('B', 0.0), bfp.get('TB', 0.05)), K['th_hat'])
    F, S = ontwerp(K)
    zt = np.log(K['tijd']) - K['theta_rijder'] - K['ref']           # z met het echte racepeil (baanfactor per rijder)
    e = (zt - K['k_zh']).to_numpy()
    X = np.c_[np.ones(len(F)), F.to_numpy(float)]; Sx = np.c_[np.ones(len(S)), S.to_numpy(float)]
    # ok: alle ritten met een voorspelling (ook wat dit model als valpartij ziet; de toets mag niet afhangen van de eigen valvlag)
    ok = (np.isfinite(K["k_zh"]) & np.isfinite(K["y"]) & ~K["verdacht"] & np.isfinite(K["th_hat"])).to_numpy().copy()
    ok &= np.isfinite(X).all(1) & np.isfinite(Sx).all(1)
    jr = K['jaar'].to_numpy(); fit = ok & ~K['k_val'].to_numpy() & (jr <= FIT_TOT) & (jr >= 2008) & np.isfinite(e)
    # aparte correctiefuncties voor jonge rijders (t/m neo) en de rest: groei en ervaring werken anders
    seg = K['groep'].isin(JONG).to_numpy() if segment else np.zeros(len(K), bool)
    sc_r = np.sqrt(K['k_vl'] + K['k_r']).to_numpy()            # eerste schaal: de Kalman-onzekerheid + ritruis
    for ronde in range(3):                                      # locatie (gewogen met 1 / schaal^2) en schaal om en om
        loc = np.zeros(len(K)); coefs = {}
        for sg in (True, False):
            m_ = fit & (seg == sg)
            if m_.sum() < 1000: continue
            b = huber(X[m_], e[m_], w=1.0 / sc_r[m_] ** 2, c=HUBER_C); coefs['jong' if sg else 'ouder'] = pd.Series(b, index=['const'] + list(F.columns))
            loc[seg == sg] = X[seg == sg] @ b
        rr = e - loc; lr = np.log(np.abs(rr) + 1e-5)
        ff = fit & np.isfinite(lr)
        gs = np.linalg.lstsq(Sx[ff], lr[ff], rcond=None)[0]; sc_r = np.exp(Sx @ gs)
        sc_r *= np.median(np.abs(rr[ff]) / sc_r[ff]) / 0.6745
    K['pred'] = K['th_hat'] + K['ref'] + K['k_zh'] + loc
    K['pred_basis'] = K['th_hat'] + K['ref'] + K['k_zh']
    # online biascorrectie: lopend gemiddelde van de recente fouten (alleen eerdere dagen), algemeen en gewogen naar de toppers
    if BIAS_HL > 0:
        wb = np.exp(-K['k_c'].clip(lower=0, upper=0.4).fillna(0.4).to_numpy() / 0.04)
        b_all, b_top = online_bias(K['dag'].to_numpy(), (np.log(K['tijd']) - K['pred']).to_numpy(), fit_b := (ok & ~K['k_val'].to_numpy()), wb, BIAS_HL)
        K['bias'] = (1 - wb) * b_all + wb * b_top
        K['pred'] = K['pred'] + K['bias']
        if BAAN_FUNCTIE:
            # per baan een correctie als functie van het niveau, geschat op ALLE rijders (alleen eerdere dagen; halfwaardetijd 2 jaar,
            # ridge 50): op snel ijs winnen zwakkere rijders meer dan de baanfactor zegt (Calgary tot -1,2% bij >30% van de top)
            K['bias_baanfunctie'] = baan_functie(K, (np.log(K['tijd']) - K['pred']).to_numpy(), fit_b)
            K['pred'] = K['pred'] + K['bias_baanfunctie']
        if BAAN_TOP_HL > 0:
            # per baan: lopende fout van de toppers (gewogen met wb, alleen eerdere dagen, halfwaardetijd BAAN_TOP_HL dagen, krimp naar 0):
            # het baangemiddelde mengt alle soorten wedstrijden; voor elite-wedstrijden is het ijs op de ene baan anders dan op de andere
            r_ = (np.log(K['tijd']) - K['pred']).to_numpy(); bt = np.zeros(len(K))
            for b_, idx in K.groupby('baan').indices.items():
                if len(idx) < 50: continue
                bb_, nb_ = online_baan(K['dag'].to_numpy()[idx], r_[idx], fit_b[idx], wb[idx], BAAN_TOP_HL)
                bt[idx] = bb_ * nb_ / (nb_ + BAAN_TOP_K)
            K['bias_baan'] = wb * bt
            K['pred'] = K['pred'] + K['bias_baan']
        if SZ_TOP:
            # seizoensverloop per baan voor de toppers: functie van de week in het seizoen, uit EERDERE seizoenen op die baan
            # (kernel 2 weken, halfwaardetijd 2 seizoenen, krimp 60); één ronde (herhalen gaat toevallige seizoenen naspelen)
            K['bias_seizoen'] = wb * seizoen_top(K, (np.log(K['tijd']) - K['pred']).to_numpy(), fit_b, wb)
            K['pred'] = K['pred'] + K['bias_seizoen']
        if SNEL_A > 0:
            # snelle correctie per baan voor de toppers: lopend gemiddelde per WEDSTRIJDDAG (elke nieuwe dag telt SNEL_A, los van de
            # kalendertijd: het vorige bezoek aan SLC telt ook als het een jaar geleden was), alleen eerdere dagen, krimp SNEL_K
            r_ = (np.log(K['tijd']) - K['pred']).to_numpy(); bs = np.zeros(len(K))
            for b_, idx in K.groupby('baan').indices.items():
                bs[idx] = per_wedstrijddag(K['dag'].to_numpy()[idx], r_[idx], fit_b[idx], wb[idx], SNEL_A, SNEL_K)
            K['bias_snel'] = wb * bs
            K['pred'] = K['pred'] + K['bias_snel']
    rc = K[fit].drop_duplicates(['dag', 'event_id', 'gender', 'afstand_idx'])
    Ix = lambda d: np.c_[np.ones(len(d)), np.log1p(d['ijs_vn']), np.log1p(d['ijs_evn']), d['hoog'], np.log(d['v_theta'].clip(1e-7)),
                         np.log1p(d['ijs_bn']), d['ijs_nieuw'] if 'ijs_nieuw' in d else np.zeros(len(d))]      # geschiedenis van de baan: nieuw vs gevestigd
    rci = (rc['theta'] - rc['th_hat_veld']).to_numpy()                 # onzekerheid van het dagpeil zelf (op raceniveau)
    gi = np.linalg.lstsq(Ix(rc), np.log(np.abs(rci) + 1e-5), rcond=None)[0]
    sc_i = np.exp(Ix(K) @ gi); sc_i *= np.median(np.abs(rci) / np.exp(Ix(rc) @ gi)) / 0.6745
    K['sd'] = np.sqrt(sc_r ** 2 + sc_i ** 2)
    K['err'] = np.log(K['tijd']) - K['pred']; K['u'] = K['err'] / K['sd']
    K['ag'] = K['afstand'].map(GROEP); K['nll'] = np.nan; K['pit'] = np.nan; par_t = {}
    Zq = kenm_verdeling(K); wq = Zq[:, 1]                        # vorm van de foutverdeling als functie van niveau en leeftijd
    for gname in ('sprint', 'midden', 'lang'):
        sf = fit & (K['ag'] == gname).to_numpy(); idx = np.flatnonzero(sf)
        idx = np.random.default_rng(0).choice(idx, min(len(idx), 150000), replace=False)
        u_f = K['u'].to_numpy()[idx]; Z_f = Zq[idx]; top = Z_f[:, 1] > 0.3
        # toppers zwaarder in de fit (anders bepalen de vele zwakkere rijders alles)
        p = twee_t_w(np.r_[u_f, u_f[top]], np.r_[Z_f, Z_f[top]]); par_t[gname] = p
        sa = ok & (K['ag'] == gname).to_numpy()
        K.loc[sa, 'nll'] = tw_nll(p, K.loc[sa, 'u'].to_numpy(), Zq[sa]) + np.log(K.loc[sa, 'sd'].to_numpy())
        K.loc[sa, 'pit'] = tw_cdf(p, K.loc[sa, 'u'].to_numpy(), Zq[sa])
    # snelle staart (voorbij Q_STAART): p' = Q (p / Q)^b met b = b0 + b1 w, geschat op de fitjaren (zonder eigen valpartijen);
    # als p' uniform is, is b (-log(p / Q)) exponentieel met rate 1: ML van rate b(w)
    pit = K['pit'].to_numpy(); Q = Q_STAART
    ft = fit & (pit < Q) & np.isfinite(pit)
    xs = -np.log(pit[ft] / Q); ws = wq[ft]
    from scipy.optimize import minimize as _min
    bb = _min(lambda c: -np.mean(np.log(np.maximum(c[0] + c[1] * ws, 1e-3)) - np.maximum(c[0] + c[1] * ws, 1e-3) * xs), np.array([1.0, 0.0]), method='Nelder-Mead').x
    b_r = np.maximum(bb[0] + bb[1] * wq, 0.2)
    st = (pit < Q) & np.isfinite(pit)
    K['pit'] = np.where(st, Q * (np.clip(pit, 1e-300, 1) / Q) ** b_r, pit)
    K['nll'] = np.where(st, K['nll'] - np.log(b_r) - (b_r - 1) * np.log(np.clip(pit, 1e-300, 1) / Q), K['nll'])
    K.attrs['staart_b'] = [float(v) for v in bb]; K['staart_br'] = b_r
    K['ok'] = ok
    if VORM:
        K = vorm_toepassen(K, fit, ok, verbose)
        # voorspelde tijd = mediaan van de nieuwe verdeling (verschuiving t.o.v. de mediaan zonder vormlaag)
        mu_, sg_ = K.pop('vorm_mu'), K.pop('vorm_sg')
        u0 = kwantiel_u(K, par_t, Zq, 0.5)
        K['vorm_mu'] = mu_; K['vorm_sg'] = sg_
        u1 = kwantiel_u(K, par_t, Zq, 0.5)
        K['bias_vorm'] = np.nan_to_num((u1 - u0) * K['sd'].to_numpy())
        K['pred'] = K['pred'] + K['bias_vorm']
        K['err'] = np.log(K['tijd']) - K['pred']
    if verbose:
        print('staart b (snel, voorbij %.0f%%): b0 %.3f  b1 (x exp(-niveau/4%%)) %.3f' % (100 * Q, bb[0], bb[1]))
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
       'th_hat', 'theta', 'k_c', 'k_val', 'rk_', 'lf']].to_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'))
