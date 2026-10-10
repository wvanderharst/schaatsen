"""Nieuwe, eenvoudige methode voor de kans per rit (los van do_analysis), getoetst op de 200 beste vrouwen en 200 beste mannen sinds 2011.

Alles chronologisch, alleen informatie van vóór de race:
  1. racepeil   : per race (event x geslacht x afstand) de mediaan van (ln tijd - eigen niveau) over de ANDERE rijders in die race met
                  minstens MIN_N eerdere ritten op die afstand (leave-one-out): het ijs en de omstandigheden van die dag.
  2. niveau     : per rijder x afstand een robuust lopend gemiddelde van ln tijd - racepeil (Huber op HUB x eigen spreiding);
                  snelle ritten wegen zwaarder (SNEL_W); valpartijen (fout > VAL_SD x spreiding en > VAL_MIN trager) tellen niet.
  3. spreiding  : per rijder x afstand een lopend gemiddelde van de gekwadrateerde fout, gekrompen naar S0 met M0 pseudo-ritten.
  4. kans       : u = fout / spreiding wordt geijkt op de elite-ritten van EERDERE seizoenen, per groep (ervaring x tijd sinds de vorige
                  rit): schaal per groep, dan de gepoolde empirische verdeling met GPD-staarten voorbij 2% en 98%.
Gebruik: python elite_model.py ritten.parquet  (schrijft elite_uitkomst.parquet en print de toets)."""
import sys, math, time
import numpy as np, pandas as pd
from numba import njit

MIN_N = 3          # eerdere ritten op de afstand om mee te tellen voor het racepeil
ALPHA = 0.12       # vloer van het leergewicht van het niveau
SNEL_W = 1.5       # snelle ritten wegen zwaarder
HUB = 3.0          # Huber-grens (in eigen spreiding)
BETA = 0.06        # vloer van het leergewicht van de spreiding
S0 = 0.012         # spreiding (ln-tijd) vooraf
M0 = 5.0           # krimp van de spreiding naar S0 (pseudo-ritten)
VAL_SD, VAL_MIN = 6.0, 0.03
GAMMA = 0.0        # vorm per rijder (alle afstanden samen): f += GAMMA * fout na elke rit
HF = 60.0          # vervaltijd van de vorm (dagen)
TAU = 0.015        # sd van het dag-effect rond het baaneffect (ln-tijd): het racepeil wordt naar het baaneffect getrokken (anker)
ETA = 0.05         # leergewicht van het baaneffect
VFLOOR = 0.01      # ondergrens van de onzekerheid van het veldpeil (gezamenlijke verschuiving van de niveaus van een groep rijders)


@njit(cache=True)
def _pass(rk, st, ltijd, dag, rid, baan, nb, tau, eta, vfloor, nst, nr, min_n, alpha, snel_w, hub, beta, s0, m0, val_sd, val_min, gamma, hf,
          o_par, o_mu, o_s, o_n, o_gap, o_e, o_val, o_goed, o_f, o_spar, o_k):
    mu = np.zeros(nst); s2 = np.full(nst, s0 * s0); n = np.zeros(nst)
    last = np.full(nr, -100000); f = np.zeros(nr); fdag = np.zeros(nr)
    B = np.zeros(nb); nB = np.zeros(nb)              # baaneffect (anker van het racepeil)
    N = len(rk); i = 0
    buf = np.empty(4096); buf2 = np.empty(4096)
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        # verschillen van rijders met genoeg historie
        k = 0; k2 = 0
        for q in range(i, j):
            r_ = rid[q]; fc = f[r_] * math.exp(-(dag[q] - fdag[r_]) / hf) if gamma > 0 else 0.0
            o_f[q] = fc
            if n[st[q]] >= min_n:
                buf[k] = ltijd[q] - mu[st[q]] - fc; k += 1
            if n[st[q]] >= 1:
                buf2[k2] = ltijd[q] - mu[st[q]] - fc; k2 += 1
        srt = np.sort(buf[:k])
        # onzekerheid van het racepeil: robuuste spreiding van het veld, mediaan van k waarden (factor 1,2533^2 / k)
        if k >= 4:
            sv = (srt[(3 * k) // 4] - srt[k // 4]) / 1.349
            spar2 = 1.5708 * sv * sv / (k - 1)
        else:
            spar2 = np.nan
        bb = baan[i]; Bb = B[bb]; tB2 = tau * tau if nB[bb] > 0 else 1.0      # onbekende baan: geen anker
        # terugval (alleen voor het bijwerken, niet voor de toets): rijders met enige historie, of aan het begin van de data peil 0
        par_terug = np.median(buf2[:k2]) if k2 >= 1 else 0.0
        terug_ok = (k2 >= 3) or (dag[i] < dag[0] + 400)     # bijwerken met het terugvalpeil alleen met >= 3 rijders met historie (of bij de start van de data)
        for q in range(i, j):
            a = st[q]
            o_n[q] = n[a]; o_gap[q] = dag[q] - last[rid[q]]
            eigen = n[a] >= min_n
            kk = k - 1 if eigen else k
            if kk < 3:
                o_par[q] = par_terug; o_e[q] = np.nan; o_mu[q] = mu[a]; o_s[q] = np.nan; o_val[q] = False; o_goed[q] = False
                continue
            o_goed[q] = True
            # leave-one-out mediaan: haal de eigen waarde uit de gesorteerde lijst
            if eigen:
                x = ltijd[q] - mu[a] - o_f[q]; pos = np.searchsorted(srt, x)
                if pos >= k:
                    pos = k - 1
                m_ = kk // 2
                if kk % 2 == 1:
                    idx = m_ if m_ < pos else m_ + 1
                    par = srt[idx]
                else:
                    i1 = m_ - 1 if m_ - 1 < pos else m_
                    i2 = m_ if m_ < pos else m_ + 1
                    par = 0.5 * (srt[i1] + srt[i2])
            else:
                m_ = kk // 2
                par = srt[m_] if kk % 2 == 1 else 0.5 * (srt[m_ - 1] + srt[m_])
            # anker: racepeil = gewogen gemiddelde van het veld (variantie spar2) en het baaneffect (variantie tau^2)
            v_ = (spar2 if spar2 == spar2 else 0.02 ** 2) + vfloor * vfloor
            w_ = tB2 / (tB2 + v_)
            par = w_ * par + (1 - w_) * Bb
            o_par[q] = par; o_spar[q] = math.sqrt(v_ * tB2 / (v_ + tB2)); o_k[q] = kk
            sp = math.sqrt((n[a] * s2[a] + m0 * s0 * s0) / (n[a] + m0))
            o_s[q] = sp; o_mu[q] = mu[a]
            if n[a] == 0:
                o_e[q] = np.nan; o_val[q] = False
            else:
                e = ltijd[q] - par - mu[a] - o_f[q]; o_e[q] = e
                o_val[q] = (e > val_sd * sp) and (e > val_min)
        # baaneffect bijwerken met het (verankerde) peil van deze race
        pr_ = np.nan
        for q in range(i, j):
            if o_goed[q]:
                pr_ = o_par[q]; break
        if pr_ == pr_:
            if nB[bb] == 0:
                B[bb] = pr_
            else:
                B[bb] += max(eta, 1.0 / (nB[bb] + 1.0)) * (pr_ - B[bb])
            nB[bb] += 1.0
        # bijwerken na de race (alle rijders van de race tegelijk: de race zelf is in de voorspelling niet gebruikt)
        for q in range(i, j):
            a = st[q]
            if o_val[q] or not (o_goed[q] or terug_ok):
                continue
            y = ltijd[q] - o_par[q]
            if n[a] == 0:
                mu[a] = y; n[a] = 1.0
            else:
                sp = math.sqrt((n[a] * s2[a] + m0 * s0 * s0) / (n[a] + m0)); e = y - mu[a] - o_f[q]
                w = max(1.0 / (n[a] + 1.0), alpha)
                if e < 0:
                    w = min(w * snel_w, 0.6)
                ec = min(max(e, -hub * sp), hub * sp)
                mu[a] += w * ec
                if o_goed[q]:                               # spreiding alleen bij een goed racepeil
                    b = max(1.0 / (n[a] + 1.0), beta)
                    s2[a] = (1 - b) * s2[a] + b * min(e * e, (hub * sp) ** 2)
                if gamma > 0 and n[a] >= min_n:
                    r_ = rid[q]; f[r_] = o_f[q] + gamma * ec; fdag[r_] = dag[q]
                n[a] += 1.0
            last[rid[q]] = dag[q]
        i = j
    return 0


@njit(cache=True)
def _pass_ext(st, ltijd, dag, rid, par, spar, rk, nst, nr, alpha, snel_w, hub, beta, s0, m0, val_sd, val_min, gamma, hf, tweezijdig,
              dth, hoog, ref, ijs,
              o_mu, o_s, o_n, o_gap, o_e, o_val, o_f, o_sf, o_ss, o_g, o_pe):
    """Zoals _pass, maar met een extern racepeil (bijv. het dag-effect theta van do_analysis): geen schaalverschuiving tussen groepen."""
    mu = np.zeros(nst); s2 = np.full(nst, s0 * s0); n = np.zeros(nst)
    s2f = np.full(nst, s0 * s0); s2s = np.full(nst, s0 * s0); nf = np.zeros(nst); ns = np.zeros(nst)    # spreiding per kant
    last = np.full(nr, -100000); f = np.zeros(nr); fdag = np.zeros(nr)
    N = len(st); i = 0
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        for q in range(i, j):
            a = st[q]; r_ = rid[q]
            fc = f[r_] * math.exp(-(dag[q] - fdag[r_]) / hf) if gamma > 0 else 0.0
            o_f[q] = fc; o_n[q] = n[a]; o_gap[q] = dag[q] - last[r_]; o_mu[q] = mu[a]
            # ijs per niveau: lineaire functie van het niveau g = mu - ref (ln-tijd t.o.v. wereldtop) voor snel ijs (dth) en hoogland
            g_ = mu[a] - ref[q] if n[a] >= 1 else 0.0
            o_g[q] = g_
            pe = par[q] + (ijs[0] + ijs[1] * g_) * dth[q] + (ijs[2] + ijs[3] * g_) * hoog[q]
            o_pe[q] = pe
            sp = math.sqrt((n[a] * s2[a] + m0 * s0 * s0) / (n[a] + m0)); o_s[q] = sp
            o_sf[q] = math.sqrt((nf[a] * s2f[a] + m0 * sp * sp) / (nf[a] + m0)); o_ss[q] = math.sqrt((ns[a] * s2s[a] + m0 * sp * sp) / (ns[a] + m0))
            if n[a] == 0 or par[q] != par[q]:
                o_e[q] = np.nan; o_val[q] = False
            else:
                e = ltijd[q] - pe - mu[a] - fc; o_e[q] = e
                o_val[q] = (e > val_sd * math.sqrt(sp * sp + spar[q] * spar[q])) and (e > val_min)
        for q in range(i, j):
            a = st[q]
            if o_val[q] or par[q] != par[q]:
                continue
            y = ltijd[q] - o_pe[q]
            if n[a] == 0:
                mu[a] = y; n[a] = 1.0
            else:
                sp = o_s[q]; e = y - mu[a] - o_f[q]
                w = max(1.0 / (n[a] + 1.0), alpha)
                if e < 0:
                    w = min(w * snel_w, 0.6)
                ec = min(max(e, -hub * sp), hub * sp)
                mu[a] += w * ec
                b = max(1.0 / (n[a] + 1.0), beta)
                s2[a] = (1 - b) * s2[a] + b * min(e * e, (hub * sp) ** 2)
                if tweezijdig:                              # per kant, ruimer geknipt: juist de grote uitschieters bepalen de staart van die rijder
                    if e < 0:
                        bf = max(1.0 / (nf[a] + 1.0), beta); s2f[a] = (1 - bf) * s2f[a] + bf * min(e * e, 0.05 ** 2); nf[a] += 1.0
                    else:
                        bs = max(1.0 / (ns[a] + 1.0), beta); s2s[a] = (1 - bs) * s2s[a] + bs * min(e * e, 0.05 ** 2); ns[a] += 1.0
                if gamma > 0:
                    f[rid[q]] = o_f[q] + gamma * ec; fdag[rid[q]] = dag[q]
                n[a] += 1.0
            last[rid[q]] = dag[q]
        i = j
    return 0


HOOGLAND = ('Salt Lake City (USA)', 'Calgary (CAN)')


def ijs_kenmerken(df, par_kol='theta'):
    """dth = racepeil min het Heerenveen-peil (mediaan per geslacht x afstand x seizoen, anders over alle seizoenen), hoogland (0/1),
    ref = ln-tijd van de wereldtop per geslacht x afstand (1e percentiel van ln tijd - racepeil): het niveau g = mu - ref."""
    th = df[par_kol]
    hv = df[df['baan'] == 'Heerenveen (NED)'].groupby(['gender', 'afstand_idx', 'seizoen'])[par_kol].median().rename('hv')
    hv_all = df[df['baan'] == 'Heerenveen (NED)'].groupby(['gender', 'afstand_idx'])[par_kol].median().rename('hv_all')
    x = df[['gender', 'afstand_idx', 'seizoen']].join(hv, on=['gender', 'afstand_idx', 'seizoen']).join(hv_all, on=['gender', 'afstand_idx'])
    dth = (th - x['hv'].fillna(x['hv_all'])).fillna(0.0).to_numpy()
    hoog = df['baan'].isin(HOOGLAND).to_numpy().astype(float)
    lt = np.log(df['tijd']) - th
    ref = lt.groupby([df['gender'], df['afstand_idx']]).transform(lambda v: np.nanquantile(v, 0.01)).to_numpy()
    return dth, hoog, ref


def draai_ext(df, alpha=ALPHA, snel_w=SNEL_W, gamma=GAMMA, hf=HF, beta=BETA, m0=M0, s0=S0, hub=HUB, par_kol='theta', var_kol='v_theta', tweezijdig=False,
              ijs=(0.0, 0.0, 0.0, 0.0)):
    """Nieuwe niveaus, spreiding en vorm met een extern racepeil (kolom par_kol, variantie var_kol)."""
    df = df.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    rk = df.groupby(['dag', 'event_id', 'gender', 'afstand_idx'], sort=False).ngroup().to_numpy().astype(np.int64)
    st = (df['skater_id'].astype(np.int64) * 8 + df['afstand_idx'].astype(np.int64))
    st_codes, st_u = pd.factorize(st); rid_codes, _ = pd.factorize(df['skater_id'])
    N = len(df); out = {k: np.zeros(N) for k in ('mu', 's', 'n', 'gap', 'e', 'f', 'sf', 'ss', 'g', 'pe')}; val = np.zeros(N, bool)
    dth, hoog, ref = ijs_kenmerken(df, par_kol); df['dth'] = dth; df['hoog'] = hoog
    par = df[par_kol].to_numpy(dtype=float); spar = np.sqrt(df[var_kol].to_numpy(dtype=float))
    _pass_ext(st_codes.astype(np.int64), np.log(df['tijd'].to_numpy()), df['dag'].to_numpy().astype(np.int64), rid_codes.astype(np.int64),
              par, spar, rk, len(st_u), rid_codes.max() + 1, alpha, snel_w, hub, beta, s0, m0, VAL_SD, VAL_MIN, gamma, hf, tweezijdig,
              dth, hoog, ref, np.asarray(ijs, float),
              out['mu'], out['s'], out['n'], out['gap'], out['e'], val, out['f'], out['sf'], out['ss'], out['g'], out['pe'])
    for k, v in out.items():
        df[k] = v
    df['par'] = par; df['spar'] = spar; df['val'] = val; df['goed'] = np.isfinite(par)
    df['k'] = df.groupby(['dag', 'event_id', 'gender', 'afstand_idx'])['skater_id'].transform('size').astype(float)    # veldgrootte
    s_kant = np.where(df['e'] < 0, df['sf'], df['ss']) if tweezijdig else df['s'].to_numpy()
    df['s_tot'] = np.sqrt(s_kant ** 2 + df['spar'] ** 2)
    df['u'] = df['e'] / df['s_tot']
    return df


def draai(df, alpha=ALPHA, snel_w=SNEL_W, gamma=GAMMA, hf=HF, beta=BETA, m0=M0, s0=S0, hub=HUB, tau=TAU, eta=ETA, vfloor=VFLOOR):
    df = df.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    rk = df.groupby(['dag', 'event_id', 'gender', 'afstand_idx'], sort=False).ngroup().to_numpy().astype(np.int64)
    st = (df['skater_id'].astype(np.int64) * 8 + df['afstand_idx'].astype(np.int64))
    st_codes, st_u = pd.factorize(st); rid_codes, _ = pd.factorize(df['skater_id'])
    N = len(df)
    out = {k: np.zeros(N) for k in ('par', 'mu', 's', 'n', 'gap', 'e', 'f', 'spar', 'k')}; val = np.zeros(N, bool); goed = np.zeros(N, bool)
    bcode, bu = pd.factorize(df['baan'].astype(str) + '|' + df['afstand'].astype(str) + '|' + df['gender'].astype(str))   # baaneffect per baan x afstand x geslacht
    _pass(rk, st_codes.astype(np.int64), np.log(df['tijd'].to_numpy()), df['dag'].to_numpy().astype(np.int64), rid_codes.astype(np.int64),
          bcode.astype(np.int64), len(bu), tau, eta, vfloor,
          len(st_u), rid_codes.max() + 1, MIN_N, alpha, snel_w, hub, beta, s0, m0, VAL_SD, VAL_MIN, gamma, hf,
          out['par'], out['mu'], out['s'], out['n'], out['gap'], out['e'], val, goed, out['f'], out['spar'], out['k'])
    for k, v in out.items():
        df[k] = v
    df['val'] = val; df['goed'] = goed
    df['s_tot'] = np.sqrt(df['s'] ** 2 + df['spar'].fillna(df['s']) ** 2)          # eigen spreiding + onzekerheid van het racepeil
    df['u'] = np.where(goed, df['e'] / df['s_tot'], np.nan)
    key = ['dag', 'event_id', 'gender', 'afstand_idx']
    return df
    return df


def kies_elite(df, n_per=200, vanaf=2011):
    """Per geslacht de n_per rijders met de beste plek in de PR-ranglijst (500 m t/m 10.000 m, vanaf `vanaf`, senioren-afstanden)."""
    x = df[(df['jaar'] >= vanaf) & df['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m', '10000m']) & ~df['verdacht']]
    pb = x.groupby(['gender', 'afstand', 'skater_id'])['tijd'].min().reset_index()
    pb['rang'] = pb.groupby(['gender', 'afstand'])['tijd'].rank(method='min')
    beste = pb.groupby(['gender', 'skater_id'])['rang'].min().reset_index().sort_values(['gender', 'rang'])
    return set(beste.groupby('gender').head(n_per)['skater_id'])


GAP_GRENZEN = (15, 61, 201)
N_GRENZEN = (10, 30)


K_GRENZEN = (6, 12, 24)           # aantal bekende rijders in de race (kwaliteit van het racepeil)


def ijk(df, elite, eerste_seizoen=2014, q=0.02):
    """Kans per rit voor de elite, per seizoen geijkt op de elite-ritten van eerdere seizoenen (vanaf 2008)."""
    from scipy.stats import genpareto as gp
    from scipy.special import ndtri
    e = df[df['skater_id'].isin(elite) & np.isfinite(df['u']) & ~df['val'] & (df['n'] >= MIN_N) & ~df['verdacht']].copy()
    e['g'] = (np.digitize(e['gap'], GAP_GRENZEN) * 100 + np.digitize(e['n'], N_GRENZEN) * 10 + np.digitize(e['k'], K_GRENZEN)
              + 1000 * e['baan'].isin(HOOGLAND).astype(int))          # hoogland: eigen locatie en (grotere) spreiding
    e['p'] = np.nan
    def staart(y):
        xi, _, sg = gp.fit(y, floc=0.0)
        if xi < 0:
            xi, sg = 0.0, float(np.mean(y))
        return min(float(xi), 0.9), float(sg)
    for S in sorted(e['seizoen'].unique()):
        if S < eerste_seizoen:
            continue
        ref = e[(e['seizoen'] < S) & (e['seizoen'] >= 2008)]
        sc = ref.groupby('g')['u'].agg(lambda x: (np.quantile(x, .84) - np.quantile(x, .16)) / 2 if len(x) >= 50 else np.nan)
        md = ref.groupby('g')['u'].agg(lambda x: x.median() if len(x) >= 50 else np.nan)
        sc_all = (np.quantile(ref['u'], .84) - np.quantile(ref['u'], .16)) / 2; md_all = ref['u'].median()
        sc = sc.fillna(sc_all); md = md.fillna(md_all)
        v_ref = np.sort(((ref['u'] - ref['g'].map(md)) / ref['g'].map(sc)).to_numpy())
        lo, hi = np.quantile(v_ref, q), np.quantile(v_ref, 1 - q)
        xl, sl = staart(lo - v_ref[v_ref < lo]); xh, sh = staart(v_ref[v_ref > hi] - hi)
        t = e['seizoen'] == S
        v = ((e.loc[t, 'u'] - e.loc[t, 'g'].map(md).fillna(md_all)) / e.loc[t, 'g'].map(sc).fillna(sc_all)).to_numpy()
        p = np.searchsorted(v_ref, v, side='right') / len(v_ref)
        p = np.where(v < lo, q * gp.sf(lo - v, xl, 0, sl), p); p = np.where(v > hi, 1 - q * gp.sf(v - hi, xh, 0, sh), p)
        e.loc[t, 'p'] = np.clip(p, 1e-9, 1 - 1e-9)
    e = e[np.isfinite(e['p'])]
    e['z'] = ndtri(e['p'])
    return e


def _mediaan_regressie(X, y, n_iter=40):
    """Mediaanregressie (kleinste absolute afwijkingen) met iteratief herwogen kleinste kwadraten."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        w = 1.0 / np.maximum(np.abs(y - X @ b), 1e-3)
        b_new = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        if np.max(np.abs(b_new - b)) < 1e-6:
            b = b_new; break
        b = b_new
    return b


def ijk_glad(df, elite, eerste_seizoen=2014, q=0.02):
    """Zonder groepen: locatie en log-schaal van u als lineaire functies van log(1+gap), log(1+n), 1/sqrt(veld) en g, per seizoen geschat
    op de elite-ritten van eerdere seizoenen; daarna één gepoolde empirische verdeling van v = (u - m(x)) / s(x) met GPD-staarten."""
    from scipy.stats import genpareto as gp
    from scipy.special import ndtri
    e = df[df['skater_id'].isin(elite) & np.isfinite(df['u']) & ~df['val'] & (df['n'] >= MIN_N) & ~df['verdacht']].copy()
    gp_ = np.clip(e['gap'].to_numpy(), 0, 3000); n_ = e['n'].to_numpy(); kk = np.maximum(e['k'].to_numpy(), 1)
    # verzadigende exponentiële functies: buiten het geziene bereik vlakken ze af (een kwadraat in log(n) schoot weg bij veteranen)
    X = np.c_[np.ones(len(e)), np.exp(-n_ / 10), np.exp(-n_ / 40), np.exp(-gp_ / 20), np.exp(-gp_ / 120), 1 / np.sqrt(kk), 1 / kk]
    u = e['u'].to_numpy(); sz = e['seizoen'].to_numpy(); p_out = np.full(len(e), np.nan)
    def staart(y):
        xi, _, sg = gp.fit(y, floc=0.0)
        if xi < 0:
            xi, sg = 0.0, float(np.mean(y))
        return min(float(xi), 0.9), float(sg)
    for S in sorted(np.unique(sz)):
        if S < eerste_seizoen:
            continue
        r = (sz < S) & (sz >= 2008); t = sz == S
        uc = np.clip(u[r], -6, 6)
        bm = _mediaan_regressie(X[r], uc)                                             # locatie: mediaan (de staarten zijn scheef)
        res = uc - X[r] @ bm
        # log-schaal per kant (de spreiding is asymmetrisch): mediaan van log|res| aan de snelle en aan de trage kant
        neg = res < 0
        bsl = _mediaan_regressie(X[r][neg], np.log(-res[neg] + 1e-4)); bsh = _mediaan_regressie(X[r][~neg], np.log(res[~neg] + 1e-4))
        def schaal(Xs, rr):
            return np.where(rr < 0, np.exp(Xs @ bsl), np.exp(Xs @ bsh)) / 0.6745
        v_ref = np.sort(res / schaal(X[r], res))
        lo, hi = np.quantile(v_ref, q), np.quantile(v_ref, 1 - q)
        xl, sl = staart(lo - v_ref[v_ref < lo]); xh, sh = staart(v_ref[v_ref > hi] - hi)
        rt = u[t] - X[t] @ bm
        v = rt / schaal(X[t], rt)
        p = np.searchsorted(v_ref, v, side='right') / len(v_ref)
        p = np.where(v < lo, q * gp.sf(lo - v, xl, 0, sl), p); p = np.where(v > hi, 1 - q * gp.sf(v - hi, xh, 0, sh), p)
        p_out[t] = np.clip(p, 1e-9, 1 - 1e-9)
    e['p'] = p_out; e = e[np.isfinite(e['p'])]; e['z'] = ndtri(e['p'])
    return e


def _gewogen_kwantiel(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(q, cw, x[o])


def ijk_kernel(df, elite, eerste_seizoen=2014, q=0.02, knopen=(6, 6, 5), bw=1.0):
    """Zonder groepen: locatie (mediaan) en schaal (helft van q84 - q16) van u als doorlopende functie van
    t1 = log(1 + dagen sinds de vorige rit), t2 = log(1 + aantal eerdere ritten), t3 = log(veldgrootte).
    Op een raster van knooppunten (kwantielen van elke factor) worden ze met Gauss-kernelgewichten geschat (bandbreedte bw x de
    knoopafstand) op de elite-ritten van eerdere seizoenen, en voor elke rit lineair geïnterpoleerd. Daarna één gepoolde
    empirische verdeling met GPD-staarten."""
    from scipy.stats import genpareto as gp
    from scipy.special import ndtri
    from scipy.interpolate import RegularGridInterpolator as RGI
    e = df[df['skater_id'].isin(elite) & np.isfinite(df['u']) & ~df['val'] & (df['n'] >= MIN_N) & ~df['verdacht']].copy()
    T = np.c_[np.log1p(np.clip(e['gap'], 0, 3000)), np.log1p(e['n']), np.log(np.maximum(e['k'], 1))]
    u = e['u'].to_numpy(); sz = e['seizoen'].to_numpy(); p_out = np.full(len(e), np.nan)
    def staart(y):
        xi, _, sg = gp.fit(y, floc=0.0)
        if xi < 0:
            xi, sg = 0.0, float(np.mean(y))
        return min(float(xi), 0.9), float(sg)
    for S in sorted(np.unique(sz)):
        if S < eerste_seizoen:
            continue
        r = (sz < S) & (sz >= 2008); t = sz == S
        Tr, ur = T[r], np.clip(u[r], -6, 6)
        ax = [np.unique(np.quantile(Tr[:, d], np.linspace(0.02, 0.98, knopen[d]))) for d in range(3)]
        h = [max(np.min(np.diff(a)) if len(a) > 1 else 1.0, 1e-3) * bw for a in ax]
        M = np.zeros([len(a) for a in ax]); SC = np.zeros_like(M)
        for idx in np.ndindex(M.shape):
            w = np.exp(-0.5 * sum(((Tr[:, d] - ax[d][idx[d]]) / h[d]) ** 2 for d in range(3)))
            if w.sum() < 50:
                w = w + 1e-6
            M[idx] = _gewogen_kwantiel(ur, w, 0.5)
            SC[idx] = (_gewogen_kwantiel(ur, w, 0.84) - _gewogen_kwantiel(ur, w, 0.16)) / 2
        fm = RGI(ax, M, bounds_error=False, fill_value=None); fs = RGI(ax, SC, bounds_error=False, fill_value=None)
        clip = lambda X: np.column_stack([np.clip(X[:, d], ax[d][0], ax[d][-1]) for d in range(3)])    # buiten het raster: rand
        v_ref = np.sort((ur - fm(clip(Tr))) / fs(clip(Tr)))
        lo, hi = np.quantile(v_ref, q), np.quantile(v_ref, 1 - q)
        xl, sl = staart(lo - v_ref[v_ref < lo]); xh, sh = staart(v_ref[v_ref > hi] - hi)
        Tt = clip(T[t]); v = (u[t] - fm(Tt)) / fs(Tt)
        p = np.searchsorted(v_ref, v, side='right') / len(v_ref)
        p = np.where(v < lo, q * gp.sf(lo - v, xl, 0, sl), p); p = np.where(v > hi, 1 - q * gp.sf(v - hi, xh, 0, sh), p)
        p_out[t] = np.clip(p, 1e-9, 1 - 1e-9)
    e['p'] = p_out; e = e[np.isfinite(e['p'])]; e['z'] = ndtri(e['p'])
    return e


def schat_ijs(df, tot_seizoen=2015, g_max=0.10):
    """Coefficienten (a, b, c, d) van het ijs per niveau: regressie van de fout op [dth, g*dth, hoog, g*hoog] (seizoenen <= tot_seizoen),
    alleen rijders tot g_max (10% trager dan de wereldtop): over alle rijders extrapoleert een lineaire functie slecht naar de top."""
    x = df[np.isfinite(df['e']) & (df['n'] >= 5) & (df['seizoen'] <= tot_seizoen) & (df['seizoen'] >= 2008) & ~df['val'] & (df['e'].abs() < 0.1)
           & (df['g'] < g_max)]
    g = x['g'].clip(-0.05, 0.3).to_numpy()
    A = np.c_[x['dth'], g * x['dth'], x['hoog'], g * x['hoog']]
    return np.linalg.lstsq(A, x['e'].to_numpy(), rcond=None)[0]


def ijs_rollend(df, jaren=4, g_max=0.10, eerste_seizoen=2010):
    """IJs per niveau, meeschuivend: voor elk seizoen een regressie van de fout op [dth, g*dth, hoog, g*hoog] over de `jaren` seizoenen
    ervoor (rijders tot g_max van de wereldtop, geen valpartijen); het geschatte effect wordt van de fout afgetrokken (u opnieuw).
    Het hooglandvoordeel van de top is sinds ~2016 gegroeid, een vaste schatting loopt daarachter."""
    df = df.copy()
    g = df['g'].clip(-0.05, 0.3).to_numpy()
    A = np.c_[df['dth'], g * df['dth'], df['hoog'], g * df['hoog']]
    ok = np.isfinite(df['e'].to_numpy()) & (df['n'].to_numpy() >= 5) & ~df['val'].to_numpy() & (np.abs(df['e'].to_numpy()) < 0.1) & (g < g_max)
    sz = df['seizoen'].to_numpy(); corr = np.zeros(len(df)); coefs = {}
    for S in np.unique(sz):
        if S < eerste_seizoen:
            continue
        r = ok & (sz < S) & (sz >= S - jaren)
        if r.sum() < 5000:
            continue
        b = np.linalg.lstsq(A[r], df['e'].to_numpy()[r], rcond=None)[0]; coefs[int(S)] = b
        t = sz == S; corr[t] = A[t] @ b
    df['e'] = df['e'] - corr; df['u'] = df['e'] / df['s_tot']; df['ijs_corr'] = corr
    return df, coefs


def hoogland_niveau(e, jaren=4):
    """Na de ijking: de top wint op het hoogland meer dan de rest van de elite. Per seizoen een meeschuivende regressie (vier seizoenen
    ervoor) van z op hoogland x (1, rang), met rang = positie van het niveau binnen geslacht x afstand x seizoen (0 = beste, doorlopend);
    de correctie wordt van z afgetrokken en p opnieuw berekend."""
    from scipy.stats import norm
    e = e.copy()
    e['rang'] = e.groupby(['gender', 'afstand', 'seizoen'])['mu'].rank(pct=True)
    h = e['baan'].isin(HOOGLAND).to_numpy().astype(float); r = e['rang'].to_numpy()
    A = np.c_[h, h * r]; z = np.clip(e['z'].to_numpy(), -5, 5); sz = e['seizoen'].to_numpy(); corr = np.zeros(len(e))
    for S in np.unique(sz):
        ref = (sz < S) & (sz >= S - jaren)
        if ref.sum() < 5000 or h[ref].sum() < 300:
            continue
        b = np.linalg.lstsq(np.c_[np.ones(ref.sum()), A[ref]], z[ref], rcond=None)[0][1:]
        t = sz == S; corr[t] = A[t] @ b
    e['z'] = e['z'] - corr; e['p'] = norm.cdf(e['z']); e['hoog_corr'] = corr
    return e


def toets(e, vanaf_seizoen=2016):
    from scipy.stats import binom
    x = e[e['seizoen'] >= vanaf_seizoen]; regels = []; fout = 0
    for kant, p in (('snel', x['p'].to_numpy()), ('traag', 1 - x['p'].to_numpy())):
        for d in (1e-3, 1e-2, 0.05):
            O = int((p <= d).sum()); E = len(x) * d; ok = abs(O - E) <= 2 * math.sqrt(E); fout += not ok
            regels.append(f"{kant:5s} kans<=1/{int(round(1 / d)):<5d}: {O:5d} ritten, verwacht {E:7.1f} ± {math.sqrt(E):5.1f} ({O / E:4.2f}x) {'ok' if ok else 'FOUT'}")
        k = pd.Series(p <= 0.02).groupby(x['skater_id'].to_numpy()).agg(['sum', 'size'])
        for kk in (3, 4, 5, 6, 8):
            E = float(binom.sf(kk - 1, k['size'], 0.02).sum()); O = int((k['sum'] >= kk).sum()); ok = abs(O - E) <= 2 * math.sqrt(max(E, 1e-9)) + 1; fout += not ok
            regels.append(f"{kant:5s} rijders met >={kk} ritten kans<=1/50: {O:4d}, verwacht {E:6.1f}  {'ok' if ok else 'FOUT'}")
    return fout, regels, len(x), x['skater_id'].nunique()


def rijder(e, naam_deel, n=20):
    x = e[e['skater_name'].astype(str).str.contains(naam_deel, na=False)].sort_values('dag').tail(n)
    return x, int((x['p'] <= 0.01).sum()), int((x['p'] <= 0.05).sum()), int((x['p'] >= 0.99).sum()), int((x['p'] >= 0.95).sum())


def zeldzaam(e, naam_deel, n=20, grens=1 / 30):
    """Aantal ritten met kans <= grens (snel of traag) in de laatste n ritten van de rijder."""
    x = e[e['skater_name'].astype(str).str.contains(naam_deel, na=False)].sort_values('dag').tail(n)
    return int(((x['p'] <= grens) | (x['p'] >= 1 - grens)).sum()), len(x)


if __name__ == '__main__':
    t = time.time()
    df = draai(pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'ritten.parquet'))
    print('pass klaar', round(time.time() - t), 's', flush=True)
    elite = kies_elite(df); print('elite', len(elite), flush=True)
    e = ijk(df, elite)
    fout, regels, nr, nk = toets(e)
    print(f'toets elite seizoenen 2016+: {nr} ritten van {nk} rijders, {fout} fouten'); print('\n'.join(regels))
    for nm in ('Jutta LEERDAM', 'Francesca LOLLOBRIGIDA', 'Femke KOK', 'Jordan STOLZ'):
        x, a, b, c, d = rijder(e, nm)
        print(f"{nm}: laatste {len(x)} ritten: snel <=1/100 {a}, <=1/20 {b} | traag <=1/100 {c}, <=1/20 {d}  (verwacht 0,2 en 1)")
    e.to_parquet('elite_uitkomst.parquet')
