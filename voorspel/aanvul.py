"""Nieuwe wedstrijden (wekelijkse scraper, data/speedskating_nieuw.parquet) aan een bestaande werkmap toevoegen zonder het model opnieuw
te fitten: alle parameters en lagen blijven zoals ze zijn, alleen de nieuwe informatie komt erbij.
  1. Ritten: de nieuwe toernooien gaan door dezelfde opschoning als de historie (do_analysis.laad_dataset, met twee seizoenen eromheen als
     context voor de onmogelijk-snel-regels); geboortejaar uit de historie van de rijder, anders uit de categoriecode (zoals prep_cat.py).
  2. Racepeil theta van elke nieuwe race uit het veld, dag voor dag: het Kalmanfilter (kalman_mv2, vaste parameters) loopt over de
     geschiedenis van de rijders in de nieuwe races; per race is theta de gewogen mediaan van ln tijd - wereldtop - verwacht niveau (gewicht
     1 / (onzekerheid niveau + ritruis)). Minder dan 3 rijders met een niveau: het gewone peil van die baan (laatste vier seizoenen).
     Daarna tellen de ritten van die dag mee voor de niveaus van de volgende dagen. Universele baanfactor per rijder als in maak_baanfactor.py.
     Een hele race > 20% sneller dan het baanpeil, of een losse rit > 40% sneller dan verwacht, is verdacht (zoals onmogelijk.py).
  3. Uitvoer in een nieuwe werkmap (de oude blijft onaangetast; overige bestanden als koppeling): ritten_cat.parquet en
     voorspel2_uitkomst.parquet met de nieuwe ritten erbij (verwachting = filterniveau + racepeil, eigen spreiding van de laatste bekende rit),
     en talent_momentopnamen.parquet: alleen de rijders met nieuwe ritten krijgen nieuwe seizoensstanden (de rest blijft zoals in de basis).
     Het filter loopt alleen over de geschiedenis van die rijders; wie niet reed, kost niets.
Toets: --toets JJJJ-MM-DD behandelt de bestaande ritten vanaf die dag als nieuw en vergelijkt het geschatte racepeil met het opgeslagen.
Gebruik: python aanvul.py <werkmap> <datamap met speedskating_nieuw.parquet> <nieuwe werkmap> <kalman_par.json> [--toets JJJJ-MM-DD]"""
import os, sys, json, shutil, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'schaatsmodel'))
import numpy as np, pandas as pd
import kalman_mv2 as kf, baanfactor as bf, talent as tal
import time

RACE_GRENS, RIT_GRENS = 0.20, 0.40
NIEUW = ['speedskating_nieuw.parquet']
EXACT = {'YF': 7, 'YE': 8, 'YD': 9, 'YC': 10, 'YB': 11, 'YA': 12, 'C1': 13, 'C2': 14, 'B1': 15, 'B2': 16, 'A1': 17, 'A2': 18,
         'N1': 19, 'N2': 20, 'N3': 21, 'N4': 22}


def nieuwe_ritten(w, datamap, R):
    """Nieuwe toernooien uit de scraper, opgeschoond en in het formaat van ritten_cat.parquet (zonder theta)."""
    import do_analysis as da
    A = pd.read_parquet(os.path.join(datamap, 'speedskating_nieuw.parquet'))
    A = A[~A['Event_ID'].isin(set((R['event_id'] % 1000000).unique()))]
    if not len(A):
        return None
    tmp = os.path.join(w + '_aanvul_tmp'); os.makedirs(tmp, exist_ok=True)
    if os.path.exists(os.path.join(w, 'uitgesloten_races.csv')):
        shutil.copy(os.path.join(w, 'uitgesloten_races.csv'), tmp)
    bron = os.path.join(w, 'speedskating_historie_racedatum.csv')
    kop = pd.read_csv(bron, nrows=0).columns.tolist()
    ctx = pd.concat([c[pd.to_datetime(c['Datum'], format='%d/%m/%Y', errors='coerce') >= pd.Timestamp('2024-07-01')]
                     for c in pd.read_csv(bron, chunksize=500000, dtype=str)], ignore_index=True)
    A = A.copy()
    for c in ('Datum', 'Race_Datum'):
        A[c] = pd.to_datetime(A[c]).dt.strftime('%d/%m/%Y')
    A['Tijd_Raw'] = A['Tijd_Raw'].astype(str)
    pad = os.path.join(tmp, 'aanvul.csv')
    pd.concat([ctx, A[[k for k in kop if k in A]].astype(str)], ignore_index=True)[kop].to_csv(pad, index=False)
    df = da.laad_dataset(pad, verbose=False)
    shutil.rmtree(tmp)
    df = df[df['toernooi_id'].isin(set(A['Event_ID']))].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd'])
    df = df[['skater_id', 'skater_name', 'gender', 'land', 'groep', 'afstand', 'afstand_idx', 'dag', 'jaar', 'seizoen', 'event_id', 'baan',
             'tijd', 'verdacht', 'categorie']].copy()
    for c in ('skater_name', 'afstand', 'categorie', 'baan', 'land', 'groep', 'gender'):
        df[c] = df[c].astype(str)
    # geboortejaar: uit de historie, anders uit de categoriecode
    hist = R.drop_duplicates('skater_id').set_index('skater_id')[['geboren', 'leeftijd_exact']]
    code = df['categorie'].str[1:]
    s = df['seizoen'].astype(float)
    ex = code.isin(list(EXACT))
    lo = code.map({**EXACT, 'SA': 23, **{str(a): a for a in range(30, 95, 5)}}).astype(float)
    hi = code.map({**EXACT, 'SA': 29, **{str(a): a + 4 for a in range(30, 95, 5)}}).astype(float)
    b_ex = (s - lo)[ex].groupby(df.loc[ex, 'skater_id']).median()
    b_mid = ((s - hi + s - lo) / 2).groupby(df['skater_id']).median()
    geb = df['skater_id'].map(hist['geboren'])
    df['geboren'] = geb.fillna(df['skater_id'].map(b_ex)).fillna(df['skater_id'].map(b_mid))
    df['leeftijd_exact'] = df['skater_id'].map(hist['leeftijd_exact']).fillna(df['skater_id'].isin(b_ex.index)).astype(bool)
    df['leeftijd'] = df['seizoen'] - df['geboren']
    return df.reset_index(drop=True)


def invoer(D, refq_tab):
    """Als kalman.voorbereid, maar met de wereldtop-referentie van de hele basis (refq_tab) i.p.v. van de deelverzameling."""
    D = D.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    rk = D.groupby(['dag', 'event_id', 'gender', 'afstand_idx'], sort=False).ngroup().to_numpy().astype(np.int64)
    rid, _ = pd.factorize(D['skater_id'])
    refq = np.array([refq_tab.get((g, k), np.nan) for g, k in zip(D['gender'], D['afstand_idx'])])
    lf = (D['leeftijd'] + ((D['dag'] - (D['seizoen'] - 1970) * 365.25 - 182) / 365.25).clip(0, 1)).to_numpy(float)
    inv = dict(rk=rk, rid=rid.astype(np.int64), a=D['afstand_idx'].to_numpy().astype(np.int64), z=np.full(len(D), np.nan),
               vth=np.nan_to_num(D['v_theta'].to_numpy(float)), dag=D['dag'].to_numpy(float), lf=lf, nr=int(rid.max() + 1), refq=refq,
               seiz=D['seizoen'].to_numpy().astype(np.int64))
    return D, inv


def gewogen_mediaan(x, w):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w)
    return x[np.searchsorted(c, 0.5 * c[-1])]


def schat_peil(R, N, par):
    """Racepeil per nieuwe race, dag voor dag. R: basis (met theta), N: nieuwe ritten (zonder theta). Geeft N met theta, v_theta,
    theta_rijder, verdacht bijgewerkt en de filteruitvoer per nieuwe rit (niveau vooraf c, verwachting zh, onzekerheid, val)."""
    lt_b = np.log(R['tijd'].to_numpy(float)); fit = (R['jaar'] <= 2019).to_numpy() & np.isfinite(R['theta'].to_numpy())
    refq_tab = pd.Series((lt_b - R['theta'].to_numpy())[fit]).groupby([R.loc[fit, 'gender'].to_numpy(), R.loc[fit, 'afstand_idx'].to_numpy()]).quantile(0.01)
    hv = bf.hv_peil(R); bpar = json.load(open(BF_PAR)) if os.path.exists(BF_PAR) else None
    # gewoon peil per baan x geslacht x afstand (laatste vier seizoenen, >= 3 races), anders per baan, anders Heerenveen
    rr = R[R['seizoen'] >= R['seizoen'].max() - 4].drop_duplicates(['event_id', 'dag', 'gender', 'afstand_idx'])
    gb = rr.groupby(['baan', 'gender', 'afstand_idx'])['theta'].agg(['median', 'size']); gb = gb[gb['size'] >= 3]['median']
    gbaan = (rr['theta'] - rr.set_index(['gender', 'afstand_idx']).index.map(hv).to_numpy(float)).groupby(rr['baan']).median()
    N = N.copy(); N['theta'] = np.nan; N['v_theta'] = 0.0; N['theta_rijder'] = np.nan; N['_nieuw'] = True
    keep = set(N['skater_id'])
    B = R[R['skater_id'].isin(keep)].copy(); B['_nieuw'] = False
    D, inv = invoer(pd.concat([B, N], ignore_index=True), refq_tab)
    lt = np.log(D['tijd'].to_numpy(float)); nieuw = D['_nieuw'].to_numpy()
    verd = D['verdacht'].to_numpy().copy()
    th_r = D['theta_rijder'].to_numpy(float).copy()
    z = np.where(verd | nieuw, np.nan, lt - th_r - inv['refq'])
    th = D['theta'].to_numpy(float).copy(); vth = inv['vth'].copy()
    uit = {k: np.full(len(D), np.nan) for k in ('c', 'zh', 'vl', 'r')}; uit['val'] = np.zeros(len(D), bool)
    for d in np.unique(inv['dag'][nieuw]):
        inv['z'] = z; inv['vth'] = vth
        o = kf.pass_(inv, par)
        dd = np.flatnonzero(nieuw & (inv['dag'] == d))
        for k in ('c', 'zh', 'vl', 'r'):
            uit[k][dd] = o[k][dd]
        for r_ in np.unique(inv['rk'][dd]):
            q = dd[inv['rk'][dd] == r_]
            e = lt[q] - inv['refq'][q] - o['zh'][q]
            ok = np.isfinite(e) & ~verd[q]
            g_, a_, b_ = D.at[q[0], 'gender'], int(D.at[q[0], 'afstand_idx']), D.at[q[0], 'baan']
            prior = gb.get((b_, g_, a_), hv.get((g_, a_), 0.0) + gbaan.get(b_, 0.0))
            if ok.sum() >= 3:
                wv = 1.0 / np.maximum(o['vl'][q][ok] + o['r'][q][ok], 1e-6)
                t_ = gewogen_mediaan(e[ok], wv)
                mad = 1.4826 * np.median(np.abs(e[ok] - t_))
                v_ = max(mad, 0.005) ** 2 * (np.pi / 2) / ok.sum()
            else:
                t_, v_ = prior, 0.01 ** 2
            th[q] = t_; vth[q] = v_
            if t_ - prior < -RACE_GRENS:                                  # hele race onmogelijk snel: verkeerde afstand of tijdwaarneming
                verd[q] = True
            verd[q] |= np.isfinite(e) & (e - t_ < -RIT_GRENS)
            # universele baanfactor per rijder (maak_baanfactor.py)
            dlt = t_ - hv.get((g_, a_), np.nan)
            if bpar is not None and np.isfinite(dlt):
                g = o['c'][q]; wv_ = 1 / np.maximum(o['vl'][q] + o['r'][q], 1e-6)
                okg = np.isfinite(g)
                gbar = np.sum(np.clip(g[okg], 0, 0.5) * wv_[okg]) / np.sum(wv_[okg]) if okg.any() else bf.G0
                gg = np.where(okg, g, gbar)
                th_r[q] = bf.factor(np.full(len(q), t_), np.full(len(q), dlt), gg, np.full(len(q), gbar), bpar['A'], bpar['TM'], bpar['B'], bpar['TB'])
            else:
                th_r[q] = t_
            z[q] = np.where(verd[q], np.nan, lt[q] - th_r[q] - inv['refq'][q])
    D['theta'] = th; D['v_theta'] = vth; D['theta_rijder'] = th_r; D['verdacht'] = verd
    D['refq'] = inv['refq']
    for k, v in uit.items():
        D['_' + k] = v
    # seizoensstanden (momentopnamen) van alleen deze rijders; wie niet reed, houdt de stand van de basis
    inv['z'] = z; inv['vth'] = vth; o = kf.pass_(inv, par, snap=True); s = o['snap']
    D['_val'] = np.where(nieuw, o['val'], False)                         # valpartijen volgens het filter (achteraf, met het peil)
    _, uniq = pd.factorize(D['skater_id'])
    S = pd.DataFrame({'skater_id': uniq[s['rid']], 'seizoen': s['seizoen'], 'n_seizoen': s['n'], 'lf': s['lf'], 'dag': s['dag']})
    for j, a in enumerate(tal.AF):
        S['x_' + a] = s['x'][:, j]; S['p_' + a] = s['p'][:, j]; S['na_' + a] = s['na'][:, j]
    S['v'] = s['x'][:, 8]; S['pv'] = s['p'][:, 8]
    info = D.sort_values('dag').groupby('skater_id').agg(naam=('skater_name', 'last'), gender=('gender', 'last'), land=('land', lambda x: x.mode().iloc[0]),
                                                          geboren=('geboren', 'last'), exact=('leeftijd_exact', 'last'), eerste=('seizoen', 'min'))
    S = tal.niveaus(S.join(info, on='skater_id').sort_values(['skater_id', 'seizoen']))
    S['leeftijd'] = S['seizoen'] - S['geboren']
    vorig = S.groupby('skater_id')[['L_jeugd', 'seizoen']].shift(1)                 # zoals talent.kenmerken
    S['verbetering'] = np.where(vorig['seizoen'] == S['seizoen'] - 1, vorig['L_jeugd'] - S['L_jeugd'], np.nan)
    return D[nieuw].drop(columns=['_nieuw']).reset_index(drop=True), S


def uitkomst_rijen(N, V):
    """Rijen in het formaat van voorspel2_uitkomst.parquet voor de nieuwe ritten (alleen wat de site en rijderschok gebruiken)."""
    sdr = V.sort_values('dag').groupby('skater_id')['sd_rijder'].last()
    X = pd.DataFrame({'skater_id': N['skater_id'], 'skater_name': N['skater_name'], 'gender': N['gender'], 'groep': N['groep'], 'dag': N['dag'],
                      'jaar': N['jaar'], 'seizoen': N['seizoen'], 'baan': N['baan'], 'afstand': N['afstand'], 'tijd': N['tijd'],
                      'pred': N['refq'] + N['_zh'] + N['theta_rijder'], 'theta': N['theta'], 'k_c': N['_c'], 'k_val': N['_val'].astype(bool),
                      'sd': np.sqrt(N['_vl'] + N['_r'] + N['v_theta']), 'sd_rijder': N['skater_id'].map(sdr)})
    X['ok'] = np.isfinite(X['pred']) & ~N['verdacht'].to_numpy()
    return X


def main(w, datamap, w2, par_pad, toets=None):
    par = json.load(open(par_pad))
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'))
    for c in ('afstand', 'skater_name', 'categorie', 'baan', 'land', 'groep', 'gender'):
        R[c] = R[c].astype(str)
    if toets:
        d0 = int((pd.Timestamp(toets) - pd.Timestamp('1970-01-01')).days)
        N = R[R['dag'] >= d0].drop(columns=['theta', 'v_theta', 'theta_rijder']); R = R[R['dag'] < d0]
        Y, _ = schat_peil(R, N, par)
        X = Y.merge(pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'theta']).astype({'afstand': str}),
                    on=['skater_id', 'dag', 'afstand', 'tijd'], suffixes=('', '_oud'))
        rc = X.drop_duplicates(['event_id', 'dag', 'gender', 'afstand_idx'])
        dif = 100 * (rc['theta'] - rc['theta_oud'])
        print(f'toets vanaf {toets}: {len(rc)} races, {len(X)} ritten | racepeil nieuw - opgeslagen (%): mediaan {dif.median():+.3f},'
              f' mediaan |verschil| {dif.abs().median():.3f}, p90 |verschil| {dif.abs().quantile(0.9):.3f}, sd {dif.std():.3f}')
        n_ = X.groupby(['event_id', 'dag', 'gender', 'afstand_idx'])['tijd'].transform('size')
        for lo_, hi_ in ((1, 5), (5, 15), (15, 1000)):
            m = (n_ >= lo_) & (n_ < hi_); r_ = X[m].drop_duplicates(['event_id', 'dag', 'gender', 'afstand_idx'])
            if len(r_):
                print(f'  races met {lo_}-{hi_ - 1} rijders: {len(r_)}, mediaan |verschil| {100 * (r_["theta"] - r_["theta_oud"]).abs().median():.3f}%')
        return
    N = nieuwe_ritten(w, datamap, R)
    if N is None:
        print('geen nieuwe toernooien'); return
    print(f'nieuwe ritten {len(N)}, rijders {N["skater_id"].nunique()} (nieuw {(~N["skater_id"].isin(set(R["skater_id"]))).sum()} ritten van onbekende rijders),'
          f' {N["dag"].nunique()} dagen', flush=True)
    t0 = time.time(); N, S = schat_peil(R, N, par)
    print(f'racepeil geschat ({time.time() - t0:.0f} s); verdacht {int(N["verdacht"].sum())}, val {int(N["_val"].sum())}', flush=True)
    os.makedirs(w2, exist_ok=True)
    for f in os.listdir(w):                                               # overige bestanden als koppeling
        doel = os.path.join(w2, f)
        if f not in ('ritten_cat.parquet', 'voorspel2_uitkomst.parquet', 'talent_momentopnamen.parquet') and not os.path.lexists(doel):
            os.symlink(os.path.realpath(os.path.join(w, f)), doel)
    kol = list(R.columns)
    R2 = pd.concat([R, N[kol]], ignore_index=True)
    R2['skater_name'] = R2['skater_name'].astype('category')
    R2.to_parquet(os.path.join(w2, 'ritten_cat.parquet'))
    V = pd.read_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'))
    X = uitkomst_rijen(N, V)
    for c in V.columns:
        if c not in X:
            X[c] = np.nan
    for c in ('afstand', 'baan', 'gender', 'skater_name', 'groep'):
        V[c] = V[c].astype(str)
    V2 = pd.concat([V, X[V.columns]], ignore_index=True)
    for c in ('ok', 'k_val'):
        V2[c] = V2[c].astype(bool)
    V2.to_parquet(os.path.join(w2, 'voorspel2_uitkomst.parquet')); del V, V2
    print('ritten en uitkomst geschreven', flush=True)
    # seizoensstanden: de rijders met nieuwe ritten vanaf het eerste seizoen met nieuwe ritten; de rest blijft
    M = pd.read_parquet(os.path.join(w, 'talent_momentopnamen.parquet'))
    s0 = int(N['seizoen'].min()); ids = set(N['skater_id'])
    weg = M['skater_id'].isin(ids) & (M['seizoen'] >= s0)
    M2 = pd.concat([M[~weg], S[S['skater_id'].isin(ids) & (S['seizoen'] >= s0)][[c for c in M.columns if c in S.columns]]], ignore_index=True)
    M2.to_parquet(os.path.join(w2, 'talent_momentopnamen.parquet'))
    oud = M[M['seizoen'] == s0 - 1].set_index('skater_id')['x_500m']; nw = S[S['seizoen'] == s0 - 1].set_index('skater_id')['x_500m']
    vg = (oud - nw.reindex(oud.index)).abs().dropna()
    print(f'momentopnamen: {int((~weg).sum())} ongewijzigd, {len(M2) - int((~weg).sum())} opnieuw (seizoen {s0}+);'
          f' controle seizoen {s0 - 1} 500 m oud vs nieuw: mediaan |verschil| {vg.median():.2e} ({time.time() - t0:.0f} s)', flush=True)

BF_PAR = ''
if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    BF_PAR = os.path.join(a[0], 'baanfactor_par.json')
    t = sys.argv[sys.argv.index('--toets') + 1] if '--toets' in sys.argv else None
    if t:
        a = [x for x in a if x != t]
    main(a[0], a[1], a[2], a[3], toets=t)
