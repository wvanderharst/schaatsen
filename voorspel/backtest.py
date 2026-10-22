"""Backtest op de internationale top: afstand tot de top 5 met een logistische verdeling, tegen onze methode.

Internationale topraces (wereldbekers, WK's, EK's, Spelen; de data heeft geen toernooinamen): minstens N_ELITE rijders binnen 5% van de
wereldtop (niveau vooraf) uit minstens N_LAND landen. Per rit de referentie R = gemiddelde ln tijd van de 5 snelsten ZONDER de rijder
zelf (anders trekt de winnaar zijn eigen referentie omlaag), afstand tot de top g = ln tijd - R (negatief = sneller dan de top 5).

Maatstaf: per rijder x afstand een ligging en schaal uit EERDERE internationale races (uitdovend, halfwaardetijd 1 jaar, gekrompen naar
de nieuwkomers). Onze methode, met dezelfde informatie achteraf: e = (ln t - voorspelling) - gemiddelde (ln t - voorspelling) van
diezelfde 5 rijders, schaal uit onze rijderonzekerheid. Voor beide een gepoolde vorm, geschat op 2008-2019, getoetst op 2020+:
  - logistisch (symmetrisch);
  - scheef logistisch (type I gegeneraliseerd: F(z) = (1 + exp(-(z - a) / b))^-alpha). Die is gedefinieerd op alle reele getallen, dus
    negatieve afstanden (sneller dan de top 5) zijn geen probleem. Een verschoven log-logistische verdeling heeft wel een ondergrens, en een
    rit daaronder krijgt kans 0.
Log loss in ln-tijd-eenheden (lager = beter), direct vergelijkbaar; daarnaast onze gewone voorspelling vooraf (met de ijsonzekerheid).
Valpartijen (filtervlag) tellen niet mee. Gebruik: python backtest.py <werkmap>"""
import os, sys, importlib.util
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import minimize

N_ELITE, N_LAND, GRENS = 10, 5, 0.05
AF6 = ['500m', '1000m', '1500m', '3000m', '5000m', '10000m']
spec = importlib.util.spec_from_file_location('em', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite', 'elite_model.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)


def laad(w):
    kol = ['skater_id', 'skater_name', 'gender', 'dag', 'jaar', 'seizoen', 'baan', 'afstand', 'tijd', 'pred', 'pit', 'nll', 'ok', 'k_val',
           'k_c', 'rk_']
    pad = os.path.join(w, 'voorspel2_uitkomst.parquet'); namen = pq.read_schema(pad).names
    V = pq.read_table(pad, columns=kol + [c for c in ('sd_rijder', 'sd') if c in namen]).to_pandas()
    if 'sd_rijder' not in V:                              # oudere uitvoer: totale onzekerheid als benadering
        V['sd_rijder'] = V['sd']
    for c in ('afstand', 'baan', 'gender', 'skater_name'):
        V[c] = V[c].astype(str)
    V = V[V['afstand'].isin(AF6) & V['ok'] & np.isfinite(V['tijd']) & np.isfinite(V['pred'])].copy()
    L = pq.read_table(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land']).to_pandas().drop_duplicates('skater_id', keep='last')
    V = V.merge(L, on='skater_id', how='left')
    V['lt'] = np.log(V['tijd'])
    return V


def internationaal(V):
    el = V[V['k_c'] < GRENS]
    a = el.groupby('rk_').agg(n_el=('skater_id', 'size'), n_land=('land', 'nunique'))
    goed = a[(a['n_el'] >= N_ELITE) & (a['n_land'] >= N_LAND)].index
    return V[V['rk_'].isin(goed)].copy()


def referentie(I):
    """Per rit: gemiddelde ln tijd en gemiddelde residu (ln t - voorspelling) van de 5 snelsten zonder de rijder zelf."""
    I = I.sort_values(['rk_', 'lt'], kind='stable').reset_index(drop=True)
    I['res'] = I['lt'] - I['pred']; I['rang'] = I.groupby('rk_').cumcount()
    n = I.groupby('rk_')['lt'].transform('size'); I = I[n >= 6].copy()
    top6 = I[I['rang'] < 6]
    s5_lt = top6[top6['rang'] < 5].groupby('rk_')['lt'].sum(); s6_lt = top6.groupby('rk_')['lt'].sum()
    s5_r = top6[top6['rang'] < 5].groupby('rk_')['res'].sum(); s6_r = top6.groupby('rk_')['res'].sum()
    v5 = top6[top6['rang'] < 5].groupby('rk_')['sd_rijder'].apply(lambda x: (x ** 2).sum()); v6 = top6.groupby('rk_')['sd_rijder'].apply(lambda x: (x ** 2).sum())
    zelf = I['rang'] < 5
    I['R'] = np.where(zelf, (I['rk_'].map(s6_lt) - I['lt']) / 5, I['rk_'].map(s5_lt) / 5)
    I['Rres'] = np.where(zelf, (I['rk_'].map(s6_r) - I['res']) / 5, I['rk_'].map(s5_r) / 5)
    I['Rvar'] = np.where(zelf, (I['rk_'].map(v6) - I['sd_rijder'] ** 2) / 25, I['rk_'].map(v5) / 25)
    I['g'] = I['lt'] - I['R']
    return I


def _lopend(I, hl, k, prior, kolom_waarde, val):
    """Uitdovende som (halfwaardetijd hl dagen) over EERDERE dagen per rijder x afstand: geeft (som w, som w x) voor elke rit."""
    x = kolom_waarde; d = I['dag'].to_numpy().astype(float); key = (I['skater_id'].astype(str) + '|' + I['afstand']).to_numpy()
    Sw_o = np.zeros(len(I)); Sx_o = np.zeros(len(I)); i = 0; n = len(I)
    while i < n:
        j = i
        while j < n and key[j] == key[i]:
            j += 1
        Sw = Sx = 0.0; laatst = None; q = i
        while q < j:
            p = q
            while p < j and d[p] == d[q]:
                p += 1
            if laatst is not None:
                f = 0.5 ** ((d[q] - laatst) / hl); Sw *= f; Sx *= f
            Sw_o[q:p] = Sw; Sx_o[q:p] = Sx
            for t in range(q, p):
                if not val[t] and np.isfinite(x[t]):
                    Sw += 1.0; Sx += x[t]
            laatst = d[q]; q = p
        i = j
    return Sw_o, Sx_o


def maatstaf(I, hl=365.0, k=2.0, k_s=2.0, knip=0.04):
    """Ligging en schaal per rijder x afstand uit eerdere internationale races (alleen eerdere dagen):
    ligging = (som w g + k mu0) / (som w + k), mu0 = mediaan van nieuwkomers;
    spreiding binnen de rijder = wortel((som w r^2 + k_s s0^2) / (som w + k_s)), s0 = typische spreiding van ervaren rijders;
    voorspellende schaal = wortel(spreiding^2 (1 + 1 / (som w + k_mu))): de ligging zelf is onzeker zolang er weinig races zijn;
    k_mu = (s0 / tau0)^2 (Bayes, tau0 = spreiding tussen nieuwkomers)."""
    I = I.sort_values(['skater_id', 'afstand', 'dag'], kind='stable').reset_index(drop=True)
    val = I['k_val'].to_numpy(); g = I['g'].to_numpy()
    nieuw = I.groupby(['skater_id', 'afstand']).cumcount() == 0; fit = I['jaar'] <= 2019
    mu0 = I.loc[nieuw & fit, 'g'].median(); tau0 = 1.4826 * (I.loc[nieuw & fit, 'g'] - mu0).abs().median()
    Sw, Sg = _lopend(I, hl, k, mu0, g, val)
    mu = (Sg + k * mu0) / (Sw + k)
    r = np.clip(g - mu, -knip, knip)
    s0 = 1.4826 * np.median(np.abs(r[(Sw >= 5) & fit.to_numpy() & ~val]))
    k_mu = (s0 / tau0) ** 2                                # Bayes: priorgewicht = spreiding binnen / tussen rijders, in races
    mu = (Sg + k_mu * mu0) / (Sw + k_mu); r = np.clip(g - mu, -knip, knip)
    Sw2, Sr2 = _lopend(I, hl, k_s, s0, r ** 2, val)
    s_in = np.sqrt((Sr2 + k_s * s0 ** 2) / (Sw2 + k_s))
    I['mu_b'] = mu; I['s_b'] = np.sqrt(s_in ** 2 + s_in ** 2 / (Sw + k_mu))     # + onzekerheid over de ligging zelf
    print(f'maatstaf: nieuwkomers mediaan {100 * mu0:.2f}%, spreiding tussen nieuwkomers {100 * tau0:.2f}%, typische spreiding binnen een rijder {100 * s0:.2f}%')
    return I


def nll_vorm(par, z, skew):
    a, lb = par[0], par[1]; b = np.exp(lb); y = (z - a) / b
    if skew:
        al = np.exp(par[2])
        return -(np.log(al) - lb - y - (al + 1) * np.logaddexp(0, -y))
    return -(-lb - y - 2 * np.logaddexp(0, -y))


def cdf_vorm(par, z, skew):
    a, b = par[0], np.exp(par[1]); y = (z - a) / b
    return np.exp(-(np.exp(par[2]) if skew else 1.0) * np.logaddexp(0, -y))


def fit_vorm(z, skew):
    p0 = np.array([0.0, np.log(0.6)] + ([0.0] if skew else []))
    return minimize(lambda p: np.mean(nll_vorm(p, z, skew)), p0, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-9}).x


def kort(r):
    if 'rijders' in r:
        return r.split('met ')[1].split(' ritten')[0] + ':' + r.split(':')[1].split(',')[0].strip() + '/' + r.split('verwacht')[1].split()[0]
    return r.split('kans<=')[1].split(':')[0].strip() + ' ' + r.split('(')[1].split(')')[0]


def main(w):
    V = laad(w); I = maatstaf(referentie(internationaal(V)))
    I['z_b'] = (I['g'] - I['mu_b']) / I['s_b']
    I['e_o'] = I['res'] - I['Rres']; I['s_o'] = np.sqrt(I['sd_rijder'] ** 2 + I['Rvar']); I['z_o'] = I['e_o'] / I['s_o']
    fitm = (I['jaar'] <= 2019) & ~I['k_val']; toets = (I['jaar'] >= 2020) & ~I['k_val']
    print(f"internationale topraces: {I['rk_'].nunique()} races, {len(I)} ritten ({toets.sum()} in de toetsjaren);"
          f" afstand tot de top 5 (mediaan): {100 * I['g'].median():.2f}%, nieuwkomers {100 * I.loc[I.groupby(['skater_id', 'afstand']).cumcount() == 0, 'g'].median():.2f}%")
    res = {}
    for nm, zc, sc in (('maatstaf', 'z_b', 's_b'), ('onze methode', 'z_o', 's_o')):
        for skew in (False, True):
            p = fit_vorm(I.loc[fitm, zc].to_numpy(), skew)
            z = I[zc].to_numpy(); ll = nll_vorm(p, z, skew) + np.log(I[sc].to_numpy())
            lab = f"{nm}, {'scheef logistisch' if skew else 'logistisch'}"
            res[lab] = (ll, cdf_vorm(p, z, skew), p)
    res['onze methode vooraf (zonder top 5)'] = (I['nll'].to_numpy(), I['pit'].to_numpy(), None)
    top = toets & (I['k_c'] < GRENS)
    print('\nlog loss (ln tijd, lager = beter), toetsjaren 2020+:')
    for lab, (ll, pit, p) in res.items():
        extra = '' if p is None else '  vorm ' + ' '.join(f'{v:+.3f}' for v in p)
        print(f'  {lab:40s} alle {np.mean(ll[toets]):.4f} | binnen 5% van de top {np.mean(ll[top]):.4f}{extra}')
    print('\nkalibratie, toetsjaren, rijders binnen 5% van de top (snel: kans <= 1/1000, 1/100, 1/20; rijders met >= k snelle ritten <= 1/50):')
    for lab, (ll, pit, p) in res.items():
        x = I.loc[top, ['skater_id', 'seizoen']].copy(); x['p'] = pit[top.to_numpy()]
        fout, rg, _, _ = em.toets(x, vanaf_seizoen=2019)
        print(f'  {lab:40s} fouten {fout:2d} | snel ' + ' '.join(kort(r) for r in rg[:8]) + ' | traag ' + ' '.join(kort(r) for r in rg[8:11]))
    # voorbeelden
    print('\nvoorbeelden (kans op zo snel of sneller):')
    nm_b = 'maatstaf, scheef logistisch'; nm_o = 'onze methode, scheef logistisch'
    I['p_b'] = res[nm_b][1]; I['p_o'] = res[nm_o][1]
    for naam, af, datum in (('Jutta LEERDAM', '1000m', '2025-11-14'), ('Jutta LEERDAM', '1000m', '2026-02-09'), ('Femke KOK', '1000m', '2026-02-09'),
                            ('Francesca LOLLOBRIGIDA', '3000m', '2026-02-07')):
        dg = (pd.Timestamp(datum) - pd.Timestamp('1970-01-01')).days
        x = I[(I['skater_name'] == naam) & (I['afstand'] == af) & (I['dag'] == dg)]
        for _, r in x.iterrows():
            f = lambda p: f'1 op {1 / p:,.0f}'.replace(',', '.') if p < 0.5 else f'trager 1 op {1 / (1 - p):,.0f}'.replace(',', '.')
            print(f"  {naam} {af} {datum} {r['baan']}: afstand tot top 5 {100 * r['g']:+.2f}% (verwacht {100 * r['mu_b']:+.2f}%, schaal {100 * r['s_b']:.2f}%)"
                  f" | maatstaf {f(r['p_b'])} | onze methode met top 5 {f(r['p_o'])} | onze methode vooraf {f(r['pit'])}")
    I.to_parquet(os.path.join(w, 'backtest_uitkomst.parquet'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
