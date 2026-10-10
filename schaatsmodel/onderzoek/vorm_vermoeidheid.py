import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, do_analysis as da
M = {'100m': 100, '200m': 200, '300m': 300, '400m': 400, '500m': 500, '700m': 700, '1000m': 1000, '1500m': 1500, '3000m': 3000, '5000m': 5000, '10000m': 10000}
o = pd.read_pickle('vv_rijen.pkl')
o = o[np.isfinite(o.u) & (o.u.abs() < 6) & (o.n_prior >= 3) & o.afstand.isin(M)][['event_id','skater_id','gender','afstand','rdag','u','niv','thdev']].copy()
o['m'] = o.afstand.map(M).astype(np.int32); o['afstand'] = o.afstand.astype('category'); o['gender'] = o.gender.astype('category')
for c in ('u','niv','thdev'): o[c] = o[c].astype(np.float32)
# veld-relatief: centreren binnen dezelfde race (event x geslacht x afstand x dag)
g = o.groupby(['event_id', 'gender', 'afstand', 'rdag'])
o['nrace'] = g.u.transform('size'); o = o[o.nrace >= 5]
o['ud'] = o.u - o.groupby(['event_id', 'gender', 'afstand', 'rdag']).u.transform('median')
# volgorde binnen het toernooi per rijder: dag, dan kortste afstand eerst
o = o.sort_values(['skater_id', 'event_id', 'rdag', 'm'])
k = o.groupby(['skater_id', 'event_id'])
o['pos'] = k.cumcount(); o['nev'] = k.u.transform('size')
o['dagnr'] = o.rdag - k.rdag.transform('min')
o['m_eerder_dag'] = o.groupby(['skater_id', 'event_id', 'rdag']).m.cumsum() - o.m
o['n_eerder_dag'] = o.groupby(['skater_id', 'event_id', 'rdag']).cumcount()
key = o.skater_id.astype(np.int64) * 100000 + o.rdag.astype(np.int64)
dagm = o.m.groupby(key.to_numpy()).sum()
o['m_gisteren'] = pd.Series((key - 1).to_numpy()).map(dagm).fillna(0).to_numpy()
# meerkamp-type per rijder x toernooi: multiset van afstanden, verspreid over 2 dagen
IDX = {a: i for i, a in enumerate(M)}
o['_c'] = 10 ** o.afstand.map(IDX).astype(np.int64)
code = o.groupby(['skater_id', 'event_id'])['_c'].transform('sum')
def _code(s):
    return sum(10 ** IDX[a] for a in s.split('/'))
ndag = k.rdag.transform('nunique')
TYPES = {'500m/500m/1000m/1000m': 'sprint', '500m/1500m/3000m/5000m': 'klein allround', '500m/1500m/5000m/10000m': 'groot allround',
         '500m/1000m/1500m/3000m': 'junior allround', '500m/500m/1000m/1000m/1500m': 'sprint+1500', '100m/300m/500m': 'jeugd 100/300/500',
         '300m/500m/1000m': 'pupillen 300/500/1000', '500m/1000m/1500m': 'mini allround 500/1000/1500', '500m/1500m/3000m': 'mini allround 500/1500/3000'}
TC = {_code(c): t for c, t in TYPES.items()}
o['type'] = np.where(ndag == 2, code.map(TC).fillna('overig 2 dagen'), np.where(o.nev == 1, 'enkele race', np.where(ndag == 1, '1 dag, meerdere races', 'overig')))
TOP = o.niv >= 90

print('=== 1. VORM: verrassing (t.o.v. veld) van een race tegen die van de vorige race van dezelfde rijder in hetzelfde toernooi ===')
o['ud_vorig'] = k.ud.shift(1); o['gap'] = o.rdag - k.rdag.shift(1)
p = o.dropna(subset=['ud_vorig'])
p['gb'] = pd.cut(p.gap, [-1, 0, 1, 2, 4, 9, 400], labels=['zelfde dag', '1 dag', '2 dagen', '3-4 d', '5-9 d', '10+ d'])
def helling(d):
    if len(d) < 200: return pd.Series({'n': len(d), 'helling': np.nan})
    x = d.ud_vorig.clip(-4, 4); y = d.ud.clip(-4, 4); return pd.Series({'n': len(d), 'helling': np.cov(x, y)[0, 1] / x.var()})
for nm, sel in (('alle', p.niv >= 0), ('toppers', p.niv >= 90), ('elite', p.niv >= 97)):
    print(nm, p[sel].groupby('gb').apply(helling).round(3).T.to_string(), sep='\n')
print('\nper meerkamp (dag 2 tegen dag 1, gemiddelde verrassing per dag):')
mk = o[o.type.isin(set(TYPES.values()))]
dd = mk[mk.dagnr <= 1].groupby(['type', 'skater_id', 'event_id', 'dagnr']).ud.mean().unstack('dagnr')[[0, 1]].dropna()
r = dd.groupby(level='type').apply(lambda d: pd.Series({'n': len(d), 'helling dag1->dag2': np.cov(d[0].clip(-4, 4), d[1].clip(-4, 4))[0, 1] / d[0].clip(-4, 4).var()}))
print(r.round(3).to_string())

print('\n=== 2. VERMOEIDHEID t.o.v. het veld (rijders in DEZELFDE race met en zonder eerdere belasting) ===')
x = o[~o.type.isin(set(TYPES.values()))].copy()
x['belasting_dag'] = pd.cut(x.m_eerder_dag, [-1, 0, 1000, 1500, 3000, 20000], labels=['geen', '≤1000 m', '1500 m', '3000 m', '≥5000 m'])
for nm, sel in (('alle', x.niv >= 0), ('toppers', x.niv >= 90)):
    t = x[sel].groupby(['afstand', 'belasting_dag']).ud.agg(['mean', 'size']).unstack('belasting_dag')
    t = t[[c for c in t.columns if c[0] == 'mean']].round(2); t.columns = [c[1] for c in t.columns]
    print(nm, '- gem. verrassing (sd, + = trager) per afstand x eerder gereden meters dezelfde dag'); print(t.loc[['500m', '1000m', '1500m', '3000m', '5000m']].to_string())
x['gisteren'] = pd.cut(x.m_gisteren, [-1, 0, 1500, 5000, 30000], labels=['geen', '≤1500 m', '1500-5000 m', '>5000 m'])
print('\nalle - gem. verrassing per eerder gereden meters de dag ervoor (geen race eerder dezelfde dag)')
print(x[x.m_eerder_dag == 0].groupby(['afstand', 'gisteren']).ud.mean().unstack().round(2).loc[['500m', '1000m', '1500m', '3000m', '5000m']].to_string())

print('\n=== 3. VERMOEIDHEID in meerkampen (hele veld): dag-effect t.o.v. de baan, % van de tijd, vs dezelfde afstand in enkele races ===')
base = o[o.type == 'enkele race'].groupby(['gender', 'afstand'], observed=True).thdev.median()
mk = mk.drop_duplicates(['event_id', 'gender', 'afstand', 'rdag'])
mk['rel'] = mk.thdev - pd.Series(list(zip(mk.gender.astype(str), mk.afstand.astype(str)))).map({(a, b): v for (a, b), v in base.items()}).to_numpy()
mk['plek'] = 'dag ' + (mk.dagnr + 1).astype(str) + ' · ' + mk.afstand.astype(str)
mk = mk[mk.dagnr <= 1]
print(mk.groupby(['type', 'plek']).rel.agg(['median', 'size']).round(2).to_string())
