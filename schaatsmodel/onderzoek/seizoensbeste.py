import numpy as np, pandas as pd, sys
from scipy.stats import norm
ZB = pd.read_pickle('zbase.pkl'); zc = pd.read_pickle(sys.argv[1] if len(sys.argv) > 1 else 'normaal_z_dag.pkl').reindex(ZB.index).astype(float).to_numpy()
niv = pd.read_pickle('gas_d.pkl')['abs'].reindex(ZB.index).to_numpy()
dt = pd.to_datetime(ZB.dag.to_numpy(), unit='D'); seiz = np.where(dt.month >= 7, dt.year, dt.year - 1)
ok = np.isfinite(zc) & (zc < 6); jr = ZB.jaar.to_numpy()
for nm, sel in (('2010-2019', ok & (jr >= 2010) & (jr <= 2019)), ('2020+', ok & (jr >= 2020))):
    d = pd.DataFrame({'sk': ZB.skater_id.to_numpy(), 's': seiz, 'niv': niv, 'z': zc})[sel]
    S = d.groupby(['sk', 's']).agg(n=('z', 'size'), zmin=('z', 'min'), niv=('niv', 'median')).reset_index()
    S = S[S.n >= 8]; S['pit'] = 1 - (1 - norm.cdf(S.zmin)) ** S.n
    S = S.sort_values(['sk', 's']); nx = S.groupby('sk').shift(-1); okk = nx.s == S.s + 1
    T = S[okk & (S.niv >= 90)]; v = nx.pit[okk & (S.niv >= 90)]
    r = lambda x: f"{100*(x<.05).mean():.1f}% / {100*(x<.10).mean():.1f}%"
    print(f"{nm}: alle {r(S.pit)} | toppers {r(S.pit[S.niv>=90])} | elite {r(S.pit[S.niv>=97])} | toppers na extreem seizoen {r(v[T.pit<.05])} | na gewoon seizoen {r(v[T.pit>.5])}")
