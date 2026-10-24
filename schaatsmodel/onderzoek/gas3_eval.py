import warnings; warnings.filterwarnings('ignore')
import glob, numpy as np, pandas as pd
d = pd.read_pickle('gas_d.pkl')[['ev', 'jr', 'abs', 'val', 'np', 'u']]
R = {k: pd.read_pickle(f'gas3_res_{k}.pkl') for k in ['basis', 'S', 'J'] if glob.glob(f'gas3_res_{k}.pkl')}
base = R['basis']
idx = base.index[(d.loc[base.index, 'jr'] >= 2020) & ~d.loc[base.index, 'val']]
ev = d.loc[idx, 'ev'].to_numpy(); top = (d.loc[idx, 'abs'] >= 90).to_numpy(); vast = (d.loc[idx, 'np'] >= 20).to_numpy()
rng = np.random.default_rng(0)


def se(x, e):
    g = pd.DataFrame({'x': x, 'e': e}).groupby('e').x.agg(['sum', 'size']); b = []
    for _ in range(200):
        k = rng.integers(0, len(g), len(g)); b.append(g['sum'].to_numpy()[k].sum() / g['size'].to_numpy()[k].sum())
    return np.std(b)


naam = {'basis': 'geen GAS', 'S': 'symmetrische volatiliteit', 'J': 'asymmetrische volatiliteit (snel/traag)'}
print('Test 2020+, valpartijen eruit; log loss t.o.v. geen GAS (negatief = beter)')
for k, r in R.items():
    dd = -(r.loc[idx, 'lp'].to_numpy() - base.loc[idx, 'lp'].to_numpy()); z = r.loc[idx, 'zN'].to_numpy()
    print(f"  {naam[k]:42s} {dd.mean():+.4f} ± {se(dd, ev):.4f} | toppers {dd[top].mean():+.4f} ± {se(dd[top], ev[top]):.4f} | "
          f"|z|>4 vaste rijders {100*(np.abs(z[vast]) > 4).mean():.4f}%")
if 'S' in R and 'J' in R:
    dd = -(R['J'].loc[idx, 'lp'].to_numpy() - R['S'].loc[idx, 'lp'].to_numpy())
    print(f"  asymmetrisch t.o.v. symmetrisch: {dd.mean():+.4f} ± {se(dd, ev):.4f} | toppers {dd[top].mean():+.4f} ± {se(dd[top], ev[top]):.4f}")

# kalibratie per volatiliteitsgroep (toppers, test): na standaardisatie met de eigen schaal moeten de kwantielen gelijk zijn
for k in [k for k in ('S', 'J') if k in R]:
    r = R[k].loc[idx]; dd = d.loc[idx]
    x = dd.u.to_numpy() - r.m.to_numpy()
    es = np.where(x < 0, x / np.exp(r.lf.to_numpy()), x / np.exp(r.ls.to_numpy()))
    T = pd.DataFrame({'pit': r.pit.to_numpy(), 'lf': r.lf.to_numpy(), 'ls': r.ls.to_numpy()})[top]
    print(f'\n{naam[k]} - toppers per groep van eigen SNELLE volatiliteit (verwacht: 10% onder P10, 5% onder P5)')
    T['bin'] = pd.qcut(T.lf.rank(method='first'), 5, labels=['laagst', '2', '3', '4', 'hoogst'])
    print(T.groupby('bin').agg(schaal=('lf', lambda s: np.exp(s).median()), onder_P5=('pit', lambda s: 100 * (s < .05).mean()),
                               onder_P10=('pit', lambda s: 100 * (s < .10).mean())).round(2).to_string())
    print(f'{naam[k]} - toppers per groep van eigen TRAGE volatiliteit (verwacht: 10% boven P90, 5% boven P95)')
    T['bin'] = pd.qcut(T.ls.rank(method='first'), 5, labels=['laagst', '2', '3', '4', 'hoogst'])
    print(T.groupby('bin').agg(schaal=('ls', lambda s: np.exp(s).median()), boven_P90=('pit', lambda s: 100 * (s > .90).mean()),
                               boven_P95=('pit', lambda s: 100 * (s > .95).mean())).round(2).to_string())
