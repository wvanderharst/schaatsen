"""Snelheidsverbetering door baan en materiaal: het peil van een race (theta, uit het veld) vangt alles op wat het HELE veld tegelijk sneller
maakt (ijs, baan, weer, materiaal); het niveau van de rijders (Kalman) vangt hun eigen vooruitgang. Hier: hoe veranderde het peil over de
seizoenen? Per race (event, dag, geslacht, afstand, >= 8 rijders) het peil; regressie peil = vast effect baan x geslacht x afstand
+ seizoenseffect (gemeenschappelijk: materiaal en algemene ijsverbetering) + per overdekte hoofdbaan een eigen afwijking per seizoen.
Ook: de top-10 van de senioren (rijdersniveau) per seizoen ter vergelijking. Gebruik: python baan_materiaal.py <werkmap>"""
import os, sys
import numpy as np, pandas as pd



def era(w, R=None):
    """Per seizoen: ln-verschil van het peil van een normale dag in Heerenveen t.o.v. het laatste seizoen (gemeenschappelijk effect van
    baan en materiaal + de eigen afwijking van Heerenveen), glad (lopend gemiddelde over 3 seizoenen). > 0: toen trager."""
    if R is None:
        R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['gender', 'afstand', 'dag', 'seizoen', 'event_id', 'baan', 'theta', 'verdacht'])
    R = R[~R['verdacht'] & R['afstand'].astype(str).isin(['500m', '1000m', '1500m', '3000m', '5000m'])].copy()
    for c in ('gender', 'afstand', 'baan'):
        R[c] = R[c].astype(str)
    rk = ['event_id', 'dag', 'gender', 'afstand']
    X = R.groupby(rk).agg(theta=('theta', 'median'), n=('theta', 'size'), baan=('baan', 'first'), seizoen=('seizoen', 'first')).reset_index()
    laatste = int(X['seizoen'].max())
    X = X[(X['n'] >= 8) & X['seizoen'].between(2007, laatste)]
    X['cel'] = X['baan'] + '|' + X['gender'] + '|' + X['afstand']
    X = X[X.groupby('cel')['theta'].transform('size') >= 20].copy()
    X['b'] = 0.0
    for _ in range(50):
        X['a'] = (X['theta'] - X['b']).groupby(X['cel']).transform('mean')
        b = (X['theta'] - X['a']).groupby(X['seizoen']).mean(); X['b'] = X['seizoen'].map(b)
    X['res'] = X['theta'] - X['a'] - X['b']
    hv = X[X['baan'].str.startswith('Heerenveen')].groupby('seizoen')['res'].mean().reindex(b.index).fillna(0.0)
    vol = X.groupby('seizoen').size() >= 1000                   # alleen volledige seizoenen; het lopende seizoen = het laatste volledige
    e = (b + hv)[vol].rolling(3, center=True, min_periods=1).mean()
    e = e - e.iloc[-1]
    return {**e.to_dict(), **{int(k): 0.0 for k in b.index if k > e.index.max()}}


if __name__ == '__main__':
    w = sys.argv[1]
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['gender', 'afstand', 'dag', 'seizoen', 'event_id', 'baan', 'theta', 'verdacht'])
    for c in ('gender', 'afstand', 'baan'):
        R[c] = R[c].astype(str)
    R = R[~R['verdacht'] & R['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m'])]
    rk = ['event_id', 'dag', 'gender', 'afstand']
    X = R.groupby(rk).agg(theta=('theta', 'median'), n=('theta', 'size'), baan=('baan', 'first'), seizoen=('seizoen', 'first')).reset_index()
    X = X[(X['n'] >= 8) & X['seizoen'].between(2006, 2025)]
    X['cel'] = X['baan'] + '|' + X['gender'] + '|' + X['afstand']
    X = X[X.groupby('cel')['theta'].transform('size') >= 20]
    # vaste effecten: cel + seizoen (afwisselend middelen, ~ kleinste kwadraten met twee factoren)
    X['a'] = 0.0; X['b'] = 0.0
    for _ in range(50):
        X['a'] = X.groupby('cel')['theta'].transform('mean') - X.groupby('cel')['b'].transform('mean') * 0
        X['a'] = (X['theta'] - X['b']).groupby(X['cel']).transform('mean')
        b = (X['theta'] - X['a']).groupby(X['seizoen']).mean(); b -= b.loc[2025]
        X['b'] = X['seizoen'].map(b)
    print('gemeenschappelijk seizoenseffect op het peil (alle banen; % sneller dan 2025-26 = negatief), seizoen: effect (aantal races)')
    cnt = X.groupby('seizoen').size()
    print('  ' + ' '.join(f'{s}:{100 * (np.exp(v) - 1):+.2f}%({cnt[s]})' for s, v in b.items()))
    X['res'] = X['theta'] - X['a'] - X['b']
    HOOFD = ['Heerenveen', 'Calgary', 'Salt Lake City', 'Inzell', 'Hamar', 'Thialf', 'Erfurt', 'Berlin', 'Kolomna', 'Nagano', 'Gangneung', 'Tomaszów']
    print('\nper baan: eigen afwijking per seizoen bovenop het gemeenschappelijke effect (%), alleen seizoenen met >= 5 races')
    for h in HOOFD:
        m = X['baan'].str.startswith(h)
        if m.sum() < 50:
            continue
        t = X[m].groupby('seizoen')['res'].agg(['mean', 'size'])
        t = t[t['size'] >= 5]
        print(f'  {h:15s} ' + ' '.join(f'{s}:{100 * (np.exp(v) - 1):+.1f}' for s, v in t['mean'].items()))
    # vergelijking: uitslagen zelf (snelste tijden) per seizoen: mediaan van de 10 snelste tijden per geslacht x afstand, t.o.v. 2025
    R2 = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['gender', 'afstand', 'seizoen', 'tijd', 'verdacht'])
    R2 = R2[~R2['verdacht'] & R2['afstand'].astype(str).isin(['500m', '1000m', '1500m', '3000m', '5000m']) & R2['seizoen'].between(2006, 2025)]
    R2['l'] = np.log(R2['tijd'])
    t10 = R2.groupby(['gender', 'afstand', 'seizoen'], observed=True)['l'].apply(lambda v: np.sort(v.to_numpy())[:10].mean()).rename('t10').reset_index()
    t10['rel'] = t10['t10'] - t10.groupby(['gender', 'afstand'], observed=True)['t10'].transform(lambda v: v.iloc[-1])
    print('\nde 10 snelste tijden van het seizoen (gemiddeld over geslacht x afstand), % t.o.v. 2025-26:')
    g = t10.groupby('seizoen')['rel'].mean()
    print('  ' + ' '.join(f'{s}:{100 * (np.exp(v) - 1):+.2f}%' for s, v in g.items()))
