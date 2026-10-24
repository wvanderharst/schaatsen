"""Onmogelijk snelle ritten als 'verdacht' markeren (tellen dan nergens mee, net als ritten met een gecorrigeerde afstand).
Twee regels, ruim buiten wat echt schaatsen doet (de grootste echte uitschieters van de top zijn enkele procenten):
- hele race: de mediaan van ln(tijd) - verwachting vooraf over het veld (>= 3 rijders) is meer dan 20% sneller (verkeerde afstand of
  tijdwaarneming, bijv. Makinsk 500 m vrouwen -55%, Tynset 2019 -23 tot -26%);
- losse rit: meer dan 40% sneller dan verwacht EN meer dan 6 sd van het model (patroon van een verkeerd afstandslabel: 3000 m als 5000 m
  is -51%, 1000 m als 1500 m -41%; bijv. Hailar 22-08-2024 '5000 m' twaalf rijders -100 tot -108%, Chita 3000 m junior -72%).
Alleen als de rit NIET bevestigd wordt: binnen 2 jaar daarna rijdt de rijder op die afstand nooit binnen 15% van die tijd. Zo blijven
echte doorbraken staan waarvan de voorspelling fout zat (bijv. Brock Miron 1000 m Hamar 2007, toen nog weinig geschiedenis), en vallen
verkeerde afstandslabels weg. De verwachting komt uit de vorige modelrun (voorspel2_uitkomst.parquet). Schrijft onmogelijk.csv (de lijst) en ritten_cat.parquet met
verdacht |= markering; het origineel blijft als ritten_cat_voor_onmogelijk.parquet.
Gebruik: python onmogelijk.py <werkmap>"""
import os, sys, shutil
import numpy as np, pandas as pd

RACE_GRENS, RIT_GRENS, RIT_Z, BEVESTIG, BEVESTIG_DAGEN = 0.20, 0.40, 6.0, 0.15, 730

w = sys.argv[1]
V = pd.read_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'),
                    columns=['skater_id', 'skater_name', 'dag', 'baan', 'afstand', 'tijd', 'pred', 'sd', 'ok'])
V['afstand'] = V['afstand'].astype(str)
bron = os.path.join(w, 'ritten_cat_voor_onmogelijk.parquet')
if not os.path.exists(bron):
    shutil.copy(os.path.join(w, 'ritten_cat.parquet'), bron)
R = pd.read_parquet(bron); R['afstand'] = R['afstand'].astype(str)
key = ['skater_id', 'dag', 'afstand', 'tijd']; rk = ['dag', 'event_id', 'gender', 'afstand_idx']
X = R[key + rk[1:]].reset_index().merge(V[V['ok'] & np.isfinite(V['pred'])].drop_duplicates(key), on=key, how='inner')
X['e'] = np.log(X['tijd']) - X['pred']; X['z'] = X['e'] / X['sd']
med = X.groupby(rk)['e'].transform('median'); n = X.groupby(rk)['e'].transform('size')
X['race'] = (n >= 3) & (med < -RACE_GRENS)
X['rit'] = (X['e'] < -RIT_GRENS) & (X['z'] < -RIT_Z)
kand = X[X['race'] | X['rit']]
# bevestiging: een latere rit van dezelfde rijder op dezelfde afstand binnen BEVESTIG_DAGEN die hoogstens BEVESTIG trager is
L = R.loc[~R['verdacht'] & np.isfinite(R['tijd']), ['skater_id', 'afstand', 'dag', 'tijd']]
L = L[L['skater_id'].isin(kand['skater_id'])]
B = kand[['index', 'skater_id', 'afstand', 'dag', 'tijd']].merge(L, on=['skater_id', 'afstand'], suffixes=('', '_l'))
B = B[(B['dag_l'] > B['dag']) & (B['dag_l'] <= B['dag'] + BEVESTIG_DAGEN) & (np.log(B['tijd_l']) - np.log(B['tijd']) <= BEVESTIG)]
M = kand[~kand['index'].isin(B['index'])].copy()
print(f"kandidaten {len(kand)}, bevestigd door latere ritten {len(kand) - len(M)}")
M['reden'] = np.where(M['race'], 'race', 'rit')
M['datum'] = pd.to_datetime(M['dag'], unit='D').dt.date
M['sneller_%'] = (100 * (1 - np.exp(M['e']))).round(1)
M.sort_values(['dag', 'baan'])[['datum', 'baan', 'afstand', 'skater_name', 'tijd', 'sneller_%', 'reden']].to_csv(os.path.join(w, 'onmogelijk.csv'), index=False)
R['verdacht'] = R['verdacht'].to_numpy() | R.index.isin(M['index'])
R.to_parquet(os.path.join(w, 'ritten_cat.parquet'))
print(f"gemarkeerd: {len(M)} ritten ({int(M['race'].sum())} via hele race, {int((~M['race']).sum())} los), "
      f"2020+: {int((M['dag'] >= 18262).sum())}; verdacht totaal {int(R['verdacht'].sum())}")
print(M.groupby(['baan', 'afstand', 'reden']).size().sort_values(ascending=False).head(15).to_string())
