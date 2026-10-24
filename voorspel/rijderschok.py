"""Alleen de schok van de rijder, nooit de gecombineerde (rijder + ijs). Het ijs wordt achteraf gecorrigeerd met het veld zelf: per race
(event, dag, geslacht, afstand, >= 5 rijders zonder val) de mediaan van ln(tijd) - verwachting. Rijderschok = (ln t - verwachting - ijs);
gestandaardiseerd met de eigen spreiding van de rijder (sd_rijder). De kans komt uit de werkelijke verdeling van die gestandaardiseerde
schok bij alle toprijders (niveau vooraf < 6%) in 2008-2019, dus zonder aangenomen staartvorm.
Gebruik als module: tabel(werkmap, uitkomst) geeft per rit ijs_achteraf, schok_pct, zr en kans (kans < 0,5: sneller dan verwacht)."""
import os
import numpy as np, pandas as pd, pyarrow.parquet as pq


def tabel(w, pad):
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'event_id', 'gender', 'jaar', 'baan', 'verdacht'])
    R['afstand'] = R['afstand'].astype(str)
    V = pq.read_table(pad, columns=['skater_id', 'skater_name', 'dag', 'afstand', 'tijd', 'pred', 'sd_rijder', 'ok', 'k_val', 'k_c']).to_pandas()
    V['afstand'] = V['afstand'].astype(str); V['skater_name'] = V['skater_name'].astype(str)
    V = V.merge(R[~R['verdacht']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'])
    V = V[V['ok'] & np.isfinite(V['pred']) & np.isfinite(V['sd_rijder'])].copy()
    V['res'] = np.log(V['tijd']) - V['pred']
    rk = ['event_id', 'dag', 'gender', 'afstand']
    g = V[~V['k_val']].groupby(rk)['res']
    ijs = pd.DataFrame({'ijs_achteraf': g.median(), 'n_veld': g.size()}).reset_index()
    V = V.merge(ijs, on=rk, how='left')
    V = V[V['n_veld'] >= 5].copy()
    V['schok'] = V['res'] - V['ijs_achteraf']; V['schok_pct'] = 100 * (np.exp(V['schok']) - 1)
    V['zr'] = V['schok'] / V['sd_rijder']
    ref = np.sort(V.loc[V['jaar'].between(2008, 2019) & (V['k_c'] < 0.06) & ~V['k_val'], 'zr'].to_numpy())
    V['kans'] = (np.searchsorted(ref, V['zr'].to_numpy(), side='right') + 0.5) / (len(ref) + 1)
    return V
