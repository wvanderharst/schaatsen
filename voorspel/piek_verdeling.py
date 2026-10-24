"""Is de foutverdeling op WK afstanden en Olympische Spelen wezenlijk anders? Daar piekt iedereen en rijdt niemand 'voor de training'.
Vergelijking voor dezelfde rijders in dezelfde seizoenen: (A) WK afstanden + Spelen, (B) hun andere ritten in een topveld (mediaan niveau
van de race < 4%, vooral World Cups), (C) hun overige ritten. Op z = Phi^-1(PIT) van het model zonder piek- en leeftijdslaag, en op de fout
in % (ln t - verwachting). Valpartijen apart geteld.
Gebruik: python piek_verdeling.py <werkmap> <uitkomst.parquet>"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk


def laad(w, pad):
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd', 'gender', 'afstand_idx'])
    R['afstand'] = R['afstand'].astype(str)
    T = pk.toernooien(R); eids = set(T.loc[T['soort'].isin(['Spelen', 'WK afstanden']), 'eid'])
    V = pq.read_table(pad, columns=['skater_id', 'skater_name', 'dag', 'jaar', 'seizoen', 'afstand', 'tijd', 'pred', 'pit', 'u', 'ok', 'k_val', 'k_c']).to_pandas()
    V['afstand'] = V['afstand'].astype(str)
    V = V.merge(R[['skater_id', 'dag', 'afstand', 'tijd', 'event_id', 'gender', 'afstand_idx']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']),
                on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    V = V[V['ok'] & np.isfinite(V['pit']) & V['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m', '10000m']) & (V['jaar'] >= 2008)].copy()
    V['A'] = pk.markering(V['event_id'].fillna(-1).astype(np.int64), eids)
    V['veld'] = V.groupby(['dag', 'event_id', 'gender', 'afstand_idx'])['k_c'].transform('median')
    V['z'] = norm.ppf(V['pit'].clip(1e-12, 1 - 1e-12)); V['e'] = 100 * (np.log(V['tijd']) - V['pred'])
    deeln = V.loc[V['A'], ['skater_id', 'seizoen']].drop_duplicates()
    V = V.merge(deeln.assign(deelnemer=True), on=['skater_id', 'seizoen'], how='left'); V['deelnemer'] = V['deelnemer'].fillna(False).astype(bool)
    V['groep'] = np.where(V['A'], 'A WK afstanden + Spelen', np.where(V['deelnemer'] & (V['veld'] < 0.04), 'B zelfde rijders, topveld (WC)',
                          np.where(V['deelnemer'], 'C zelfde rijders, overig', '')))
    return V[V['groep'] != '']


def kentallen(x):
    zz = x.loc[~x['k_val'], 'z']; ee = x.loc[~x['k_val'], 'e']
    q = np.percentile(zz, [5, 25, 50, 75, 95])
    return dict(ritten=len(x), val_pct=round(100 * x['k_val'].mean(), 2), gem_z=round(zz.clip(-4, 4).mean(), 3), sd_z=round(zz.clip(-4, 4).std(), 3),
                robuust_sd=round((q[3] - q[1]) / 1.349, 3), scheef=round(((q[4] - q[2]) - (q[2] - q[0])) / (q[4] - q[0]), 3),
                snel20=round((x['pit'] <= 0.05).mean() / 0.05, 2), snel100=round((x['pit'] <= 0.01).mean() / 0.01, 2),
                traag20=round((x['pit'] >= 0.95).mean() / 0.05, 2), traag100=round((x['pit'] >= 0.99).mean() / 0.01, 2),
                traag20_zonder_val=round((x.loc[~x['k_val'], 'pit'] >= 0.95).mean() / 0.05, 2),
                gem_fout_pct=round(ee.mean(), 3), sd_fout_pct=round(ee.std(), 3), robuust_sd_fout=round((np.percentile(ee, 75) - np.percentile(ee, 25)) / 1.349, 3))


if __name__ == '__main__':
    V = laad(sys.argv[1], sys.argv[2])
    pd.set_option('display.width', 250)
    for per, s in (('2008-2019', V['jaar'] <= 2019), ('2020+', V['jaar'] >= 2020)):
        T = pd.DataFrame({g: kentallen(V[s & (V['groep'] == g)]) for g in sorted(V['groep'].unique())}).T
        print(f'\n{per}  (z en fout zonder valpartijen; snel/traag: waargenomen / verwacht, met valpartijen tenzij vermeld)'); print(T.to_string())
    # alleen de top < 2% (niveau vooraf)
    for per, s in (('top < 2%, 2008+', V['k_c'] < 0.02),):
        T = pd.DataFrame({g: kentallen(V[s & (V['groep'] == g)]) for g in sorted(V['groep'].unique())}).T
        print(f'\n{per}'); print(T.to_string())
    V.to_parquet(os.path.join(sys.argv[1], 'piek_verdeling.parquet'))
