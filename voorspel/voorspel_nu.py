"""Voorspelde tijden voor een gekozen dag op Heerenveen en Salt Lake City (gemiddelde dag op die baan) voor de top van nu.
Voegt denkbeeldige ritten (zonder tijd) toe aan de data, draait dezelfde passes (kenmerken, Kalman) en het model met alle jaren
als fit (FIT_TOT = 2026), en geeft per rijder x afstand x baan de mediaan en de 10%- en 90%-grens.
Gebruik: python voorspel_nu.py <map met ritten_cat.parquet> <kalman_par.json> <uitmap> [datum JJJJ-MM-DD] [aantal per geslacht]"""
import os, sys
os.environ['FIT_TOT'] = '2026'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import kenmerken as km
import model2 as m2

BANEN = ['Heerenveen (NED)', 'Salt Lake City (USA)']
AF = ['500m', '1000m', '1500m', '3000m', '5000m', '10000m']


def kies_top(R, n_per, vanaf_dag):
    """Per geslacht de n_per rijders met de beste plek in de PR-ranglijst sinds 2011, die sinds vanaf_dag nog gereden hebben."""
    x = R[(R['jaar'] >= 2011) & R['afstand'].astype(str).isin(AF) & ~R['verdacht']]
    actief = set(R.loc[R['dag'] >= vanaf_dag, 'skater_id'])
    pb = x[x['skater_id'].isin(actief)].groupby(['gender', 'afstand', 'skater_id'])['tijd'].min().reset_index()
    pb['rang'] = pb.groupby(['gender', 'afstand'])['tijd'].rank(method='min')
    b = pb.sort_values('rang').groupby(['gender', 'skater_id']).first().reset_index().sort_values(['gender', 'rang', 'tijd'])
    return b.groupby('gender').head(n_per)['skater_id'].tolist()


def main(w, par_pad, uit, datum='2026-10-20', n_per=50):
    os.makedirs(uit, exist_ok=True)
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet')); R['afstand'] = R['afstand'].astype(str)
    D = int((pd.Timestamp(datum) - pd.Timestamp('1970-01-01')).days); S = int(pd.Timestamp(datum).year if pd.Timestamp(datum).month >= 7 else pd.Timestamp(datum).year - 1)
    top = kies_top(R, n_per, D - 2 * 365)
    laatste = R[R['skater_id'].isin(top)].sort_values('dag').groupby('skater_id').tail(1).set_index('skater_id')
    recent = R[R['skater_id'].isin(top) & (R['dag'] >= D - 3 * 365) & R['afstand'].isin(AF)].groupby(['skater_id', 'afstand']).size()
    rijen = []
    for (sid, a), n in recent.items():
        if n < 3:
            continue
        l = laatste.loc[sid]
        for bi, baan in enumerate(BANEN):
            rijen.append(dict(skater_id=sid, skater_name=l['skater_name'], gender=l['gender'], land=l['land'], groep=l['groep'], categorie=l['categorie'],
                              afstand=a, afstand_idx=AF.index(a) + 2, dag=D, jaar=pd.Timestamp(datum).year, seizoen=S, event_id=-(bi + 1), baan=baan,
                              tijd=np.nan, verdacht=False, theta=np.nan, v_theta=0.0, geboren=l['geboren'], leeftijd=S - l['geboren'] if np.isfinite(l['geboren']) else np.nan,
                              leeftijd_exact=l['leeftijd_exact']))
    N = pd.DataFrame(rijen)
    A = pd.concat([R, N[R.columns]], ignore_index=True)
    for c in ('groep', 'categorie', 'afstand', 'baan', 'gender', 'land', 'skater_name'):
        A[c] = A[c].astype(str)
    A.to_parquet(os.path.join(uit, 'ritten_cat.parquet'))
    km.bouw(A).to_parquet(os.path.join(uit, 'kenmerken.parquet'))
    K, coefs, gs, gi, par_t = m2.draai(uit, json.load(open(par_pad)), verbose=False)
    x = K[(K['dag'] == D) & (K['event_id'] < 0)].copy()
    Zq = m2.kenm_verdeling(x)
    loc0 = x['pred'].to_numpy() - (x['bias_vorm'].to_numpy() if 'bias_vorm' in x else 0.0)     # locatie zonder de vormlaag
    for q, nm in ((0.1, 'p10'), (0.5, 'mediaan'), (0.9, 'p90')):
        x[nm] = np.exp(loc0 + m2.kwantiel_u(x, par_t, Zq, q) * x['sd'].to_numpy())     # exact, met staartcorrectie en vormlaag
    pr = R[~R['verdacht']].groupby(['skater_id', 'afstand'])['tijd'].min().rename('PR')
    sb = R[~R['verdacht'] & (R['seizoen'] == S - 1)].groupby(['skater_id', 'afstand'])['tijd'].min().rename('SB_vorig')
    x = x.join(pr, on=['skater_id', 'afstand']).join(sb, on=['skater_id', 'afstand'])
    x['baan'] = x['baan'].str.replace(r' \(.*\)$', '', regex=True)
    uitk = x[['gender', 'skater_name', 'afstand', 'baan', 'mediaan', 'p10', 'p90', 'PR', 'SB_vorig', 'lf']].rename(columns={'skater_name': 'naam', 'lf': 'leeftijd'})
    uitk['leeftijd'] = uitk['leeftijd'].round(1)
    for c in ('mediaan', 'p10', 'p90'):
        uitk[c] = uitk[c].round(2)
    uitk = uitk.sort_values(['gender', 'afstand', 'baan', 'mediaan'])
    uitk.to_csv(os.path.join(uit, 'voorspelling_nu.csv'), index=False)
    return uitk


if __name__ == '__main__':
    a = sys.argv
    u = main(a[1], a[2], a[3], a[4] if len(a) > 4 else '2026-10-20', int(a[5]) if len(a) > 5 else 50)
    print(u.groupby(['gender', 'afstand', 'baan']).head(3).to_string(index=False))
