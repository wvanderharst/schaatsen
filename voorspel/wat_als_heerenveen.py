"""Wat als de rit in Heerenveen was gereden? Voor gekozen rijders en afstanden: per echte rit een denkbeeldige rit (zonder tijd) op
dezelfde dag in Heerenveen, in een eigen race met alleen die rijder (dus zonder veldaanpassing: geen veld- of elite-ijsterm en geen
veldgeheugen in de vormlaag). Het hele model rekent die rit vooraf door: het Heerenveen-ijs zoals het die dag vooraf werd voorspeld,
de baancorrecties van Heerenveen en de vorm van de rijder op dat moment.
Gebruik: python wat_als_heerenveen.py <werkmap> <kalman_par.json> <uitmap> "Naam|afstand|n" ...
De echte ritten komen uit <werkmap>/voorspel2_uitkomst.parquet (zelfde model, zonder de denkbeeldige ritten)."""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import kenmerken as km
import model2 as m2

HV = 'Heerenveen (NED)'


def tijd_s(t):
    return f'{int(t // 60)}:{t % 60:05.2f}'.replace('.', ',') if t >= 60 else f'{t:.2f}'.replace('.', ',')


def kans(p):
    return f'sneller 1 op {1 / p:.0f}' if p < 0.5 else f'trager 1 op {1 / (1 - p):.0f}'


def main(w, par_pad, uit, keuzes):
    os.makedirs(uit, exist_ok=True)
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet')); R['afstand'] = R['afstand'].astype(str)
    V = pd.read_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'skater_name', 'dag', 'baan', 'afstand', 'tijd', 'pred', 'pit', 'ok'])
    V['afstand'] = V['afstand'].astype(str); V['skater_name'] = V['skater_name'].astype(str)
    echt = []
    for naam, af, n in keuzes:
        echt.append(V[(V['skater_name'] == naam) & (V['afstand'] == af) & V['ok']].sort_values('dag').tail(n))
    E = pd.concat(echt, ignore_index=True)
    # denkbeeldige ritten: kopie van de echte rit, in Heerenveen, zonder tijd en dagpeil, eigen race per rit
    sel = R.merge(E[['skater_id', 'dag', 'afstand', 'tijd']], on=['skater_id', 'dag', 'afstand', 'tijd'])
    sel = sel.drop_duplicates(['skater_id', 'dag', 'afstand']).reset_index(drop=True)
    N = sel.copy(); N['baan'] = HV; N['tijd'] = np.nan; N['theta'] = np.nan; N['v_theta'] = 0.0; N['verdacht'] = False
    if 'theta_rijder' in N:
        N['theta_rijder'] = np.nan
    N['event_id'] = m2.WAT_ALS_EVENT - np.arange(len(N))
    A = pd.concat([R, N[R.columns]], ignore_index=True)
    A = A.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    for c in ('groep', 'categorie', 'afstand', 'baan', 'gender', 'land', 'skater_name'):
        A[c] = A[c].astype(str)
    A.to_parquet(os.path.join(uit, 'ritten_cat.parquet'))
    shutil.copy(os.path.join(w, 'baanfactor_par.json'), os.path.join(uit, 'baanfactor_par.json'))
    km.bouw(A).to_parquet(os.path.join(uit, 'kenmerken.parquet'))
    K, coefs, gs, gi, par_t = m2.draai(uit, json.load(open(par_pad)), verbose=False)
    x = K[K['event_id'] <= m2.WAT_ALS_EVENT][['skater_id', 'dag', 'afstand', 'pred']].rename(columns={'pred': 'pred_hv'})
    E = E.merge(x, on=['skater_id', 'dag', 'afstand'], how='left')
    E['datum'] = pd.to_datetime(E['dag'], unit='D').dt.strftime('%d-%m-%Y'); E['baan'] = E['baan'].astype(str).str.replace(r' \(.*\)$', '', regex=True)
    E['gereden'] = E['tijd'].map(tijd_s); E['verwacht'] = np.exp(E['pred']).map(tijd_s); E['kans'] = E['pit'].map(kans)
    E['verwacht Heerenveen'] = np.exp(E['pred_hv']).map(tijd_s)
    E[['skater_name', 'afstand', 'datum', 'baan', 'tijd', 'pred', 'pit', 'pred_hv']].to_csv(os.path.join(uit, 'wat_als_heerenveen.csv'), index=False)
    for (naam, af), g in E.groupby(['skater_name', 'afstand'], sort=False):
        print(f'\n{naam}, {af}'); print(g[['datum', 'baan', 'gereden', 'verwacht', 'kans', 'verwacht Heerenveen']].to_string(index=False))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], a[3], [(k.split('|')[0], k.split('|')[1], int(k.split('|')[2])) for k in a[4:]])
