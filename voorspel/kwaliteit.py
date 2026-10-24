"""Kwaliteit van een rit, gecorrigeerd voor alles: q = ln(tijd) - theta, met theta het peil van die race achteraf (uit het veld: baan,
ijs, weer, hoogte, dag). Omgerekend naar Heerenveen: tijd_HV = exp(q + theta_HV), theta_HV = mediaan peil van Heerenveen-races met een
topveld (mediaan niveau < 5%) in de laatste drie seizoenen, per geslacht x afstand. Ten opzichte van het wereldrecord (snelste geldige tijd
in de data), op dezelfde manier naar Heerenveen omgerekend: pct_wr = 100 (exp(q - q_wr) - 1).
Gebruik: python kwaliteit.py <werkmap> ['naam|afstand' ...]"""
import os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq


def tabel(w):
    V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'skater_name', 'gender', 'dag', 'seizoen', 'baan', 'afstand', 'tijd', 'pred',
                                                                          'pit', 'theta', 'th_hat_veld', 'k_c', 'ok', 'k_val']).to_pandas()
    for c in ('skater_name', 'afstand', 'baan', 'gender'):
        V[c] = V[c].astype(str)
    V = V[V['ok'] & np.isfinite(V['theta']) & np.isfinite(V['tijd'])].copy()
    V['q'] = np.log(V['tijd']) - V['theta']
    rk = ['dag', 'baan', 'gender', 'afstand']
    V['veld_niv'] = V.groupby(rk)['k_c'].transform('median')
    hv = V[V['baan'].str.startswith('Heerenveen') & (V['seizoen'] >= V['seizoen'].max() - 2) & (V['veld_niv'] < 0.05)]
    th_hv = hv.drop_duplicates(rk).groupby(['gender', 'afstand'])['theta'].median().rename('theta_hv')
    wr = V[~V['k_val']].sort_values('tijd').drop_duplicates(['gender', 'afstand'])[['gender', 'afstand', 'tijd', 'q', 'skater_name', 'baan', 'dag']]
    wr = wr.rename(columns={'tijd': 'wr', 'q': 'q_wr', 'skater_name': 'wr_naam', 'baan': 'wr_baan', 'dag': 'wr_dag'})
    V = V.join(th_hv, on=['gender', 'afstand']).merge(wr, on=['gender', 'afstand'], how='left')
    V['tijd_hv'] = np.exp(V['q'] + V['theta_hv']); V['wr_hv'] = np.exp(V['q_wr'] + V['theta_hv'])
    V['pct_wr'] = 100 * (np.exp(V['q'] - V['q_wr']) - 1)
    V['ijs_verrassing'] = V['theta'] - V['th_hat_veld']                 # < 0: ijs sneller dan vooraf verwacht
    V['eigen'] = np.log(V['tijd']) - V['pred'] - V['ijs_verrassing']    # < 0: rijder zelf sneller dan verwacht (bij dit ijs)
    return V


def ts(s):
    m = int(s // 60); r = s - 60 * m
    return (f'{m}:{r:05.2f}' if m else f'{r:.2f}').replace('.', ',')


if __name__ == '__main__':
    w = sys.argv[1]; V = tabel(w)
    for arg in sys.argv[2:]:
        naam, af = arg.split('|')
        x = V[(V['skater_name'] == naam) & (V['afstand'] == af) & ~V['k_val']].sort_values('dag').tail(16)
        r0 = x.iloc[0]
        print(f"\n{naam}, {af}  (wereldrecord in de data: {ts(r0['wr'])}, {r0['wr_naam']}, {r0['wr_baan']} "
              f"{pd.to_datetime(r0['wr_dag'], unit='D').strftime('%d-%m-%Y')}; in Heerenveen-termen {ts(r0['wr_hv'])})")
        print(f"{'datum':10s} {'baan':16s} {'gereden':>8s} {'kwaliteit HV':>12s} {'+% WR':>6s} {'ijs t.o.v. vooraf':>18s} {'rijder t.o.v. verwacht':>22s}")
        for _, r in x.iterrows():
            print(f"{pd.to_datetime(r['dag'], unit='D').strftime('%d-%m-%Y')} {r['baan'].split(' (')[0][:16]:16s} {ts(r['tijd']):>8s} {ts(r['tijd_hv']):>12s} "
                  f"{r['pct_wr']:6.2f} {100 * (np.exp(r['ijs_verrassing']) - 1):+17.2f}% {100 * (np.exp(r['eigen']) - 1):+21.2f}%")
