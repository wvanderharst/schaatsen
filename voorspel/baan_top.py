"""Baancorrectie voor de top bij het omrekenen naar Heerenveen. Het racepeil komt uit het hele veld; op sommige banen (vooral hoogland:
Calgary, Salt Lake City) winnen toprijders meer dan het veld, zodat hun omgerekende tijd daar te snel uitvalt. Per baan b: c_b x w(niveau),
w = exp(-max(niveau, 0) / TW) (1 voor de wereldtop, ~0 ver daarachter). c_b uit dezelfde rijder, hetzelfde seizoen en dezelfde afstand op
baan b en in Heerenveen, alleen WK, Spelen en World Cups (geen verschil in motivatie): gewogen kleinste kwadraten, gekrompen naar 0
(prior-sd PRIOR_SD; banen met weinig paren houden ~0). Schrijft baan_top.json naast deze code. Gebruik: python baan_top.py <werkmap>"""
import os, sys, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piek as pk

TW, PRIOR_SD, HV = 0.05, 0.003, 'Heerenveen (NED)'


def main(w):
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'gender', 'afstand', 'dag', 'baan', 'tijd', 'theta', 'event_id', 'seizoen', 'verdacht', 'land'])
    for c in ('gender', 'afstand', 'baan'):
        R[c] = R[c].astype(str)
    Tn = pk.toernooien(R); intl = set(Tn['eid']) | set(pk.internationaal(R, set(Tn['eid'])))
    V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'k_c']).to_pandas(); V['afstand'] = V['afstand'].astype(str)
    R = R.merge(V.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    R = R[~R['verdacht'] & R['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m', '10000m']) & (R['seizoen'] >= 2010) & R['k_c'].notna()
          & (R['event_id'] % 1000000).isin(intl)]
    R['y'] = np.log(R['tijd']) - R['theta']
    g = R.groupby(['skater_id', 'seizoen', 'afstand', 'baan']).agg(y=('y', 'median'), c=('k_c', 'median')).reset_index()
    hv = g[g['baan'] == HV].set_index(['skater_id', 'seizoen', 'afstand'])['y'].rename('yhv')
    P = g[g['baan'] != HV].join(hv, on=['skater_id', 'seizoen', 'afstand']).dropna()
    P['d'] = P['y'] - P['yhv']; P['w'] = np.exp(-np.clip(P['c'], 0, None) / TW)
    uit = {}
    for b, x in P.groupby('baan'):
        d = x['d'].clip(-0.05, 0.05).to_numpy(); w_ = x['w'].to_numpy()
        if len(x) < 20 or w_.sum() < 5:
            continue
        a = np.sum(w_ * d) / np.sum(w_ ** 2)                      # d ~ c_b x w
        res = d - a * w_; s2 = 1.4826 ** 2 * np.median(np.abs(res)) ** 2
        se2 = s2 / np.sum(w_ ** 2)
        k = PRIOR_SD ** 2 / (PRIOR_SD ** 2 + se2)
        uit[b] = {'c': round(float(k * a), 5), 'ruw': round(float(a), 5), 'paren': int(len(x)), 'krimp': round(float(k), 2)}
    json.dump({'TW': TW, 'banen': uit, 'uitleg': __doc__.split('\n')[0]}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baan_top.json'), 'w'), indent=1, ensure_ascii=False)
    for b, v in sorted(uit.items(), key=lambda t: t[1]['c'])[:8] + sorted(uit.items(), key=lambda t: t[1]['c'])[-4:]:
        print(f"{b:28s} correctie {100 * v['c']:+.2f}% (ruw {100 * v['ruw']:+.2f}%, {v['paren']} paren, krimp {v['krimp']})")


if __name__ == '__main__':
    main(sys.argv[1])
