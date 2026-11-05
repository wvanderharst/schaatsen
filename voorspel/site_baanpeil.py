"""Peil per baan voor de simulatie van regionale wedstrijden: per baan x geslacht x afstand het typische peil van een race (mediaan van het
racepeil theta, uit het veld) t.o.v. een normale dag in Heerenveen (meta.peil_hv), en de spreiding van het racepeil (hoe sterk het ijs per
wedstrijd wisselt; robuust: 1,4826 x MAD). Laatste vier seizoenen als er >= 5 races zijn, anders alle seizoenen (>= 5 races).
Heerenveen is per definitie 0 (zelfde normaal als 'Heerenveen nu'). Schrijft meta.baanpeil = {baan-index: {"g|afstand": [ln-verschil, spreiding, races]}} in de index. Gebruik: python site_baanpeil.py <werkmap> <sitemap>"""
import gzip, json, base64, os, sys
import numpy as np, pandas as pd

w, S = sys.argv[1], sys.argv[2]
I = json.load(gzip.open(os.path.join(S, 'index.json.gz')))
banen = I['meta']['banen']; bidx = {b: i for i, b in enumerate(banen)}; hv = I['meta']['peil_hv']
R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['gender', 'afstand', 'dag', 'seizoen', 'event_id', 'baan', 'theta', 'verdacht'])
for c in ('gender', 'afstand', 'baan'):
    R[c] = R[c].astype(str)
R = R[~R['verdacht']]
X = R.groupby(['event_id', 'dag', 'gender', 'afstand']).agg(theta=('theta', 'median'), n=('theta', 'size'), baan=('baan', 'first'), seizoen=('seizoen', 'first')).reset_index()
X = X[X['n'] >= 5]
laatste = int(X['seizoen'].max())
uit = {}
for (b, g, a), x in X.groupby(['baan', 'gender', 'afstand']):
    h = hv.get(f'{g}|{a}')
    if h is None:
        continue
    r = x[x['seizoen'] >= laatste - 4]
    if len(r) < 5:
        r = x
    if len(r) < 5:
        continue
    t = r['theta'].to_numpy(); med = float(np.median(t)); sd = float(1.4826 * np.median(np.abs(t - med)))
    uit.setdefault(str(bidx[b]), {})[f'{g}|{a}'] = [round(med - h, 5), round(max(sd, 0.002), 5), int(len(r))]
# Heerenveen per definitie 0: alle banen t.o.v. hetzelfde gewone peil van Heerenveen (zelfde maat: mediaan van alle races). Anders
# rekenen 'Heerenveen nu' (normale dag met topveld, meta.peil_hv) en de kolom IJs of de simulatie in Heerenveen met twee verschillende
# normalen (10 km mannen 0,51% verschil).
hvk = uit.get(str(bidx.get('Heerenveen (NED)', -1)), {})
for k_, v_ in uit.items():
    for ga, d in v_.items():
        if ga in hvk and k_ != str(bidx.get('Heerenveen (NED)', -1)):
            d[0] = round(d[0] - hvk[ga][0], 5)
for ga in hvk:
    hvk[ga][0] = 0.0
I['meta']['baanpeil'] = uit
open(os.path.join(S, 'index.json.gz'), 'wb').write(gzip.compress(json.dumps(I, ensure_ascii=False, separators=(',', ':')).encode()))
open(os.path.join(S, 't', 'index.txt'), 'w').write(base64.b64encode(open(os.path.join(S, 'index.json.gz'), 'rb').read()).decode())
for b in ('Heerenveen (NED)', 'Den Haag (NED)', 'Utrecht (NED)', 'Haarlem (NED)', 'Salt Lake City (USA)', 'Calgary (CAN)', 'Inzell (GER)'):
    v = uit.get(str(bidx.get(b, -1)), {})
    print(b, {k: (round(100 * (np.exp(d[0]) - 1), 2), round(100 * d[1], 2), d[2]) for k, d in v.items() if k in ('v|500m', 'v|1000m', 'm|1500m')})
print('banen met peil', len(uit))
