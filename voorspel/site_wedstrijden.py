"""Wedstrijdbestanden voor de website, uit de rijderblokken van site_data.py: per wedstrijd (dag, baan, geslacht, afstand) alle ritten.
w/{dag // 60}.txt: {"dag|baan|geslacht|afstand": [[rijder, tijd, Heerenveen-tijd nu, plaats in de eigen race, veld, soort, rijderschok %, kans, verdacht], ...]}.
Verder de referentieverdeling van de rijderschok (ln(1 + %/100)) van rijders binnen 5% van de top-10, seizoenen 2020+, als 201 kwantielen
(voor de simulatie). Base64 van gzip-JSON. Gebruik: python site_wedstrijden.py <sitemap met index.json.gz en r/> <uitmap>"""
import gzip, json, base64, collections, os, sys
import numpy as np


def main(S, uit):
    os.makedirs(os.path.join(uit, 'w'), exist_ok=True)
    I = json.load(gzip.open(os.path.join(S, 'index.json.gz')))
    g = {r[0]: r[2] for r in I['rijders']}
    W = collections.defaultdict(lambda: collections.defaultdict(list)); ref = []
    for b in range(I['meta']['nb']):
        d = json.load(gzip.open(os.path.join(S, 'r', f'{b}.json.gz')))
        for sid, x in d.items():
            sid = int(sid)
            top = {int(s) for s, e in x['s'].items() if any(v[1] is not None and v[1] < 5 and v[4] >= 3 for af, v in e['af'].items() if af not in ('100m', '300m'))}
            for v in x['r']:
                dag, a, bi = v[0], v[1], v[2]
                W[dag // 60][f'{dag}|{bi}|{g[sid]}|{a}'].append([sid, v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]])
                if v[8] is not None and not v[10] and dag >= 18080:
                    s = (np.datetime64(int(dag), 'D').astype(object))
                    sz = s.year if s.month >= 7 else s.year - 1
                    if sz in top:
                        ref.append(np.log1p(v[8] / 100))
    for k, races in W.items():
        for r in races.values():
            r.sort(key=lambda z: (z[1] is None, z[1]))
        raw = gzip.compress(json.dumps(races, separators=(',', ':')).encode(), 9)
        open(os.path.join(uit, 'w', f'{k}.txt'), 'w').write(base64.b64encode(raw).decode())
    q = np.quantile(np.array(ref), np.linspace(0.0025, 0.9975, 201))
    json.dump({'n': len(ref), 'q': [round(float(v), 5) for v in q]}, open(os.path.join(uit, 'schokref.json'), 'w'))
    print('wedstrijdbestanden', len(W), 'referentie', len(ref), 'sd', float(np.std(ref)), 'iqr/1.349', float((q[150] - q[50]) / 1.349))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
