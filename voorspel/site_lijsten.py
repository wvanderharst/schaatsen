"""Compacte lijstbestanden voor de website (ranglijsten en talenten), gemaakt uit de rijderblokken van site_data.py.
l/{geslacht}_{afstand}.txt: [rijder, seizoen, leeftijd, Heerenveen-tijd nu, achter top-10, wereldrang, ritten] (>= 2 ritten);
l/kans.txt: [rijder, seizoen, leeftijd, p_wk, e_wk, p_os, e_os, p_wc, e_wc, groei, beste achter top-10 (500-10.000 m, >= 3 ritten), beste afstand].
Alles als base64 van gzip-JSON. Gebruik: python site_lijsten.py <sitemap met index.json.gz en r/> <uitmap>"""
import gzip, json, base64, collections, os, sys

SEN = ['500m', '1000m', '1500m', '3000m', '5000m', '10000m']


def main(S, uit):
    os.makedirs(os.path.join(uit, 'l'), exist_ok=True)
    I = json.load(gzip.open(os.path.join(S, 'index.json.gz')))
    g = {r[0]: r[2] for r in I['rijders']}
    lijst = collections.defaultdict(list); kans = []
    for b in range(I['meta']['nb']):
        d = json.load(gzip.open(os.path.join(S, 'r', f'{b}.json.gz')))
        for sid, x in d.items():
            sid = int(sid)
            for s, e in x['s'].items():
                s = int(s); l = e.get('lft'); best = None
                for af, v in e['af'].items():
                    if v[4] >= 2:
                        lijst[(g[sid], af)].append([sid, s, l, v[0], v[1], v[3], v[4]])
                    if af in SEN and v[4] >= 3 and v[1] is not None and (best is None or v[1] < best[0]):
                        best = (v[1], SEN.index(af))
                k = e.get('kans')
                if k or e.get('groei') is not None:
                    kans.append([sid, s, l] + ([k['p_wk'], k['e_wk'], k['p_os'], k['e_os'], k['p_wc'], k['e_wc']] if k else [None] * 6)
                                + [e.get('groei'), best[0] if best else None, best[1] if best else None])

    def schrijf(naam, obj):
        raw = gzip.compress(json.dumps(obj, separators=(',', ':')).encode(), 9)
        open(os.path.join(uit, 'l', f'{naam}.txt'), 'w').write(base64.b64encode(raw).decode())

    for (gg, af), rows in lijst.items():
        schrijf(f'{gg}_{af}', rows)
    schrijf('kans', kans)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
