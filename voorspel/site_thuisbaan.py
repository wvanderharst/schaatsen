"""Thuisbaan (de baan met de meeste ritten) per rijder in de index van de website zetten (kolom 'thuisbaan', index in meta.banen).
Gebruik: python site_thuisbaan.py <sitemap met index.json.gz en r/>"""
import gzip, json, base64, collections, os, sys

S = sys.argv[1]
I = json.load(gzip.open(os.path.join(S, 'index.json.gz')))
thuis = {}
for b in range(I['meta']['nb']):
    d = json.load(gzip.open(os.path.join(S, 'r', f'{b}.json.gz')))
    for sid, x in d.items():
        c = collections.Counter(v[2] for v in x['r'])
        if c:
            thuis[int(sid)] = c.most_common(1)[0][0]
kol = I['meta']['kol']
if 'thuisbaan' not in kol:
    kol.append('thuisbaan')
    for r in I['rijders']:
        r.append(thuis.get(r[0]))
else:
    k = kol.index('thuisbaan')
    for r in I['rijders']:
        r[k] = thuis.get(r[0])
raw = json.dumps(I, ensure_ascii=False, separators=(',', ':')).encode()
open(os.path.join(S, 'index.json.gz'), 'wb').write(gzip.compress(raw))
open(os.path.join(S, 't', 'index.txt'), 'w').write(base64.b64encode(open(os.path.join(S, 'index.json.gz'), 'rb').read()).decode())
c = collections.Counter(thuis.values())
print(len(thuis), [(I['meta']['banen'][b], n) for b, n in c.most_common(8)])
