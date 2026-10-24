"""Titelkansen in de rijderblokken van de website zetten: tot 18 jaar uit titels.py (titels_alle.parquet), vanaf 18 jaar uit titels_sen.py
(titels_alle_sen.parquet). Schrijft de blokken (gzip en base64) opnieuw. Gebruik: python site_kansen.py <werkmap> <sitemap>"""
import gzip, json, base64, os, sys
import pandas as pd

K = ['p_wk', 'e_wk', 'p_os', 'e_os', 'p_wc', 'e_wc']


def main(w, S):
    kans = {}
    for f in ('titels_alle.parquet', 'titels_alle_sen.parquet'):       # senioren (18+) overschrijven junioren
        for r in pd.read_parquet(os.path.join(w, f)).itertuples(index=False):
            kans[(int(r.skater_id), int(r.seizoen))] = [getattr(r, k) for k in K]
    nb = json.load(gzip.open(os.path.join(S, 'index.json.gz')))['meta']['nb']
    for b in range(nb):
        d = json.load(gzip.open(os.path.join(S, 'r', f'{b}.json.gz')))
        for sid, x in d.items():
            for s, e in x['s'].items():
                e.pop('kans', None)
                v = kans.get((int(sid), int(s)))
                if v is not None:
                    e['kans'] = {k: round(float(a), 4) for k, a in zip(K, v)}
        open(os.path.join(S, 'r', f'{b}.json.gz'), 'wb').write(gzip.compress(json.dumps(d, ensure_ascii=False, separators=(',', ':')).encode()))
        os.makedirs(os.path.join(S, 't', 'r'), exist_ok=True)
        open(os.path.join(S, 't', 'r', f'{b}.txt'), 'w').write(base64.b64encode(open(os.path.join(S, 'r', f'{b}.json.gz'), 'rb').read()).decode())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
