"""Zet de kans per rit van de nieuwe elite-methode (../elite/web_kans.py) in web_model.json, na export_web.py en vóór bouw_pagina.py.
Voor de elite-rijders (200 beste vrouwen + 200 beste mannen sinds 2011) komt de kans per rit uit de nieuwe methode en de
volatiliteit uit hun eigen spreiding over alle afstanden samen; de overige rijders houden de kans uit dit model.
Gebruik: python web_elite_kans.py elite_kans.parquet elite_spreiding.parquet"""
import sys, json, pandas as pd

kans = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'elite_kans.parquet')
fac = pd.read_parquet(sys.argv[2] if len(sys.argv) > 2 else 'elite_spreiding.parquet')['fac']
d = json.load(open('web_model.json', encoding='utf-8'))
z = {(int(r.skater_id), int(r.dag), r.afstand, round(float(r.tijd), 2)): float(r.z) for r in kans.itertuples()}
elite = set(fac.index.astype(int))
n_el = n_rit = n_mis = 0
for r in d['rijders']:
    if r['id'] not in elite:
        continue
    r['nm'] = 1; n_el += 1                       # nieuwe methode
    for q in r['res']:                           # [dag, baan, afstand, tijd, z, verwacht]
        v = z.get((r['id'], q[0], q[2], round(q[3], 2)))
        q[4] = None if v is None else round(v, 2); n_rit += 1; n_mis += v is None
    f = round(float(fac[r['id']]), 2)
    for p in r['pr'].values():
        p['vol'] = f
d['meta']['elite_methode'] = 1
json.dump(d, open('web_model.json', 'w', encoding='utf-8'), separators=(',', ':'), ensure_ascii=False)
print(f'{n_el} elite-rijders op de site, {n_rit} ritten, waarvan {n_mis} zonder kans (minder dan 3 eerdere ritten op de afstand)')
