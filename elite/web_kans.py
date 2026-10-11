"""Kans per rit volgens de nieuwe methode voor de elite (200 beste vrouwen + 200 beste mannen sinds 2011), voor de website.
Schrijft elite_kans.parquet (skater_id, dag, afstand, tijd, z; z = Phi^-1(kans), negatief = snel, 9 = valpartij) en
elite_spreiding.parquet (eigen spreiding per rijder over alle afstanden, als factor op de typische spreiding CD).
Alleen de elite: voor zwakkere en jonge rijders klopt de ijking niet (het niveau loopt achter bij snelle groei; zie README).
Gebruik: python web_kans.py ritten_theta.parquet"""
import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd, elite_model as em

df = em.draai_ext(pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else 'ritten_theta.parquet'), alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
elite = set(em.kies_elite(df))
e = em.ijk(df, elite)
fout, regels, nr, nk = em.toets(e)
print(f'elite: {nr} ritten van {nk} rijders 2016+, {fout} fouten in de toets', *regels, sep='\n  ')
uit = e[['skater_id', 'dag', 'afstand', 'tijd', 'z']].copy()
val = df[df['val'] & df['skater_id'].isin(elite)][['skater_id', 'dag', 'afstand', 'tijd']].assign(z=9.0)
uit = pd.concat([uit, val], ignore_index=True)
df['fac'] = df['s'] / np.asarray(em.CD)[df['afstand_idx'].to_numpy()]      # stand vóór de laatste rit
fac = df[df['skater_id'].isin(elite)].sort_values('dag').groupby('skater_id')['fac'].last()
uit.to_parquet('elite_kans.parquet'); fac.rename('fac').to_frame().to_parquet('elite_spreiding.parquet')
print('ritten', len(uit), 'rijders', uit['skater_id'].nunique(), '| eigen spreiding p10/50/90:', np.round(np.quantile(fac, [.1, .5, .9]), 2))
