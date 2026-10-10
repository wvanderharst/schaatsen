import warnings; warnings.filterwarnings('ignore')
import sys, itertools, numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten.parquet'); elite=None
res=[]
grid=[dict(alpha=a,snel_w=sw,gamma=g,hf=h) for a in (0.12,0.2,0.3) for sw in (1.0,1.5) for g in (0.0,0.2,0.4) for h in (30.0,90.0) if not (g==0 and h==90.0)]
for kw in grid:
    df=em.draai(R.copy(),**kw)
    if elite is None: elite=em.kies_elite(df)
    e=em.ijk(df,elite); fout,regels,nr,nk=em.toets(e)
    x=e[e.seizoen>=2016]; k=pd.Series(x.p<=0.02).groupby(x.skater_id).sum()
    L=em.rijder(e,'Jutta LEERDAM')
    print(kw, 'fouten', fout, '| snel 1/1000 %.2fx 1/100 %.2fx 1/20 %.2fx | traag 1/1000 %.2fx 1/100 %.2fx' % ((x.p<=1e-3).mean()/1e-3,(x.p<=1e-2).mean()/1e-2,(x.p<=.05).mean()/.05,(x.p>=1-1e-3).mean()/1e-3,(x.p>=.99).mean()/.01),
          '| rijders >=8 snel', int((k>=8).sum()), '| Leerdam 20: snel1/100', L[1], 'snel1/20', L[2], flush=True)
