import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df0=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df0); top50=em.kies_elite(df0,n_per=50)
c=em.schat_ijs(df0); print('ijs (a,b,c,d) een stap:', c.round(4), flush=True)
for lab,df in (('zonder ijs',df0),('ijs per niveau',em.draai_ext(R.copy(), ijs=tuple(c), **kw))):
    for cal,fn in (('groepen',em.ijk),('glad',em.ijk_glad)):
        e=fn(df,el); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016]; t=z.skater_id.isin(top50)
        L=em.zeldzaam(e,'Jutta LEERDAM')[0]; F=em.zeldzaam(e,'Francesca LOLLOBRIGIDA')[0]
        print(f"{lab:14s} {cal:7s}: fouten {fout:2d} | top50/rest snel1/100 {(z[t].p<=.01).mean()/.01:.2f}/{(z[~t].p<=.01).mean()/.01:.2f} traag1/100 {(z[t].p>=.99).mean()/.01:.2f}/{(z[~t].p>=.99).mean()/.01:.2f} | Leerdam20 <=1/30: {L} Lollo20: {F} |", ' ; '.join(x[:60] for x in regels if 'FOUT' in x), flush=True)
