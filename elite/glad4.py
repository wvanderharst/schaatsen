import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df0=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df0); top50=em.kies_elite(df0,n_per=50)
c=em.schat_ijs(df0); print('ijs (a,b,c,d), rijders tot 10% van de top:', c.round(4), flush=True)
df1=em.draai_ext(R.copy(), ijs=tuple(c), **kw)
x=df1[df1.skater_id.isin(el)&(df1.seizoen>=2016)&np.isfinite(df1.e)&(df1.n>=5)&~df1.val&(df1.e.abs()<0.1)]
b=np.linalg.lstsq(np.c_[np.ones(len(x)),x.dth,x.hoog],x.e,rcond=None)[0]; print(f'controle elite 2016+ na ijs: fout = {b[0]:+.4f} {b[1]:+.3f}*dth {b[2]:+.4f}*hoog')
for lab,df in (('zonder ijs',df0),('ijs per niveau',df1)):
    for cal,fn in (('groepen',em.ijk),('kernel',em.ijk_kernel)):
        e=fn(df,el); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016]; t=z.skater_id.isin(top50)
        print(f"{lab:14s} {cal:7s}: fouten {fout:2d} | top50/rest snel1/100 {(z[t].p<=.01).mean()/.01:.2f}/{(z[~t].p<=.01).mean()/.01:.2f} traag1/100 {(z[t].p>=.99).mean()/.01:.2f}/{(z[~t].p>=.99).mean()/.01:.2f} | Leerdam20 <=1/30: {em.zeldzaam(e,'Jutta LEERDAM')[0]} |", ' ; '.join(x[:55] for x in regels if 'FOUT' in x), flush=True)
        e.to_parquet(f'elite_{lab.replace(" ","_")}_{cal}.parquet')
