import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
# ijs bij de elite: fout tegen dth en hoogland, voor en na 2016
for lab,s in (('elite <=2015',df.skater_id.isin(el)&(df.seizoen<=2015)),('elite 2016+',df.skater_id.isin(el)&(df.seizoen>=2016)),('top50 2016+',df.skater_id.isin(top50)&(df.seizoen>=2016)),('rest (niet elite) 2016+',~df.skater_id.isin(el)&(df.seizoen>=2016))):
    x=df[s&np.isfinite(df.e)&(df.n>=5)&~df.val&(df.e.abs()<0.1)]
    A=np.c_[np.ones(len(x)),x.dth,x.hoog]; b=np.linalg.lstsq(A,x.e,rcond=None)[0]
    print(f"{lab:24s}: fout = {b[0]:+.4f} {b[1]:+.3f}*dth {b[2]:+.4f}*hoogland   (n={len(x)}, gem. g {x.g.median():.3f})")
e=em.ijk_glad(df,el); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016]; t=z.skater_id.isin(top50)
print(f"glad (mediaanregressie): fouten {fout} | top50/rest snel1/100 {(z[t].p<=.01).mean()/.01:.2f}/{(z[~t].p<=.01).mean()/.01:.2f} traag1/100 {(z[t].p>=.99).mean()/.01:.2f}/{(z[~t].p>=.99).mean()/.01:.2f} | Leerdam20 <=1/30: {em.zeldzaam(e,'Jutta LEERDAM')[0]}")
print('\n'.join(x for x in regels if 'FOUT' in x))
