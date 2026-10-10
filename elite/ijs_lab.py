import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet')
kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
ijs=np.zeros(4)
for it in range(3):
    c=em.schat_ijs(df); ijs=ijs+c; print('ijs-coef (a, b, c, d):', ijs.round(4), flush=True)
    df=em.draai_ext(R.copy(), ijs=tuple(ijs), **kw)
for lab,fn in (('groepen',em.ijk),('glad',em.ijk_glad)):
    e=fn(df,el); fout,regels,nr,nk=em.toets(e); z=e[e.seizoen>=2016]; t=z.skater_id.isin(top50)
    print(f"{lab}: fouten {fout} | top50/rest snel1/100 {(z[t].p<=.01).mean()/.01:.2f}/{(z[~t].p<=.01).mean()/.01:.2f} traag1/100 {(z[t].p>=.99).mean()/.01:.2f}/{(z[~t].p>=.99).mean()/.01:.2f}", flush=True)
    print('   ', ' ; '.join(x for x in regels if 'FOUT' in x))
    for nm in ('Jutta LEERDAM','Francesca LOLLOBRIGIDA','Femke KOK','Jordan STOLZ','Joep WENNEMARS'):
        k,n=em.zeldzaam(e,nm); print(f"    {nm}: laatste {n} ritten, kans <= 1/30: {k}  (verwacht {n/15:.1f})")
    e.to_parquet(f'elite_{lab}.parquet')
