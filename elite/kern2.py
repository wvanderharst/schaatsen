import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
for kn,bw in (((8,8,6),0.5),((8,8,6),0.35),((10,10,8),0.35)):
    e=em.ijk_kernel(df,el,knopen=kn,bw=bw); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016]; t=z.skater_id.isin(top50)
    print(f"knopen {kn} bw {bw}: fouten {fout} | top50/rest snel1/100 {(z[t].p<=.01).mean()/.01:.2f}/{(z[~t].p<=.01).mean()/.01:.2f} traag1/100 {(z[t].p>=.99).mean()/.01:.2f}/{(z[~t].p>=.99).mean()/.01:.2f} | Leerdam20 <=1/30: {em.zeldzaam(e,'Jutta LEERDAM')[0]} |", ' ; '.join(x[:58] for x in regels if 'FOUT' in x), flush=True)
    e.to_parquet(f'elite_kernel_{bw}_{kn[0]}.parquet')
