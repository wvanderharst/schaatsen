import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
from scipy.stats import norm
R=pd.read_parquet('ritten_theta.parquet')
df=em.draai_ext(R.copy(), alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0); el=em.kies_elite(df)
e=em.ijk(df,el); fout,regels,_,_=em.toets(e); print('fouten', fout)
for nm in ('Jutta LEERDAM','Francesca LOLLOBRIGIDA','Femke KOK','Jordan STOLZ'):
    k,n=em.zeldzaam(e,nm); print(f"{nm}: laatste {n}: kans<=1/30 {k} (verwacht {n/15:.1f})")
x=e[e.skater_name.astype(str).str.contains('Jutta LEERDAM')].sort_values('dag').tail(20)
x=x.assign(d=pd.to_datetime(x.dag,unit='D').dt.date, kans=np.where(x.p<.5, '1 op '+(1/x.p).round(0).astype(int).astype(str)+' snel', '1 op '+(1/(1-x.p)).round(0).astype(int).astype(str)+' traag'))
print(x[['d','baan','afstand','tijd','e','s_tot','kans']].round(4).to_string(index=False))
e.to_parquet('elite_beste.parquet')
