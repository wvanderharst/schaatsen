import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
df=em.draai(pd.read_parquet('ritten.parquet'), alpha=0.2, snel_w=1.0, gamma=0.2, hf=30.0)
el=em.kies_elite(df); e=em.ijk(df,el); x=e[e.seizoen>=2016].copy()
top50=em.kies_elite(df,n_per=50)
x['tier']=np.where(x.skater_id.isin(top50),'top 50','51-200')
for t,g in x.groupby('tier'):
    print(f"{t:7s} ritten {len(g):6d} | snel 1/20 {(g.p<=.05).mean()/.05:.2f}x 1/100 {(g.p<=.01).mean()/.01:.2f}x 1/1000 {(g.p<=1e-3).mean()/1e-3:.2f}x | traag 1/20 {(g.p>=.95).mean()/.05:.2f}x 1/100 {(g.p>=.99).mean()/.01:.2f}x")
k=x.groupby('skater_id').agg(k=('p',lambda p:(p<=.02).sum()),n=('p','size'),s=('s','median'),spar=('spar','median'),rms=('e',lambda e:np.sqrt(np.mean(np.clip(e,-.1,.1)**2))),k_el=('k','median'),naam=('skater_name','first'))
k['rms/s']=k.rms/np.sqrt(k.s**2+k.spar.fillna(0)**2)
print(k.sort_values('k',ascending=False).head(10)[['naam','n','k','s','spar','rms/s','k_el']].round(4).to_string())
print('rms/s mediaan alle elite', round(k['rms/s'].median(),2), '| bij >=8 schokken', round(k[k.k>=8]['rms/s'].median(),2))
