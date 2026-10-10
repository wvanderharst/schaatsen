import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
df=em.draai(pd.read_parquet('ritten.parquet'), alpha=0.12, snel_w=1.0, gamma=0.0)
elite=em.kies_elite(df)
# veldsterkte: aantal elite-rijders in de race (model-vrij) en mediane mu van de bekende rijders
df['is_el']=df.skater_id.isin(elite)
key=['dag','event_id','gender','afstand_idx']
g=df.groupby(key)
df['n_el']=g.is_el.transform('sum')
e=df[df.is_el&df.goed&np.isfinite(df.u)&~df.val&(df.n>=3)&(df.seizoen>=2012)]
e=e.assign(nb=pd.cut(e.n_el,[0,3,8,15,24,1000],labels=['1-3','4-8','9-15','16-24','25+']))
print(e.groupby('nb').apply(lambda x: pd.Series({'ritten':len(x),'gem u':x.u.mean(),'mediaan u':x.u.median(),'u<-2.05 (%)':100*(x.u<-2.054).mean(),'u<-3.1 (‰)':1000*(x.u<-3.09).mean(),'u>2.05 (%)':100*(x.u>2.054).mean()})).round(2).to_string())
