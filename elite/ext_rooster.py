import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); el=None
grid=[dict(alpha=a,snel_w=sw,gamma=g,hf=h,tweezijdig=tz) for tz in (True,False) for a in (0.2,0.3) for sw in (1.5,2.0) for g in (0.2,0.4) for h in (30.0,60.0)]
for kw in grid:
    df=em.draai_ext(R.copy(),**kw)
    if el is None: el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
    e=em.ijk(df,el); fout,regels,nr,nk=em.toets(e); z=e[e.seizoen>=2016]
    t=np.where(z.skater_id.isin(top50),'t','r'); a_=z[t=='t']; b_=z[t=='r']
    r=em.rijder(e,'Jutta LEERDAM'); l=em.rijder(e,'Francesca LOLLOBRIGIDA')
    print(f"{kw} fouten {fout} | top50/rest snel1/100 {(a_.p<=.01).mean()/.01:.2f}/{(b_.p<=.01).mean()/.01:.2f} traag1/100 {(a_.p>=.99).mean()/.01:.2f}/{(b_.p>=.99).mean()/.01:.2f} | Leerdam20 snel 1/100 {r[1]} 1/20 {r[2]} | Lollo20 1/20 {l[2]} |", ' ; '.join(x.split(':')[0]+':'+x.split(':')[1][:22] for x in regels if 'FOUT' in x), flush=True)
