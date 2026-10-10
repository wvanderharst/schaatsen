import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
x=df[df.skater_id.isin(el)&(df.seizoen>=2012)&np.isfinite(df.u)&(df.n>=10)]
for hl in (True,False):
    y=x[x.baan.isin(em.HOOGLAND)==hl]; print(f"{'hoogland' if hl else 'overig'}: spreiding van u (q84-q16)/2 = {(np.quantile(y.u,.84)-np.quantile(y.u,.16))/2:.2f}, mediaan {y.u.median():+.2f}")
for lab,d in (('zonder ijs-correctie',df),('met meeschuivende ijs-correctie',em.ijs_rollend(df,jaren=4)[0])):
    e=em.ijk(d,el); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016].copy(); z['hl']=z.baan.isin(em.HOOGLAND)
    print(lab, f'fouten {fout}:', ' ; '.join(x[:58] for x in regels if 'FOUT' in x))
    for grp,s in (('elite',np.ones(len(z),bool)),('top 50',z.skater_id.isin(top50).to_numpy())):
        for hl in (True,False):
            y=z[s&(z.hl==hl)]
            print(f"   {grp:6s} {'hoogland' if hl else 'overig  '}: {len(y)} | snel 1/20 {(y.p<=.05).mean()/.05:.2f}x 1/100 {(y.p<=.01).mean()/.01:.2f}x | traag 1/20 {(y.p>=.95).mean()/.05:.2f}x 1/100 {(y.p>=.99).mean()/.01:.2f}x")
    print('   Leerdam laatste 20, kans <= 1/30:', em.zeldzaam(e,'Jutta LEERDAM')[0], flush=True)
    e.to_parquet(f"elite_hoogcal_{'ijs' if 'met' in lab else 'basis'}.parquet")
