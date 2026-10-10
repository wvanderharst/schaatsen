import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); kw=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
df=em.draai_ext(R.copy(), **kw); el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
for lam,m in ((0.05,2.0),(0.1,2.0),(0.1,5.0),(0.2,3.0)):
    d=em.dag_effect_niveau(df, lam=lam, m=m)
    e=em.ijk(d,el); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016].copy(); z['hl']=z.baan.isin(em.HOOGLAND); t=z.skater_id.isin(top50)
    print(f"lam {lam} m {m}: fouten {fout} |", ' ; '.join(x[:50] for x in regels if 'FOUT' in x))
    for grp,s in (('top 50',t.to_numpy()),('51-200',~t.to_numpy())):
        for hl in (True,False):
            y=z[s&(z.hl==hl)]
            print(f"   {grp:6s} {'hoogland' if hl else 'overig  '}: snel 1/20 {(y.p<=.05).mean()/.05:.2f}x 1/100 {(y.p<=.01).mean()/.01:.2f}x | traag 1/20 {(y.p>=.95).mean()/.05:.2f}x 1/100 {(y.p>=.99).mean()/.01:.2f}x")
    print('   Leerdam laatste 20, kans <= 1/30:', em.zeldzaam(e,'Jutta LEERDAM')[0], '| Lollobrigida:', em.zeldzaam(e,'Francesca LOLLOBRIGIDA')[0], flush=True)
    e.to_parquet(f'elite_dagniv_{lam}_{m}.parquet')
