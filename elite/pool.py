import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); basis=dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
el=top50=None
for lam,m0r in ((1.0,5.0),(0.0,5.0),(0.0,15.0),(0.0,25.0),(0.0,40.0),(0.15,20.0),(0.3,10.0),(0.5,10.0)):
    df=em.draai_ext(R.copy(), lam_af=lam, m0r=m0r, **basis)
    if el is None: el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
    e=em.ijk(df,el); fout,regels,_,_=em.toets(e); z=e[e.seizoen>=2016]; t=z.skater_id.isin(top50).to_numpy()
    print(f"lam_af {lam} m0r {m0r}: fouten {fout} |", ' ; '.join(x[:48] for x in regels if 'FOUT' in x))
    print('   alle: snel 1/100 %.2fx 1/1000 %.2fx | traag 1/100 %.2fx 1/1000 %.2fx' % ((z.p<=.01).mean()/.01,(z.p<=1e-3).mean()/1e-3,(z.p>=.99).mean()/.01,(z.p>=1-1e-3).mean()/1e-3))
    for g,s in (('top50',t),('51-200',~t)):
        y=z[s]; print(f"   {g:6s}: snel 1/20 {(y.p<=.05).mean()/.05:.2f}x 1/100 {(y.p<=.01).mean()/.01:.2f}x | traag 1/20 {(y.p>=.95).mean()/.05:.2f}x 1/100 {(y.p>=.99).mean()/.01:.2f}x")
    # spreiding voorspelt de gerealiseerde spreiding? kwartielen
    zz=z.copy(); zz['q']=pd.qcut(zz.s/np.asarray(em.CD)[zz.afstand_idx],4,labels=False); zz['ae']=np.abs(zz.e)/np.asarray(em.CD)[zz.afstand_idx]
    print('   s/cd per kwart vooraf vs 1.25*mediaan|e|/cd:', ' '.join(f"{a:.2f}->{b:.2f}" for a,b in zip(zz.groupby('q').apply(lambda x:(x.s/np.asarray(em.CD)[x.afstand_idx]).median()), 1.4826*zz.groupby('q').ae.median())))
    print('   Leerdam laatste 20 <=1/30:', em.zeldzaam(e,'Jutta LEERDAM')[0], '| Lollobrigida:', em.zeldzaam(e,'Francesca LOLLOBRIGIDA')[0], '| Kok:', em.zeldzaam(e,'Femke KOK')[0], '| Stolz:', em.zeldzaam(e,'Jordan STOLZ')[0], flush=True)
    e.to_parquet(f'elite_pool_{lam}_{m0r}.parquet')
