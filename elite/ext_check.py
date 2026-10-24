import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten_theta.parquet'); el=None
for kw in (dict(alpha=0.12,snel_w=1.0,gamma=0.0),dict(alpha=0.2,snel_w=1.0,gamma=0.2,hf=30.0),dict(alpha=0.2,snel_w=1.5,gamma=0.2,hf=30.0)):
    df=em.draai_ext(R.copy(),**kw)
    if el is None: el=em.kies_elite(df); top50=em.kies_elite(df,n_per=50)
    y=df[df.skater_id.isin(el)&(df.n>=10)&(df.seizoen>=2016)&np.isfinite(df.e)]
    print(kw, 'elite MAD-sd fout %.4f mediane s %.4f spar %.4f' % (1.4826*np.median(np.abs(y.e-y.e.median())), y.s.median(), y.spar.median()), flush=True)
    e=em.ijk(df,el); fout,regels,nr,nk=em.toets(e); print('  fouten',fout, '|', ' ; '.join(r for r in regels if 'FOUT' in r)[:600], flush=True)
    z=e[e.seizoen>=2016]
    for t,g in z.groupby(np.where(z.skater_id.isin(top50),'top 50','51-200')):
        print(f"   {t:7s} snel 1/20 {(g.p<=.05).mean()/.05:.2f}x 1/100 {(g.p<=.01).mean()/.01:.2f}x 1/1000 {(g.p<=1e-3).mean()/1e-3:.2f}x | traag 1/20 {(g.p>=.95).mean()/.05:.2f}x 1/100 {(g.p>=.99).mean()/.01:.2f}x")
    r=em.rijder(e,'Jutta LEERDAM'); print('   Leerdam laatste 20: snel <=1/100', r[1], '<=1/20', r[2], '| traag', r[3], r[4], flush=True)
