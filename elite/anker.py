import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R=pd.read_parquet('ritten.parquet'); el=None
for tau,vf,eta in ((0.015,0.01,0.05),(0.015,0.02,0.02),(0.01,0.02,0.02),(0.01,0.03,0.01)):
    df=em.draai(R.copy(), alpha=0.2, snel_w=1.0, gamma=0.2, hf=30.0, tau=tau, vfloor=vf, eta=eta)
    if el is None: el=em.kies_elite(df)
    h=df[(df.baan=='Heerenveen (NED)')&(df.afstand=='500m')&(df.gender=='v')&df.goed&(df.seizoen==2024)].groupby(['dag','event_id']).par.first()
    c=df[(df.baan=='Calgary (CAN)')&(df.afstand=='500m')&(df.gender=='v')&df.goed&(df.seizoen==2024)].groupby(['dag','event_id']).par.first()
    y=df[df.skater_id.isin(el)&df.goed&(df.n>=10)&(df.seizoen>=2016)]
    print(f"tau={tau} vfloor={vf} eta={eta}: Heerenveen-peil sd {h.std():.4f} (gem {h.mean():.3f}), Calgary gem {c.mean():.3f} -> Calgary-Hv {100*(c.mean()-h.mean()):+.1f}% | elite MAD-sd fout {1.4826*np.median(np.abs(y.e-y.e.median())):.4f}, mediane s {y.s.median():.4f}", flush=True)
