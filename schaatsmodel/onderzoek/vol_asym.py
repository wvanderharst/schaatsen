import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
from scipy import stats, optimize
from scipy.special import expit
d=pd.read_pickle('gas_d.pkl'); d=d[np.isfinite(d.u)&~d.val]
r=pd.read_pickle('gas3_res_S.pkl')[['m','lf']]; d=d.join(r)
d['u2']=(d.u-d.m)/np.exp(d.lf); d['lsd2']=d.lsd+d.lf
d['band']=pd.cut(d['abs'],[0,50,75,90,95,98,100.1],labels=False,right=False)
d['rel']=d.lsd2-d.groupby(['ai','band']).lsd2.transform('median')
fit=((d.jr<=2019)&(d.jr>=2010)).to_numpy(); test=(d.jr>=2020).to_numpy()
U=d.u2.to_numpy(); REL=d.rel.to_numpy(); G=d.grp.to_numpy(); ev=d.ev.to_numpy(); top=(d['abs']>=90).to_numpy()
def tp(th,u):
    p0=expit(th[0]); sL,nL,sU,nU=np.exp(th[1]),1.5+28.5*expit(th[2]),np.exp(th[3]),1.5+28.5*expit(th[4])
    return np.where(u<0,np.log(2*p0)+stats.t.logpdf(u/sL,nL)-np.log(sL),np.log(2*(1-p0))+stats.t.logpdf(u/sU,nU)-np.log(sU))
rng=np.random.default_rng(0)
def ll(bf,bs):
    lam=np.where(U<0,bf*REL,bs*REL); e=U/np.exp(lam); out=np.zeros(len(U))
    for g in range(4):
        mg=G==g; pool=e[fit&mg]; s=rng.choice(pool,min(100000,len(pool)),replace=False)
        th=optimize.minimize(lambda t:-np.mean(tp(t,s)),[0,0,1,0,1],method='Nelder-Mead',options={'maxiter':1500,'xatol':1e-4,'fatol':1e-8}).x
        out[mg]=-(tp(th,e[mg])-lam[mg])
    return out
res={}
for bf,bs in [(0,0),(-0.2,0),(-0.3,0),(-0.4,0),(-0.5,0),(-0.3,0.1),(-0.3,-0.1)]:
    v=ll(bf,bs); res[(bf,bs)]=v; print(f'bf {bf:+.1f} bs {bs:+.1f}: fit {v[fit].mean():.5f} test {v[test].mean():.5f}',flush=True)
best=min(res,key=lambda k:res[k][fit].mean()); print('beste op fitjaren',best)
def se(x,e):
    g=pd.DataFrame({'x':x,'e':e}).groupby('e').x.agg(['sum','size']); b=[]
    for _ in range(200):
        k=rng.integers(0,len(g),len(g)); b.append(g['sum'].to_numpy()[k].sum()/g['size'].to_numpy()[k].sum())
    return np.std(b)
dd=(res[best]-res[(0,0)])[test]; tt=top[test]
print('test t.o.v. symmetrisch: %+.4f ± %.4f, toppers %+.4f ± %.4f'%(dd.mean(),se(dd,ev[test]),dd[tt].mean(),se(dd[tt],ev[test][tt])),flush=True)
json.dump({'bf':best[0],'bs':best[1]},open('vol_asym.json','w'))
