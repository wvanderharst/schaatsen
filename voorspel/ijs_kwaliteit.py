"""IJskwaliteit per race: racepeil min het gewone peil van die baan x geslacht x afstand (mediaan van het volledige model, 2021+).
Verdeling (moet rond 0 liggen) voor het volledige model en de wekelijkse aanvulling, en hoe sterk races op dezelfde dag en baan
hetzelfde ijs hebben (aandeel gedeelde variantie, correlatie van paren). Gebruik: python ijs_kwaliteit.py <werkmap na aanvul.py>"""
import sys
import numpy as np, pandas as pd, scipy.stats as st, json
R=pd.read_parquet(sys.argv[1] + '/ritten_cat.parquet',columns=['gender','afstand','dag','baan','theta','event_id','seizoen','verdacht'])
for c in ('gender','afstand','baan'): R[c]=R[c].astype(str)
rk=['event_id','dag','gender','afstand']
X=R[~R.verdacht].groupby(rk).agg(theta=('theta','first'),n=('theta','size'),baan=('baan','first'),seizoen=('seizoen','first')).reset_index()
X['bron']=np.where(X.event_id%1000000>32104,'aanvulling','model')
X=X[X.seizoen>=2021]
# basiswaarde: mediaan per baan x geslacht x afstand over de 4 seizoenen ervoor (zoals de aanvulling), anders alle eerdere
X['cel']=X.baan+'|'+X.gender+'|'+X.afstand
base=X[X.bron=='model'].groupby('cel').theta.median()
X['q']=100*(X.theta-X.cel.map(base))
X=X[X.cel.map(X[X.bron=='model'].groupby('cel').size()).fillna(0)>=10]
out={}
for (b,groot),x in X.groupby(['bron',X.n>=14]):
    q=x.q.to_numpy(); med=np.median(q); s=1.4826*np.median(np.abs(q-med))
    out[f'{b}|{"14+" if groot else "<14"}']=dict(races=len(q),gemiddelde=round(q.mean(),3),mediaan=round(med,3),sd=round(q.std(),3),robuuste_sd=round(s,3),scheefheid=round(st.skew(q),2),kurtosis=round(st.kurtosis(q),2),
        binnen_1sd=round(np.mean(np.abs(q-med)<s),3),binnen_2sd=round(np.mean(np.abs(q-med)<2*s),3))
print(pd.DataFrame(out).T.to_string())
X.to_parquet(sys.argv[1] + '/ijs_q.parquet')
# zelfde dag, zelfde baan: races met >= 14 rijders
Y=X[X.n>=14].copy(); Y['dagbaan']=Y.dag.astype(str)+'|'+Y.baan
Y=Y[Y.groupby('dagbaan').q.transform('size')>=2]
for b,y in Y.groupby('bron'):
    g=y.groupby('dagbaan').q
    tot=y.q.var(); binnen=(y.q-g.transform('mean')).pow(2).sum()/(len(y)-g.ngroups)
    # paren
    p=[]
    for _,z in y.groupby('dagbaan'):
        v=z.q.to_numpy()
        for i in range(len(v)):
            for j in range(i+1,len(v)): p.append((v[i],v[j]))
    p=np.array(p); r=np.corrcoef(np.r_[p[:,0],p[:,1]],np.r_[p[:,1],p[:,0]])[0,1]
    d=np.abs(p[:,0]-p[:,1])
    # andere dag zelfde baan, zelfde week: ter vergelijking willekeurige paren zelfde baan andere dag
    print(f'{b}: {g.ngroups} dag x baan met >=2 races ({len(y)} races) | sd tussen races {np.sqrt(tot):.3f}%, binnen dezelfde dag+baan {np.sqrt(binnen):.3f}% | aandeel van de variantie dat de dag+baan deelt {1-binnen/tot:.2f} | correlatie paren {r:.2f} | mediaan |verschil| paar {np.median(d):.3f}%')
    # zelfde baan, andere dag binnen 7 dagen
    yy=y.assign(baan_=y.baan).sort_values('dag')
    dd=[]
    for _,z in yy.groupby('baan_'):
        m=z.groupby('dag').q.mean()
        dl=m.index.to_numpy(); v=m.to_numpy()
        for i in range(len(dl)-1):
            if dl[i+1]-dl[i]<=7: dd.append(abs(v[i+1]-v[i]))
    print(f'   ter vergelijking: zelfde baan, volgende wedstrijddag binnen 7 dagen (daggemiddelden): mediaan |verschil| {np.median(dd):.3f}% ({len(dd)} paren)')
