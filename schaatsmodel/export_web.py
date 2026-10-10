import warnings; warnings.filterwarnings('ignore')
import pickle, json, math, time, numpy as np, pandas as pd, do_analysis as da
t0=time.time()
m=da.laad_volledig('model_full_tuned.pkl')
h=da.Historie(m,dict(np.load('hist_nieuw.npz',allow_pickle=True)))
ZN=pd.read_pickle('normaal_z_dag.pkl')
O=m.obs
VANDAAG=int((pd.Timestamp('2026-10-10')-pd.Timestamp('1970-01-01')).days)
MAXD=int(O.dag.max()); AF=['500m','1000m','1500m','3000m','5000m','10000m']
laatste=O[~O.verdacht].groupby('skater_id').dag.max()
actief=set(laatste[laatste>=MAXD-730].index.astype(int))
print('actief',len(actief),'max dag',pd.to_datetime(MAXD,unit='D').date(),flush=True)
# kandidaten per geslacht x afstand
sel=set(); rang={}
for a in AF:
    j=da.NAAR_INDEX[a]
    rows=[]
    for s in actief:
        st=m.staten[s]
        if st.n[j]<3: continue
        try: v=m.voorspel(s,a,VANDAAG)
        except Exception: continue
        rows.append((s,st.gender,v['normale_tijd']))
    df=pd.DataFrame(rows,columns=['s','g','t'])
    for g,x in df.groupby('g'):
        top=x.sort_values('t').head(150)
        sel|=set(top.s); rang[(g,a)]=list(top.s)
    print(a,len(df),'kandidaten',round(time.time()-t0),flush=True)
print('geselecteerd',len(sel),flush=True)
pool=da.empirische_z(m)
zq={a:[round(float(x),4) for x in np.quantile(pool[a],np.linspace(0,1,401))] for a in AF}
PLO={a_:float(m.piek_drempel[da.PIEK_GIDX[a_]][0]) for a_ in AF} if getattr(m,'piek_drempel',None) else {}
def zk(a,p):          # zelfde interpolatie als zKwantiel in de pagina
    q=zq[a]; x=p*(len(q)-1); i=min(int(x),len(q)-2); f=x-i; return q[i]+f*(q[i+1]-q[i])
def t_bij(v,a,p,anker):   # zelfde formule als tijdBij in de pagina (eigen piekdiepte en snelle schaal)
    z=zk(a,p); lo=PLO.get(a)
    if lo is not None and z<lo: z=lo+(z-lo)*v.get('piek_diepte',1.0)
    if z<0: z*=v.get('snel_fac',1.0)
    return (v['t_lim']+math.exp(v['mu_log']+v['sigma_log']*z))*math.exp(anker)
def rijder(s):
    """Alles voor één rijder: voorspellingen nu, uitslagen met verwachting vooraf en het traject met 80%-band."""
    st=m.staten[s]; o=O[(O.skater_id==s)&(~O.verdacht)]
    pr={}
    for a in AF:
        j=da.NAAR_INDEX[a]
        if st.n[j]<1: continue
        v=m.voorspel(s,a,VANDAAG)
        oa=o[o.afstand==a]
        pb=float(oa.tijd.min()) if len(oa) else None
        sb=oa[oa.dag>=MAXD-330].tijd.min() if len(oa) else None
        pr[a]=dict(dp=round(v.get('piek_diepte',1.0),3),n=int(st.n[j]),norm=round(v['normale_tijd'],3),piek=round(v['piek_tijd'],3),mu=round(v['mu_log'],5),sd=round(v['sigma_log'],5),sf=round(v.get('snel_fac',1.0),4),vol=(round(v.get('snel_fac_vol',1.0)**(1/m.params.vol_asym_bf),3) if m.params.vol_asym_bf else None),vf=round(v.get('vorm_snel',1.0),3),tl=round(v['t_lim'],3),
                   pb=round(pb,2) if pb else None,sb=(round(float(sb),2) if sb==sb and sb is not None else None))
    # resultaten laatste 3 jaar + verwachting (normaal) na elk event op de gereden afstand
    r=o[o.dag>=MAXD-3*365].sort_values('dag')
    res=[]
    for _,q in r.iterrows():
        a=q.afstand
        if a not in AF: continue
        try: v=h.voorspel(s,a,int(q.dag))          # verwachting vooraf (voor de race)
        except KeyError: v=None
        res.append([int(q.dag),q.baan,a,round(float(q.tijd),2),None if pd.isna(ZN.get(q.Index if hasattr(q,'Index') else q.name,np.nan)) else round(float(ZN.get(q.name,np.nan)),2),None if v is None else round(v['normale_tijd'],2)])
    # verloop: verwachting (incl. tijdelijke vorm) en vaste rating, op racedagen (voor en na) en maandelijks tot vandaag
    tr={}
    dagen_r=sorted(set(int(d) for d in r.dag))
    grid=list(range(MAXD-3*365, VANDAAG, 30))+[VANDAAG]
    for a in AF:
        ra=r[r.afstand==a]
        if len(ra)<2: continue
        d_start=int(ra.dag.min())-14
        ds=sorted(set([d for d in dagen_r]+[d+1 for d in dagen_r]+grid))
        pts=[]
        for d in ds:
            if d<d_start: continue
            try: v=h.voorspel(s,a,int(d))
            except KeyError: continue
            nt=round(v['normale_tijd'],2); ak=m.baan_effect(da.STANDAARD_BAAN,int(d))[0]
            # 80%-band (10% en 90%) als afwijking van de verwachting in honderdsten
            pts.append([int(d),nt,round(v['normale_basis'],2),int(round(100*(t_bij(v,a,.1,ak)-nt))),int(round(100*(t_bij(v,a,.9,ak)-nt)))])
        tr[a]=pts
    kans=da.piek_kansen(m,s)
    return dict(id=int(s),naam=st.naam,land=st.land,g=st.gender,groep=st.groep,pr=pr,res=res,tr=tr,pk=round(float(kans[0]),4),sl=round(float(kans[2]),4))
# rijders parallel (fork: het model wordt gedeeld); eerst één voorspelling zodat de caches (empirische z, baan per dag) er al zijn
m.voorspel(next(iter(sel)),'500m',VANDAAG); h.voorspel(next(iter(sel)),'500m',VANDAAG)
import multiprocessing as mp, os
NPROC=int(os.environ.get('EXPORT_PROCESSEN',min(4,os.cpu_count() or 1)))
if NPROC>1:
    with mp.get_context('fork').Pool(NPROC) as pl: rijders=list(pl.imap(rijder,sorted(sel),chunksize=4))
else:
    rijders=[rijder(s) for s in sorted(sel)]
print('rijders klaar',round(time.time()-t0),flush=True)
# banen
ev=m.events; recent=ev[ev.dag>=MAXD-3*365].groupby('baan').size()
banen=[]
for b,n in recent.sort_values(ascending=False).items():
    if n<2 or not b: continue
    mu,sd,ne=m.baan_effect(b,VANDAAG)
    banen.append(dict(naam=b,mu=round(mu,5),sd=round(sd,5),n=int(n),pct=round(100*(math.exp(mu-m.anker_std)-1),2)))
pk_mu,pk_sd=m.piekdag_statistiek()
meta=dict(piek_lo={a_:round(float(m.piek_drempel[da.PIEK_GIDX[a_]][0]),4) for a_ in AF} if getattr(m,'piek_drempel',None) else {},vol_bf=float(m.params.vol_asym_bf),piek_q=list(m.params.piek_q),vandaag=VANDAAG,data_tot=MAXD,anker=round(m.anker_std,6),piek_mu=round(pk_mu,5),piek_sd=round(pk_sd,5),
          loc_schaal_jaren=list(m.loc_schaal_tab['jaren']),afstanden=AF,
          ranglijst={f'{g}|{a}':v for (g,a),v in rang.items()})
t=m.loc_schaal_tab
meta['loc']={g:{a:[round(float(v),3) for v in t['loc'][gi,da.NAAR_INDEX[a]]] for a in AF} for gi,g in enumerate(['m','v'])}
meta['sc']={g:{a:[round(float(v),3) for v in t['sc'][gi,da.NAAR_INDEX[a]]] for a in AF} for gi,g in enumerate(['m','v'])}
meta['banden']=['0–25','25–50','50–75','75–90','90–95','95–98','98–100']
meta['trend_coef']=dict(zip(da.TREND_KOL,[round(float(c),3) for c in m.trend_coef]))
json.dump(dict(meta=meta,rijders=rijders,banen=banen,zq=zq),open('web_model.json','w'),separators=(',',':'),ensure_ascii=False)
import os; print('json MB',round(os.path.getsize('web_model.json')/1e6,2),'tijd',round(time.time()-t0))
