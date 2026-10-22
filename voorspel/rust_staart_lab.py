"""Proef op de opgeslagen uitkomst: rustlaag (taper) en niet-parametrische staartijking na het model.
Log loss op dezelfde ritten als het oude model (2020+), plus de niveaus vooraf (k_c) want niveau_pct van het oude model is de rating NA de race.
Tellingen snel/traag per niveau vooraf en per afstand (2020+).
Gebruik: python rust_staart_lab.py <uitkomst.parquet>"""
import os, sys, importlib.util
import numpy as np, pandas as pd, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rustlaag as rl, staart_np as sn
spec = importlib.util.spec_from_file_location('em', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite', 'elite_model.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)

pad = sys.argv[1]; w = os.path.dirname(pad) or '.'
V = pq.read_table(pad, columns=['skater_id', 'groep', 'dag', 'jaar', 'seizoen', 'afstand', 'tijd', 'pit', 'nll', 'ok', 'k_val', 'k_c', 'pred', 'theta', 'th_hat_veld']).to_pandas()
V['afstand'] = V['afstand'].astype(str)
K = pd.read_parquet(os.path.join(w, 'kenmerken.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'gap']); K['afstand'] = K['afstand'].astype(str)
V = V.merge(K.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
O = pd.read_parquet(os.path.join(w, 'oud_vooraf.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'll', 'niveau_pct'])
O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']); O['afstand'] = O['afstand'].astype(str)
V = V.merge(O, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
if os.environ.get('VERDACHT', '0') == '1':                  # ritten die in ritten_cat.parquet als verdacht staan (onmogelijk.py) weglaten
    Rv = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'verdacht']); Rv['afstand'] = Rv['afstand'].astype(str)
    Rv = Rv[Rv['verdacht']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd'])
    V = V.merge(Rv, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left'); V = V[V['verdacht'] != True].drop(columns='verdacht')
    print('zonder verdachte ritten', flush=True)
V = V[V['ok'] & np.isfinite(V['pit']) & V['afstand'].isin(sn.AFSTANDEN)].reset_index(drop=True)
del K, O
dag = V['dag'].to_numpy(); g = V['k_c'].to_numpy(); gap = V['gap'].to_numpy(); af = V['afstand'].to_numpy()
import plafond as pf
pgat = pf.gat(V['skater_id'].to_numpy(), V['afstand'].to_numpy(), V['dag'].to_numpy(), V['tijd'].to_numpy(), V['theta'].to_numpy(),
              (V['pred'] - V['th_hat_veld']).to_numpy(), ~V['k_val'].to_numpy() & np.isfinite(V['theta'].to_numpy()),
              W=float(os.environ.get('PLAFOND_W', '730')), k=int(os.environ.get('PLAFOND_K', '2')))
pgat2 = pf.gat(V['skater_id'].to_numpy(), V['afstand'].to_numpy(), V['dag'].to_numpy(), V['tijd'].to_numpy(), V['theta'].to_numpy(),
               (V['pred'] - V['th_hat_veld']).to_numpy(), ~V['k_val'].to_numpy() & np.isfinite(V['theta'].to_numpy()),
               W=float(os.environ.get('PLAFOND2_W', '1095')), k=int(os.environ.get('PLAFOND2_K', '2')))
import piek as pk
Rp = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan', 'tijd']); Rp['afstand'] = Rp['afstand'].astype(str)
eids = set(pk.toernooien(Rp)['eid'])
ev = V[['skater_id', 'dag', 'afstand', 'tijd']].merge(Rp[['skater_id', 'dag', 'afstand', 'tijd', 'event_id']].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']),
                                                    on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')['event_id'].fillna(-1).astype(np.int64).to_numpy()
xpiek = np.where(pk.markering(ev, eids), 1.0, np.nan); del Rp
print('piekritten', int(np.isfinite(xpiek).sum()), flush=True)
print('bewezen niveau: rijen met waarde', int(np.isfinite(pgat).sum()), 'mediaan %.2f%%' % np.nanmedian(pgat), flush=True)
geb = ~V['k_val'].to_numpy()
p0 = V['pit'].to_numpy(); n0 = V['nll'].to_numpy()

var = {'model': (p0, n0)}
RUST = {'rust': {}, 'rust schaal': {'schaal': True}, 'rust hl=365': {'hl': 365.0}, 'rust bw_r=.5': {'bw_r': 0.5}, 'rust k0=30': {'k0': 30.0}}
NP = {'np': {}, 'np k1=30': {'k1': 30.0}, 'np k1=10': {'k1': 10.0}, 'np bw=.01': {'bw': 0.01}, 'np k0=10': {'k0': 10.0},
      'np oud': {'a0': 0.0, 'a1': 0.0}, 'np a1=.25': {'a1': 0.25}, 'np a1=.5': {'a1': 0.5}, 'np a1=.5 a0=.25': {'a1': 0.5, 'a0': 0.25}, 'np a1=.5 a0=.5': {'a1': 0.5, 'a0': 0.5},
      'np a1=.75 a0=.25': {'a1': 0.75, 'a0': 0.25}, 'np a0=.25': {'a0': 0.25}, 'np a0=.5': {'a0': 0.5}, 'np a1=.25 a0=.25': {'a1': 0.25, 'a0': 0.25}, 'np a1=.5 a0=.25 hl=365': {'a1': 0.5, 'a0': 0.25, 'hl': 365.0}}
VARS = os.environ.get('VARS', 'rust,np,rust+np').split(',')
TOON = os.environ.get('TOON', 'model,rust,np,rust+np').split(',')
PLAFOND = {'plafond': {}, 'plafond bw=1': {'bw_x': 1.0}, 'plafond k0=30': {'k0': 30.0}, 'plafond hl=730': {'hl': 730.0}, 'plafond zonder schaal': {'schaal': False}}
RIJDER = {'rijder': {}, 'rijder Q=.1': {'Q': 0.1}, 'rijder k=10': {'k': 10.0}, 'rijder hl=365': {'hl': 365.0}, 'rijder hl=1460': {'hl': 1460.0}}
sid = V['skater_id'].to_numpy()
cache = {}
for v in VARS:
    p_, d_ = p0, np.zeros(len(p0)); pre = ''
    for laag in v.split('+'):
        pre = pre + '+' + laag
        if pre not in cache:
            if laag in RUST:
                pr, dr, _, _ = rl.rust(dag, g, gap, p_, geb, **RUST[laag])
            elif laag.startswith('piek'):
                kw_ = {'piek': {}, 'piek k0=10': {'k0': 10.0}, 'piek k0=100': {'k0': 100.0}, 'piek hl=2190': {'hl': 2190.0}, 'piek zonder schaal': {'schaal': False}}[laag]
                pr, dr, _, _ = rl.laag(dag, g, xpiek, p_, geb, np.array([0.0, 1.0]), 0.5, **{'k0': 30.0, **kw_})
            elif laag == 'plafond2':
                pr, dr, _, _ = rl.laag(dag, g, pgat2, p_, geb, pf.GX, 0.5, k0=30.0)
            elif laag in PLAFOND:
                kw_ = dict(PLAFOND[laag]); bwx = kw_.pop('bw_x', 0.5)
                pr, dr, _, _ = rl.laag(dag, g, pgat, p_, geb, pf.GX, bwx, **kw_)
            elif laag in RIJDER:
                pr, dr, _ = sn.rijder(dag, sid, p_, geb, **RIJDER[laag])
            else:
                pr, dr, _ = sn.ijk(dag, g, af, p_, geb, **NP[laag])
            cache[pre] = (pr, d_ + dr)
        p_, d_ = cache[pre]
    var[v] = (p_, n0 + d_)
print('varianten klaar', flush=True)

T = (V['jaar'].to_numpy() >= 2020)
M = T & np.isfinite(V['ll'].to_numpy())
pct = V['niveau_pct'].to_numpy(); sen = (V['groep'] == 'senior').to_numpy()
print('\nlog loss 2020+ (ritten van het oude model; toppers/elite naar rating na de race; vooraf: k_c < 6% / < 2%)')
for nm, (p, nl) in var.items():
    print(f"  {nm:12s} | alle {np.nanmean(nl[M]):.4f} | toppers {np.nanmean(nl[M & (pct >= 90)]):.4f} | elite {np.nanmean(nl[M & (pct >= 97)]):.4f}"
          f" | senior {np.nanmean(nl[M & sen]):.4f} | toppers vooraf {np.nanmean(nl[M & (g < 0.06)]):.4f} | elite vooraf {np.nanmean(nl[M & (g < 0.02)]):.4f}")

pk_ = np.isfinite(xpiek)
print('\nlog loss 2020+ op WK en Spelen (ritten van het oude model):')
for nm, (p, nl) in var.items():
    s_ = M & pk_
    print(f"  {nm:12s} | alle {np.nanmean(nl[s_]):.4f} (n={s_.sum()}) | top vooraf <2% {np.nanmean(nl[s_ & (g < 0.02)]):.4f} (n={(s_ & (g < 0.02)).sum()}) | snel 1/20 {(p[s_ & geb] < 0.05).mean() / 0.05:.2f} traag 1/20 {(p[s_ & geb] > 0.95).mean() / 0.05:.2f}")
niv = pd.cut(V['k_c'], [-1, 0.01, 0.02, 0.035, 0.06, 0.1, 0.18, 9], labels=['<1%', '1-2%', '2-3,5%', '3,5-6%', '6-10%', '10-18%', '>18%']).to_numpy()
Tc = T & geb


def tel(p, s):
    x = p[s]
    return ' '.join(f'{(x < q).mean() / q:4.2f}' for q in (0.1, 0.05, 0.01, 0.001, 2e-4, 1e-4)) + ' | ' + ' '.join(f'{(x > 1 - q).mean() / q:4.2f}' for q in (0.05, 0.01))


def deviantie(p):
    """Poisson-deviantie van de aantallen snelle ritten (kans <= 1/100, 1/1000, 1/5000) over niveau x afstand (2020+, zonder valpartijen)."""
    D = 0.0
    for lab in ['<1%', '1-2%', '2-3,5%', '3,5-6%', '6-10%', '10-18%', '>18%']:
        for a in sn.AFSTANDEN:
            x = p[Tc & (niv == lab) & (af == a)]
            for q in (0.01, 0.001, 2e-4):
                O = (x <= q).sum(); E = len(x) * q
                D += 2 * ((O * np.log(O / E) if O > 0 else 0.0) - (O - E))
    return D


print('\nPoisson-deviantie snelle staart (1/100, 1/1000, 1/5000) over 7 niveaus x 6 afstanden (126 cellen; ~126 bij perfecte kalibratie):')
for nm, (p, nl) in var.items():
    print(f'  {nm:28s} {deviantie(p):7.1f}')
print('\ntellingen 2020+ (waargenomen / verwacht): snel 1/10 1/20 1/100 1/1000 1/5000 1/10000 | traag 1/20 1/100')
for nm, (p, nl) in var.items():
    if nm not in TOON:
        continue
    print(f'  {nm}')
    for lab in ['<1%', '1-2%', '2-3,5%', '3,5-6%', '6-10%', '10-18%', '>18%']:
        s = Tc & (niv == lab); print(f'    niveau {lab:7s} n={s.sum():7d}  {tel(p, s)}')
    for a in sn.AFSTANDEN:
        s = Tc & (af == a); print(f'    {a:13s} n={s.sum():7d}  {tel(p, s)}')
    for a in sn.AFSTANDEN:
        s = Tc & (af == a) & (g < 0.035); print(f'    top {a:9s} n={s.sum():7d}  {tel(p, s)}')

print('\nkalibratietoets 2020+, niveau vooraf (incl. aantallen per rijder):')
for nm, (p, nl) in var.items():
    if nm not in TOON:
        continue
    for lab, s in (('top <2%', Tc & (g < 0.02)), ('top <6%', Tc & (g < 0.06))):
        x = V.loc[s, ['skater_id', 'seizoen']].copy(); x['p'] = p[s]
        fout, rg, nr, nk = em.toets(x, vanaf_seizoen=2019)
        print(f'  {nm:8s} {lab}: {nr} ritten, {fout} fouten')
        print('    ' + '\n    '.join(r for r in rg if r.startswith('snel')))
