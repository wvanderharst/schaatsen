"""Trend voor jonge rijders (jeugd t/m neo): verwachte groei per categorie als lineaire functie van het niveau, geschat op de
voorspelfouten (seizoenen <= 2015, herhaald), plus optioneel eigen groei per rijder (kd). Toets: elite (ook apart hun jonge en latere
ritten) en niet-elite groepen gekozen op informatie vooraf (categorie en niveau vooraf), geijkt op alle niet-elite rijders per niveauband."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, elite_model as em
R = pd.read_parquet('ritten_theta.parquet'); basis = dict(alpha=0.2, snel_w=2.0, gamma=0.4, hf=30.0)
NIV = (0.02, 0.04, 0.06, 0.09, 0.13, 0.20); J = list(em.JONG)
def r4(y): return f"snel1/100 {(y.p <= .01).mean() / .01:.2f} traag1/100 {(y.p >= .99).mean() / .01:.2f} snel1/20 {(y.p <= .05).mean() / .05:.2f} traag1/20 {(y.p >= .95).mean() / .05:.2f}"
def toets_alles(df, label, elite):
    jd = df['groep'].isin(J)
    e_el = pd.concat([em.ijk(df[jd], elite), em.ijk(df[~jd], elite)]) if APART else em.ijk(df, elite)       # jonge ritten: eigen ijking en staart
    fe = em.toets(e_el)[0]; x = e_el[e_el.seizoen >= 2016]; jj = x.groep.isin(J)
    nb = np.digitize(df['g'], NIV); niet = set(df['skater_id'].unique())      # alle rijders: de elite is gekozen op later (selectie op de toekomst)
    ea = pd.concat([em.ijk(df[(nb == b) & (jd == j_)], niet, per_categorie=True).assign(nb=b) for b in range(len(NIV) + 1) for j_ in (True, False)])
    rg = [f"elite jong: {r4(x[jj])}", f"elite later: {r4(x[~jj])}"]
    for nm, t in (('senior <6%', ea[(ea.groep == 'senior') & (ea.nb <= 2)]), ('senior 6-13%', ea[(ea.groep == 'senior') & ea.nb.isin([3, 4])]),
                  ('jong <6%', ea[ea.groep.isin(J) & (ea.nb <= 2)]), ('jong 6-13%', ea[ea.groep.isin(J) & ea.nb.isin([3, 4])])):
        rg.append(f"{nm}: fouten {em.toets(t)[0]:2d} {r4(t[t.seizoen >= 2016])}")
    z = df[(df.seizoen >= 2016) & (df.n >= 3) & ~df.val & np.isfinite(df.e) & (df.e.abs() < .1)]
    print(f"{label}: elite fouten {fe}\n   " + '\n   '.join(rg) + f"\n   gemiddelde fout 2016+ (%): {z.groupby(z.groep.astype(str)).e.mean().mul(100).round(2).to_dict()}"
          f"\n   Leerdam laatste 20 <=1/30: {em.zeldzaam(e_el, 'Jutta LEERDAM')[0]}", flush=True)
import sys
APART = True
df = em.draai_ext(R.copy(), **basis); elite = set(em.kies_elite(df)); toets_alles(df, 'zonder trend, jong apart geijkt', elite)
kat = None
for ronde in range(3):
    kat = em.schat_trend(df, kat=kat, groepen=J); df = em.draai_ext(R.copy(), kat=kat, **basis)
    print(f'ronde {ronde + 1}:', {k: (round(v[0], 4), round(v[1], 3)) for k, v in sorted(kat.items())}, flush=True)
toets_alles(df, 'trend a + b x niveau, jong apart geijkt', elite)
for kd in (1.0,):
    toets_alles(em.draai_ext(R.copy(), kat=kat, kd=kd, **basis), f'trend + eigen groei kd={kd}', elite)
