"""Niveau per rijder met een Kalmanfilter: gemeenschappelijk niveau c (alle afstanden samen) + afwijking per afstand d_a (specialisatie),
gemeten als z = ln tijd - theta - ref (ln-tijd boven de wereldtop op die afstand, zodat afstanden vergelijkbaar zijn).
  voorspellen : c groeit met de verwachte verandering per jaar g(leeftijd) = G1 exp(-(leeftijd - 12) / TG) + G2 max(leeftijd - 35, 0) / 10,
                de onzekerheid van c met Q(leeftijd) = (Q0 + Q1 exp(-(leeftijd - 12) / TQ)) per jaar, die van d_a met QA per jaar;
                vorm f (alle afstanden, uitdovend met HF dagen) komt erbij.
  bijwerken   : fout e = z - c - d_a - f, met variantie S = Pc + Pa + R (+ onzekerheid van theta); R = KR x (eigen spreiding)^2.
                c en d_a krijgen elk hun deel (Pc / S en Pa / S), Huber-geknipt op HUB x wortel(S); snelle ritten tellen SNEL x zo zwaar;
                valpartijen (e > 6 wortel(S) en > 3%) tellen niet.
Q0, KR, SNEL en GAMMA zijn functies van het niveau: p x exp(PG x exp(-niveau / GW)) (PG = QG, KRG, SNG, GAG), zodat toppers eigen waarden krijgen.
Alles chronologisch: per rit worden alleen ritten van eerdere races gebruikt."""
import math
import numpy as np, pandas as pd
from numba import njit

AF = ['100m', '300m', '500m', '1000m', '1500m', '3000m', '5000m', '10000m']
CD = np.array([0.012, 0.012, 0.011, 0.0134, 0.015, 0.0161, 0.0154, 0.016])
# standaardparameters (afgestemd in kalman_tune.py op de jaren <= 2019)
PAR = dict(G1=-0.075, TG=2.2, G2=0.005, Q0=0.0001, Q1=0.0016, TQ=3.0, QA=0.00005, P0C=0.0009, P0A=0.0004, KR=0.8, HUB=3.0, SNEL=1.3,
           GAMMA=0.3, HF=30.0, M0R=25.0, BETA=0.06, QG=0.0, KRG=0.0, SNG=0.0, GAG=0.0, GW=0.04)
NAMEN = list(PAR)


@njit(cache=True)
def _kalman(rk, rid, a, z, vth, dag, lf, nr, p, o_zh, o_vl, o_r, o_e, o_val, o_na, o_c, o_d, o_f, o_pc):
    G1, TG, G2, Q0_, Q1, TQ, QA, P0C, P0A, KR_, HUB, SNEL_, GAMMA_, HF, M0R, BETA = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]
    QG, KRG, SNG, GAG, GW = p[16], p[17], p[18], p[19], p[20]
    NA = 8
    c = np.zeros(nr); pc = np.zeros(nr); tc = np.full(nr, -1.0); started = np.zeros(nr, np.bool_)
    d = np.zeros((nr, NA)); pa = np.full((nr, NA), P0A); ta = np.full((nr, NA), -1.0); na = np.zeros((nr, NA))
    f = np.zeros(nr); fd = np.zeros(nr); r2 = np.ones(nr); nr2 = np.zeros(nr)
    N = len(rk); i = 0
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        for q in range(i, j):
            r_ = rid[q]; k = a[q]; o_na[q] = na[r_, k]
            if not started[r_]:
                o_zh[q] = np.nan; o_vl[q] = np.nan; o_e[q] = np.nan; o_val[q] = False; o_r[q] = np.nan
                o_c[q] = np.nan; o_d[q] = np.nan; o_f[q] = 0.0; o_pc[q] = np.nan
                continue
            age = lf[q] if lf[q] == lf[q] else 25.0
            wg = math.exp(-max(c[r_] + d[r_, k], 0.0) / GW)          # 1 bij de wereldtop, ~0 ver erachter: parameters als functie van het niveau
            Q0 = Q0_ * math.exp(QG * wg); KR = KR_ * math.exp(KRG * wg); SNEL = SNEL_ * math.exp(SNG * wg); GAMMA = GAMMA_ * math.exp(GAG * wg)
            dtc = max(dag[q] - tc[r_], 0.0) / 365.0
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * math.exp(-(age - 12.0) / TQ)
            cp = c[r_] + gr * min(dtc, 1.5); pcp = pc[r_] + qc * dtc
            if ta[r_, k] < 0:
                pap = P0A
            else:
                pap = pa[r_, k] + QA * max(dag[q] - ta[r_, k], 0.0) / 365.0
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R)
            R = KR * CD[k] * CD[k] * rr
            o_zh[q] = cp + d[r_, k] + fc; o_vl[q] = pcp + pap; o_r[q] = R; o_c[q] = cp; o_d[q] = d[r_, k]; o_f[q] = fc; o_pc[q] = pcp
            if z[q] != z[q]:
                o_e[q] = np.nan; o_val[q] = False
            else:
                S = pcp + pap + R + vth[q]
                e = z[q] - o_zh[q]; o_e[q] = e
                o_val[q] = (e > 6.0 * math.sqrt(S)) and (e > 0.03)
        for q in range(i, j):
            r_ = rid[q]; k = a[q]
            if z[q] != z[q] or o_val[q]:
                continue
            if not started[r_]:                     # eerste rit ooit: start van het gemeenschappelijke niveau
                c[r_] = z[q]; pc[r_] = P0C; tc[r_] = dag[q]; started[r_] = True
                d[r_, k] = 0.0; pa[r_, k] = P0A; ta[r_, k] = dag[q]; na[r_, k] = 1.0
                continue
            # (twee ritten van dezelfde rijder in één race: de tweede gebruikt de stand na de eerste)
            age = lf[q] if lf[q] == lf[q] else 25.0
            wg = math.exp(-max(c[r_] + d[r_, k], 0.0) / GW)          # 1 bij de wereldtop, ~0 ver erachter: parameters als functie van het niveau
            Q0 = Q0_ * math.exp(QG * wg); KR = KR_ * math.exp(KRG * wg); SNEL = SNEL_ * math.exp(SNG * wg); GAMMA = GAMMA_ * math.exp(GAG * wg)
            dtc = max(dag[q] - tc[r_], 0.0) / 365.0
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * math.exp(-(age - 12.0) / TQ)
            cp = c[r_] + gr * min(dtc, 1.5); pcp = pc[r_] + qc * dtc
            pap = P0A if ta[r_, k] < 0 else pa[r_, k] + QA * max(dag[q] - ta[r_, k], 0.0) / 365.0
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R)
            R = KR * CD[k] * CD[k] * rr
            S = pcp + pap + R + vth[q]
            e = z[q] - cp - d[r_, k] - fc
            ec = min(max(e, -HUB * math.sqrt(S)), HUB * math.sqrt(S))
            kc = pcp / S; ka = pap / S
            if e < 0:
                kc = min(kc * SNEL, 0.9); ka = min(ka * SNEL, 0.9)
            c[r_] = cp + kc * ec; pc[r_] = pcp * (1.0 - min(kc, 0.95)); tc[r_] = dag[q]
            d[r_, k] = d[r_, k] + ka * ec; pa[r_, k] = pap * (1.0 - min(ka, 0.95)); ta[r_, k] = dag[q]; na[r_, k] += 1.0
            rest = ec * (1.0 - kc - ka)
            f[r_] = fc + GAMMA * rest; fd[r_] = dag[q]
            br = max(1.0 / (nr2[r_] + 1.0), BETA)
            r2[r_] = (1 - br) * r2[r_] + br * min(e * e, (HUB * CD[k]) ** 2 * 4) / (CD[k] * CD[k]); nr2[r_] += 1.0
        i = j
    return 0


def voorbereid(df):
    """Sorteren en de vaste invoer voor de pass klaarzetten (eenmalig; daarna kan de pass vaak draaien bij het afstemmen)."""
    df = df.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    rk = df.groupby(['dag', 'event_id', 'gender', 'afstand_idx'], sort=False).ngroup().to_numpy().astype(np.int64)
    rid, _ = pd.factorize(df['skater_id'])
    lt = np.log(df['tijd'].to_numpy(float)); y = np.where(df['verdacht'].to_numpy(), np.nan, lt - df['theta'].to_numpy(float))
    fitset = (df['jaar'] <= 2019) & np.isfinite(y)
    ref = pd.Series(y[fitset.to_numpy()]).groupby([df.loc[fitset, 'gender'].to_numpy(), df.loc[fitset, 'afstand_idx'].to_numpy()]).quantile(0.01)
    refq = np.array([ref.get((g, k), np.nan) for g, k in zip(df['gender'], df['afstand_idx'])])
    lf = (df['leeftijd'] + ((df['dag'] - (df['seizoen'] - 1970) * 365.25 - 182) / 365.25).clip(0, 1)).to_numpy(float)
    inv = dict(rk=rk, rid=rid.astype(np.int64), a=df['afstand_idx'].to_numpy().astype(np.int64), z=y - refq,
               vth=np.nan_to_num(df['v_theta'].to_numpy(float)), dag=df['dag'].to_numpy(float), lf=lf, nr=int(rid.max() + 1), refq=refq,
               seiz=df['seizoen'].to_numpy().astype(np.int64))
    return df, inv


def pass_(inv, par=None):
    p = dict(PAR); p.update(par or {})
    pv = np.array([p[k] for k in NAMEN], float)
    N = len(inv['rk']); o = {k: np.zeros(N) for k in ('zh', 'vl', 'r', 'e', 'na', 'c', 'd', 'f', 'pc')}; val = np.zeros(N, np.bool_)
    _kalman(inv['rk'], inv['rid'], inv['a'], inv['z'], inv['vth'], inv['dag'], inv['lf'], inv['nr'], pv,
            o['zh'], o['vl'], o['r'], o['e'], val, o['na'], o['c'], o['d'], o['f'], o['pc'])
    o['val'] = val
    return o
