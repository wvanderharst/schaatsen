"""Niveau per rijder met een multivariaat Kalmanfilter over alle 8 afstanden (100 m .. 10 km), gemeten als z = ln tijd - theta - ref
(ln-tijd boven de wereldtop op die afstand). Toestand x (8) met covariantie P (8 x 8) per rijder.
  start       : bij de eerste rit ooit x = z op alle afstanden, P = P0C x C(RHO0) + P0A x I, met C(rho)_ij = rho^|i-j|.
  voorspellen : x groeit met g(leeftijd) = G1 exp(-(leeftijd - 12) / TG) + G2 max(leeftijd - 35, 0) / 10 per jaar (op alle afstanden),
                P += dt x (Q(leeftijd) C(RHO) + QA I), Q(leeftijd) = Q0 + Q1 exp(-(leeftijd - 12) / TQ); vorm f (alle afstanden, HF dagen).
  bijwerken   : fout e = z - x_j - f, S = P_jj + R (+ onzekerheid theta), R = KR x CD_j^2 x eigen spreiding; K = P[:, j] / S;
                Huber-geknipt op HUB x wortel(S); snelle ritten tellen SNEL x zo zwaar; valpartijen (e > 6 wortel(S) en > 3%) tellen niet.
Zelfde invoer en uitvoer als kalman.py (voorbereid, pass_)."""
import math
import numpy as np
from numba import njit
from kalman import voorbereid, CD

PAR = dict(G1=-0.062, TG=2.6, G2=0.0044, Q0=0.00012, Q1=0.00066, TQ=3.4, QA=0.00005, RHO=0.85, RHO0=0.9, P0C=0.0146, P0A=0.0028,
           KR=0.69, HUB=3.0, SNEL=1.57, GAMMA=0.38, HF=2.4, M0R=25.0, BETA=0.06)
NAMEN = list(PAR)


@njit(cache=True)
def _kalman_mv(rk, rid, a, z, vth, dag, lf, nr, p, o_zh, o_vl, o_r, o_e, o_val, o_na, o_c, o_d, o_f, o_pc):
    G1, TG, G2, Q0, Q1, TQ, QA, RHO, RHO0, P0C, P0A, KR, HUB, SNEL, GAMMA, HF, M0R, BETA = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                                                                                         p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17])
    NA = 8
    C = np.zeros((NA, NA)); C0 = np.zeros((NA, NA))
    for i_ in range(NA):
        for j_ in range(NA):
            C[i_, j_] = RHO ** abs(i_ - j_); C0[i_, j_] = P0C * RHO0 ** abs(i_ - j_) + (P0A if i_ == j_ else 0.0)
    x = np.zeros((nr, NA)); P = np.zeros((nr, NA, NA)); tl = np.full(nr, -1.0); started = np.zeros(nr, np.bool_); na = np.zeros((nr, NA))
    f = np.zeros(nr); fd = np.zeros(nr); r2 = np.ones(nr); nr2 = np.zeros(nr)
    xp = np.zeros(NA); Pp = np.zeros((NA, NA)); Kv = np.zeros(NA)
    N = len(rk); i = 0
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        for q in range(i, j):                      # voorspellen (stand vóór de race)
            r_ = rid[q]; k = a[q]; o_na[q] = na[r_, k]
            if not started[r_]:
                o_zh[q] = np.nan; o_vl[q] = np.nan; o_e[q] = np.nan; o_val[q] = False; o_r[q] = np.nan
                o_c[q] = np.nan; o_d[q] = np.nan; o_f[q] = 0.0; o_pc[q] = np.nan
                continue
            age = lf[q] if lf[q] == lf[q] else 25.0
            dt = max(dag[q] - tl[r_], 0.0) / 365.0
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * math.exp(-(age - 12.0) / TQ)
            xk = x[r_, k] + gr * min(dt, 1.5); pkk = P[r_, k, k] + dt * (qc + QA)
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R); R = KR * CD[k] * CD[k] * rr
            o_zh[q] = xk + fc; o_vl[q] = pkk; o_r[q] = R; o_c[q] = xk; o_d[q] = 0.0; o_f[q] = fc; o_pc[q] = pkk
            if z[q] != z[q]:
                o_e[q] = np.nan; o_val[q] = False
            else:
                S = pkk + R + vth[q]; e = z[q] - o_zh[q]; o_e[q] = e
                o_val[q] = (e > 6.0 * math.sqrt(S)) and (e > 0.03)
        for q in range(i, j):                      # bijwerken
            r_ = rid[q]; k = a[q]
            if z[q] != z[q] or o_val[q]:
                continue
            if not started[r_]:
                for m_ in range(NA):
                    x[r_, m_] = z[q]
                    for n_ in range(NA):
                        P[r_, m_, n_] = C0[m_, n_]
                tl[r_] = dag[q]; started[r_] = True; na[r_, k] = 1.0
                continue
            age = lf[q] if lf[q] == lf[q] else 25.0
            dt = max(dag[q] - tl[r_], 0.0) / 365.0
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * math.exp(-(age - 12.0) / TQ)
            for m_ in range(NA):
                xp[m_] = x[r_, m_] + gr * min(dt, 1.5)
                for n_ in range(NA):
                    Pp[m_, n_] = P[r_, m_, n_] + dt * (qc * C[m_, n_] + (QA if m_ == n_ else 0.0))
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R); R = KR * CD[k] * CD[k] * rr
            S = Pp[k, k] + R + vth[q]
            e = z[q] - xp[k] - fc
            ec = min(max(e, -HUB * math.sqrt(S)), HUB * math.sqrt(S))
            boost = 1.0
            if e < 0:
                boost = min(SNEL, 0.9 * S / max(Pp[k, k], 1e-12))
            for m_ in range(NA):
                Kv[m_] = Pp[m_, k] / S
            for m_ in range(NA):
                x[r_, m_] = xp[m_] + boost * Kv[m_] * ec
            for m_ in range(NA):
                for n_ in range(NA):
                    P[r_, m_, n_] = Pp[m_, n_] - Kv[m_] * Pp[k, n_]
            for m_ in range(NA):                    # symmetrisch en positief houden
                for n_ in range(m_ + 1, NA):
                    v_ = 0.5 * (P[r_, m_, n_] + P[r_, n_, m_]); P[r_, m_, n_] = v_; P[r_, n_, m_] = v_
                if P[r_, m_, m_] < 1e-8:
                    P[r_, m_, m_] = 1e-8
            tl[r_] = dag[q]; na[r_, k] += 1.0
            rest = ec * (1.0 - boost * Kv[k])
            f[r_] = fc + GAMMA * rest; fd[r_] = dag[q]
            br = max(1.0 / (nr2[r_] + 1.0), BETA)
            r2[r_] = (1 - br) * r2[r_] + br * min(e * e, (HUB * CD[k]) ** 2 * 4) / (CD[k] * CD[k]); nr2[r_] += 1.0
        i = j
    return 0


def pass_(inv, par=None):
    p = dict(PAR); p.update({k: v for k, v in (par or {}).items() if k in PAR})
    pv = np.array([p[k] for k in NAMEN], float)
    N = len(inv['rk']); o = {k: np.zeros(N) for k in ('zh', 'vl', 'r', 'e', 'na', 'c', 'd', 'f', 'pc')}; val = np.zeros(N, np.bool_)
    _kalman_mv(inv['rk'], inv['rid'], inv['a'], inv['z'], inv['vth'], inv['dag'], inv['lf'], inv['nr'], pv,
               o['zh'], o['vl'], o['r'], o['e'], val, o['na'], o['c'], o['d'], o['f'], o['pc'])
    o['val'] = val
    return o
