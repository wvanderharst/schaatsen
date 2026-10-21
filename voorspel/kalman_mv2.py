"""Als kalman_mv.py, met een eigen groeisnelheid per rijder: toestand (x_100m .. x_10km, v), v = afwijking van de verwachte groei per jaar
(alle afstanden samen; lokaal lineaire trend). Tussen twee races: x += (g(leeftijd) + v) x dt (dt max 1,5 jaar), v *= exp(-LV x dt);
de onzekerheid van v start op P0V x exp(-(leeftijd - 12) / TQ) en groeit met QV x exp(-(leeftijd - 12) / TQ) per jaar (jonge rijders
verschillen het meest in groeitempo). Zelfde invoer en uitvoer als kalman.py."""
import math
import numpy as np
from numba import njit
from kalman import voorbereid, CD

PAR = dict(G1=-0.062, TG=2.6, G2=0.0044, Q0=0.00012, Q1=0.00066, TQ=3.4, QA=0.00005, RHO=0.85, RHO0=0.9, P0C=0.0146, P0A=0.0028,
           KR=0.69, HUB=3.0, SNEL=1.57, GAMMA=0.38, HF=2.4, M0R=25.0, BETA=0.06, P0V=0.0004, QV=0.0002, LV=0.5)
NAMEN = list(PAR)


@njit(cache=True)
def _kalman_mv2(rk, rid, a, z, vth, dag, lf, nr, p, o_zh, o_vl, o_r, o_e, o_val, o_na, o_c, o_d, o_f, o_pc,
                seiz, snap, s_r, s_s, s_x, s_p, s_na, s_n, s_lf, s_dag, cnt):
    G1, TG, G2, Q0, Q1, TQ, QA, RHO, RHO0, P0C, P0A, KR, HUB, SNEL, GAMMA, HF, M0R, BETA, P0V, QV, LV = (
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19], p[20])
    NA = 8; NS = 9
    C = np.zeros((NA, NA)); C0 = np.zeros((NA, NA))
    for i_ in range(NA):
        for j_ in range(NA):
            C[i_, j_] = RHO ** abs(i_ - j_); C0[i_, j_] = P0C * RHO0 ** abs(i_ - j_) + (P0A if i_ == j_ else 0.0)
    x = np.zeros((nr, NS)); P = np.zeros((nr, NS, NS)); tl = np.full(nr, -1.0); started = np.zeros(nr, np.bool_); na = np.zeros((nr, NA))
    f = np.zeros(nr); fd = np.zeros(nr); r2 = np.ones(nr); nr2 = np.zeros(nr)
    xp = np.zeros(NS); Pp = np.zeros((NS, NS)); Kv = np.zeros(NS); Fm = np.zeros((NS, NS)); T1 = np.zeros((NS, NS))
    cur_s = np.full(nr, -1); n_s = np.zeros(nr); last_lf = np.full(nr, np.nan)     # momentopnamen aan het eind van elk seizoen
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
            dt = max(dag[q] - tl[r_], 0.0) / 365.0; dtc = min(dt, 1.5)
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * math.exp(-(age - 12.0) / TQ); wa = math.exp(-(age - 12.0) / TQ)
            phi = math.exp(-LV * dt); h = dtc * (1.0 + phi) / 2.0
            xk = x[r_, k] + gr * dtc + x[r_, 8] * h
            pkk = P[r_, k, k] + 2 * h * P[r_, k, 8] + h * h * P[r_, 8, 8] + dt * (qc + QA)
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R); R = KR * CD[k] * CD[k] * rr
            o_zh[q] = xk + fc; o_vl[q] = pkk; o_r[q] = R; o_c[q] = xk; o_d[q] = x[r_, 8]; o_f[q] = fc; o_pc[q] = pkk
            if z[q] != z[q]:
                o_e[q] = np.nan; o_val[q] = False
            else:
                S = pkk + R + vth[q]; e = z[q] - o_zh[q]; o_e[q] = e
                o_val[q] = (e > 6.0 * math.sqrt(S)) and (e > 0.03)
        for q in range(i, j):
            r_ = rid[q]; k = a[q]
            if z[q] != z[q] or o_val[q]:
                continue
            if snap and started[r_] and seiz[q] != cur_s[r_]:      # eerste rit van een nieuw seizoen: stand na het vorige seizoen bewaren
                c_ = cnt[0]
                s_r[c_] = r_; s_s[c_] = cur_s[r_]; s_n[c_] = n_s[r_]; s_lf[c_] = last_lf[r_]; s_dag[c_] = tl[r_]
                for m_ in range(NS):
                    s_x[c_, m_] = x[r_, m_]; s_p[c_, m_] = P[r_, m_, m_]
                for m_ in range(NA):
                    s_na[c_, m_] = na[r_, m_]
                cnt[0] = c_ + 1; n_s[r_] = 0.0
            cur_s[r_] = seiz[q]; n_s[r_] += 1.0; last_lf[r_] = lf[q]
            age = lf[q] if lf[q] == lf[q] else 25.0
            wa = math.exp(-(age - 12.0) / TQ)
            if not started[r_]:
                for m_ in range(NA):
                    x[r_, m_] = z[q]
                    for n_ in range(NA):
                        P[r_, m_, n_] = C0[m_, n_]
                    P[r_, m_, 8] = 0.0; P[r_, 8, m_] = 0.0
                x[r_, 8] = 0.0; P[r_, 8, 8] = P0V * wa + 1e-6
                tl[r_] = dag[q]; started[r_] = True; na[r_, k] = 1.0
                continue
            dt = max(dag[q] - tl[r_], 0.0) / 365.0; dtc = min(dt, 1.5)
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * wa
            phi = math.exp(-LV * dt); h = dtc * (1.0 + phi) / 2.0
            # overgang F: x_m += h v (m < 8), v *= phi
            for m_ in range(NS):
                for n_ in range(NS):
                    Fm[m_, n_] = 1.0 if m_ == n_ else 0.0
            for m_ in range(NA):
                Fm[m_, 8] = h
            Fm[8, 8] = phi
            for m_ in range(NA):
                xp[m_] = x[r_, m_] + gr * dtc + h * x[r_, 8]
            xp[8] = phi * x[r_, 8]
            for m_ in range(NS):                    # T1 = F P
                for n_ in range(NS):
                    s_ = 0.0
                    for l_ in range(NS):
                        s_ += Fm[m_, l_] * P[r_, l_, n_]
                    T1[m_, n_] = s_
            for m_ in range(NS):                    # Pp = T1 F' + Q
                for n_ in range(NS):
                    s_ = 0.0
                    for l_ in range(NS):
                        s_ += T1[m_, l_] * Fm[n_, l_]
                    Pp[m_, n_] = s_
            for m_ in range(NA):
                for n_ in range(NA):
                    Pp[m_, n_] += dt * (qc * C[m_, n_] + (QA if m_ == n_ else 0.0))
            Pp[8, 8] += QV * wa * dt
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R); R = KR * CD[k] * CD[k] * rr
            S = Pp[k, k] + R + vth[q]
            e = z[q] - xp[k] - fc
            ec = min(max(e, -HUB * math.sqrt(S)), HUB * math.sqrt(S))
            boost = 1.0
            if e < 0:
                boost = min(SNEL, 0.9 * S / max(Pp[k, k], 1e-12))
            for m_ in range(NS):
                Kv[m_] = Pp[m_, k] / S
            for m_ in range(NS):
                x[r_, m_] = xp[m_] + boost * Kv[m_] * ec
            for m_ in range(NS):
                for n_ in range(NS):
                    P[r_, m_, n_] = Pp[m_, n_] - Kv[m_] * Pp[k, n_]
            for m_ in range(NS):
                for n_ in range(m_ + 1, NS):
                    v_ = 0.5 * (P[r_, m_, n_] + P[r_, n_, m_]); P[r_, m_, n_] = v_; P[r_, n_, m_] = v_
                if P[r_, m_, m_] < 1e-9:
                    P[r_, m_, m_] = 1e-9
            if x[r_, 8] > 0.2: x[r_, 8] = 0.2
            if x[r_, 8] < -0.2: x[r_, 8] = -0.2
            tl[r_] = dag[q]; na[r_, k] += 1.0
            rest = ec * (1.0 - boost * Kv[k])
            f[r_] = fc + GAMMA * rest; fd[r_] = dag[q]
            br = max(1.0 / (nr2[r_] + 1.0), BETA)
            r2[r_] = (1 - br) * r2[r_] + br * min(e * e, (HUB * CD[k]) ** 2 * 4) / (CD[k] * CD[k]); nr2[r_] += 1.0
        i = j
    if snap:                                       # stand aan het eind van de data
        for r_ in range(nr):
            if started[r_] and n_s[r_] > 0:
                c_ = cnt[0]
                s_r[c_] = r_; s_s[c_] = cur_s[r_]; s_n[c_] = n_s[r_]; s_lf[c_] = last_lf[r_]; s_dag[c_] = tl[r_]
                for m_ in range(NS):
                    s_x[c_, m_] = x[r_, m_]; s_p[c_, m_] = P[r_, m_, m_]
                for m_ in range(NA):
                    s_na[c_, m_] = na[r_, m_]
                cnt[0] = c_ + 1
    return 0


def pass_(inv, par=None, snap=False):
    """snap=True: ook een momentopname per rijder x seizoen (stand na de laatste rit van dat seizoen) in o['snap']."""
    p = dict(PAR); p.update({k: v for k, v in (par or {}).items() if k in PAR})
    pv = np.array([p[k] for k in NAMEN], float)
    N = len(inv['rk']); o = {k: np.zeros(N) for k in ('zh', 'vl', 'r', 'e', 'na', 'c', 'd', 'f', 'pc')}; val = np.zeros(N, np.bool_)
    M = N if snap else 1
    s_r = np.zeros(M, np.int64); s_s = np.zeros(M, np.int64); s_x = np.zeros((M, 9)); s_p = np.zeros((M, 9)); s_na = np.zeros((M, 8))
    s_n = np.zeros(M); s_lf = np.zeros(M); s_dag = np.zeros(M); cnt = np.zeros(1, np.int64)
    seiz = inv['seiz'] if 'seiz' in inv else np.zeros(N, np.int64)
    _kalman_mv2(inv['rk'], inv['rid'], inv['a'], inv['z'], inv['vth'], inv['dag'], inv['lf'], inv['nr'], pv,
                o['zh'], o['vl'], o['r'], o['e'], val, o['na'], o['c'], o['d'], o['f'], o['pc'],
                seiz, snap, s_r, s_s, s_x, s_p, s_na, s_n, s_lf, s_dag, cnt)
    o['val'] = val
    if snap:
        c = int(cnt[0])
        o['snap'] = dict(rid=s_r[:c], seizoen=s_s[:c], x=s_x[:c], p=s_p[:c], na=s_na[:c], n=s_n[:c], lf=s_lf[:c], dag=s_dag[:c])
    return o
