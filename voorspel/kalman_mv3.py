"""Eén universele baanfactor-methode + niveau per rijder in één chronologische pass (vervangt het racepeil theta van het oude model).

Racepeil van rit i in race r op dag/baan E (alle races van één toernooidag op één baan delen het ijs):
    effect_i = (theta_E + delta_r) x m(g_i)
  anker      : Heerenveen = 0 met sd TAUHV per dag (anders drijft de schaal weg: snelle ritten tellen zwaarder, het veld leest dat als traag ijs).
  theta_E    : dagpeil van de baan, geschat uit ALLE races van die dag (alle categorieën, afstanden, geslachten) -> jeugd en senioren op
               hetzelfde ijs verbinden de niveaus; vooraf: baangemiddelde + CEV x eerdere toernooidagen + CREC x recente afwijking van de baan
               + CGL x algemene stand, onzekerheid TAU^2 + TAUB^2 / (1 + aantal dagen op de baan).
  delta_r    : afwijking van één race (afstand x geslacht x tijdstip; hier past het veld het peil deels aan), vooraf het gewone verschil
               voor die afstand op die baan (lopend gemiddelde; op Heerenveen vast 0: dat legt de schaal per afstand vast), sd TAUD.
  m(g)       : niet lineair in het niveau g (ln-tijd boven de wereldtop): m = 1 + A1 (exp(-g / TM) - exp(-G0 / TM)), G0 = 10%.
  schatting  : afwisselend theta_E en delta_r (Huber-gewogen, gewicht 1 / ((onzekerheid niveau + volle ritruis) x (1 + JW x max(18 - leeftijd, 0))):
               jonge rijders tellen minder mee voor het peil van de dag); voor de eigen rit van een
               rijder geldt het peil zonder die rit (leave-one-out).
  niveau     : als kalman_mv2 (8 afstanden, groei per leeftijd, eigen groeisnelheid, vorm), meting z = ln t - wereldtop - effect.
               Ritruis KR x CD^2 x (eigen spreiding)^BR: BR < 1 = minder week-op-week, meer de typische spreiding per afstand.
  uitschieters: per rijder (alle afstanden, lang geheugen HO ritten) hoe vaak een rit snel / traag buiten UO sd viel (geen valpartijen),
               gekrompen naar het basisniveau (M0O ritten): de historische deviatie als uitschieters (pf, ps).
De baanstatus wordt pas na afloop van een dag bijgewerkt."""
import math
import numpy as np, pandas as pd
from numba import njit
from kalman import CD

PAR = dict(G1=-0.0518, TG=2.149, G2=0.0024, Q0=0.000109, Q1=0.00224, TQ=1.541, QA=1e-06, RHO=0.944, RHO0=0.990, P0C=0.0239, P0A=0.000188,
           KR=0.351, HUB=3.0, SNEL=1.675, GAMMA=0.427, HF=3.49, M0R=25.0, BETA=0.06, P0V=0.00266, QV=0.000352, LV=2.236,
           A1=0.15, TM=0.04, TAU=0.006, TAUB=0.02, TAUD=0.004, CEV=0.5, CREC=0.3, CGL=0.3, WMIN=0.05, HUBT=2.5, BR=0.5, UO=2.0, HO=150.0, M0O=30.0, JW=1.0, TAUHV=0.002)
NAMEN = list(PAR)
G0 = 0.10


@njit(cache=True)
def _mv3(evk, rk, rid, a, lt, ref, dag, lf, seiz, baan, gsl, nr, nb, hv, maxe, p,
         o_zh, o_vl, o_r, o_e, o_ev, o_val, o_na, o_c, o_f, o_th0, o_tv0, o_thl, o_tvl, o_thr, o_the, o_m, o_pf, o_ps, o_v,
         snap, s_r, s_s, s_x, s_p, s_na, s_n, s_lf, s_dag, s_pf, s_ps, cnt):
    (G1, TG, G2, Q0, Q1, TQ, QA, RHO, RHO0, P0C, P0A, KR, HUB, SNEL, GAMMA, HF, M0R, BETA, P0V, QV, LV,
     A1, TM, TAU, TAUB, TAUD, CEV, CREC, CGL, WMIN, HUBT, BR, UO, HO, M0O, JW, TAUHV) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], p[25], p[26], p[27], p[28], p[29],
        p[30], p[31], p[32], p[33], p[34], p[35], p[36])
    NA = 8; NS = 9
    C = np.zeros((NA, NA)); C0 = np.zeros((NA, NA))
    for i_ in range(NA):
        for j_ in range(NA):
            C[i_, j_] = RHO ** abs(i_ - j_); C0[i_, j_] = P0C * RHO0 ** abs(i_ - j_) + (P0A if i_ == j_ else 0.0)
    pbase = 0.5 * math.erfc(UO / math.sqrt(2.0)) * 1.5
    lam_o = math.exp(-1.0 / HO)
    tE2 = TAU * TAU; tD2 = TAUD * TAUD
    # rijders
    x = np.zeros((nr, NS)); P = np.zeros((nr, NS, NS)); tl = np.full(nr, -1.0); started = np.zeros(nr, np.bool_); na = np.zeros((nr, NA))
    f = np.zeros(nr); fd = np.zeros(nr); r2 = np.ones(nr); nr2 = np.zeros(nr)
    cf = np.zeros(nr); cs = np.zeros(nr); cn = np.zeros(nr)
    cur_s = np.full(nr, -1); n_s = np.zeros(nr); last_lf = np.full(nr, np.nan)
    xp = np.zeros(NS); Pp = np.zeros((NS, NS)); Kv = np.zeros(NS); Fm = np.zeros((NS, NS)); T1 = np.zeros((NS, NS))
    # banen: dagpeil (lopend gemiddelde per baan), afstandsverschil per baan x geslacht x afstand, toernooi- en recente afwijking
    vmE = np.zeros(nb); vnE = np.zeros(nb); gmE = 0.0; gnE = 0.0
    md = np.zeros((nb, 2, NA)); mdn = np.zeros((nb, 2, NA))
    rb_d = np.full((nb, 64), -100000.0); rb_s = np.zeros((nb, 64)); rb_k = np.zeros((nb, 64)); rb_i = np.zeros(nb, np.int64)
    rec = np.zeros(nb); recn = np.zeros(nb); gl = 0.0; gl_d = -100000.0
    MB = 8192
    be_b = np.zeros(MB, np.int64); be_th = np.zeros(MB); be_dev = np.zeros(MB); be_n = 0
    br_b = np.zeros(MB, np.int64); br_g = np.zeros(MB, np.int64); br_a = np.zeros(MB, np.int64); br_d = np.zeros(MB); br_n = 0
    dag_nu = -1.0
    xs = np.zeros(maxe); ms = np.zeros(maxe); vs = np.zeros(maxe); ws = np.zeros(maxe); okq = np.zeros(maxe, np.bool_); lr = np.zeros(maxe, np.int64)
    r_g = np.zeros(maxe, np.int64); r_a = np.zeros(maxe, np.int64); md0 = np.zeros(maxe); num_r = np.zeros(maxe); den_r = np.zeros(maxe); dl = np.zeros(maxe)
    N = len(rk); i = 0
    while i < N:
        j = i
        while j < N and evk[j] == evk[i]:
            j += 1
        b = baan[i]; d0 = dag[i]
        if d0 != dag_nu:                                     # nieuwe dag: de baanstatus bijwerken met de dagen ervoor
            for t_ in range(be_n):
                b_ = be_b[t_]; th_ = be_th[t_]; dv = be_dev[t_]
                gevonden = False
                for u_ in range(64):
                    if rb_d[b_, u_] == dag_nu:
                        rb_s[b_, u_] += dv; rb_k[b_, u_] += 1.0; gevonden = True; break
                if not gevonden:
                    u_ = rb_i[b_] % 64; rb_d[b_, u_] = dag_nu; rb_s[b_, u_] = dv; rb_k[b_, u_] = 1.0; rb_i[b_] += 1
                rec[b_] = 0.8 * rec[b_] + 0.2 * dv; recn[b_] += 1.0
                dec = math.exp(-(dag_nu - gl_d) / 30.0 * math.log(2.0)) if gl_d > -100000 else 0.0
                gl = gl * dec * 0.95 + 0.05 * dv; gl_d = dag_nu
                w_ = max(1.0 / (vnE[b_] + 1.0), WMIN)
                vmE[b_] = th_ if vnE[b_] == 0 else vmE[b_] + w_ * (th_ - vmE[b_]); vnE[b_] += 1.0
                wg = max(1.0 / (gnE + 1.0), 0.01); gmE = th_ if gnE == 0 else gmE + wg * (th_ - gmE); gnE += 1.0
            for t_ in range(br_n):
                b_ = br_b[t_]; g_ = br_g[t_]; a_ = br_a[t_]
                if b_ != hv:
                    w_ = max(1.0 / (mdn[b_, g_, a_] + 1.0), WMIN)
                    md[b_, g_, a_] = br_d[t_] if mdn[b_, g_, a_] == 0 else md[b_, g_, a_] + w_ * (br_d[t_] - md[b_, g_, a_])
                mdn[b_, g_, a_] += 1.0
            be_n = 0; br_n = 0; dag_nu = d0
        # ---- dagpeil vooraf
        basis = vmE[b] if vnE[b] > 0 else gmE
        if b == hv and TAUHV > 0:                           # anker: Heerenveen is de standaardbaan (0), met een kleine eigen dagafwijking
            basis = 0.0
        s_ = 0.0; kk = 0.0
        for u_ in range(64):
            if rb_d[b, u_] < d0 and rb_d[b, u_] >= d0 - 5:
                s_ += rb_s[b, u_]; kk += rb_k[b, u_]
        ev = s_ / kk if kk > 0 else 0.0
        glc = gl * math.exp(-(d0 - gl_d) / 30.0 * math.log(2.0)) if gl_d > -100000 else 0.0
        thE0 = basis + CEV * ev * kk / (kk + 1.0) + CREC * rec[b] * (1.0 - math.exp(-recn[b] / 3.0)) + CGL * glc
        tvE0 = tE2 + TAUB * TAUB / (1.0 + vnE[b])
        if b == hv and TAUHV > 0:
            thE0 = 0.0; tvE0 = TAUHV * TAUHV
        # ---- races binnen de dag en rijders voorspellen
        nloc = 0; vorige_rk = -1
        for q in range(i, j):
            t_ = q - i; r_ = rid[q]; kq = a[q]; g = gsl[q]
            if rk[q] != vorige_rk:
                r_g[nloc] = g; r_a[nloc] = kq
                if b == hv:
                    md0[nloc] = 0.0
                else:
                    md0[nloc] = md[b, g, kq] * mdn[b, g, kq] / (mdn[b, g, kq] + 3.0)
                nloc += 1; vorige_rk = rk[q]
            lr[t_] = nloc - 1
            o_na[q] = na[r_, kq]
            o_pf[q] = (cf[r_] + M0O * pbase) / (cn[r_] + M0O); o_ps[q] = (cs[r_] + M0O * pbase) / (cn[r_] + M0O)
            o_th0[q] = thE0 + md0[nloc - 1]; o_tv0[q] = tvE0 + tD2
            okq[t_] = False; xs[t_] = 0.0; ms[t_] = 1.0; vs[t_] = 1.0; ws[t_] = 0.0
            if not started[r_]:
                o_zh[q] = np.nan; o_vl[q] = np.nan; o_r[q] = np.nan; o_c[q] = np.nan; o_f[q] = 0.0; o_m[q] = 1.0; o_v[q] = 0.0
                continue
            age = lf[q] if lf[q] == lf[q] else 25.0
            dt = max(dag[q] - tl[r_], 0.0) / 365.0; dtc = min(dt, 1.5)
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * math.exp(-(age - 12.0) / TQ)
            phi = math.exp(-LV * dt); h = dtc * (1.0 + phi) / 2.0
            xk = x[r_, kq] + gr * dtc + x[r_, 8] * h
            pkk = P[r_, kq, kq] + 2 * h * P[r_, kq, 8] + h * h * P[r_, 8, 8] + dt * (qc + QA)
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R); R = KR * CD[kq] * CD[kq] * rr ** BR
            gq = min(max(xk, 0.0), 0.5)
            m_ = 1.0 + A1 * (math.exp(-gq / TM) - math.exp(-G0 / TM)); m_ = min(max(m_, 0.3), 3.0)
            o_zh[q] = xk + fc; o_vl[q] = pkk; o_r[q] = R; o_c[q] = xk; o_f[q] = fc; o_m[q] = m_; o_v[q] = x[r_, 8]
            if lt[q] == lt[q]:
                # gewicht in het racepeil: volle eigen spreiding (niet ^BR) en minder voor jonge rijders (hun niveau loopt achter bij groei)
                xs[t_] = lt[q] - ref[q] - xk - fc; ms[t_] = m_
                vs[t_] = max((pkk + KR * CD[kq] * CD[kq] * rr) * (1.0 + JW * max(18.0 - age, 0.0)), 1e-8); okq[t_] = xs[t_] == xs[t_]
        # ---- dagpeil en raceafwijkingen uit het veld (afwisselend, Huber)
        thE = thE0
        for l_ in range(nloc):
            dl[l_] = md0[l_]
        numE = 0.0; denE = 0.0
        for it in range(4):
            numE = 0.0; denE = 0.0
            for t_ in range(j - i):
                if okq[t_]:
                    tot = thE + dl[lr[t_]]
                    rres = xs[t_] - tot * ms[t_]; sd_ = math.sqrt(vs[t_])
                    hw = 1.0 if abs(rres) <= HUBT * sd_ else HUBT * sd_ / abs(rres)
                    ws[t_] = hw / vs[t_]
                    numE += ws[t_] * ms[t_] * (xs[t_] - dl[lr[t_]] * ms[t_]); denE += ws[t_] * ms[t_] * ms[t_]
                else:
                    ws[t_] = 0.0
            thE = (thE0 / tvE0 + numE) / (1.0 / tvE0 + denE)
            for l_ in range(nloc):
                num_r[l_] = 0.0; den_r[l_] = 0.0
            for t_ in range(j - i):
                if okq[t_]:
                    num_r[lr[t_]] += ws[t_] * ms[t_] * (xs[t_] - thE * ms[t_]); den_r[lr[t_]] += ws[t_] * ms[t_] * ms[t_]
            for l_ in range(nloc):
                dl[l_] = (md0[l_] / tD2 + num_r[l_]) / (1.0 / tD2 + den_r[l_])
        tvE = 1.0 / (1.0 / tvE0 + denE)
        for q in range(i, j):
            t_ = q - i; l_ = lr[t_]
            o_thr[q] = thE + dl[l_]; o_the[q] = thE
            if okq[t_]:
                wq = ws[t_]; mq = ms[t_]
                nE = numE - wq * mq * (xs[t_] - dl[l_] * mq); dE = denE - wq * mq * mq
                thE_q = (thE0 / tvE0 + nE) / (1.0 / tvE0 + dE)
                nr_ = num_r[l_] + (thE - thE_q) * den_r[l_] - wq * mq * (xs[t_] - thE_q * mq); dr_ = den_r[l_] - wq * mq * mq
                dq = (md0[l_] / tD2 + nr_) / (1.0 / tD2 + dr_)
                o_thl[q] = thE_q + dq; o_tvl[q] = 1.0 / (1.0 / tvE0 + dE) + 1.0 / (1.0 / tD2 + dr_)
            else:
                o_thl[q] = thE + dl[l_]; o_tvl[q] = tvE + 1.0 / (1.0 / tD2 + den_r[l_])
            if started[rid[q]] and lt[q] == lt[q] and okq[t_]:
                m_ = o_m[q]
                e = xs[t_] - o_thl[q] * m_; o_e[q] = e; o_ev[q] = xs[t_] - o_th0[q] * m_
                S = o_vl[q] + o_r[q] + o_tvl[q] * m_ * m_
                o_val[q] = (e > 6.0 * math.sqrt(S)) and (e > 0.03)
            else:
                o_e[q] = np.nan; o_ev[q] = np.nan; o_val[q] = False
        # ---- rijders bijwerken (in volgorde; een tweede rit op dezelfde dag gebruikt de stand na de eerste)
        for q in range(i, j):
            r_ = rid[q]; kq = a[q]
            if lt[q] != lt[q] or o_val[q]:
                continue
            m_ = o_m[q]; zq = lt[q] - ref[q] - o_thl[q] * m_; vth = o_tvl[q] * m_ * m_
            if snap and started[r_] and seiz[q] != cur_s[r_]:
                c_ = cnt[0]
                s_r[c_] = r_; s_s[c_] = cur_s[r_]; s_n[c_] = n_s[r_]; s_lf[c_] = last_lf[r_]; s_dag[c_] = tl[r_]
                s_pf[c_] = (cf[r_] + M0O * pbase) / (cn[r_] + M0O); s_ps[c_] = (cs[r_] + M0O * pbase) / (cn[r_] + M0O)
                for m2 in range(NS):
                    s_x[c_, m2] = x[r_, m2]; s_p[c_, m2] = P[r_, m2, m2]
                for m2 in range(NA):
                    s_na[c_, m2] = na[r_, m2]
                cnt[0] = c_ + 1; n_s[r_] = 0.0
            cur_s[r_] = seiz[q]; n_s[r_] += 1.0; last_lf[r_] = lf[q]
            age = lf[q] if lf[q] == lf[q] else 25.0
            wa = math.exp(-(age - 12.0) / TQ)
            if not started[r_]:
                for m2 in range(NA):
                    x[r_, m2] = zq
                    for n2 in range(NA):
                        P[r_, m2, n2] = C0[m2, n2]
                    P[r_, m2, 8] = 0.0; P[r_, 8, m2] = 0.0
                x[r_, 8] = 0.0; P[r_, 8, 8] = P0V * wa + 1e-6
                tl[r_] = dag[q]; started[r_] = True; na[r_, kq] = 1.0
                continue
            rr = (nr2[r_] * r2[r_] + M0R) / (nr2[r_] + M0R); R = KR * CD[kq] * CD[kq] * rr ** BR
            if o_vl[q] == o_vl[q] and o_e[q] == o_e[q]:
                uu = o_e[q] / math.sqrt(o_vl[q] + R + vth)
                cf[r_] = cf[r_] * lam_o + (1.0 if uu < -UO else 0.0); cs[r_] = cs[r_] * lam_o + (1.0 if uu > UO else 0.0); cn[r_] = cn[r_] * lam_o + 1.0
            dt = max(dag[q] - tl[r_], 0.0) / 365.0; dtc = min(dt, 1.5)
            gr = G1 * math.exp(-(age - 12.0) / TG) + G2 * max(age - 35.0, 0.0) / 10.0
            qc = Q0 + Q1 * wa
            phi = math.exp(-LV * dt); h = dtc * (1.0 + phi) / 2.0
            for m2 in range(NS):
                for n2 in range(NS):
                    Fm[m2, n2] = 1.0 if m2 == n2 else 0.0
            for m2 in range(NA):
                Fm[m2, 8] = h
            Fm[8, 8] = phi
            for m2 in range(NA):
                xp[m2] = x[r_, m2] + gr * dtc + h * x[r_, 8]
            xp[8] = phi * x[r_, 8]
            for m2 in range(NS):
                for n2 in range(NS):
                    s2 = 0.0
                    for l_ in range(NS):
                        s2 += Fm[m2, l_] * P[r_, l_, n2]
                    T1[m2, n2] = s2
            for m2 in range(NS):
                for n2 in range(NS):
                    s2 = 0.0
                    for l_ in range(NS):
                        s2 += T1[m2, l_] * Fm[n2, l_]
                    Pp[m2, n2] = s2
            for m2 in range(NA):
                for n2 in range(NA):
                    Pp[m2, n2] += dt * (qc * C[m2, n2] + (QA if m2 == n2 else 0.0))
            Pp[8, 8] += QV * wa * dt
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / HF)
            S = Pp[kq, kq] + R + vth
            e = zq - xp[kq] - fc
            ec = min(max(e, -HUB * math.sqrt(S)), HUB * math.sqrt(S))
            boost = 1.0
            if e < 0:
                boost = min(SNEL, 0.9 * S / max(Pp[kq, kq], 1e-12))
            for m2 in range(NS):
                Kv[m2] = Pp[m2, kq] / S
            for m2 in range(NS):
                x[r_, m2] = xp[m2] + boost * Kv[m2] * ec
            for m2 in range(NS):
                for n2 in range(NS):
                    P[r_, m2, n2] = Pp[m2, n2] - Kv[m2] * Pp[kq, n2]
            for m2 in range(NS):
                for n2 in range(m2 + 1, NS):
                    v_ = 0.5 * (P[r_, m2, n2] + P[r_, n2, m2]); P[r_, m2, n2] = v_; P[r_, n2, m2] = v_
                if P[r_, m2, m2] < 1e-9:
                    P[r_, m2, m2] = 1e-9
            if x[r_, 8] > 0.2: x[r_, 8] = 0.2
            if x[r_, 8] < -0.2: x[r_, 8] = -0.2
            tl[r_] = dag[q]; na[r_, kq] += 1.0
            rest = ec * (1.0 - boost * Kv[kq])
            f[r_] = fc + GAMMA * rest; fd[r_] = dag[q]
            br = max(1.0 / (nr2[r_] + 1.0), BETA)
            r2[r_] = (1 - br) * r2[r_] + br * min(e * e, (HUB * CD[kq]) ** 2 * 4) / (CD[kq] * CD[kq]); nr2[r_] += 1.0
        # ---- baanstatus na de dag: dagpeil van de baan en afstandsverschillen
        if denE > 0 and be_n < MB:
            be_b[be_n] = b; be_th[be_n] = thE; be_dev[be_n] = thE - basis; be_n += 1
        for l_ in range(nloc):
            if den_r[l_] > 0 and br_n < MB:
                br_b[br_n] = b; br_g[br_n] = r_g[l_]; br_a[br_n] = r_a[l_]; br_d[br_n] = dl[l_]; br_n += 1
        i = j
    if snap:
        for r_ in range(nr):
            if started[r_] and n_s[r_] > 0:
                c_ = cnt[0]
                s_r[c_] = r_; s_s[c_] = cur_s[r_]; s_n[c_] = n_s[r_]; s_lf[c_] = last_lf[r_]; s_dag[c_] = tl[r_]
                s_pf[c_] = (cf[r_] + M0O * pbase) / (cn[r_] + M0O); s_ps[c_] = (cs[r_] + M0O * pbase) / (cn[r_] + M0O)
                for m2 in range(NS):
                    s_x[c_, m2] = x[r_, m2]; s_p[c_, m2] = P[r_, m2, m2]
                for m2 in range(NA):
                    s_na[c_, m2] = na[r_, m2]
                cnt[0] = c_ + 1
    return 0


def voorbereid(df):
    """Sorteren en de vaste invoer voor de pass. Wereldtop (ref) per geslacht x afstand: 1e percentiel van ln tijd - theta_oud (jaren
    <= 2019), verschoven met het mediane oude racepeil in Heerenveen, zodat de niveaus ongeveer op de Heerenveen-schaal staan."""
    df = df.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    rk = df.groupby(['dag', 'event_id', 'gender', 'afstand_idx'], sort=False).ngroup().to_numpy().astype(np.int64)
    evk = df.groupby(['dag', 'event_id'], sort=False).ngroup().to_numpy().astype(np.int64)
    rid, _ = pd.factorize(df['skater_id'])
    lt = np.where(df['verdacht'].to_numpy(), np.nan, np.log(df['tijd'].to_numpy(float)))
    yo = lt - df['theta'].to_numpy(float)
    fitset = (df['jaar'] <= 2019).to_numpy() & np.isfinite(yo)
    ref = pd.Series(yo[fitset]).groupby([df.loc[fitset, 'gender'].to_numpy(), df.loc[fitset, 'afstand_idx'].to_numpy()]).quantile(0.01)
    hvm = df[(df['baan'] == 'Heerenveen (NED)') & (df['jaar'] <= 2019)].groupby(['gender', 'afstand_idx'])['theta'].median()
    refq = np.array([ref.get((g, k), np.nan) + hvm.get((g, k), 0.0) for g, k in zip(df['gender'], df['afstand_idx'])])
    lf = (df['leeftijd'] + ((df['dag'] - (df['seizoen'] - 1970) * 365.25 - 182) / 365.25).clip(0, 1)).to_numpy(float)
    bc, bu = pd.factorize(df['baan'].astype(str))
    hv = int(np.flatnonzero(np.asarray(bu) == 'Heerenveen (NED)')[0])
    inv = dict(evk=evk, rk=rk, rid=rid.astype(np.int64), a=df['afstand_idx'].to_numpy().astype(np.int64), lt=lt, ref=refq,
               dag=df['dag'].to_numpy(float), lf=lf, seiz=df['seizoen'].to_numpy().astype(np.int64), baan=bc.astype(np.int64),
               gsl=(df['gender'] == 'v').to_numpy().astype(np.int64), nr=int(rid.max() + 1), nb=len(bu), hv=hv,
               maxe=int(np.bincount(evk).max()) + 1)
    inv['z'] = lt - refq; inv['refq'] = refq; inv['vth'] = np.zeros(len(df))
    return df, inv


def pass_(inv, par=None, snap=False):
    p = dict(PAR); p.update({k: v for k, v in (par or {}).items() if k in PAR})
    pv = np.array([p[k] for k in NAMEN], float)
    N = len(inv['rk'])
    o = {k: np.zeros(N) for k in ('zh', 'vl', 'r', 'e', 'ev', 'na', 'c', 'f', 'th0', 'tv0', 'thl', 'tvl', 'thr', 'the', 'm', 'pf', 'ps', 'v')}
    val = np.zeros(N, np.bool_)
    M = N if snap else 1
    s = dict(r=np.zeros(M, np.int64), s=np.zeros(M, np.int64), x=np.zeros((M, 9)), p=np.zeros((M, 9)), na=np.zeros((M, 8)), n=np.zeros(M),
             lf=np.zeros(M), dag=np.zeros(M), pf=np.zeros(M), ps=np.zeros(M))
    cnt = np.zeros(1, np.int64)
    _mv3(inv['evk'], inv['rk'], inv['rid'], inv['a'], inv['lt'], inv['ref'], inv['dag'], inv['lf'], inv['seiz'], inv['baan'], inv['gsl'],
         inv['nr'], inv['nb'], inv['hv'], inv['maxe'], pv,
         o['zh'], o['vl'], o['r'], o['e'], o['ev'], val, o['na'], o['c'], o['f'], o['th0'], o['tv0'], o['thl'], o['tvl'], o['thr'], o['the'],
         o['m'], o['pf'], o['ps'], o['v'],
         snap, s['r'], s['s'], s['x'], s['p'], s['na'], s['n'], s['lf'], s['dag'], s['pf'], s['ps'], cnt)
    o['val'] = val; o['d'] = np.zeros(N); o['pc'] = o['vl']
    if snap:
        c = int(cnt[0])
        o['snap'] = dict(rid=s['r'][:c], seizoen=s['s'][:c], x=s['x'][:c], p=s['p'][:c], na=s['na'][:c], n=s['n'][:c], lf=s['lf'][:c],
                         dag=s['dag'][:c], pf=s['pf'][:c], ps=s['ps'][:c])
    return o
