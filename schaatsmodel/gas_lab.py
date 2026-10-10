"""GAS-laag (score-gedreven, Creal-Koopman-Lucas) bovenop de voorspelfouten van het schaatsmodel.

Per rijder (of per rijder x afstand) een tijdsvariabele verschuiving m en log-schaal lam:
    u_t = m_t + exp(lam_t) * eps_t,   eps ~ twee-delige Student-t (vaste vorm per afstandsgroep, snel/traag eigen schaal en df)
    m_{t+1}   = phi_m^(gap/30)   * m_t   + a_m[kant]   * score_m
    lam_{t+1} = phi_l^(gap/30)   * lam_t + a_l[kant]   * score_l
score_m = w*eps/s^2 en score_l = w*eps^2/s^2 - 1 (w = (nu+1)/(nu+(eps/s)^2)): de score van de eigen (scheve) verdeling.
'kant' = snel (eps<0) of traag (eps>0): aparte gevoeligheden (scheve score).
Varianten voor de volatiliteitsscore aan de snelle kant: 't' (begrensd, robuust) of 'normaal' (onbegrensd: een uitschieter
laat de volatiliteit exploderen); trage kant kan begrensd worden op 'kap'.
Valpartijen tellen niet mee (geen score, geen update). Gebruik: functies hieronder vanuit gas_run.py."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def _tlogpdf(x, nu):
    return (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi)
            - 0.5 * (nu + 1.0) * math.log1p(x * x / nu))


@njit(cache=True)
def gas_pass(key, gap, u, grp, val, scoor, vorm, par, modus_vol_snel, kap_traag, lam_min, lam_max,
             pw, drempel_lo, drempel_hi, massa, uit_m, uit_l):
    """Een filterpass. key: rijder- (of rijder x afstand-) id, gesorteerd op tijd; gap: dagen sinds vorige rij van dezelfde key
    (-1 = eerste). vorm[g] = (p0, sL, nuL, sU, nuU). par = (phi_m, phi_l, am_snel, am_traag, al_snel, al_traag).
    modus_vol_snel: 0 = t-score, 1 = normale score (onbegrensd). kap_traag: grens op de volatiliteitsscore aan de trage kant (0 = geen).
    pw: eigen kansen (piek, normaal, slecht) per rij; massa[g] = massa van de gebieden onder de vorm (voor de herweging).
    Geeft de som van de log-scores (alleen rijen met scoor) en vult uit_m/uit_l met de toestand VOOR elke rij."""
    phi_m, phi_l, am_s, am_t, al_s, al_t = par[0], par[1], par[2], par[3], par[4], par[5]
    tot = 0.0
    m = 0.0
    lam = 0.0
    vorige = -1
    for i in range(len(u)):
        if key[i] != vorige:
            m = 0.0; lam = 0.0; vorige = key[i]
        elif gap[i] > 0:
            f = gap[i] / 30.0
            m *= phi_m ** f
            lam *= phi_l ** f
        uit_m[i] = m; uit_l[i] = lam
        if val[i]:
            continue
        g = grp[i]
        p0 = vorm[g, 0]; sL = vorm[g, 1]; nuL = vorm[g, 2]; sU = vorm[g, 3]; nuU = vorm[g, 4]
        sig = math.exp(lam)
        eps = (u[i] - m) / sig
        if eps < 0:
            s = sL; nu = nuL; lp = math.log(2.0 * p0) + _tlogpdf(eps / s, nu) - math.log(s)
        else:
            s = sU; nu = nuU; lp = math.log(2.0 * (1.0 - p0)) + _tlogpdf(eps / s, nu) - math.log(s)
        lp -= lam
        # herweging met de eigen piekkans (gebieden in eps-ruimte)
        if eps < drempel_lo[g]:
            lp += math.log(pw[i, 0] / massa[g, 0])
        elif eps > drempel_hi[g]:
            lp += math.log(pw[i, 2] / massa[g, 2])
        else:
            lp += math.log(pw[i, 1] / massa[g, 1])
        if scoor[i]:
            tot += lp
        r2 = (eps / s) ** 2
        w = (nu + 1.0) / (nu + r2)
        sc_m = w * eps / (s * s)
        if eps < 0:
            sc_l = (r2 - 1.0) if modus_vol_snel == 1 else (w * r2 - 1.0)
            m += am_s * sc_m
            lam += al_s * sc_l
        else:
            sc_l = w * r2 - 1.0
            if kap_traag > 0 and sc_l > kap_traag:
                sc_l = kap_traag
            m += am_t * sc_m
            lam += al_t * sc_l
        if lam < lam_min:
            lam = lam_min
        if lam > lam_max:
            lam = lam_max
    return tot


@njit(cache=True)
def gas_pass2(sk, dag, ai, u, grp, val, scoor, vorm, par, dagblok, lam_min, lam_max,
              pw, drempel_lo, drempel_hi, massa, uit_m, uit_l):
    """Twee lagen: laag 1 per rijder (alle afstanden), laag 2 per rijder x afstand. Rijen gesorteerd op rijder, dag, evenement.
    par = (phi_m1, phi_l1, am_s1, am_t1, al_s1, al_t1, phi_m2, phi_l2, am_s2, am_t2, al_s2, al_t2); verval per 30 dagen.
    dagblok: True = races van dezelfde dag gebruiken de toestand van het begin van die dag (geen informatie binnen de dag)."""
    tot = 0.0
    m1 = 0.0; l1 = 0.0; d1 = -1
    m2 = np.zeros(8); l2 = np.zeros(8); d2 = np.full(8, -1)
    m1d = 0.0; l1d = 0.0
    m2d = np.zeros(8); l2d = np.zeros(8)
    vorige = -1; vorige_dag = -1
    for i in range(len(u)):
        j = ai[i]
        if sk[i] != vorige:
            m1 = 0.0; l1 = 0.0; d1 = dag[i]
            for k in range(8):
                m2[k] = 0.0; l2[k] = 0.0; d2[k] = -1
            vorige = sk[i]; vorige_dag = -1
        if dag[i] > d1:
            f = (dag[i] - d1) / 30.0
            m1 *= par[0] ** f; l1 *= par[1] ** f; d1 = dag[i]
        if d2[j] >= 0 and dag[i] > d2[j]:
            f = (dag[i] - d2[j]) / 30.0
            m2[j] *= par[6] ** f; l2[j] *= par[7] ** f
        d2[j] = dag[i]
        if dag[i] != vorige_dag:
            for k in range(8):                    # laag 2 voor alle afstanden laten vervallen tot vandaag
                if d2[k] >= 0 and dag[i] > d2[k]:
                    f = (dag[i] - d2[k]) / 30.0
                    m2[k] *= par[6] ** f; l2[k] *= par[7] ** f; d2[k] = dag[i]
            m1d = m1; l1d = l1
            for k in range(8):
                m2d[k] = m2[k]; l2d[k] = l2[k]
            vorige_dag = dag[i]
        if dagblok:
            m = m1d + m2d[j]; lam = l1d + l2d[j]
        else:
            m = m1 + m2[j]; lam = l1 + l2[j]
        if lam < lam_min:
            lam = lam_min
        if lam > lam_max:
            lam = lam_max
        uit_m[i] = m; uit_l[i] = lam
        if val[i]:
            continue
        g = grp[i]
        p0 = vorm[g, 0]; sL = vorm[g, 1]; nuL = vorm[g, 2]; sU = vorm[g, 3]; nuU = vorm[g, 4]
        eps = (u[i] - m) / math.exp(lam)
        if eps < 0:
            s = sL; nu = nuL; lp = math.log(2.0 * p0) + _tlogpdf(eps / s, nu) - math.log(s)
        else:
            s = sU; nu = nuU; lp = math.log(2.0 * (1.0 - p0)) + _tlogpdf(eps / s, nu) - math.log(s)
        lp -= lam
        if eps < drempel_lo[g]:
            lp += math.log(pw[i, 0] / massa[g, 0])
        elif eps > drempel_hi[g]:
            lp += math.log(pw[i, 2] / massa[g, 2])
        else:
            lp += math.log(pw[i, 1] / massa[g, 1])
        if scoor[i]:
            tot += lp
        r2 = (eps / s) ** 2
        w = (nu + 1.0) / (nu + r2)
        sc_m = w * eps / (s * s)
        sc_l = w * r2 - 1.0
        if eps < 0:
            m1 += par[2] * sc_m; l1 += par[4] * sc_l; m2[j] += par[8] * sc_m; l2[j] += par[10] * sc_l
        else:
            m1 += par[3] * sc_m; l1 += par[5] * sc_l; m2[j] += par[9] * sc_m; l2[j] += par[11] * sc_l
    return tot


@njit(cache=True)
def gas_toestanden(sk, dag, ai, u, grp, val, vorm, par, lam_min, lam_max, uit_m, uit_l, dag_sk, dag_dag, dag_st):
    """Zoals gas_pass2 met dagblok, zonder score; schrijft per rij de toestand vooraf (uit_m, uit_l) en per (rijder, dag)
    de toestand NA die dag (dag_st[k] = m1, l1, m2[0..7], l2[0..7]) voor voorspellen vanaf een willekeurige datum. Geeft het aantal dagen."""
    m1 = 0.0; l1 = 0.0; d1 = -1
    m2 = np.zeros(8); l2 = np.zeros(8); d2 = np.full(8, -1)
    m1d = 0.0; l1d = 0.0
    m2d = np.zeros(8); l2d = np.zeros(8)
    vorige = -1; vorige_dag = -1; nd = 0
    for i in range(len(u)):
        j = ai[i]
        if sk[i] != vorige:
            m1 = 0.0; l1 = 0.0; d1 = dag[i]
            for k in range(8):
                m2[k] = 0.0; l2[k] = 0.0; d2[k] = -1
            vorige = sk[i]; vorige_dag = -1
        if dag[i] > d1:
            f = (dag[i] - d1) / 30.0
            m1 *= par[0] ** f; l1 *= par[1] ** f; d1 = dag[i]
        if d2[j] >= 0 and dag[i] > d2[j]:
            f = (dag[i] - d2[j]) / 30.0
            m2[j] *= par[6] ** f; l2[j] *= par[7] ** f
        d2[j] = dag[i]
        if dag[i] != vorige_dag:
            for k in range(8):                    # laag 2 voor alle afstanden laten vervallen tot vandaag
                if d2[k] >= 0 and dag[i] > d2[k]:
                    f = (dag[i] - d2[k]) / 30.0
                    m2[k] *= par[6] ** f; l2[k] *= par[7] ** f; d2[k] = dag[i]
            m1d = m1; l1d = l1
            for k in range(8):
                m2d[k] = m2[k]; l2d[k] = l2[k]
            vorige_dag = dag[i]
        m = m1d + m2d[j]; lam = l1d + l2d[j]
        if lam < lam_min:
            lam = lam_min
        if lam > lam_max:
            lam = lam_max
        uit_m[i] = m; uit_l[i] = lam
        if not val[i]:
            g = grp[i]
            p0 = vorm[g, 0]; sL = vorm[g, 1]; nuL = vorm[g, 2]; sU = vorm[g, 3]; nuU = vorm[g, 4]
            eps = (u[i] - m) / math.exp(lam)
            if eps < 0:
                s = sL; nu = nuL
            else:
                s = sU; nu = nuU
            r2 = (eps / s) ** 2
            w = (nu + 1.0) / (nu + r2)
            sc_m = w * eps / (s * s)
            sc_l = w * r2 - 1.0
            if eps < 0:
                m1 += par[2] * sc_m; l1 += par[4] * sc_l; m2[j] += par[8] * sc_m; l2[j] += par[10] * sc_l
            else:
                m1 += par[3] * sc_m; l1 += par[5] * sc_l; m2[j] += par[9] * sc_m; l2[j] += par[11] * sc_l
        laatste = (i == len(u) - 1) or (sk[i + 1] != sk[i]) or (dag[i + 1] != dag[i])
        if laatste:
            dag_sk[nd] = sk[i]; dag_dag[nd] = dag[i]
            dag_st[nd, 0] = m1; dag_st[nd, 1] = l1
            for k in range(8):
                dag_st[nd, 2 + k] = m2[k]; dag_st[nd, 10 + k] = l2[k]; dag_st[nd, 18 + k] = d2[k]
            nd += 1
    return nd


@njit(cache=True)
def gas_pass3(sk, dag, ai, u, grp, val, scoor, vorm, par, lam_min, lam_max,
              pw, drempel_lo, drempel_hi, massa, uit_m, uit_lf, uit_ls):
    """Twee lagen met ASYMMETRISCHE volatiliteit per rijder: aparte log-schaal voor de snelle (lf) en trage (ls) kant.
    par = (phi_m1, phi_l1, am_s1, am_t1, a_f1, a_s1, phi_m2, phi_l2, am_s2, am_t2, a_f2, a_s2). Alleen eerdere dagen (dagblok)."""
    tot = 0.0
    m1 = 0.0; f1 = 0.0; s1 = 0.0; d1 = -1
    m2 = np.zeros(8); f2 = np.zeros(8); s2 = np.zeros(8); d2 = np.full(8, -1)
    m1d = 0.0; f1d = 0.0; s1d = 0.0
    m2d = np.zeros(8); f2d = np.zeros(8); s2d = np.zeros(8)
    vorige = -1; vorige_dag = -1
    for i in range(len(u)):
        j = ai[i]
        if sk[i] != vorige:
            m1 = 0.0; f1 = 0.0; s1 = 0.0; d1 = dag[i]
            for k in range(8):
                m2[k] = 0.0; f2[k] = 0.0; s2[k] = 0.0; d2[k] = -1
            vorige = sk[i]; vorige_dag = -1
        if dag[i] > d1:
            fct = (dag[i] - d1) / 30.0
            m1 *= par[0] ** fct; f1 *= par[1] ** fct; s1 *= par[1] ** fct; d1 = dag[i]
        if d2[j] < 0:
            d2[j] = dag[i]
        if dag[i] != vorige_dag:
            for k in range(8):
                if d2[k] >= 0 and dag[i] > d2[k]:
                    fct = (dag[i] - d2[k]) / 30.0
                    m2[k] *= par[6] ** fct; f2[k] *= par[7] ** fct; s2[k] *= par[7] ** fct; d2[k] = dag[i]
            m1d = m1; f1d = f1; s1d = s1
            for k in range(8):
                m2d[k] = m2[k]; f2d[k] = f2[k]; s2d[k] = s2[k]
            vorige_dag = dag[i]
        m = m1d + m2d[j]
        lf = min(max(f1d + f2d[j], lam_min), lam_max)
        ls = min(max(s1d + s2d[j], lam_min), lam_max)
        uit_m[i] = m; uit_lf[i] = lf; uit_ls[i] = ls
        if val[i]:
            continue
        g = grp[i]
        p0 = vorm[g, 0]; sL = vorm[g, 1]; nuL = vorm[g, 2]; sU = vorm[g, 3]; nuU = vorm[g, 4]
        x = u[i] - m
        if x < 0:
            lam = lf; eps = x / math.exp(lf); s = sL; nu = nuL
            lp = math.log(2.0 * p0) + _tlogpdf(eps / s, nu) - math.log(s) - lam
        else:
            lam = ls; eps = x / math.exp(ls); s = sU; nu = nuU
            lp = math.log(2.0 * (1.0 - p0)) + _tlogpdf(eps / s, nu) - math.log(s) - lam
        if eps < drempel_lo[g]:
            lp += math.log(pw[i, 0] / massa[g, 0])
        elif eps > drempel_hi[g]:
            lp += math.log(pw[i, 2] / massa[g, 2])
        else:
            lp += math.log(pw[i, 1] / massa[g, 1])
        if scoor[i]:
            tot += lp
        r2 = (eps / s) ** 2
        w = (nu + 1.0) / (nu + r2)
        sc_m = w * eps / (s * s)
        sc_l = w * r2 - 1.0
        if x < 0:
            m1 += par[2] * sc_m; f1 += par[4] * sc_l; m2[j] += par[8] * sc_m; f2[j] += par[10] * sc_l
        else:
            m1 += par[3] * sc_m; s1 += par[5] * sc_l; m2[j] += par[9] * sc_m; s2[j] += par[11] * sc_l
    return tot


@njit(cache=True)
def vorm_pass(sk, dg, z, val, Hk, c, Hl, cs, uit, dsk, ddg, dst, P=0.0):
    """Vormtoestand op dag-z (t.o.v. het veld), dagblok per rijder (alle afstanden). Per rij de toestand VOOR de dag in uit[:, 0:4]:
    Zk = som exp(-dt/Hk) * gemiddelde dag-z (geknipt op +-4), F = som exp(-dt/Hl) * max(0, -z - c) (snel),
    S = som exp(-dt/Hl) * max(0, z - cs) (traag), Zl = som exp(-dt/Hl) * gemiddelde dag-z. Na elke rijder-dag de toestand erna
    in dsk/ddg/dst (voor voorspellingen vooruit). uit[:, 4] = N = som exp(-dt/Hl) * 1[dag met z < -3]. Geeft het aantal rijder-dagen."""
    n = len(sk); zk = 0.0; vf = 0.0; vs = 0.0; zl = 0.0; vn = 0.0; ld = -1; i = 0; nd = 0
    # plateau P > 0: dag-gemiddelden jonger dan P dagen tellen volledig mee (buffer), daarna dooft hun bijdrage uit met exp(-(leeftijd - P) / Hk)
    nb = 64; bdg = np.zeros(nb, np.int64); bz = np.zeros(nb); kb = 0
    while i < n:
        j = i
        while j < n and sk[j] == sk[i] and dg[j] == dg[i]:
            j += 1
        if i == 0 or sk[i] != sk[i - 1]:
            zk = 0.0; vf = 0.0; vs = 0.0; zl = 0.0; vn = 0.0; kb = 0
        else:
            fk = math.exp(-(dg[i] - ld) / Hk); fl = math.exp(-(dg[i] - ld) / Hl)
            zk *= fk; vf *= fl; vs *= fl; zl *= fl; vn *= fl
        ld = dg[i]
        zkb = 0.0
        if P > 0.0:
            k2 = 0
            for b in range(kb):
                age = dg[i] - bdg[b]
                if age > P:
                    zk += bz[b] * math.exp(-(age - P) / Hk)
                else:
                    bdg[k2] = bdg[b]; bz[k2] = bz[b]; k2 += 1; zkb += bz[b]
            kb = k2
        for k in range(i, j):
            uit[k, 0] = zk + zkb; uit[k, 1] = vf; uit[k, 2] = vs; uit[k, 3] = zl; uit[k, 4] = vn
        af = 0.0; asl = 0.0; az = 0.0; nn = 0; ind = 0.0
        for k in range(i, j):
            if val[k]:
                continue
            if z[k] < -3.0:
                ind = 1.0
            if -z[k] - c > af:
                af = -z[k] - c
            if z[k] - cs > asl:
                asl = z[k] - cs
            az += min(max(z[k], -4.0), 4.0); nn += 1
        vf += af; vs += asl; vn += ind
        if nn > 0:
            if P > 0.0:
                if kb == nb:                     # buffer vol (kan alleen bij P > ~60 dagen): oudste naar de exponentiële toestand
                    zk += bz[0] * math.exp(-max(dg[i] - bdg[0] - P, 0.0) / Hk)
                    for b in range(nb - 1):
                        bdg[b] = bdg[b + 1]; bz[b] = bz[b + 1]
                    kb -= 1
                bdg[kb] = dg[i]; bz[kb] = az / nn; kb += 1; zkb += az / nn
            else:
                zk += az / nn
            zl += az / nn
        dsk[nd] = sk[i]; ddg[nd] = dg[i]
        dst[nd, 0] = zk + zkb; dst[nd, 1] = vf; dst[nd, 2] = vs; dst[nd, 3] = zl; dst[nd, 4] = vn
        nd += 1
        i = j
    return nd


@njit(cache=True)
def diepte_pass(sk, dg, z, val, zlo, H, uit, dsk, ddg, dst):
    """Piekdiepte: per rijder vervallen som van overschrijdingen in de piekzone (zlo - z voor z < zlo) en het aantal zulke races.
    uit[:, 0:2] = toestand VOOR de dag; dsk/ddg/dst = toestand NA elke rijder-dag. Geeft het aantal rijder-dagen."""
    n = len(sk); se = 0.0; sn = 0.0; ld = -1; i = 0; nd = 0
    while i < n:
        j = i
        while j < n and sk[j] == sk[i] and dg[j] == dg[i]:
            j += 1
        if i == 0 or sk[i] != sk[i - 1]:
            se = 0.0; sn = 0.0
        else:
            f = math.exp(-(dg[i] - ld) / H); se *= f; sn *= f
        ld = dg[i]
        for q in range(i, j):
            uit[q, 0] = se; uit[q, 1] = sn
        for q in range(i, j):
            if (not val[q]) and z[q] < zlo:
                se += zlo - z[q]; sn += 1.0
        dsk[nd] = sk[i]; ddg[nd] = dg[i]; dst[nd, 0] = se; dst[nd, 1] = sn; nd += 1
        i = j
    return nd
