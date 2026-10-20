"""Kenmerken vooraf voor het voorspellen van een rittijd (alles chronologisch, alleen informatie van vóór de race).

Twee delen:
  ijs     : per race (dag x event x geslacht x afstand) wat er vooraf over het racepeil theta bekend is: het lopende gemiddelde
            van theta op deze baan voor deze afstand en geslacht, op deze baan over alle afstanden, de afwijking op de eerdere dagen
            van hetzelfde toernooi (zelfde baan, binnen 5 dagen) en de algemene afwijking van de laatste weken.
  rijder  : per rit het niveau op de afstand (robuust lopend gemiddelde van ln tijd - theta, snelle ritten wegen zwaarder), de vorm
            over alle afstanden (snel en traag uitdovend), de eigen spreiding (alle afstanden samen), de laatste fout op deze en op
            elke afstand, het bewezen PR (beste ln tijd - theta, en de ruwe beste tijd), de seizoensbeste van dit en de twee vorige
            seizoenen, het aantal ritten, de tijd sinds de vorige rit en de niveaus op alle afstanden.
Gebruik: python kenmerken.py <map met ritten_cat.parquet>  (schrijft kenmerken.parquet)"""
import sys, os, math, time
import numpy as np, pandas as pd
from numba import njit

AF = ['100m', '300m', '500m', '1000m', '1500m', '3000m', '5000m', '10000m']
CD = np.array([0.012, 0.012, 0.011, 0.0134, 0.015, 0.0161, 0.0154, 0.016])     # typische spreiding van de elite per afstand
HOOGLAND = ('Salt Lake City (USA)', 'Calgary (CAN)')


@njit(cache=True)
def _ijs(rk_start, rk_end, baan, gsl, af, dag, theta, nb, w_min, w_baan, ev_dagen, glob_hl,
         o_vm, o_vn, o_ba, o_bn, o_ev, o_evn, o_gl):
    """Per race (blokken rk_start..rk_end, gesorteerd op dag): lopend gemiddelde theta per baan x geslacht x afstand en per baan,
    afwijking op eerdere dagen van hetzelfde toernooi en de algemene afwijking (halfwaardetijd glob_hl dagen)."""
    vm = np.zeros((nb, 2, 8)); vn = np.zeros((nb, 2, 8)); bm = np.zeros(nb); bn = np.zeros(nb)
    # eerdere dagen van een toernooi: per baan de afwijkingen van de laatste dagen (dag, som, aantal), ringbuffer van 64
    rb_d = np.full((nb, 64), -100000); rb_s = np.zeros((nb, 64)); rb_k = np.zeros((nb, 64)); rb_i = np.zeros(nb, np.int64)
    gl = 0.0; gl_d = -100000.0
    R = len(rk_start); i = 0
    while i < R:
        # alle races van deze dag eerst voorspellen, dan pas bijwerken (volgorde binnen een dag is onbekend)
        j = i
        while j < R and dag[j] == dag[i]:
            j += 1
        for q in range(i, j):
            b = baan[q]; g = gsl[q]; a = af[q]
            o_vm[q] = vm[b, g, a]; o_vn[q] = vn[b, g, a]; o_ba[q] = bm[b]; o_bn[q] = bn[b]
            s = 0.0; k = 0.0
            for t in range(64):
                if rb_d[b, t] < dag[q] and rb_d[b, t] >= dag[q] - ev_dagen:
                    s += rb_s[b, t]; k += rb_k[b, t]
            o_ev[q] = s / k if k > 0 else 0.0; o_evn[q] = k
            o_gl[q] = gl * math.exp(-(dag[q] - gl_d) / glob_hl * math.log(2.0)) if gl_d > -100000 else 0.0
        for q in range(i, j):
            th = theta[q]
            if th != th:
                continue
            b = baan[q]; g = gsl[q]; a = af[q]
            ref = vm[b, g, a] if vn[b, g, a] >= 3 else (bm[b] if bn[b] >= 3 else th)
            d = th - ref
            # toernooi-buffer: tel deze afwijking op bij de ingang van deze dag
            gevonden = False
            for t in range(64):
                if rb_d[b, t] == dag[q]:
                    rb_s[b, t] += d; rb_k[b, t] += 1.0; gevonden = True; break
            if not gevonden:
                t = rb_i[b] % 64; rb_d[b, t] = dag[q]; rb_s[b, t] = d; rb_k[b, t] = 1.0; rb_i[b] += 1
            # algemene afwijking (lopend, uitdovend)
            dec = math.exp(-(dag[q] - gl_d) / glob_hl * math.log(2.0)) if gl_d > -100000 else 0.0
            gl = gl * dec * 0.98 + 0.02 * d; gl_d = dag[q]
            w = max(1.0 / (vn[b, g, a] + 1.0), w_min); vm[b, g, a] += w * (th - vm[b, g, a]) if vn[b, g, a] > 0 else 0.0
            if vn[b, g, a] == 0:
                vm[b, g, a] = th
            vn[b, g, a] += 1.0
            wb = max(1.0 / (bn[b] + 1.0), w_baan); bm[b] = th if bn[b] == 0 else bm[b] + wb * (th - bm[b]); bn[b] += 1.0
        i = j
    return 0


def ijs_kenmerken(df, w_min=0.05, w_baan=0.02, ev_dagen=5, glob_hl=30.0):
    """Kenmerken van het racepeil vooraf per race; geeft een DataFrame op race-niveau (sleutel dag, event_id, gender, afstand_idx)."""
    r = df.drop_duplicates(['dag', 'event_id', 'gender', 'afstand_idx'])[['dag', 'event_id', 'gender', 'afstand_idx', 'baan', 'theta', 'v_theta', 'seizoen']]
    r = r.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    bc, bu = pd.factorize(r['baan']); n = len(r)
    o = {k: np.zeros(n) for k in ('vm', 'vn', 'ba', 'bn', 'ev', 'evn', 'gl')}
    _ijs(np.arange(n), np.arange(n), bc.astype(np.int64), (r['gender'] == 'v').to_numpy().astype(np.int64), r['afstand_idx'].to_numpy().astype(np.int64),
         r['dag'].to_numpy().astype(np.int64), r['theta'].to_numpy(float), len(bu), w_min, w_baan, ev_dagen, glob_hl,
         o['vm'], o['vn'], o['ba'], o['bn'], o['ev'], o['evn'], o['gl'])
    for k, v in o.items():
        r['ijs_' + k] = v
    r['hoog'] = r['baan'].isin(HOOGLAND).astype(float)
    return r


@njit(cache=True)
def _rijder(rk, rid, a, y, lt, spar, dag, seiz, nr, alpha, snel_w, hub, gamma, hf, hf2, beta, m0r, m0, cd, val_sd, val_min,
            o_mu, o_n, o_gapa, o_gap, o_f, o_f2, o_s, o_sa, o_pr, o_prraw, o_sb3, o_sbc, o_ela, o_elany, o_elage, o_e, o_val,
            o_mus, o_ns, o_eerste_dag):
    NA = 8
    mu = np.zeros((nr, NA)); n = np.zeros((nr, NA)); tl = np.full((nr, NA), -100000.0); el = np.zeros((nr, NA))
    pr = np.full((nr, NA), 9.9); prraw = np.full((nr, NA), 99.0); s2a = np.zeros((nr, NA))
    sb0 = np.full((nr, NA), 9.9); sb1 = np.full((nr, NA), 9.9); sb2 = np.full((nr, NA), 9.9); sbs = np.full((nr, NA), -1)
    f = np.zeros(nr); fd = np.zeros(nr); f2 = np.zeros(nr); f2d = np.zeros(nr); r2 = np.ones(nr); nr2 = np.zeros(nr)
    lastany = np.full(nr, -100000.0); elany = np.zeros(nr); elany_d = np.full(nr, -100000.0); eerste = np.full(nr, -1.0)
    N = len(rk); i = 0
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        for q in range(i, j):
            r_ = rid[q]; d_ = a[q]
            if eerste[r_] < 0:
                eerste[r_] = dag[q]
            o_eerste_dag[q] = eerste[r_]
            if sbs[r_, d_] != seiz[q]:                     # nieuw seizoen voor deze afstand: seizoensbesten doorschuiven
                if sbs[r_, d_] == seiz[q] - 1:
                    sb2[r_, d_] = sb1[r_, d_]; sb1[r_, d_] = sb0[r_, d_]
                elif sbs[r_, d_] == seiz[q] - 2:
                    sb2[r_, d_] = sb0[r_, d_]; sb1[r_, d_] = 9.9
                else:
                    sb1[r_, d_] = 9.9; sb2[r_, d_] = 9.9
                sb0[r_, d_] = 9.9; sbs[r_, d_] = seiz[q]
            fc = f[r_] * math.exp(-(dag[q] - fd[r_]) / hf)
            fc2 = f2[r_] * math.exp(-(dag[q] - f2d[r_]) / hf2)
            o_f[q] = fc; o_f2[q] = fc2; o_n[q] = n[r_, d_]; o_mu[q] = mu[r_, d_]
            o_gapa[q] = dag[q] - tl[r_, d_]; o_gap[q] = dag[q] - lastany[r_]
            rr = (nr2[r_] * r2[r_] + m0r) / (nr2[r_] + m0r)
            sp = cd[d_] * math.sqrt(rr); o_s[q] = sp
            o_sa[q] = math.sqrt((n[r_, d_] * s2a[r_, d_] + m0 * sp * sp) / (n[r_, d_] + m0))
            o_pr[q] = pr[r_, d_]; o_prraw[q] = prraw[r_, d_]
            o_sb3[q] = min(sb0[r_, d_], min(sb1[r_, d_], sb2[r_, d_])); o_sbc[q] = sb0[r_, d_]
            o_ela[q] = el[r_, d_]; o_elany[q] = elany[r_]; o_elage[q] = dag[q] - elany_d[r_]
            for k in range(NA):
                o_mus[q, k] = mu[r_, k]; o_ns[q, k] = n[r_, k]
            if n[r_, d_] == 0 or y[q] != y[q]:
                o_e[q] = np.nan; o_val[q] = False
            else:
                e = y[q] - mu[r_, d_] - fc; o_e[q] = e
                o_val[q] = (e > val_sd * math.sqrt(sp * sp + spar[q] * spar[q])) and (e > val_min)
        for q in range(i, j):
            r_ = rid[q]; d_ = a[q]
            if o_val[q] or y[q] != y[q]:
                lastany[r_] = dag[q]
                continue
            if lt[q] < prraw[r_, d_]:
                prraw[r_, d_] = lt[q]
            if y[q] < pr[r_, d_]:
                pr[r_, d_] = y[q]
            if y[q] < sb0[r_, d_]:
                sb0[r_, d_] = y[q]
            if n[r_, d_] == 0:
                mu[r_, d_] = y[q]; n[r_, d_] = 1.0; el[r_, d_] = 0.0
            else:
                sp = o_s[q]; e = y[q] - mu[r_, d_] - o_f[q]      # opnieuw: een tweede rit in dezelfde race had vooraf nog geen niveau
                w = max(1.0 / (n[r_, d_] + 1.0), alpha)
                if e < 0:
                    w = min(w * snel_w, 0.6)
                ec = min(max(e, -hub * sp), hub * sp)
                mu[r_, d_] += w * ec
                b = max(1.0 / (n[r_, d_] + 1.0), beta)
                s2a[r_, d_] = (1 - b) * s2a[r_, d_] + b * min(e * e, (hub * sp) ** 2)
                br = max(1.0 / (nr2[r_] + 1.0), beta)
                r2[r_] = (1 - br) * r2[r_] + br * min(e * e, (hub * sp) ** 2) / (cd[d_] * cd[d_]); nr2[r_] += 1.0
                f[r_] = o_f[q] + gamma * ec; fd[r_] = dag[q]
                f2[r_] = o_f2[q] + 0.15 * ec; f2d[r_] = dag[q]
                el[r_, d_] = e; elany[r_] = e; elany_d[r_] = dag[q]
                n[r_, d_] += 1.0
            tl[r_, d_] = dag[q]; lastany[r_] = dag[q]
        i = j
    return 0


def rijder_kenmerken(df, alpha=0.2, snel_w=2.0, hub=3.0, gamma=0.4, hf=30.0, hf2=120.0, beta=0.06, m0r=25.0, m0=5.0, val_sd=6.0, val_min=0.03):
    df = df.sort_values(['dag', 'event_id', 'gender', 'afstand_idx'], kind='stable').reset_index(drop=True)
    rk = df.groupby(['dag', 'event_id', 'gender', 'afstand_idx'], sort=False).ngroup().to_numpy().astype(np.int64)
    rid, _ = pd.factorize(df['skater_id']); N = len(df)
    lt = np.log(df['tijd'].to_numpy(float)); th = df['theta'].to_numpy(float)
    y = np.where(df['verdacht'].to_numpy(), np.nan, lt - th)
    names = ('mu', 'n', 'gapa', 'gap', 'f', 'f2', 's', 'sa', 'pr', 'prraw', 'sb3', 'sbc', 'ela', 'elany', 'elage', 'e', 'eerste_dag')
    o = {k: np.zeros(N) for k in names}; val = np.zeros(N, bool)
    mus = np.zeros((N, 8), np.float32); ns = np.zeros((N, 8), np.float32)
    _rijder(rk, rid.astype(np.int64), df['afstand_idx'].to_numpy().astype(np.int64), y, lt, np.sqrt(df['v_theta'].to_numpy(float)),
            df['dag'].to_numpy().astype(np.float64), df['seizoen'].to_numpy().astype(np.int64), rid.max() + 1,
            alpha, snel_w, hub, gamma, hf, hf2, beta, m0r, m0, CD, val_sd, val_min,
            o['mu'], o['n'], o['gapa'], o['gap'], o['f'], o['f2'], o['s'], o['sa'], o['pr'], o['prraw'], o['sb3'], o['sbc'],
            o['ela'], o['elany'], o['elage'], o['e'], val, mus, ns, o['eerste_dag'])
    for k in names:
        df[k] = o[k].astype(np.float32) if k not in ('e',) else o[k]
    df['val'] = val; df['y'] = y
    for k, nm in enumerate(AF):
        df['mu_' + nm] = mus[:, k]; df['n_' + nm] = ns[:, k]
    return df


def bouw(df):
    t = time.time()
    df = rijder_kenmerken(df); print('rijder-pass', round(time.time() - t), 's', flush=True)
    r = ijs_kenmerken(df); print('ijs-pass', round(time.time() - t), 's', flush=True)
    df = df.merge(r[['dag', 'event_id', 'gender', 'afstand_idx'] + [c for c in r.columns if c.startswith('ijs_')] + ['hoog']],
                  on=['dag', 'event_id', 'gender', 'afstand_idx'], how='left')
    return df


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else '.'
    df = bouw(pd.read_parquet(os.path.join(w, 'ritten_cat.parquet')))
    df.to_parquet(os.path.join(w, 'kenmerken.parquet'))
    print(len(df), 'ritten', flush=True)
