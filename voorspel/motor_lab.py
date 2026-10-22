"""Proef: andere 'motoren' voor het rijderniveau dan het Kalman-filter, op hetzelfde doel: z = ln(tijd) - theta_rijder - ref (de prestatie
t.o.v. het peil van de race, zoals het filter die ziet), ex ante voorspeld, per race eerst voorspellen en dan bijwerken, valpartijen
(filtervlag) tellen niet mee. Parameters met maximale aannemelijkheid op 2008-2019, toets 2020+.
  kalman : kalman_mv2 (huidig), voorspelling N(zh, vl + r + vth); 'kalman-t': dezelfde met een Student-t (nu en schaal geschat).
  gas    : score-gedreven robuust filter: per rijder een niveau per afstand (gekoppeld via rho^|afstand|), een eigen toestandsonzekerheid
           die groeit met de tijd sinds de vorige rit, en twee eigen spreidingen (snelle en trage kant, log-schaal score-gedreven bijgewerkt
           met terugtrekking naar 0); voorspelling twee-delige Student-t. Bijwerken met gewicht w = (nu + 1) / (nu + u^2): uitschieters tellen minder.
  kde    : 'bootstrap': de voorspelling is een gewogen mengsel van de eigen eerdere ritten van de rijder (laatste NB), elk verschoven met de
           verwachte groei, gewicht exp(-dagen / tau) en lager voor andere afstanden (omx rho^|afstand|), met per rit een extra variantielaag
           (ch CD)^2 + lam x jaren + vth; plus een brede Student-t-component (pi) rond het gewogen midden.
Gebruik: python motor_lab.py <werkmap> [fit|toets]"""
import os, sys, json, time
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as T, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kalman as kb
import kalman_mv2 as km2

CD = kb.CD.astype(np.float64)
NB = 40


@njit(cache=True)
def _t_logpdf(u, nu, lg):
    # lg = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 log(nu pi)
    return lg - (nu + 1) / 2 * np.log1p(u * u / nu)


@njit(cache=True)
def _gas(rk, rid, a, z, vth, dag, lf, val, nr, p, lg, CD, o_m, o_sL, o_sU):
    P0, Q, RHO, C0, NU, B, PHI, G1, TG, TWEE = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    NA = 8; N = len(rk)
    mu = np.zeros((nr, NA)); Pv = np.zeros(nr); hL = np.zeros(nr); hU = np.zeros(nr); tl = np.full(nr, -1.0); st = np.zeros(nr, np.bool_)
    i = 0
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        for q in range(i, j):
            r = rid[q]; k = a[q]
            if not st[r]:
                o_m[q] = np.nan; o_sL[q] = np.nan; o_sU[q] = np.nan
                continue
            dt = max(dag[q] - tl[r], 0.0) / 365.0; dtc = min(dt, 1.5)
            age = lf[q] if lf[q] == lf[q] else 25.0
            gr = G1 * np.exp(-(age - 12.0) / TG)
            sb = C0 * CD[k]; Pq = Pv[r] + Q * dt
            o_m[q] = mu[r, k] + gr * dtc
            o_sL[q] = np.sqrt((sb * np.exp(hL[r])) ** 2 + Pq + vth[q]); o_sU[q] = np.sqrt((sb * np.exp(hU[r])) ** 2 + Pq + vth[q])
        for q in range(i, j):
            r = rid[q]; k = a[q]
            if z[q] != z[q]:
                continue
            if not st[r]:
                for d in range(NA):
                    mu[r, d] = z[q]
                Pv[r] = P0; hL[r] = 0.0; hU[r] = 0.0; tl[r] = dag[q]; st[r] = True
                continue
            if val[q]:
                continue
            dt = max(dag[q] - tl[r], 0.0) / 365.0; dtc = min(dt, 1.5)
            age = lf[q] if lf[q] == lf[q] else 25.0
            gr = G1 * np.exp(-(age - 12.0) / TG)
            for d in range(NA):
                mu[r, d] += gr * dtc
            Pv[r] += Q * dt
            e = z[q] - mu[r, k]
            sb = C0 * CD[k]
            h = hL[r] if e < 0 else hU[r]
            S2 = (sb * np.exp(h)) ** 2 + Pv[r] + vth[q]
            u2 = e * e / S2; w = (NU + 1.0) / (NU + u2)
            K = Pv[r] / S2
            for d in range(NA):
                mu[r, d] += K * w * e * RHO ** abs(d - k)
            Pv[r] = max(Pv[r] * max(1.0 - K * w, 0.05), 1e-8)
            sc = min(w * u2, 25.0) - 1.0
            if TWEE > 0.5:
                if e < 0:
                    hL[r] = PHI * hL[r] + B * sc
                else:
                    hU[r] = PHI * hU[r] + B * sc
            else:
                hL[r] = PHI * hL[r] + B * sc; hU[r] = hL[r]
            hL[r] = min(max(hL[r], -2.0), 2.0); hU[r] = min(max(hU[r], -2.0), 2.0)
            tl[r] = dag[q]
        i = j


@njit(cache=True)
def _kde(rk, rid, a, z, vth, dag, lf, val, nr, p, lgb, CD, NB, o_ll, o_pit, o_m, o_s):
    TAU, OMX, RHO, CH, LAM, PI, BREED, G1, TG, NUB = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    N = len(rk)
    bd = np.zeros((nr, NB)); bk = np.zeros((nr, NB), np.int64); bz = np.zeros((nr, NB)); cnt = np.zeros(nr, np.int64); pos = np.zeros(nr, np.int64)
    i = 0
    while i < N:
        j = i
        while j < N and rk[j] == rk[i]:
            j += 1
        for q in range(i, j):
            r = rid[q]; k = a[q]; n = min(cnt[r], NB)
            if n == 0:
                o_ll[q] = np.nan; o_pit[q] = np.nan; o_m[q] = np.nan; o_s[q] = np.nan
                continue
            age = lf[q] if lf[q] == lf[q] else 25.0
            gr = G1 * np.exp(-(age - 12.0) / TG)
            sw = 0.0; sm = 0.0; sd_ = 0.0; sc = 0.0; sm2 = 0.0; sh = 0.0
            for b in range(n):
                dtd = dag[q] - bd[r, b]; dt = dtd / 365.0
                wgt = np.exp(-dtd / TAU)
                if bk[r, b] != k:
                    wgt *= OMX * RHO ** abs(bk[r, b] - k)
                if wgt < 1e-12:
                    continue
                c = bz[r, b] + gr * min(dt, 1.5)
                h2 = (CH * CD[k]) ** 2 + LAM * dt + vth[q]
                sw += wgt; sm += wgt * c; sm2 += wgt * c * c; sh += wgt * h2
                if z[q] == z[q]:
                    u = (z[q] - c) / np.sqrt(h2)
                    sd_ += wgt * np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi * h2)
                    sc += wgt * 0.5 * (1.0 + _erf(u / np.sqrt(2.0)))
            if sw <= 0:
                o_ll[q] = np.nan; o_pit[q] = np.nan; o_m[q] = np.nan; o_s[q] = np.nan
                continue
            m = sm / sw; o_m[q] = m; o_s[q] = np.sqrt(max(sm2 / sw - m * m, 0.0) + sh / sw)
            if z[q] == z[q]:
                sb = BREED * CD[k]; ub = (z[q] - m) / sb
                fb = np.exp(lgb - (NUB + 1) / 2 * np.log1p(ub * ub / NUB)) / sb
                f = (1 - PI) * sd_ / sw + PI * fb
                o_ll[q] = -np.log(max(f, 1e-300))
                o_pit[q] = (1 - PI) * sc / sw + PI * _tcdf_approx(ub, NUB)
        for q in range(i, j):
            r = rid[q]
            if z[q] != z[q] or val[q]:
                continue
            b = pos[r] % NB
            bd[r, b] = dag[q]; bk[r, b] = a[q]; bz[r, b] = z[q]; pos[r] += 1; cnt[r] += 1
        i = j


@njit(cache=True)
def _erf(x):
    # Abramowitz-Stegun 7.1.26 (max fout 1.5e-7)
    s = 1.0 if x >= 0 else -1.0; x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


@njit(cache=True)
def _tcdf_approx(u, nu):
    # benadering via een normale met aangepaste schaal (alleen voor de brede component, gewicht pi klein)
    return 0.5 * (1.0 + _erf(u * np.sqrt((nu - 2.0) / nu) / np.sqrt(2.0))) if nu > 2.0 else 0.5 * (1.0 + _erf(u / np.sqrt(2.0)))


def lgam(nu):
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)


def gas_par(x):
    return np.array([np.exp(x[0]), np.exp(x[1]), 1 / (1 + np.exp(-x[2])), np.exp(x[3]), 2 + np.exp(x[4]), np.exp(x[5]), 1 / (1 + np.exp(-x[6])),
                     -np.exp(x[7]), np.exp(x[8]), 1.0])


def kde_par(x):
    return np.array([np.exp(x[0]), 1 / (1 + np.exp(-x[1])), 1 / (1 + np.exp(-x[2])), np.exp(x[3]), np.exp(x[4]), 1 / (1 + np.exp(-x[5])),
                     np.exp(x[6]), -np.exp(x[7]), np.exp(x[8]), 2 + np.exp(x[9])])


def gas_ms(inv, val, p):
    N = len(inv['rk']); m = np.zeros(N); sL = np.zeros(N); sU = np.zeros(N)
    _gas(inv['rk'], inv['rid'], inv['a'], inv['z'], inv['vth'], inv['dag'], inv['lf'], val, inv['nr'], p, lgam(p[4]), CD, m, sL, sU)
    return m, sL, sU


def gas_run(inv, val, p):
    N = len(inv['rk']); m = np.zeros(N); sL = np.zeros(N); sU = np.zeros(N)
    _gas(inv['rk'], inv['rid'], inv['a'], inv['z'], inv['vth'], inv['dag'], inv['lf'], val, inv['nr'], p, lgam(p[4]), CD, m, sL, sU)
    e = inv['z'] - m; nu = p[4]; lo = e < 0; s = np.where(lo, sL, sU)
    ll = -(np.log(2.0 / (sL + sU)) + lgam(nu) - (nu + 1) / 2 * np.log1p((e / s) ** 2 / nu))
    pit = np.where(lo, 2 * sL / (sL + sU) * T.cdf(e / sL, nu), sL / (sL + sU) + 2 * sU / (sL + sU) * (T.cdf(e / sU, nu) - 0.5))
    return ll, pit, m


def kde_run(inv, val, p, schaal=False):
    N = len(inv['rk']); ll = np.zeros(N); pit = np.zeros(N); m = np.zeros(N); sc = np.zeros(N)
    _kde(inv['rk'], inv['rid'], inv['a'], inv['z'], inv['vth'], inv['dag'], inv['lf'], val, inv['nr'], p, lgam(p[9]), CD, NB, ll, pit, m, sc)
    return (ll, pit, m, sc) if schaal else (ll, pit, m)


if __name__ == '__main__':
    w = sys.argv[1]; wat = sys.argv[2] if len(sys.argv) > 2 else 'toets'
    R = pd.read_parquet(os.path.join(w, 'ritten_cat.parquet'))
    df, inv = kb.voorbereid(R); del R
    par = json.load(open(os.path.join(w, 'kalman_par_mv2b.json')))
    o = km2.pass_(inv, par)
    val = o['val'].astype(np.bool_)
    jr = df['jaar'].to_numpy()
    S = o['vl'] + o['r'] + inv['vth']; e_k = inv['z'] - o['zh']
    basis = np.isfinite(inv['z']) & ~val & np.isfinite(o['zh']) & (S > 0)
    fit = basis & (jr >= 2008) & (jr <= 2019)
    if wat in ('gas', 'kde'):
        # schatten op een derde van de rijders en alleen t/m 2019 (rijders zijn onafhankelijk in deze motoren): 3x zo snel
        m_ = (inv['rid'] % 3 == 0) & (jr <= 2019)
        inv = {k: (v[m_] if isinstance(v, np.ndarray) and len(v) == len(m_) else v) for k, v in inv.items()}
        val = val[m_]; sub = fit[m_]
        if wat == 'gas':
            x0 = np.array([np.log(0.02), np.log(0.002), 2.0, np.log(0.6), np.log(4.0), np.log(0.05), 2.0, np.log(0.05), np.log(2.0)])
            def f(x):
                ll, _, _ = gas_run(inv, val, gas_par(x)); v = np.mean(ll[sub & np.isfinite(ll)]); return v if np.isfinite(v) else 1e9
        else:
            x0 = np.array([np.log(200.0), -1.0, 1.0, np.log(0.6), np.log(0.0004), -3.0, np.log(3.0), np.log(0.05), np.log(2.0), np.log(2.0)])
            def f(x):
                ll, _, _ = kde_run(inv, val, kde_par(x)); v = np.mean(ll[sub & np.isfinite(ll)]); return v if np.isfinite(v) else 1e9
        t0 = time.time(); print('start', wat, f(x0), round(time.time() - t0, 1), 's per pass', flush=True)
        it = [0]
        def cb(x):
            it[0] += 1
            if it[0] % 20 == 0:
                print(it[0], round(f(x), 5), np.round(x, 3).tolist(), flush=True)
        res = minimize(f, x0, method='Nelder-Mead', callback=cb, options={'maxiter': 700, 'xatol': 1e-3, 'fatol': 1e-6, 'adaptive': True})
        print('klaar', res.fun, res.x.tolist(), flush=True)
        json.dump(res.x.tolist(), open(os.path.join(w, f'motor_{wat}.json'), 'w'))
        sys.exit(0)
    if wat == 'eerlijk':
        # elke motor levert midden m en schaal s; daarop voor alle motoren dezelfde gekalibreerde twee-delige t (parameters lineair in
        # Z = [1, exp(-niveau / 4%), niveau], geschat op de fitjaren, per afstandsgroep), plus mengsels
        import verdeling_lab as vl
        V = pd.read_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'k_c']); V['afstand'] = V['afstand'].astype(str)
        D = df[['skater_id', 'dag', 'afstand', 'tijd']].copy(); D['afstand'] = D['afstand'].astype(str)
        kc = D.merge(V.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')['k_c'].to_numpy()
        gq = np.clip(np.nan_to_num(kc, nan=0.2), 0, 0.4); Z = np.column_stack([np.ones(len(gq)), np.exp(-gq / 0.04), gq])
        groep = pd.Series(D['afstand']).map({'500m': 0, '1000m': 0, '1500m': 1, '3000m': 1, '5000m': 2, '10000m': 2}).fillna(-1).to_numpy()
        ms = {'kalman': (o['zh'], np.sqrt(S))}
        mg, sL, sU = gas_ms(inv, val, gas_par(np.array(json.load(open(os.path.join(w, 'motor_gas.json'))))))
        ms['gas'] = (mg, 0.5 * (sL + sU)); ms['gas twee-delig'] = (mg, (sL, sU))
        _, _, mk, sk = kde_run(inv, val, kde_par(np.array(json.load(open(os.path.join(w, 'motor_kde.json'))))), schaal=True); ms['kde'] = (mk, sk)
        toets = basis & (jr >= 2020) & np.isfinite(kc) & (groep >= 0)
        for v in ms.values():
            toets &= np.isfinite(v[0])
        fitm = fit & np.isfinite(kc) & (groep >= 0)
        for v in ms.values():
            fitm &= np.isfinite(v[0])
        uit = {}
        rng = np.random.default_rng(0)
        for nm, (m, s_) in ms.items():
            if isinstance(s_, tuple):                       # twee-delig: e/sL links, e/sU rechts (de vorm daarna gekalibreerd)
                e = inv['z'] - m; u = np.where(e < 0, e / s_[0], e / s_[1]); lsc = np.where(e < 0, np.log(s_[0]), np.log(s_[1]))
            else:
                u = (inv['z'] - m) / s_; lsc = np.log(s_)
            ll = np.full(len(u), np.nan); pit = np.full(len(u), np.nan)
            for gidx in (0, 1, 2):
                idx = np.flatnonzero(fitm & (groep == gidx)); idx = rng.choice(idx, min(len(idx), 150000), replace=False)
                f = lambda th: -np.mean(vl.logpdf('tpt', th, u[idx], Z[idx]))
                th = minimize(f, vl.start('tpt', Z.shape[1]), method='L-BFGS-B', options={'maxiter': 3000}).x
                sg = (groep == gidx) & np.isfinite(u)
                ll[sg] = -vl.logpdf('tpt', th, u[sg], Z[sg]) + lsc[sg]; pit[sg] = vl.cdf('tpt', th, u[sg], Z[sg])
            uit[nm] = (ll, pit); print(nm, 'gekalibreerd', flush=True)
        dens = {k: np.exp(-v[0]) for k, v in uit.items()}
        for combi in (('kalman', 'gas twee-delig'), ('kalman', 'kde'), ('kalman', 'gas twee-delig', 'kde')):
            beste = (None, 1e9)
            roost = [(a_,) for a_ in np.arange(0, 1.01, 0.05)] if len(combi) == 2 else [(a_, b_) for a_ in np.arange(0, 1.01, 0.1) for b_ in np.arange(0, 1.01 - a_, 0.1)]
            for wts in roost:
                ws = list(wts) + [1 - sum(wts)]
                f_ = sum(wi * dens[c] for wi, c in zip(ws, combi)); v = np.mean(-np.log(np.maximum(f_[fitm], 1e-300)))
                if v < beste[1]:
                    beste = (ws, v)
            ws = beste[0]
            f_ = sum(wi * dens[c] for wi, c in zip(ws, combi)); pit_ = sum(wi * uit[c][1] for wi, c in zip(ws, combi))
            uit['mengsel ' + '+'.join(combi) + ' ' + '/'.join(f'{x:.2f}' for x in ws)] = (-np.log(np.maximum(f_, 1e-300)), pit_)
        print(f'\neerlijk (zelfde gekalibreerde verdeling), toets 2020+: {toets.sum()} ritten; log loss van z en kalibratie top<2% snel/traag 1/20 1/100 1/1000')
        for nm, (ll, pit) in uit.items():
            r = f'  {nm:48s}'
            for lab, s in (('alle', toets), ('top<2%', toets & (kc < 0.02)), ('top<6%', toets & (kc < 0.06)), ('6-18%', toets & (kc >= 0.06) & (kc < 0.18)), ('>18%', toets & (kc >= 0.18))):
                r += f' | {lab} {np.mean(ll[s]):.4f}'
            x = pit[toets & (kc < 0.02)]
            r += ' | snel ' + ' '.join(f'{(x <= q).mean() / q:.2f}' for q in (0.05, 0.01, 0.001)) + ' traag ' + ' '.join(f'{(x >= 1 - q).mean() / q:.2f}' for q in (0.05, 0.01, 0.001))
            print(r, flush=True)
        sys.exit(0)
    # toets: alle motoren op dezelfde ritten
    uit = {}
    ll_k = 0.5 * np.log(2 * np.pi * S) + 0.5 * e_k ** 2 / S; uit['kalman'] = (ll_k, norm.cdf(e_k / np.sqrt(S)))
    uk = e_k / np.sqrt(S)
    g = minimize(lambda x: np.mean(-(lgam(2 + np.exp(x[0])) - np.log(np.exp(x[1])) - (3 + np.exp(x[0])) / 2 * np.log1p((uk[fit] / np.exp(x[1])) ** 2 / (2 + np.exp(x[0]))))),
                 np.array([np.log(3.0), 0.0]), method='Nelder-Mead').x
    nu_k, c_k = 2 + np.exp(g[0]), np.exp(g[1])
    uit['kalman-t'] = (-(lgam(nu_k) - np.log(c_k * np.sqrt(S)) - (nu_k + 1) / 2 * np.log1p((uk / c_k) ** 2 / nu_k)), T.cdf(uk / c_k, nu_k))
    print(f'kalman-t: nu {nu_k:.2f}, schaal x{c_k:.3f}', flush=True)
    for nm, run, pf in (('gas', gas_run, gas_par), ('kde', kde_run, kde_par)):
        pad = os.path.join(w, f'motor_{nm}.json')
        if os.path.exists(pad):
            p = pf(np.array(json.load(open(pad)))); ll, pit, _ = run(inv, val, p); uit[nm] = (ll, pit)
            print(nm, 'parameters', np.round(p, 5).tolist(), flush=True)
    # mengsels (gewichten op de fitjaren)
    if 'gas' in uit and 'kde' in uit:
        dens = {k: np.exp(-v[0]) for k, v in uit.items()}
        beste = (None, 1e9)
        for wk in np.arange(0, 1.01, 0.1):
            for wg in np.arange(0, 1.01 - wk, 0.1):
                f_ = wk * dens['kalman-t'] + wg * dens['gas'] + (1 - wk - wg) * dens['kde']
                v = np.mean(-np.log(np.maximum(f_[fit & np.isfinite(f_)], 1e-300)))
                if v < beste[1]:
                    beste = ((wk, wg), v)
        wk, wg = beste[0]
        f_ = wk * dens['kalman-t'] + wg * dens['gas'] + (1 - wk - wg) * dens['kde']
        pit_ = wk * uit['kalman-t'][1] + wg * uit['gas'][1] + (1 - wk - wg) * uit['kde'][1]
        uit[f'mengsel {wk:.1f}/{wg:.1f}/{1 - wk - wg:.1f}'] = (-np.log(np.maximum(f_, 1e-300)), pit_)
    # niveau vooraf uit de modeluitkomst
    V = pd.read_parquet(os.path.join(w, 'voorspel2_uitkomst.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'k_c']); V['afstand'] = V['afstand'].astype(str)
    D = df[['skater_id', 'dag', 'afstand', 'tijd']].copy(); D['afstand'] = D['afstand'].astype(str)
    kc = D.merge(V.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']), on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')['k_c'].to_numpy()
    toets = basis & (jr >= 2020)
    for v in uit.values():
        toets &= np.isfinite(v[0]) & np.isfinite(v[1])
    print(f'\ntoets 2020+: {toets.sum()} ritten (gemeenschappelijk); log loss van z (lager is beter), en kalibratie snel/traag 1/20 1/100 1/1000')
    for nm, (ll, pit) in uit.items():
        r = f'  {nm:22s}'
        for lab, s in (('alle', toets), ('top<2%', toets & (kc < 0.02)), ('top<6%', toets & (kc < 0.06)), ('6-18%', toets & (kc >= 0.06) & (kc < 0.18))):
            r += f' | {lab} {np.mean(ll[s]):.4f}'
        s = toets & (kc < 0.02); x = pit[s]
        r += ' | top<2% snel ' + ' '.join(f'{(x <= q).mean() / q:.2f}' for q in (0.05, 0.01, 0.001)) + ' traag ' + ' '.join(f'{(x >= 1 - q).mean() / q:.2f}' for q in (0.05, 0.01, 0.001))
        print(r)
    np.savez(os.path.join(w, 'motor_toets.npz'), **{k.replace(' ', '_').replace('/', '-'): np.c_[v[0], v[1]] for k, v in uit.items()}, toets=toets, kc=kc)
