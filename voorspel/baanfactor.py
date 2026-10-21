"""Universele baanfactor: één methode voor alle onderdelen (Kalman-meting, PR/seizoensbeste, voorspelling vooraf, omrekenen naar een baan).
    baanfactor(rit i in race r) = theta_r + d_r x (m(g_i) / m(gbar_r) - 1) + B x (f(g_i) - f(gbar_r)),   f(g) = exp(-g / TB)
  theta_r : dagpeil van de race zoals gemeten uit het veld (achteraf) of voorspeld uit de baan (vooraf),
  d_r     : snelheid van het ijs t.o.v. de standaardbaan Heerenveen (theta_r - mediaan theta in Heerenveen voor geslacht x afstand),
  m(g)    : niet lineair in het niveau g (ln-tijd boven de wereldtop): m(g) = 1 + A (exp(-g / TM) - exp(-G0 / TM)), G0 = 10%,
  gbar_r  : gewogen gemiddeld niveau van het veld: het gemeten dagpeil geldt voor het veld; een rijder sterker of zwakker dan het veld
            krijgt meer of minder van het verschil met Heerenveen (deels aangepast door het niveau van het veld).
  B-term : elite rijden vaak op beter geprepareerd ijs (en zijn scherper in een sterk veld): het dagpeil van een race wordt gemeten op het
            veld; een rijder die sterker is dan het veld rijdt op dat ijs relatief trager dan elders (f(g) > f(gbar) -> positief), een
            zwakkere rijder in een sterk veld profiteert. Binnen dezelfde toprijders: ~0,18% verschil tussen elite- en zwakke velden.
A, TM, B en TB worden samen geschat op de fitjaren uit de fouten van de rijders t.o.v. hun niveau (schat_m)."""
import numpy as np, pandas as pd

G0 = 0.10


def m_func(g, A, TM):
    g = np.clip(np.asarray(g, float), 0.0, 0.5)
    return 1.0 + A * (np.exp(-g / TM) - np.exp(-G0 / TM))


def factor(theta, d, g, gbar, A, TM, B=0.0, TB=0.05):
    gc = np.clip(np.asarray(g, float), 0.0, 0.5); gb = np.clip(np.asarray(gbar, float), 0.0, 0.5)
    return theta + d * (m_func(g, A, TM) / m_func(gbar, A, TM) - 1.0) + B * (np.exp(-gc / TB) - np.exp(-gb / TB))


def veldniveau(rk, g, w=None):
    """Gewogen gemiddeld niveau per race (rk), voor elke rij."""
    w = np.ones(len(g)) if w is None else w
    ok = np.isfinite(g) & np.isfinite(w)
    s = pd.DataFrame({'rk': rk, 'gw': np.where(ok, np.clip(g, 0, 0.5) * w, 0.0), 'w': np.where(ok, w, 0.0)}).groupby('rk')[['gw', 'w']].transform('sum')
    return (s['gw'] / s['w'].replace(0, np.nan)).fillna(G0).to_numpy()


def hv_peil(df, theta_kol='theta', tot_jaar=2019):
    """Mediaan dagpeil in Heerenveen per geslacht x afstand (de standaardbaan), op de fitjaren."""
    return df[(df['baan'] == 'Heerenveen (NED)') & (df['jaar'] <= tot_jaar)].groupby(['gender', 'afstand_idx'])[theta_kol].median()


def schat_m(e, d, g, gbar, w=None, tm_grid=(0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12), tb_grid=None, hulp=True):
    """Schat A en TM: de fout e (t.o.v. het niveau, met het veldpeil) ~ A x d x (h(g) - h(gbar)) / (1 + A h(gbar)) met h(g) = exp(-g/TM) - exp(-G0/TM);
    eerste orde in A: e ~ A x d x (h(g) - h(gbar)). Huber-achtig: |e| geknipt op 3%."""
    ok = np.isfinite(e) & np.isfinite(d) & np.isfinite(g) & np.isfinite(gbar)
    e = np.clip(e[ok], -0.03, 0.03); d = d[ok]; g = np.clip(g[ok], 0, 0.5); gbar = np.clip(gbar[ok], 0, 0.5)
    w = np.ones(len(e)) if w is None else w[ok]
    beste = None; rss0 = float(np.sum(w * (e - np.average(e, weights=w)) ** 2))
    for tm in tm_grid:
        h = lambda x: np.exp(-x / tm) - np.exp(-G0 / tm)
        X1 = d * (h(g) - h(gbar))
        for tb in (tb_grid or (None,)):
            # hulptermen voor een afwijking die alleen van het eigen niveau afhangt (filter, niet het ijs): constante, g, g^2, f(g);
            # de baanfactor-termen zijn dan de afhankelijkheid van het VELD bij gelijk eigen niveau
            N_ = [np.ones(len(e)), g, g * g] if hulp else [np.ones(len(e))]
            if tb is None:
                X = np.column_stack(N_ + [X1])
            else:
                X = np.column_stack(N_ + [np.exp(-g / tb), X1, -np.exp(-gbar / tb)])
            c = np.linalg.lstsq(X * np.sqrt(w)[:, None], e * np.sqrt(w), rcond=None)[0]
            rss = float(np.sum(w * (e - X @ c) ** 2))
            if beste is None or rss < beste[2]:
                beste = (float(c[-2] if tb is not None else c[-1]), tm, rss, 1 - rss / rss0, float(c[-1]) if tb is not None else 0.0, tb if tb is not None else 0.05)
    return beste


def schat_a(e, d, g, gbar, tm_grid=(0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.2, 0.3)):
    """Stap 1: alleen de snelheidsterm (zonder hulptermen en veldterm); A >= 0 (toppers winnen niet minder op snel ijs dan het veld)."""
    A, TM, _, _, _, _ = schat_m(e, d, g, gbar, tm_grid=tm_grid, hulp=False)
    return max(A, 0.0), TM


def schat_b(e, d, g, gbar, A, TM, tb_grid=(0.02, 0.03, 0.05, 0.08, 0.12, 0.2)):
    """Stap 2: veldterm B x (f(g) - f(gbar)) met A, TM vast, met hulptermen voor het eigen niveau."""
    ok = np.isfinite(e) & np.isfinite(d) & np.isfinite(g) & np.isfinite(gbar)
    e = np.clip(e[ok], -0.03, 0.03); d = d[ok]; g = np.clip(g[ok], 0, 0.5); gbar = np.clip(gbar[ok], 0, 0.5)
    r = e - d * (m_func(g, A, TM) / m_func(gbar, A, TM) - 1.0)
    beste = None
    for tb in tb_grid:
        X = np.column_stack([np.ones(len(e)), g, g * g, np.exp(-g / tb), -np.exp(-gbar / tb)])
        c = np.linalg.lstsq(X, r, rcond=None)[0]; rss = float(np.sum((r - X @ c) ** 2))
        if beste is None or rss < beste[2]:
            beste = (max(float(c[-1]), 0.0), tb, rss)
    return beste[0], beste[1], 1 - beste[2] / float(np.sum((r - r.mean()) ** 2))
