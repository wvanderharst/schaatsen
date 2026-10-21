"""Universele baanfactor: één methode voor alle onderdelen (Kalman-meting, PR/seizoensbeste, voorspelling vooraf, omrekenen naar een baan).
    baanfactor(rit i in race r) = theta_r + d_r x (m(g_i) / m(gbar_r) - 1)
  theta_r : dagpeil van de race zoals gemeten uit het veld (achteraf) of voorspeld uit de baan (vooraf),
  d_r     : snelheid van het ijs t.o.v. de standaardbaan Heerenveen (theta_r - mediaan theta in Heerenveen voor geslacht x afstand),
  m(g)    : niet lineair in het niveau g (ln-tijd boven de wereldtop): m(g) = 1 + A (exp(-g / TM) - exp(-G0 / TM)), G0 = 10%,
  gbar_r  : gewogen gemiddeld niveau van het veld: het gemeten dagpeil geldt voor het veld; een rijder sterker of zwakker dan het veld
            krijgt meer of minder van het verschil met Heerenveen (deels aangepast door het niveau van het veld).
A en TM worden geschat op de fitjaren uit de fouten van de rijders t.o.v. hun niveau (schat_m)."""
import numpy as np, pandas as pd

G0 = 0.10


def m_func(g, A, TM):
    g = np.clip(np.asarray(g, float), 0.0, 0.5)
    return 1.0 + A * (np.exp(-g / TM) - np.exp(-G0 / TM))


def factor(theta, d, g, gbar, A, TM):
    return theta + d * (m_func(g, A, TM) / m_func(gbar, A, TM) - 1.0)


def veldniveau(rk, g, w=None):
    """Gewogen gemiddeld niveau per race (rk), voor elke rij."""
    w = np.ones(len(g)) if w is None else w
    ok = np.isfinite(g) & np.isfinite(w)
    s = pd.DataFrame({'rk': rk, 'gw': np.where(ok, np.clip(g, 0, 0.5) * w, 0.0), 'w': np.where(ok, w, 0.0)}).groupby('rk')[['gw', 'w']].transform('sum')
    return (s['gw'] / s['w'].replace(0, np.nan)).fillna(G0).to_numpy()


def hv_peil(df, theta_kol='theta', tot_jaar=2019):
    """Mediaan dagpeil in Heerenveen per geslacht x afstand (de standaardbaan), op de fitjaren."""
    return df[(df['baan'] == 'Heerenveen (NED)') & (df['jaar'] <= tot_jaar)].groupby(['gender', 'afstand_idx'])[theta_kol].median()


def schat_m(e, d, g, gbar, w=None, tm_grid=(0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12)):
    """Schat A en TM: de fout e (t.o.v. het niveau, met het veldpeil) ~ A x d x (h(g) - h(gbar)) / (1 + A h(gbar)) met h(g) = exp(-g/TM) - exp(-G0/TM);
    eerste orde in A: e ~ A x d x (h(g) - h(gbar)). Huber-achtig: |e| geknipt op 3%."""
    ok = np.isfinite(e) & np.isfinite(d) & np.isfinite(g) & np.isfinite(gbar)
    e = np.clip(e[ok], -0.03, 0.03); d = d[ok]; g = np.clip(g[ok], 0, 0.5); gbar = np.clip(gbar[ok], 0, 0.5)
    w = np.ones(len(e)) if w is None else w[ok]
    beste = None
    for tm in tm_grid:
        h = lambda x: np.exp(-x / tm) - np.exp(-G0 / tm)
        X = d * (h(g) - h(gbar))
        A = float(np.sum(w * X * e) / max(np.sum(w * X * X), 1e-12))
        rss = float(np.sum(w * (e - A * X) ** 2)); rss0 = float(np.sum(w * e ** 2))
        if beste is None or rss < beste[2]:
            beste = (A, tm, rss, 1 - rss / rss0)
    return beste
