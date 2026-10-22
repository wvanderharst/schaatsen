"""Proef: lopende ijking van de snelle staart. Voor elke rit met PIT p < Q: p' = Q (p/Q)^b, met b een gladde functie van het niveau g
(kernel over g, geen groepen), geschat uit EERDERE dagen: als p' uniform is, is x = -ln(p/Q) exponentieel met rate b, dus
b(g0) = (sum K(g, g0) w + k0) / (sum K(g, g0) w x + k0) over de ritten met p < Q, w = 0,5^(leeftijd / hl).
De log loss verandert exact met -ln b - (b - 1) ln(p/Q) voor p < Q. Eigen valpartijen tellen niet mee in de schatting.
Gebruik: python staart_lab.py <uitkomst.parquet> [Q] [hl] [bandbreedte]"""
import os, sys, importlib.util
import numpy as np, pandas as pd, pyarrow.parquet as pq
from numba import njit

spec = importlib.util.spec_from_file_location('em', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'elite', 'elite_model.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
G = np.linspace(-0.03, 0.45, 49)


@njit(cache=True)
def _ijk(dag, g, x, gebruik, G, bw, hl, k0, out):
    """dag gesorteerd. out[i] = b(g_i) uit de ritten van eerdere dagen."""
    n = len(dag); m = len(G); S1 = np.zeros(m); Sx = np.zeros(m); laatst = -1e9; i = 0
    while i < n:
        j = i
        while j < n and dag[j] == dag[i]:
            j += 1
        if laatst > -1e8:
            f = 0.5 ** ((dag[i] - laatst) / hl)
            for k in range(m):
                S1[k] *= f; Sx[k] *= f
        for q in range(i, j):                       # b op het rooster, lineair geinterpoleerd naar g
            pos = (g[q] - G[0]) / (G[1] - G[0])
            k = int(min(max(pos, 0.0), m - 1.001)); t = min(max(pos - k, 0.0), 1.0)
            b0 = (S1[k] + k0) / (Sx[k] + k0); b1 = (S1[k + 1] + k0) / (Sx[k + 1] + k0)
            out[q] = (1 - t) * b0 + t * b1
        for q in range(i, j):
            if gebruik[q]:
                for k in range(m):
                    kk = math_exp(-0.5 * ((g[q] - G[k]) / bw) ** 2)
                    S1[k] += kk; Sx[k] += kk * x[q]
        laatst = dag[i]; i = j


@njit(cache=True)
def math_exp(v):
    return np.exp(v)


def ijk(V, Q=0.1, hl=365.0, bw=0.02, k0=200.0):
    V = V.sort_values('dag', kind='stable')
    p = V['pit'].to_numpy(); g = V['k_c'].clip(-0.03, 0.45).fillna(0.2).to_numpy()
    staart = p < Q; x = np.where(staart, -np.log(np.clip(p, 1e-300, 1) / Q), 0.0)
    b = np.ones(len(V))
    _ijk(V['dag'].to_numpy().astype(np.float64), g, x, staart & ~V['k_val'].to_numpy(), G, bw, hl, k0, b)
    p2 = np.where(staart, Q * (np.clip(p, 1e-300, 1) / Q) ** b, p)
    d = np.where(staart, -np.log(b) - (b - 1) * np.log(np.clip(p, 1e-300, 1) / Q), 0.0)
    V = V.assign(pit2=p2, d=d, b=b)
    return V


def kort(r):
    if 'rijders' in r:
        return r.split('met ')[1].split(' ritten')[0] + ':' + r.split(':')[1].split(',')[0].strip() + '/' + r.split('verwacht')[1].split()[0]
    return r.split('kans<=')[1].split(':')[0].strip() + ' ' + r.split('(')[1].split(')')[0]


if __name__ == '__main__':
    pad = sys.argv[1]; Q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    hl = float(sys.argv[3]) if len(sys.argv) > 3 else 365.0; bw = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    V = pq.read_table(pad, columns=['skater_id', 'dag', 'afstand', 'tijd', 'jaar', 'seizoen', 'pit', 'nll', 'ok', 'k_val', 'k_c']).to_pandas()
    V = V[V['ok'] & np.isfinite(V['pit'])].copy(); V['afstand'] = V['afstand'].astype(str)
    O = pd.read_parquet(os.path.join(os.path.dirname(pad) or '.', 'oud_vooraf.parquet'), columns=['skater_id', 'dag', 'afstand', 'tijd', 'll', 'niveau_pct'])
    O = O[np.isfinite(O['ll'])].drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd']); O['afstand'] = O['afstand'].astype(str)
    V = ijk(V, Q, hl, bw)
    if len(sys.argv) > 5:                                    # geneste tweede stap dieper in de staart (bijv. Q2 = 0,01)
        Q2 = float(sys.argv[5]); d1 = V['d'].to_numpy(); b1 = V['b'].to_numpy()
        V2 = ijk(V.assign(pit=V['pit2']), Q2, hl, bw)
        V = V2.assign(pit=V['pit'], d=d1 + V2['d'].to_numpy(), b=b1 * 0 + V2['b'].to_numpy())
    V = V.merge(O, on=['skater_id', 'dag', 'afstand', 'tijd'], how='left')
    T = V[(V['jaar'] >= 2020) & np.isfinite(V['ll'])]
    print(f'Q {Q} hl {hl} bw {bw}')
    for nm, s in (('alle', T['niveau_pct'] > -1), ('toppers', T['niveau_pct'] >= 90), ('elite', T['niveau_pct'] >= 97)):
        x = T[s]; print(f"{nm:8s} log loss {x['nll'].mean():.4f} -> {(x['nll'] + x['d']).mean():.4f} | b gem {x['b'].mean():.3f}")
    for nm, s in (('toppers', T['niveau_pct'] >= 90), ('elite', T['niveau_pct'] >= 97)):
        for k, c in (('voor', 'pit'), ('na', 'pit2')):
            x = T[s][['skater_id', 'seizoen']].copy(); x['p'] = T[s][c].to_numpy()
            fout, rg, _, _ = em.toets(x, vanaf_seizoen=2019)
            print(f'   {nm:8s} {k:4s} | snel ' + ' '.join(kort(r) for r in rg[:8]))
    T = T.assign(niv=pd.cut(T['niveau_pct'], [-1, 50, 80, 90, 97, 101], labels=['<50', '50-80', '80-90', '90-97', '97+']))
    f = lambda x: pd.Series({'n': len(x), 's100 voor': (x.pit <= .01).mean() / .01, 's100 na': (x.pit2 <= .01).mean() / .01,
                             's20 voor': (x.pit <= .05).mean() / .05, 's20 na': (x.pit2 <= .05).mean() / .05, 'b': x.b.mean(), 'dLL': x.d.mean()})
    print(T[~T['k_val']].groupby('niv', observed=True).apply(f).round(3).to_string())
