"""Bewezen niveau (lange-termijngeheugen): per rit het verschil tussen de op een na beste kwaliteit van de rijder op dezelfde afstand in
de laatste 730 dagen (q = ln tijd - theta, achteraf gecorrigeerd voor baan, ijs, weer en hoogte; alleen EERDERE dagen, geen valpartijen of
verdachte ritten) en de verwachting vooraf (pred - th_hat_veld), in %. Negatief: de rijder reed recent al duidelijk sneller dan nu verwacht.
De op een na beste (niet de beste) maakt het robuust tegen een enkele meevaller. Wordt gebruikt als kenmerk x in rustlaag.laag."""
import numpy as np, pandas as pd
from numba import njit

GX = np.linspace(-6.0, 4.0, 41)                       # rooster in %


@njit(cache=True)
def _tweede(gid, dag, q, geldig, W, k, uit):
    n = len(q)
    for i in range(n):
        m1 = 1e9; m2 = 1e9; j = i - 1
        while j >= 0 and gid[j] == gid[i] and dag[j] >= dag[i] - W:
            if dag[j] < dag[i] and geldig[j]:
                if q[j] < m1:
                    m2 = m1; m1 = q[j]
                elif q[j] < m2:
                    m2 = q[j]
            j -= 1
        m = m2 if k == 2 else m1
        uit[i] = m if m < 1e8 else np.nan


def gat(skater_id, afstand, dag, tijd, theta, verwacht_q, geldig, W=730.0, k=2):
    """verwacht_q = pred - th_hat_veld (ln-tijd op een neutrale baan); geldig = rit telt mee in het geheugen. Geeft het verschil in %."""
    df = pd.DataFrame({'s': np.asarray(skater_id), 'a': np.asarray(afstand).astype(str), 'd': np.asarray(dag, float)})
    o = np.lexsort((df['d'].to_numpy(), pd.factorize(df['a'])[0], pd.factorize(df['s'])[0]))
    gid = (df.iloc[o].groupby(['s', 'a'], sort=False).ngroup()).to_numpy().astype(np.int64)
    q = np.log(np.asarray(tijd, float)) - np.asarray(theta, float)
    ok = np.asarray(geldig, bool) & np.isfinite(q)
    u = np.full(len(o), np.nan)
    _tweede(gid, df['d'].to_numpy()[o], np.nan_to_num(q[o]), ok[o], float(W), int(k), u)
    q2 = np.full(len(q), np.nan); q2[o] = u
    return 100 * (np.exp(q2 - np.asarray(verwacht_q, float)) - 1)
