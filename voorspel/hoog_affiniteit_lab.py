"""Achteraf: eigen hooglandaffiniteit per rijder. Voor elke hooglandrit: gemiddelde fout van de rijder op EERDERE hooglanddagen
min de gemiddelde fout elders (eerdere dagen), gekrompen: n / (n + k). Kiezen op 2012-2019, tonen op 2020+."""
import numpy as np, pandas as pd
V = pd.read_parquet('voorspel2_uitkomst.parquet', columns=['skater_id', 'dag', 'jaar', 'baan', 'tijd', 'pred', 'k_c', 'ok', 'k_val'])
V = V[V['ok'] & np.isfinite(V['pred'])].sort_values(['skater_id', 'dag'], kind='stable').reset_index(drop=True)
V['r'] = np.clip(np.log(V['tijd']) - V['pred'], -0.03, 0.03); V.loc[V['k_val'], 'r'] = np.nan
V['hoog'] = V['baan'].isin(['Salt Lake City (USA)', 'Calgary (CAN)'])
# per rijder cumulatief (alleen eerdere dagen): som en aantal op hoogland en elders
for h, nm in ((True, 'h'), (False, 'l')):
    x = V['r'].where(V['hoog'] == h); c = x.notna().astype(float); x = x.fillna(0.0)
    g = V.groupby('skater_id')
    s_ = x.groupby(V['skater_id']).cumsum() - x.groupby([V['skater_id'], V['dag']]).cumsum()         # eerdere dagen
    n_ = c.groupby(V['skater_id']).cumsum() - c.groupby([V['skater_id'], V['dag']]).cumsum()
    V['s' + nm] = s_; V['n' + nm] = n_
H = V[V['hoog'] & V['r'].notna()].copy()
H['dif'] = H['sh'] / H['nh'].clip(lower=1) - H['sl'] / H['nl'].clip(lower=1)
H.loc[H['nh'] == 0, 'dif'] = 0.0
top = H['k_c'] < 0.06
for k in (0, 3, 5, 10, 20, 50):
    corr = H['dif'] * H['nh'] / (H['nh'] + k)
    uit = []
    for nm, s in (('fit', H['jaar'].between(2012, 2019)), ('toets', H['jaar'] >= 2020)):
        for gr, t in (('alle', np.ones(len(H), bool)), ('top <6%', top.to_numpy())):
            m = s.to_numpy() & t
            uit.append(f"{nm} {gr}: |fout| {100 * np.median(np.abs(H['r'][m])):.3f} -> {100 * np.median(np.abs(H['r'][m] - corr[m])):.3f}")
    print(f'k {k}:', ' | '.join(uit))
k = 10; corr = H['dif'] * H['nh'] / (H['nh'] + k)
L = H[H['skater_id'] == V.loc[V['skater_id'].isin(H['skater_id']), 'skater_id'].iloc[0]] if False else None
print('--- samenhang tussen eerdere en huidige hooglandafwijking, naar aantal eerdere hooglanddagen')
H['rc'] = H['r'] - H['sl'] / H['nl'].clip(lower=1)
for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 999)):
    for nm, s in (('fit', H['jaar'].between(2012, 2019)), ('toets', H['jaar'] >= 2020)):
        m = s & H['nh'].between(lo, hi - 1) & (H['k_c'] < 0.06)
        if m.sum() > 50:
            x = H.loc[m, 'sh'] / H.loc[m, 'nh'] - H.loc[m, 'sl'] / H.loc[m, 'nl'].clip(lower=1)
            b = np.polyfit(x, H.loc[m, 'r'], 1)[0]
            print(f'{lo}-{hi - 1} eerdere hooglanddagen, toppers, {nm}: n {int(m.sum())}, helling {b:.2f} (1 = volledig doorzetten, 0 = geen voorspellende waarde), corr {np.corrcoef(x, H.loc[m, "r"])[0, 1]:.3f}')
print('--- correctie = beta x (eerdere hooglandafwijking), gekrompen n/(n+3)')
dif = (H['sh'] / H['nh'].clip(lower=1) - H['sl'] / H['nl'].clip(lower=1)).where(H['nh'] > 0, 0.0) * H['nh'] / (H['nh'] + 3)
for beta in (0.05, 0.1, 0.15, 0.2, 0.3):
    uit = []
    for nm, s in (('fit', H['jaar'].between(2012, 2019)), ('toets', H['jaar'] >= 2020)):
        for gr, t in (('alle', np.ones(len(H), bool)), ('top <6%', (H['k_c'] < 0.06).to_numpy())):
            m = s.to_numpy() & t
            uit.append(f"{nm} {gr} {100 * np.median(np.abs(H['r'][m])):.4f}->{100 * np.median(np.abs(H['r'][m] - beta * dif[m])):.4f}")
    print(f'beta {beta}:', ' | '.join(uit))
sid = 3332 if False else None
L = H[H['skater_id'].isin(pd.read_parquet('voorspel2_uitkomst.parquet', columns=['skater_id', 'skater_name']).query('skater_name.str.contains("Jutta LEERDAM")', engine='python')['skater_id'].unique())]
print('Leerdam: eigen hooglandafwijking vooraf (laatste ritten, %):', (100 * dif[L.index]).round(2).tail(6).tolist())
