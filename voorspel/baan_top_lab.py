"""Achteraf: extra lopende correctie per baan op de residuen van voorspel2_uitkomst.parquet (alleen eerdere dagen), rooster over
halfwaardetijd, krimp en de breedte van de niveauweging; kiezen op 2014-2019 (resterende baanbias van de toppers), tonen op 2020+."""
import sys; sys.path.insert(0, '/home/user/schaatsen/voorspel')
import numpy as np, pandas as pd
from model2 import online_baan
V = pd.read_parquet('voorspel2_uitkomst.parquet', columns=['dag', 'jaar', 'baan', 'tijd', 'pred', 'k_c', 'ok', 'k_val'])
V = V[V['ok'] & np.isfinite(V['pred'])].reset_index(drop=True)
r = (np.log(V['tijd']) - V['pred']).to_numpy(); g = V['k_c'].clip(0, 0.4).fillna(0.4).to_numpy()
BANEN = ['Salt Lake City (USA)', 'Calgary (CAN)', 'Heerenveen (NED)', 'Inzell (GER)', 'Hamar (NOR)', 'Tomaszow Mazowiecki (POL)', 'Milano (ITA)']
def bias(rr, jaren):
    s = (g < 0.04) & V['jaar'].between(*jaren).to_numpy() & V['baan'].isin(BANEN).to_numpy()
    m = pd.Series(100 * rr[s]).groupby(V['baan'].to_numpy()[s]).median()
    return m, float(np.sqrt((m ** 2).mean()))
m0, b0 = bias(r, (2014, 2019)); t0, bt0 = bias(r, (2020, 2026))
print(f'huidig: fit rms {b0:.3f}  toets rms {bt0:.3f} | toets', t0.round(2).to_dict())
res = []
for T in (0.04, 0.08, 0.15):
    w = np.exp(-g / T)
    for HL in (90, 180, 365):
        for K in (2.0, 10.0, 30.0):
            corr = np.zeros(len(V))
            for b, idx in V.groupby('baan').indices.items():
                if len(idx) < 50: continue
                bb, nb = online_baan(V['dag'].to_numpy()[idx], r[idx], (~V['k_val']).to_numpy()[idx], w[idx], HL)
                corr[idx] = bb * nb / (nb + K)
            rr = r - w * corr
            _, bf_ = bias(rr, (2014, 2019)); t_, bt_ = bias(rr, (2020, 2026))
            res.append((bf_, bt_, T, HL, K, t_.round(2).to_dict()))
res.sort(key=lambda x: x[0])
for x in res[:5]:
    print(f'T {x[2]} HL {x[3]} K {x[4]}: fit rms {x[0]:.3f}  toets rms {x[1]:.3f} | toets', x[5])
