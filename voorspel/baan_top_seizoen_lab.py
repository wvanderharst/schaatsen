"""Achteraf: correctie per baan x week van het seizoen voor de toppers (gewicht exp(-niveau/4%)), uit EERDERE seizoenen op die baan
(kernel bw weken, halfwaardetijd hl seizoenen, krimp k0), bovenop de huidige voorspelling. Kiezen op 2014-2019, tonen op 2020+."""
import numpy as np, pandas as pd
V = pd.read_parquet('voorspel2_uitkomst.parquet', columns=['dag', 'jaar', 'seizoen', 'baan', 'afstand', 'tijd', 'pred', 'k_c', 'ok', 'k_val', 'pit'])
V = V[V['ok'] & np.isfinite(V['pred'])].reset_index(drop=True)
r = (np.log(V['tijd']) - V['pred']).clip(-0.03, 0.03).to_numpy(); g = V['k_c'].clip(0, 0.4).fillna(0.4).to_numpy()
V['wk'] = ((V['dag'] - ((V['seizoen'] - 1970) * 365.25 + 181)) // 7).clip(0, 51).astype(int)
seiz = np.sort(V['seizoen'].unique()); W = np.arange(52)
def corr(bw, hl, k0, T=0.04, rv=None):
    rv = r if rv is None else rv
    w = np.exp(-g / T) * (~V['k_val']).to_numpy(); Kw = np.exp(-(W[:, None] - W[None, :]) ** 2 / (2 * bw * bw))
    out = np.zeros(len(V))
    for b, idx in V.groupby('baan').indices.items():
        if len(idx) < 200: continue
        si = np.searchsorted(seiz, V['seizoen'].to_numpy()[idx]); wk = V['wk'].to_numpy()[idx]
        S_ = np.zeros((len(seiz), 52)); C_ = np.zeros((len(seiz), 52))
        np.add.at(S_, (si, wk), w[idx] * rv[idx]); np.add.at(C_, (si, wk), w[idx])
        for j in np.unique(si):
            if j == 0: continue
            a = 0.5 ** ((seiz[j] - 1 - seiz[:j]) / hl)
            sw = (a[:, None] * S_[:j]).sum(0) @ Kw.T; cw = (a[:, None] * C_[:j]).sum(0) @ Kw.T
            m = idx[si == j]; out[m] = (sw / (cw + k0))[V['wk'].to_numpy()[m]]
    return np.exp(-g / T) * out
def bias(rr, jaren, top=0.04):
    s = (g < top) & V['jaar'].between(*jaren).to_numpy()
    d = pd.DataFrame({'b': V['baan'].to_numpy()[s], 'm': pd.to_datetime(V['dag'].to_numpy()[s], unit='D').month, 'r': 100 * rr[s]})
    pm = d.groupby(['b', 'm'])['r'].agg(['median', 'size']); pm = pm[pm['size'] >= 300]
    return float(np.sqrt(np.average(pm['median'] ** 2, weights=pm['size']))), d
b0, _ = bias(r, (2014, 2019)); t0, d0 = bias(r, (2020, 2026))
print(f'huidig: rms baan x maand fit {b0:.3f} toets {t0:.3f}')
best = []
for bw in (2.0, 4.0):
    for hl in (2.0, 4.0):
        for k0 in (5.0, 20.0, 60.0):
            c = corr(bw, hl, k0); rr = r - c
            bf, _ = bias(rr, (2014, 2019)); bt, d = bias(rr, (2020, 2026))
            best.append((bf, bt, bw, hl, k0))
best.sort()
for x in best[:4]: print('bw %.0f hl %.0f k0 %.0f: fit %.3f toets %.3f' % (x[2], x[3], x[4], x[0], x[1]))
bw, hl, k0 = best[0][2:]
rr = r - corr(bw, hl, k0); _, d = bias(rr, (2020, 2026))
for b in ('Heerenveen (NED)', 'Salt Lake City (USA)', 'Calgary (CAN)'):
    print(b, 'voor', d0[d0.b == b].groupby('m').r.median().reindex([10, 11, 12, 1, 2, 3]).round(2).to_dict())
    print(b, 'na  ', d[d.b == b].groupby('m').r.median().reindex([10, 11, 12, 1, 2, 3]).round(2).to_dict())
print('--- herhaald toepassen (elke ronde op de restfout van de vorige)')
rr = r.copy()
for ronde in range(1, 5):
    c = corr(bw, hl, k0, rv=np.clip(rr, -0.03, 0.03))
    rr = rr - c
    bf, _ = bias(rr, (2014, 2019)); bt, d = bias(rr, (2020, 2026))
    print(f'ronde {ronde}: fit {bf:.3f} toets {bt:.3f} | Heerenveen', d[d.b == 'Heerenveen (NED)'].groupby('m').r.median().reindex([10, 11, 12, 1, 2, 3]).round(2).to_dict(),
          '| SLC', d[d.b == 'Salt Lake City (USA)'].groupby('m').r.median().reindex([11, 12, 2]).round(2).to_dict())
