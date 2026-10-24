"""Achteraf: extra lopende correctie per baan voor de toppers op de residuen van de huidige voorspelling. Leert per wedstrijddag (EW met
gewicht a per dag, onafhankelijk van de kalendertijd) of per kalendertijd (halfwaardetijd). Alleen eerdere dagen. Kiezen op 2014-2019."""
import numpy as np, pandas as pd
V = pd.read_parquet('voorspel2_uitkomst.parquet', columns=['dag', 'jaar', 'baan', 'tijd', 'pred', 'pit', 'k_c', 'ok', 'k_val'])
V = V[V['ok'] & np.isfinite(V['pred'])].reset_index(drop=True)
r = np.clip((np.log(V['tijd']) - V['pred']).to_numpy(), -0.03, 0.03); g = V['k_c'].clip(0, 0.4).fillna(0.4).to_numpy()
w = np.exp(-g / 0.04) * (~V['k_val']).to_numpy()
BANEN = ['Salt Lake City (USA)', 'Calgary (CAN)', 'Heerenveen (NED)', 'Inzell (GER)', 'Hamar (NOR)', 'Milano (ITA)', 'Tomaszow Mazowiecki (POL)', 'Stavanger (NOR)']
def corr(mode, par, k0):
    out = np.zeros(len(V))
    for b, idx in V.groupby('baan').indices.items():
        d = pd.DataFrame({'dag': V['dag'].to_numpy()[idx], 'sw': w[idx] * r[idx], 'w': w[idx]}).groupby('dag')[['sw', 'w']].sum()
        dagen = d.index.to_numpy(); S = d.to_numpy(); acc = np.zeros(2); prev = None; st = np.zeros((len(dagen), 2))
        for i, dg in enumerate(dagen):
            if prev is not None:
                acc *= (0.5 ** ((dg - prev) / par)) if mode == 'tijd' else (1 - par)
            st[i] = acc; acc = acc + S[i]; prev = dg
        pos = np.searchsorted(dagen, V['dag'].to_numpy()[idx]); a_ = st[pos]
        out[idx] = a_[:, 0] / (a_[:, 1] + k0)
    return np.exp(-g / 0.04) * out
def bias(rr, jaren):
    s = (g < 0.04) & V['jaar'].between(*jaren).to_numpy() & V['baan'].isin(BANEN).to_numpy()
    m = pd.Series(100 * rr[s]).groupby(V['baan'].to_numpy()[s]).agg(['median', 'size'])
    return float(np.sqrt(np.average(m['median'] ** 2, weights=np.sqrt(m['size'])))), m['median']
b0, _ = bias(r, (2014, 2019)); t0, m0 = bias(r, (2020, 2026))
print(f'huidig: fit {b0:.3f} toets {t0:.3f} | SLC {m0.get("Salt Lake City (USA)"):.2f} Calgary {m0.get("Calgary (CAN)"):.2f} Heerenveen {m0.get("Heerenveen (NED)"):.2f}')
res = []
for mode, pars in (('tijd', (30, 90, 180)), ('dag', (0.05, 0.1, 0.2, 0.35))):
    for p in pars:
        for k0 in (1.0, 5.0, 20.0):
            rr = r - corr(mode, p, k0); bf, _ = bias(rr, (2014, 2019)); bt, m = bias(rr, (2020, 2026))
            res.append((bf, bt, mode, p, k0, m))
res.sort(key=lambda x: x[0])
for x in res[:6]:
    m = x[5]; print(f'{x[2]} {x[3]} k0 {x[4]}: fit {x[0]:.3f} toets {x[1]:.3f} | SLC {m.get("Salt Lake City (USA)"):.2f} Calgary {m.get("Calgary (CAN)"):.2f} Heerenveen {m.get("Heerenveen (NED)"):.2f} Inzell {m.get("Inzell (GER)"):.2f}')
