"""Kalibratie van de kans per rit bij de profs (niveau vooraf >= 90; ook elite >= 97), jaren 2020+ (buiten de fit).
Per kant (snel / traag, valpartijen niet): hoe vaak een kans <= 1/1000 en <= 1/100 voorkomt t.o.v. verwacht, en per rijder hoeveel
rijders 3, 4, 5 of 6+ ritten met kans <= 1/50 hebben t.o.v. een binomiale verdeling met dezelfde aantallen ritten (plus dispersie).
Profs: niveau vooraf >= 90 op 500 m en langer, geen jeugdcategorie (alleen_senior=True). Gebruik: python kalibratie_staart.py [z.pkl] [model.pkl]  (z: normaal-equivalente dag-z per obs-rij; kans = Phi(z) snel, Phi(-z) traag)."""
import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd
from scipy.stats import norm, binom
import do_analysis as da


PROF_AFSTANDEN = ('500m', '1000m', '1500m', '3000m', '5000m', '10000m')     # 100 m en 300 m zijn jeugdafstanden


def toets(z, m, vanaf=2020, label='', alleen_senior=True):
    """alleen_senior: profs = niveau vooraf >= 90 op 500 m en langer en geen jeugdcategorie (de beste kinderen op hun eerste 100 m tellen niet)."""
    o = m.obs.loc[z.index]
    niv = pd.Series(da._niv_pre(o), index=o.index)
    ok = (~o['verdacht']).to_numpy() & ~(o['z'] > m.params.dip_val_z).to_numpy() & (o['jaar'] >= vanaf).to_numpy() & np.isfinite(z.to_numpy())
    if alleen_senior:
        ok &= o['afstand'].isin(PROF_AFSTANDEN).to_numpy() & ~o['groep'].isin(da.JEUGD_GROEPEN).to_numpy()
    goed = True; regels = []
    for nm, grens in (('profs (>=90)', 90.0), ('elite (>=97)', 97.0)):
        s = ok & (niv.to_numpy() >= grens)
        zz = z.to_numpy()[s]; sk = o['skater_id'].to_numpy()[s]; n = len(zz)
        for kant, p in (('snel', norm.cdf(zz)), ('traag', norm.cdf(-zz))):
            for drempel in (1e-3, 1e-2):
                O = int((p <= drempel).sum()); E = n * drempel; se = np.sqrt(E)
                v = abs(O - E) <= 2 * se; goed &= v
                regels.append(f"{label}{nm:13s} {kant:5s} kans<=1/{int(1 / drempel):<5d}: {O:5d} ritten, verwacht {E:7.1f} ± {se:5.1f}  ({O / E:4.2f}x) {'ok' if v else 'FOUT'}")
            k = pd.Series(p <= 0.02).groupby(sk).agg(['sum', 'size'])
            ni = k['size'].to_numpy(); ki = k['sum'].to_numpy()
            for kk in (3, 4, 5, 6):
                E = float(binom.sf(kk - 1, ni, 0.02).sum()); O = int((ki >= kk).sum()); se = np.sqrt(max(E, 1e-9))
                v = abs(O - E) <= 2 * se + 1; goed &= v
                regels.append(f"{label}{nm:13s} {kant:5s} rijders met >={kk} ritten kans<=1/50: {O:4d}, verwacht {E:6.1f} ± {se:4.1f}  {'ok' if v else 'FOUT'}")
            lam = ni * 0.02; disp = float(((ki - lam) ** 2 / (lam * 0.98)).sum() / max(len(ki) - 1, 1))
            regels.append(f"{label}{nm:13s} {kant:5s} dispersie per rijder (1 = geen extra verschil tussen rijders): {disp:.2f}  ({len(ki)} rijders, {n} ritten)")
    return goed, regels


if __name__ == '__main__':
    zp = sys.argv[1] if len(sys.argv) > 1 else 'normaal_z_dag.pkl'
    mp = sys.argv[2] if len(sys.argv) > 2 else 'model_full_tuned.pkl'
    m = da.laad_volledig(mp, extra=False); z = pd.read_pickle(zp).astype(float)
    goed, regels = toets(z, m)
    print('\n'.join(regels)); print('ALLES BINNEN DE MARGE' if goed else 'NIET GOED')
