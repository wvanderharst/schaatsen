"""Proef: verklaart de luchtdruk op de wedstrijddag de fout in het dagpeil vooraf?

Per race: e = gemeten dagpeil (theta, achteraf uit het veld) - verwacht dagpeil vooraf (mediaan th_hat van de rijders), in ln tijd.
Luchtdruk: dp = ln(druk op de dag) - ln(gewone druk op die baan) (gemiddelde over alle wedstrijddagen van de baan; de hoogte zit al
in de baanbasis). Natuurkunde: luchtweerstand ~ dichtheid ~ druk / temperatuur; als ~75% van het vermogen naar de luchtweerstand gaat,
is de tijd ~ dichtheid^(0,25), dus een coefficient van ~0,25 op dp.
Gebruik: python luchtdruk_lab.py <werkmap> <weermap>"""
import os, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq


def races(w):
    V = pq.read_table(os.path.join(w, 'voorspel2_uitkomst.parquet'),
                      columns=['rk_', 'baan', 'dag', 'jaar', 'gender', 'afstand', 'theta', 'th_hat', 'k_c', 'ok']).to_pandas()
    V = V[V['ok'] & np.isfinite(V['theta']) & np.isfinite(V['th_hat'])]
    for c in ('baan', 'afstand', 'gender'):
        V[c] = V[c].astype(str)
    return V.groupby('rk_').agg(baan=('baan', 'first'), dag=('dag', 'first'), jaar=('jaar', 'first'), gender=('gender', 'first'),
                                afstand=('afstand', 'first'), theta=('theta', 'first'), th_hat=('th_hat', 'median'),
                                n=('theta', 'size'), veld=('k_c', 'median')).reset_index()


def main(w, wm):
    R = races(w)
    W = pd.read_csv(os.path.join(wm, 'weer_dag.csv')).drop_duplicates(['baan', 'dag'])
    W['ln_p'] = np.log(W['druk'])
    W['dp'] = W['ln_p'] - W.groupby('baan')['ln_p'].transform('mean')              # t.o.v. de gewone druk op die baan
    W['dT'] = W['temp'] - W.groupby('baan')['temp'].transform('mean')
    R = R.merge(W, on=['baan', 'dag'], how='inner')
    R['e'] = R['theta'] - R['th_hat']
    R = R[np.isfinite(R['e']) & (R['e'].abs() < 0.05) & (R['n'] >= 5)]
    print(len(R), 'races met weer;', R['baan'].nunique(), 'banen; sd dp', round(R['dp'].std() * 100, 2), '%')
    # binnen (weinig temperatuurvariatie in de hal is onbekend): banen met kleine spreiding van de fout gelden als binnenbanen
    for nm, s in (('alle', R['jaar'] > 0), ('fit t/m 2019', R['jaar'] <= 2019), ('toets 2020+', R['jaar'] >= 2020), ('elite (veld < 5%)', R['veld'] < 0.05)):
        x = R[s]; X = np.c_[np.ones(len(x)), x['dp']]; wts = np.sqrt(x['n'].clip(upper=30))
        c, *_ = np.linalg.lstsq(X * wts.to_numpy()[:, None], x['e'].to_numpy() * wts, rcond=None)
        r = x['e'] - X @ c; sd0 = np.sqrt(np.average((x['e'] - np.average(x['e'], weights=wts)) ** 2, weights=wts)); sd1 = np.sqrt(np.average(r ** 2, weights=wts))
        print(f'{nm:18s} n {len(x):6d} | coef dp {c[1]:+.3f} | sd fout {100 * sd0:.3f}% -> {100 * sd1:.3f}%')
    # per afstand (langere afstanden: meer tijd in de lucht, kleiner aandeel start)
    print(R.groupby('afstand').apply(lambda x: pd.Series({'n': len(x), 'coef': np.polyfit(x['dp'], x['e'], 1)[0]})).round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
