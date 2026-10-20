import sys, os, time, importlib, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/schaatsen/voorspel')
import numpy as np, pandas as pd
kf = importlib.import_module(os.environ.get('KALMAN', 'kalman'))
par = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
df = pd.read_parquet('ritten_cat.parquet'); df, inv = kf.voorbereid(df)
for _ in range(2):
    t = time.time(); o = kf.pass_(inv, par); print('pass', round(time.time() - t, 1), 's', flush=True)
sen = df['afstand_idx'].to_numpy() >= 2; ok = sen & np.isfinite(o['e']) & ~o['val']; jr = df['jaar'].to_numpy()
for nm, s in (('fit 2012-2019', ok & (jr >= 2012) & (jr <= 2019)), ('toets 2020+', ok & (jr >= 2020))):
    e = np.abs(o['e'])
    print(nm, 'mediaan |e| alle %.3f%%  top6 %.3f%%  top3 %.3f%%  som %.4f' % (100 * np.median(e[s]), 100 * np.median(e[s & (o['zh'] < 0.06)]), 100 * np.median(e[s & (o['zh'] < 0.03)]),
          100 * (np.median(e[s]) + np.median(e[s & (o['zh'] < 0.06)]) + np.median(e[s & (o['zh'] < 0.03)]))))
