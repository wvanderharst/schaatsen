"""Per race: verrassing t.o.v. de verwachting VOOR het toernooi (x_prior, fase A) en het veld (theta), plus de echte racedatum."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, do_analysis as da
m = da.laad_volledig('model_full_tuned.pkl', extra=False)
o = m.obs[~m.obs['verdacht']].copy()
tl = da._tl_vector(m.t_lims, o['gender'].to_numpy(), o['afstand_idx'].to_numpy())
dl = np.exp(o['x_prior'].to_numpy()); T = tl + dl
sd = np.sqrt(o['v_prior'].to_numpy()) * dl / T                         # eigen sd in ln-tijd (zonder ijs)
o['u'] = (np.log(o['tijd'].to_numpy()) - o['theta'].to_numpy() - np.log(T)) / sd
o['niv'] = da._niv_pre(o)
o['thdev'] = (o['theta'] - o['mu_baan']) * 100
o = o[['event_id', 'dag', 'skater_id', 'gender', 'afstand', 'tijd', 'u', 'niv', 'n_prior', 'groep', 'thdev']]
n = pd.read_csv('/home/claude/data_nieuw/speedskating_historie_racedatum.csv', low_memory=False,
                usecols=['Event_ID', 'Atleet_ID', 'Afstand', 'Tijd_Sec', 'Race_Datum'])
n['rdag'] = (pd.to_datetime(n.Race_Datum, format='%d/%m/%Y', errors='coerce') - pd.Timestamp('1970-01-01')).dt.days
n = n.dropna(subset=['rdag', 'Atleet_ID']).rename(columns={'Event_ID': 'event_id', 'Atleet_ID': 'skater_id', 'Afstand': 'afstand', 'Tijd_Sec': 'tijd'})
n['skater_id'] = n['skater_id'].astype(np.int64); n['tijd'] = n['tijd'].round(2)
n = n.drop_duplicates(['event_id', 'skater_id', 'afstand', 'tijd'])
o['tijd'] = o['tijd'].round(2)
o = o.merge(n[['event_id', 'skater_id', 'afstand', 'tijd', 'rdag']], on=['event_id', 'skater_id', 'afstand', 'tijd'], how='left')
print('gekoppeld:', round(o.rdag.notna().mean(), 4), flush=True)
o['rdag'] = o['rdag'].fillna(o['dag']).astype(int)
o.to_pickle('vv_rijen.pkl'); print(len(o))
