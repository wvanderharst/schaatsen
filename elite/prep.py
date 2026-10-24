import sys; sys.path.insert(0, '/home/user/schaatsen/schaatsmodel')
import warnings; warnings.filterwarnings('ignore')
import do_analysis as da, pandas as pd
df = da.laad_dataset('speedskating_historie_racedatum.csv', verbose=True)
df = df.drop_duplicates(['skater_id', 'dag', 'afstand', 'tijd'])     # zelfde rit onder twee event-id's
df[['skater_id', 'skater_name', 'gender', 'land', 'groep', 'afstand', 'afstand_idx', 'dag', 'jaar', 'seizoen', 'event_id', 'baan', 'tijd', 'verdacht']].to_parquet('ritten.parquet')
print(len(df))
