"""Nieuwe brondata (data/speedskating_historie_1_tot_32104_part*.parquet) omzetten naar het invoerformaat van het model
(speedskating_historie_racedatum.csv: Event_ID, Toernooi, Datum, Race_Datum, Baan, Atleet_ID, Naam, Categorie, Land, Geslacht,
Afstand, Tijd_Raw, Tijd_Sec; datums als dd/mm/jjjj).

In de nieuwe data is Datum de startdag van het toernooi; de echte racedag per afstand (Race_Datum) ontbreekt. Die komt uit
data/racedatum_map.parquet (gemaakt uit de oude data met `maak_map`):
  - meerdaagse toernooien: per rit (Event_ID, Atleet_ID, Afstand, Tijd_Raw), want bij sprinttoernooien is dezelfde afstand op twee dagen;
  - daarna per Event_ID x Geslacht x Afstand als daar maar een racedag bij hoort;
  - anders de startdag (nieuwe toernooien: racedag onbekend, zie kolom Race_Datum_bron).
Gebruik:
  python nieuwe_data.py maak_map <oude speedskating_historie_racedatum.csv> data/racedatum_map.parquet
  python nieuwe_data.py zet_om data <uit.csv>"""
import os, sys, glob
import numpy as np, pandas as pd

K_RIT = ['Event_ID', 'Atleet_ID', 'Afstand', 'Tijd_Raw']


def laad_nieuw(map_):
    delen = sorted(glob.glob(os.path.join(map_, 'speedskating_historie_1_tot_*_part*.parquet')))
    N = pd.concat([pd.read_parquet(f) for f in delen], ignore_index=True)
    return N.drop_duplicates().reset_index(drop=True)                 # volledig dubbele rijen weg


def maak_map(oud_csv, uit):
    O = pd.read_csv(oud_csv, low_memory=False, usecols=K_RIT + ['Geslacht', 'Datum', 'Race_Datum'])
    O['Race_Datum'] = pd.to_datetime(O['Race_Datum'], format='%d/%m/%Y', errors='coerce')
    O['Datum'] = pd.to_datetime(O['Datum'], format='%d/%m/%Y', errors='coerce')
    meer = O.groupby('Event_ID')['Race_Datum'].transform('nunique') > 1
    rit = O[meer].drop_duplicates(K_RIT)[K_RIT + ['Race_Datum']].assign(niveau='rit')
    af = O.groupby(['Event_ID', 'Geslacht', 'Afstand'])['Race_Datum'].agg(['nunique', 'first']).reset_index()
    af = af[af['nunique'] == 1].rename(columns={'first': 'Race_Datum'}).drop(columns='nunique').assign(niveau='afstand')
    M = pd.concat([rit, af], ignore_index=True)
    M['Atleet_ID'] = M['Atleet_ID'].astype('Int64')
    M.to_parquet(uit, index=False)
    print(len(rit), 'ritten (meerdaagse toernooien) +', len(af), 'afstanden ->', uit)


def racedatum(N, M):
    """Race_Datum per rij van de nieuwe data + de bron ('rit', 'afstand' of 'startdag')."""
    rd = pd.Series(pd.NaT, index=N.index, dtype='datetime64[ns]'); bron = pd.Series('startdag', index=N.index)
    rit = M[M['niveau'] == 'rit'].drop(columns='niveau').drop_duplicates(K_RIT)
    rit['Atleet_ID'] = rit['Atleet_ID'].astype('int64')
    x = N[K_RIT].merge(rit, on=K_RIT, how='left')['Race_Datum'].to_numpy()
    ok = ~pd.isna(x); rd[ok] = x[ok]; bron[ok] = 'rit'
    af = M[M['niveau'] == 'afstand'][['Event_ID', 'Geslacht', 'Afstand', 'Race_Datum']].drop_duplicates(['Event_ID', 'Geslacht', 'Afstand'])
    y = N[['Event_ID', 'Geslacht', 'Afstand']].merge(af, on=['Event_ID', 'Geslacht', 'Afstand'], how='left')['Race_Datum'].to_numpy()
    ok2 = rd.isna().to_numpy() & ~pd.isna(y); rd[ok2] = y[ok2]; bron[ok2] = 'afstand'
    rest = rd.isna(); rd[rest] = N.loc[rest, 'Datum']
    return rd, bron


def zet_om(map_, uit):
    N = laad_nieuw(map_); M = pd.read_parquet(os.path.join(map_, 'racedatum_map.parquet'))
    rd, bron = racedatum(N, M)
    U = N.copy(); U.insert(3, 'Race_Datum', rd.dt.strftime('%d/%m/%Y')); U['Datum'] = U['Datum'].dt.strftime('%d/%m/%Y')
    U.to_csv(uit, index=False)
    print(len(U), 'rijen ->', uit); print('racedag uit:', bron.value_counts().to_dict())
    print('toernooien met alleen de startdag:', N.loc[bron == 'startdag', 'Event_ID'].nunique())


if __name__ == '__main__':
    a = sys.argv
    {'maak_map': lambda: maak_map(a[2], a[3]), 'zet_om': lambda: zet_om(a[2], a[3])}[a[1]]()
