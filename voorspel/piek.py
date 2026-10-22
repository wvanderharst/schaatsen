"""Piekmomenten: alleen de Olympische Spelen en de WK's (senioren: afstanden, allround, sprint). De brondata heeft geen toernooinamen
(overal 'SpeedskatingResults'), dus een kalender (jaar, baan, soort) plus herkenning in de data: een internationaal veld (>= 10 landen,
>= 2 van USA/CAN/JPN/CHN/KOR, zodat een EK niet meetelt) met het afstandspatroon van het soort toernooi:
  sprint: alleen 500 m en 1000 m;  allround: 500 / 1500 / 3000 / 5000 / 10000 m, geen 1000 m;
  afstanden en Spelen: alle zes afstanden (allround en sprint samen in een toernooi, zoals Hamar 2022, ook).
Bij meerdere kandidaten het toernooi met de meeste landen.
Gebruik: python piek.py <werkmap>  (drukt de herkende toernooien af)"""
import os, sys
import numpy as np, pandas as pd

OS = [(2010, 'Richmond'), (2014, 'Sochi'), (2018, 'Gangneung'), (2022, 'Beijing-National'), (2026, 'Milano')]
WK_AFSTANDEN = [(2007, 'Salt Lake City'), (2008, 'Nagano'), (2009, 'Richmond'), (2011, 'Inzell'), (2012, 'Heerenveen'), (2013, 'Sochi'),
                (2015, 'Heerenveen'), (2016, 'Kolomna'), (2017, 'Gangneung'), (2019, 'Inzell'), (2020, 'Salt Lake City'), (2021, 'Heerenveen'),
                (2023, 'Heerenveen'), (2024, 'Calgary'), (2025, 'Hamar')]
WK_ALLROUND = [(2007, 'Heerenveen'), (2008, 'Berlin'), (2009, 'Hamar'), (2010, 'Heerenveen'), (2011, 'Calgary'), (2012, 'Moscow'), (2013, 'Hamar'),
               (2014, 'Heerenveen'), (2015, 'Calgary'), (2016, 'Berlin'), (2017, 'Hamar'), (2018, 'Amsterdam'), (2019, 'Calgary'), (2020, 'Hamar'),
               (2022, 'Hamar'), (2024, 'Inzell'), (2026, 'Heerenveen')]
WK_SPRINT = [(2007, 'Hamar'), (2008, 'Heerenveen'), (2009, 'Moscow'), (2010, 'Obihiro'), (2011, 'Heerenveen'), (2012, 'Calgary'),
             (2013, 'Salt Lake City'), (2014, 'Nagano'), (2015, 'Astana'), (2016, 'Seoul'), (2017, 'Calgary'), (2018, 'Changchun'),
             (2019, 'Heerenveen'), (2020, 'Hamar'), (2022, 'Hamar'), (2024, 'Inzell'), (2026, 'Heerenveen')]
ZES = {'500m', '1000m', '1500m', '3000m', '5000m', '10000m'}
BUITEN_EU = {'USA', 'CAN', 'JPN', 'CHN', 'KOR'}


def toernooien(R):
    """R: ritten (skater_id, land, afstand, dag, event_id, baan). Geeft de herkende piektoernooien (eid = event_id mod 10^6)."""
    x = R[R['afstand'].astype(str).isin(ZES)].copy()
    x['eid'] = x['event_id'] % 1000000; x['datum'] = pd.to_datetime(x['dag'], unit='D'); x['baan'] = x['baan'].astype(str)
    E = x.groupby('eid').agg(baan=('baan', 'first'), van=('datum', 'min'), tot=('datum', 'max'), n=('skater_id', 'nunique'),
                             landen=('land', 'nunique'), buiten=('land', lambda s: len(set(s) & BUITEN_EU)),
                             afst=('afstand', lambda s: frozenset(map(str, s)))).reset_index()
    E = E[(E['landen'] >= 10) & (E['buiten'] >= 2) & E['van'].dt.month.isin([1, 2, 3])]
    uit = []
    for soort, lijst in (('Spelen', OS), ('WK afstanden', WK_AFSTANDEN), ('WK allround', WK_ALLROUND), ('WK sprint', WK_SPRINT)):
        for jaar, plaats in lijst:
            c = E[(E['van'].dt.year == jaar) & E['baan'].str.startswith(plaats)]
            alle = c
            if soort == 'WK sprint':
                c = c[c['afst'].apply(lambda a: a == {'500m', '1000m'})]
            elif soort == 'WK allround':
                c = c[c['afst'].apply(lambda a: '1000m' not in a and {'500m', '1500m', '5000m'} <= a)]
            else:
                c = c[c['afst'].apply(lambda a: a == ZES)]
            if not len(c) and soort in ('WK allround', 'WK sprint'):
                c = alle[alle['afst'].apply(lambda a: a == ZES)]            # allround en sprint in een toernooi (Hamar 2022)
            if len(c):
                c = c.sort_values(['landen', 'n'], ascending=False).head(1)      # het WK heeft het breedste veld
                for _, r in c.iterrows():
                    uit.append(dict(soort=soort, jaar=jaar, eid=int(r['eid']), baan=r['baan'], van=r['van'].date(), tot=r['tot'].date(), rijders=r['n'], landen=r['landen']))
            else:
                uit.append(dict(soort=soort, jaar=jaar, eid=-1, baan=plaats, van=None, tot=None, rijders=0, landen=0))
    return pd.DataFrame(uit)


def markering(event_id, eids):
    return np.isin(np.asarray(event_id) % 1000000, np.asarray(list(eids)))


if __name__ == '__main__':
    R = pd.read_parquet(os.path.join(sys.argv[1], 'ritten_cat.parquet'), columns=['skater_id', 'land', 'afstand', 'dag', 'event_id', 'baan'])
    T = toernooien(R)
    pd.set_option('display.width', 200)
    print(T.to_string(index=False))
    print('niet gevonden:', int((T['eid'] < 0).sum()))
