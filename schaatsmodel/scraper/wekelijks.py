"""Wekelijkse aanvulling van de brondata met nieuwe wedstrijden van SpeedskatingResults.com.

Wat het doet (python wekelijks.py <datamap>, met de datamap van de repository, standaard 'data'):
  1. Nieuwe wedstrijden: vanaf het hoogste bekende Event_ID oplopend het toernooi-overzicht (index.php?p=2&e=<id>) laden. Een toernooi
     telt als 'met uitslagen' als het overzicht links naar races heeft (p=3&e=<id>&r=...); dan worden alle ritten opgehaald met
     scraper_schaatsen.scrape_event (met de racedag per afstand). Stoppen na STOP opeenvolgende nummers zonder uitslagen.
  2. Opnieuw ophalen: toernooien die eerder nog geen uitslagen hadden ('open', tot 26 weken) en toernooien van de laatste
     HERHAAL_DAGEN dagen (late of verbeterde uitslagen). Een opnieuw opgehaald toernooi vervangt de oude rijen.
  3. Opslaan: <datamap>/speedskating_nieuw.parquet (zelfde kolommen als de historie plus Race_Datum), <datamap>/scraper_stand.json
     (hoogste nummer met uitslagen, open nummers) en een regel in <datamap>/scraper_log.md.
De historie (speedskating_historie_1_tot_*_part*.parquet) blijft ongewijzigd; voorspel/nieuwe_data.py leest het nieuwe bestand mee
(nieuwere versie van een toernooi gaat voor). Beleefd: een paar gelijktijdige verzoeken en een korte pauze.
Testen zonder netwerk: python wekelijks.py <datamap> --droog (toont alleen wat er zou gebeuren)."""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scraper_schaatsen as ss

STOP = 40                    # opeenvolgende nummers zonder uitslagen voordat we stoppen met zoeken
HERHAAL_DAGEN = 30           # toernooien die in de laatste 30 dagen begonnen opnieuw ophalen
OPEN_WEKEN = 26              # nummers zonder uitslagen zo lang blijven controleren
WERKERS = 4
KOL = ['Event_ID', 'Toernooi', 'Datum', 'Race_Datum', 'Baan', 'Atleet_ID', 'Naam', 'Categorie', 'Land', 'Geslacht', 'Afstand', 'Tijd_Raw', 'Tijd_Sec']


def heeft_uitslagen(session, eid):
    """True als het toernooi-overzicht links naar races heeft, False als de pagina er is maar (nog) leeg, None als hij niet laadt."""
    html = ss.haal(session, f"{ss.BASIS}index.php?p=2&e={eid}")
    if html is None:
        return None
    return re.search(rf'p=3(?:&amp;|&)e={eid}(?:&amp;|&)r=\d+', html) is not None


def naar_tabel(rijen):
    if not rijen:
        return pd.DataFrame(columns=KOL)
    d = pd.DataFrame(rijen)
    d['Datum'] = pd.to_datetime(d['Datum'], format='%d/%m/%Y', errors='coerce')
    d['Race_Datum'] = pd.to_datetime(d['Race_Datum'], format='%d/%m/%Y', errors='coerce').fillna(d['Datum'])
    d['Atleet_ID'] = pd.to_numeric(d['Atleet_ID'], errors='coerce').astype('Int64')
    d['Tijd_Sec'] = pd.to_numeric(d['Tijd_Sec'], errors='coerce')
    d['Event_ID'] = d['Event_ID'].astype('int64')
    d['Toernooi'] = 'SpeedskatingResults'            # zoals in de historie
    return d[KOL].dropna(subset=['Atleet_ID', 'Tijd_Sec'])


def scrape_lijst(ids):
    uit = []
    with ThreadPoolExecutor(max_workers=WERKERS) as ex:
        for eid, rijen in ex.map(lambda e: (time.sleep(0.5), ss.scrape_event(e))[1], ids):
            uit.extend(rijen)
    return naar_tabel(uit)


def main(map_, droog=False):
    pad_nieuw = os.path.join(map_, 'speedskating_nieuw.parquet')
    pad_stand = os.path.join(map_, 'scraper_stand.json')
    hist = [os.path.join(map_, f) for f in sorted(os.listdir(map_)) if f.startswith('speedskating_historie_1_tot_') and f.endswith('.parquet')]
    hoogste_hist = max(int(pd.read_parquet(f, columns=['Event_ID'])['Event_ID'].max()) for f in hist)
    nieuw = pd.read_parquet(pad_nieuw) if os.path.exists(pad_nieuw) else pd.DataFrame(columns=KOL)
    stand = json.load(open(pad_stand)) if os.path.exists(pad_stand) else {'hoogste': hoogste_hist, 'open': {}}
    hoogste = max(stand['hoogste'], hoogste_hist, int(nieuw['Event_ID'].max()) if len(nieuw) else 0)
    vandaag = date.today()
    # oude open nummers laten vallen
    stand['open'] = {k: v for k, v in stand['open'].items() if date.fromisoformat(v) >= vandaag - timedelta(weeks=OPEN_WEKEN)}
    print(f'hoogste bekende toernooi {hoogste}, open {len(stand["open"])}, al aangevuld {nieuw["Event_ID"].nunique() if len(nieuw) else 0} toernooien', flush=True)
    if droog:
        recent = sorted(nieuw.loc[nieuw['Datum'] >= pd.Timestamp(vandaag - timedelta(days=HERHAAL_DAGEN)), 'Event_ID'].unique()) if len(nieuw) else []
        print(f'zou zoeken vanaf {hoogste + 1}; opnieuw ophalen: open {sorted(map(int, stand["open"]))}, recent {list(map(int, recent))}')
        return
    session = requests.Session()
    # 1. nieuwe nummers
    gevonden, leeg, mis, eid = [], [], 0, hoogste
    while mis < STOP:
        eid += 1
        u = heeft_uitslagen(session, eid)
        if u:
            gevonden.append(eid); mis = 0
        else:
            mis += 1
            if u is False:
                leeg.append(eid)
        time.sleep(0.3)
    nieuwe_hoogste = max(gevonden) if gevonden else hoogste
    for e in leeg:                                   # bestaat maar nog zonder uitslagen: de komende weken opnieuw kijken
        stand['open'].setdefault(str(e), vandaag.isoformat())
    # 2. opnieuw: open nummers die nu uitslagen hebben, en recente toernooien
    nu_vol = [int(e) for e in list(stand['open']) if int(e) not in gevonden and heeft_uitslagen(session, int(e))]
    recent = sorted(nieuw.loc[nieuw['Datum'] >= pd.Timestamp(vandaag - timedelta(days=HERHAAL_DAGEN)), 'Event_ID'].unique().tolist()) if len(nieuw) else []
    te_halen = sorted(set(gevonden) | set(nu_vol) | set(recent))
    print(f'nieuw met uitslagen: {len(gevonden)}; open nu gevuld: {len(nu_vol)}; recent opnieuw: {len(recent)}; totaal op te halen: {len(te_halen)}', flush=True)
    T = scrape_lijst(te_halen)
    for e in T['Event_ID'].unique():
        stand['open'].pop(str(int(e)), None)
    # 3. samenvoegen: opgehaalde toernooien vervangen de oude rijen
    if len(T):
        nieuw = pd.concat([nieuw[~nieuw['Event_ID'].isin(T['Event_ID'].unique())], T], ignore_index=True)
        nieuw = nieuw.drop_duplicates().sort_values(['Event_ID', 'Race_Datum', 'Geslacht', 'Afstand', 'Tijd_Sec']).reset_index(drop=True)
        nieuw.to_parquet(pad_nieuw, index=False)
    stand['hoogste'] = max(nieuwe_hoogste, stand['hoogste'])
    stand['bijgewerkt'] = datetime.now().isoformat(timespec='seconds')
    json.dump(stand, open(pad_stand, 'w'), indent=1, sort_keys=True)
    nieuwe_toernooien = T[T['Event_ID'].isin(gevonden)] if len(T) else T
    regel = (f"| {vandaag.isoformat()} | {len(set(nieuwe_toernooien['Event_ID']))} | {len(nieuwe_toernooien)} | {len(set(T['Event_ID'])) - len(set(nieuwe_toernooien['Event_ID']))} |"
             f" {stand['hoogste']} | {len(stand['open'])} | "
             + (', '.join(f"{b} {d:%d-%m}" for (b, d) in nieuwe_toernooien.groupby('Event_ID')[['Baan', 'Datum']].first().itertuples(index=False))[:600] if len(nieuwe_toernooien) else '') + ' |\n')
    log = os.path.join(map_, 'scraper_log.md')
    if not os.path.exists(log):
        open(log, 'w').write('# Wekelijkse aanvulling van SpeedskatingResults\n\n| datum | nieuwe toernooien | nieuwe ritten | opnieuw opgehaald | hoogste nummer | open | banen |\n|---|---|---|---|---|---|---|\n')
    open(log, 'a').write(regel)
    print(regel, flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'data', droog='--droog' in sys.argv)
