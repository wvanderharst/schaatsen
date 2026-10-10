"""Zet in een bestaande CSV de echte racedatum per afstand, zonder alles opnieuw te scrapen.

    python backfill.py speedskating_historie_clean.csv speedskating_historie_racedatum.csv --cache html_cache

Voegt twee kolommen toe: Race_Datum (dd/mm/jjjj) en Race_ID (r op SpeedskatingResults). De kolom Datum blijft de
startdatum van het toernooi, zodat bestaande code niet breekt.

Koppeling: de CSV staat per toernooi in raceprogramma-volgorde (bij de Spelen 2026: 3000, 1000, 5000, 500, 1500).
Voor elke rijder telt de k-de race op een afstand binnen een toernooi als de k-de race op die afstand (en dat geslacht)
op de toernooipagina. Zo krijgen ook 2x500 m op twee dagen elk hun eigen datum. Past het niet (andere aantallen,
geen datumkoppen), dan blijft Race_Datum gelijk aan Datum en komt het toernooi in het logbestand.

Alleen toernooien waarin rijders meer dan één race rijden worden opgehaald (de rest kan maar één datum hebben).
Hervatbaar: met --cache wordt elke toernooipagina bewaard en niet opnieuw opgehaald.
"""
import argparse
import sys

import pandas as pd

from ssr_racedatums import race_datums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('invoer'); ap.add_argument('uitvoer')
    ap.add_argument('--cache', default='html_cache')
    ap.add_argument('--pauze', type=float, default=1.0)
    ap.add_argument('--max', type=int, default=0, help='alleen de eerste N toernooien (om te testen)')
    a = ap.parse_args()

    d = pd.read_csv(a.invoer, low_memory=False)
    d['Race_Datum'] = d['Datum']; d['Race_ID'] = pd.NA
    k = d.groupby(['Event_ID', 'Atleet_ID', 'Afstand']).cumcount()          # k-de race van deze rijder op deze afstand
    meer = d.groupby('Event_ID')['Atleet_ID'].transform('size') > d.groupby('Event_ID')['Atleet_ID'].transform('nunique')
    events = d.loc[meer, 'Event_ID'].drop_duplicates().tolist()
    if a.max:
        events = events[:a.max]
    print(f'{len(events)} toernooien met meerdere races per rijder', flush=True)
    log = []
    for i, e in enumerate(events):
        try:
            races = list(race_datums(e, cache_dir=a.cache, pauze=a.pauze).values())
        except Exception as ex:                                              # netwerkfout: later opnieuw proberen
            log.append((e, f'fout: {ex}')); continue
        races = [r for r in races if r['afstand'] and r['datum']]
        if not races:
            log.append((e, 'geen datums gevonden')); continue
        if len({r['datum'] for r in races}) == 1:
            continue                                                         # eendaags toernooi
        rij = d.index[d['Event_ID'] == e]
        for (g, af), sub in d.loc[rij].groupby(['Geslacht', 'Afstand'], observed=True):
            lijst = [r for r in races if r['afstand'] == af and (r['geslacht'] in (g, None))]
            if not lijst:
                log.append((e, f'{g} {af}: niet op de pagina')); continue
            kk = k.loc[sub.index].to_numpy()
            if kk.max() >= len(lijst):
                log.append((e, f'{g} {af}: {kk.max() + 1} races in csv, {len(lijst)} op de pagina'))
                kk = kk.clip(max=len(lijst) - 1)
            d.loc[sub.index, 'Race_Datum'] = [lijst[j]['datum'].strftime('%d/%m/%Y') for j in kk]
            d.loc[sub.index, 'Race_ID'] = [lijst[j]['race_id'] for j in kk]
        if i % 200 == 0:
            print(f'{i}/{len(events)}', flush=True)
    d.to_csv(a.uitvoer, index=False)
    pd.DataFrame(log, columns=['Event_ID', 'melding']).to_csv(a.uitvoer + '.log.csv', index=False)
    verschoven = (d['Race_Datum'] != d['Datum']).mean()
    print(f'klaar: {100 * verschoven:.1f}% van de rijen kreeg een andere datum dan de toernooistart; {len(log)} meldingen', flush=True)


if __name__ == '__main__':
    sys.exit(main())
