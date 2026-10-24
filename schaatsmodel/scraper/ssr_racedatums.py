"""Racedatum per afstand voor SpeedskatingResults.com-toernooien.

Probleem: een toernooi van meerdere dagen krijgt in de data één datum (de startdatum), terwijl de afstanden over
meerdere dagen verspreid liggen (Spelen 2026 vrouwen: 3000 m 7 feb, 1000 m 9 feb, 5000 m 12 feb, 500 m 15 feb,
1500 m 20 feb). Het overzicht van een toernooi (index.php?p=2&e=<event>) zet elke afstand onder een datumkop:

    <h4>7 February 2026</h4>   ... <a href="index.php?p=3&e=31588&r=1">3000m</a>
    <h4>9 February 2026</h4>   ... <a href="index.php?p=3&e=31588&r=3">1000m</a>

Deze module leest die pagina één keer per toernooi en geeft per race (r) de datum, het geslacht en de afstand.

Gebruik in de scraper (bij het ophalen van een race p=3&e=<event>&r=<race>):

    kaart = race_datums(event_id)                      # één keer per toernooi
    info = kaart.get(race_id)
    rij['Race_ID'] = race_id
    rij['Race_Datum'] = info['datum'].strftime('%d/%m/%Y') if info and info['datum'] else rij['Datum']

Bestaande CSV bijwerken (zonder alles opnieuw te scrapen): zie backfill.py.
"""
import re
import time
from datetime import date
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

BASIS = 'https://speedskatingresults.com/'
MAANDEN = {m: i + 1 for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                           'september', 'october', 'november', 'december'])}
MAANDEN.update({m: i + 1 for i, m in enumerate(['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus',
                                                'september', 'oktober', 'november', 'december'])})
MAANDEN.update({k[:3]: v for k, v in list(MAANDEN.items())})
_RE_TEKST = re.compile(r'\b(\d{1,2})\s+([A-Za-z]{3,10})\.?\s+(\d{4})\b')          # 7 February 2026 / 7 feb 2026
_RE_ISO = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')                                 # 2026-02-07
_RE_NUM = re.compile(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b')                     # 07-02-2026 / 07/02/2026 (dag eerst)
_VROUW = re.compile(r'\b(women|ladies|vrouwen|dames|female|girls|meisjes)\b', re.I)
_MAN = re.compile(r'\b(men|gentlemen|mannen|heren|male|boys|jongens)\b', re.I)
_AFSTAND = re.compile(r'^\s*(\d{3,5})\s*m\b', re.I)


def lees_datum(tekst):
    """Datum uit een kop als '7 February 2026', '2026-02-07' of '07-02-2026'; None als er geen datum in staat."""
    t = tekst.strip()
    m = _RE_ISO.search(t)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = _RE_TEKST.search(t)
    if m and m[2].lower() in MAANDEN:
        return date(int(m[3]), MAANDEN[m[2].lower()], int(m[1]))
    m = _RE_NUM.search(t)
    if m:
        return date(int(m[3]), int(m[2]), int(m[1]))
    return None


def _race_id(href):
    q = parse_qs(urlparse(urljoin(BASIS, href)).query)
    if q.get('p', [''])[0] == '3' and 'r' in q:
        try:
            return int(q['r'][0])
        except ValueError:
            return None
    return None


def parse_toernooi(html):
    """Lijst van races in paginavolgorde: dict(race_id, geslacht ('Man'/'Vrouw'/None), label, afstand ('500m' of None), datum).
    Loopt de pagina in documentvolgorde af: de laatst geziene datumkop en geslachtskop gelden voor de volgende racelinks."""
    soup = BeautifulSoup(html, 'html.parser')
    datum, geslacht, uit, gezien = None, None, [], set()
    for el in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'th', 'caption', 'strong', 'b', 'a']):
        if el.name != 'a':
            tekst = el.get_text(' ', strip=True)
            if not tekst or len(tekst) > 80:
                continue
            d = lees_datum(tekst)
            if d:
                datum = d
                continue
            if _VROUW.search(tekst):
                geslacht = 'Vrouw'
            elif _MAN.search(tekst):
                geslacht = 'Man'
            continue
        r = _race_id(el.get('href', ''))
        if r is None or r in gezien:
            continue
        gezien.add(r)
        label = el.get_text(' ', strip=True)
        m = _AFSTAND.match(label)
        uit.append(dict(race_id=r, geslacht=geslacht, label=label, afstand=(m[1] + 'm') if m else None, datum=datum))
    return uit


_SESSIE = None


def race_datums(event_id, sessie=None, cache_dir=None, pauze=1.0):
    """{race_id: info} voor een toernooi. Met cache_dir wordt de HTML bewaard (handig bij de backfill van duizenden toernooien)."""
    import os
    global _SESSIE
    html = None
    pad = os.path.join(cache_dir, f'{event_id}.html') if cache_dir else None
    if pad and os.path.exists(pad):
        with open(pad, encoding='utf-8') as f:
            html = f.read()
    if html is None:
        sessie = sessie or _SESSIE or requests.Session()
        _SESSIE = sessie
        r = sessie.get(urljoin(BASIS, f'index.php?p=2&e={int(event_id)}'), timeout=30,
                       headers={'User-Agent': 'schaatsmodel/1.0 (persoonlijk onderzoek)'})
        r.raise_for_status()
        html = r.text
        if pad:
            os.makedirs(cache_dir, exist_ok=True)
            with open(pad, 'w', encoding='utf-8') as f:
                f.write(html)
        time.sleep(pauze)                    # netjes blijven tegen de server
    return {x['race_id']: x for x in parse_toernooi(html)}
