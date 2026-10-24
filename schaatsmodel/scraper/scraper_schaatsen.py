import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Nieuw bestand: de oude CSV heeft geen kolom Race_Datum (alles in één bestand mengen geeft een kapotte kop)
FILENAME = "speedskating_historie_racedatum.csv"
BASIS = "https://speedskatingresults.com/"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

MONTH_MAP = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
    # Nederlandse maanden (voor de zekerheid)
    'mrt': '03', 'mei': '05', 'okt': '10',
}
_MAANDEN_LANG = {'januari': 1, 'februari': 2, 'maart': 3, 'april': 4, 'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
                 'september': 9, 'oktober': 10, 'november': 11, 'december': 12}

_RE_DATUM_TEKST = re.compile(r'\b(\d{1,2})\s+([A-Za-z]{3,10})\.?\s+(\d{4})\b')    # 7 February 2026
_RE_DATUM_ISO = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')                         # 2026-02-07
_RE_DATUM_NUM = re.compile(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b')             # 07-02-2026
_RE_VROUW = re.compile(r'\b(women|ladies|vrouwen|dames|female|girls|meisjes)\b', re.I)
_RE_MAN = re.compile(r'\b(men|gentlemen|mannen|heren|male|boys|jongens)\b', re.I)
# Een echte afstandskop: '500m', '1000 m', '500m (2)'. NIET 'Team pursuit', 'Mass start', 'Team sprint' enz.
_RE_AFSTAND_KOP = re.compile(r'^\s*(\d{3,5})\s*m\b', re.I)
_RE_GEEN_INDIVIDUEEL = re.compile(r'team|pursuit|ploeg|mass|massa|sprint\s*team|relay|estafette|punten|points', re.I)


def parse_schaats_tijd(tijd_str):
    if not tijd_str:
        return None
    clean = str(tijd_str).strip()

    match_min = re.search(r'(\d+)[\.:](\d{2}),(\d{2})', clean)
    if match_min:
        m, s, ms = match_min.groups()
        return round(int(m) * 60 + int(s) + float(f"0.{ms}"), 2)

    match_sec = re.search(r'^(\d{1,2}),(\d{2})$', clean)          # ook '9,87' (100 m)
    if match_sec:
        s, ms = match_sec.groups()
        return round(int(s) + float(f"0.{ms}"), 2)

    return None


def lees_datum(tekst):
    """Datum (datetime.date) uit tekst als '7 February 2026', '2026-02-07' of '07-02-2026'; anders None."""
    t = str(tekst)
    m = _RE_DATUM_ISO.search(t)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = _RE_DATUM_TEKST.search(t)
    if m:
        w = m[2].lower()
        mnd = _MAANDEN_LANG.get(w) or (int(MONTH_MAP[w[:3]]) if w[:3] in MONTH_MAP else None)
        if mnd:
            return date(int(m[3]), mnd, int(m[1]))
    m = _RE_DATUM_NUM.search(t)
    if m and 1 <= int(m[2]) <= 12:
        return date(int(m[3]), int(m[2]), int(m[1]))
    return None


def extract_event_metadata(soup):
    datum = ""
    rink = "Onbekend"
    toernooi = "Onbekend"

    teksten = []
    for elem in soup.find_all(['h1', 'h2', 'h3', 'p']):
        teksten.extend([t.strip() for t in elem.stripped_strings if t.strip()])

    if teksten:
        toernooi = teksten[0]  # De toernooinaam is vrijwel altijd het allereerste tekstblok (h1)

    for i, text in enumerate(teksten):
        match_date = re.search(r'(\d{1,2})\s*(?:-\s*\d{1,2})?\s+([a-zA-Z]{3})[a-z]*\s+(\d{4})', text, re.IGNORECASE)
        match_num = re.search(r'(\d{2})[\.-](\d{2})[\.-](\d{4})', text)

        if match_date or match_num:
            if match_date:
                dag, maand_str, jaar = match_date.groups()
                dag_str = f"{int(dag):02d}"
                maand_nr = MONTH_MAP.get(maand_str.lower()[:3], "01")
                datum = f"{dag_str}/{maand_nr}/{jaar}"
            elif match_num:
                d, m, y = match_num.groups()
                datum = f"{d}/{m}/{y}"

            # Locatie staat meestal direct vóór de datum
            if i > 0 and teksten[i - 1] != toernooi:
                potential_rink = teksten[i - 1]
                if len(potential_rink) > 2 and not re.search(r'\d{4}', potential_rink):
                    rink = potential_rink
            break

    if rink == "Onbekend":
        for a in soup.find_all('a', href=True):
            if re.search(r'[?&]r=\d+', a['href']):
                rink = a.get_text(strip=True)
                break

    return datum, rink, toernooi


def haal(session, url, pogingen=3):
    """GET met een paar herkansingen; None als het niet lukt (in plaats van stilletjes data te verliezen)."""
    for k in range(pogingen):
        try:
            res = session.get(url, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                return res.text
            if res.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def race_datums_toernooi(html):
    """Uit het toernooi-overzicht (p=2): {geslacht: {afstand: [datum race 1, datum race 2, ...]}} in programmavolgorde.
    De pagina zet elke afstand onder een datumkop; de laatst geziene datum- en geslachtskop gelden voor de volgende link."""
    soup = BeautifulSoup(html, 'html.parser')
    datum, geslacht, gezien = None, None, set()
    uit = {'Vrouw': {}, 'Man': {}, None: {}}
    for el in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'th', 'caption', 'strong', 'b', 'a']):
        if el.name != 'a':
            tekst = el.get_text(' ', strip=True)
            if not tekst or len(tekst) > 80:
                continue
            d = lees_datum(tekst)
            if d:
                datum = d
            elif _RE_VROUW.search(tekst):
                geslacht = 'Vrouw'
            elif _RE_MAN.search(tekst):
                geslacht = 'Man'
            continue
        q = parse_qs(urlparse(el.get('href', '')).query)
        if q.get('p', [''])[0] != '3' or 'r' not in q:
            continue
        r = q['r'][0]
        if r in gezien:
            continue
        gezien.add(r)
        label = el.get_text(' ', strip=True)
        m = _RE_AFSTAND_KOP.match(label)
        if not m or _RE_GEEN_INDIVIDUEEL.search(label) or datum is None:
            continue
        uit[geslacht].setdefault(f"{m.group(1)}m", []).append(datum)
    return uit


def _datum_str(d):
    return d.strftime('%d/%m/%Y') if d else ''


def scrape_event(event_id):
    races = []
    session = requests.Session()

    # 1x per toernooi: datum per afstand uit het overzicht
    overzicht = haal(session, f"{BASIS}index.php?p=2&e={event_id}")
    datums = race_datums_toernooi(overzicht) if overzicht else {'Vrouw': {}, 'Man': {}, None: {}}

    urls = [
        (f"{BASIS}index.php?p=4&e={event_id}", 'Vrouw'),
        (f"{BASIS}index.php?p=4&e={event_id}&g=1", 'Man'),
    ]

    for url, gender_label in urls:
        html = haal(session, url)
        if html is None:
            continue
        try:
            soup = BeautifulSoup(html, 'html.parser')
            event_datum, event_rink, event_toernooi = extract_event_metadata(soup)
            current_distance = None
            race_datum = event_datum
            telling = {}                       # hoeveelste blok van deze afstand (2x500 m: blok 1 en 2)

            for elem in soup.find_all(['h2', 'table']):
                if elem.name == 'h2':
                    h2_text = elem.get_text(' ', strip=True)
                    match_dist = _RE_AFSTAND_KOP.match(h2_text)
                    if match_dist and not _RE_GEEN_INDIVIDUEEL.search(h2_text):
                        current_distance = f"{match_dist.group(1)}m"
                        k = telling.get(current_distance, 0)
                        telling[current_distance] = k + 1
                        # 1) datum in de kop zelf, 2) k-de race van deze afstand in het overzicht, 3) startdatum toernooi
                        d_kop = lees_datum(h2_text)
                        lijst = datums.get(gender_label, {}).get(current_distance) or datums.get(None, {}).get(current_distance) or []
                        if d_kop:
                            race_datum = _datum_str(d_kop)
                        elif k < len(lijst):
                            race_datum = _datum_str(lijst[k])
                        else:
                            race_datum = event_datum
                    else:
                        # Ploegenachtervolging, massastart enz.: NIET onder de vorige afstand laten vallen
                        current_distance = None

                elif elem.name == 'table':
                    if current_distance is None:
                        continue
                    for tr in elem.find_all('tr'):
                        td_name = tr.find('td', class_='name')
                        td_time = tr.find('td', class_='time')

                        if not td_name or not td_time:
                            continue

                        a_tag = td_name.find('a')
                        if not a_tag or 'href' not in a_tag.attrs:
                            continue

                        match_id = re.search(r's=(\d+)', a_tag['href'])
                        skater_id = match_id.group(1) if match_id else None
                        skater_name = a_tag.get_text(strip=True)

                        td_age = tr.find('td', class_='age')
                        categorie = td_age.get_text(strip=True) if td_age else ''

                        td_ioc = tr.find('td', class_='ioc')
                        land = td_ioc.get_text(strip=True) if td_ioc else ''

                        tijd_raw = td_time.get_text(strip=True)
                        tijd_sec = parse_schaats_tijd(tijd_raw)

                        if skater_name and tijd_sec:
                            races.append({
                                'Event_ID': event_id,
                                'Toernooi': event_toernooi,
                                'Datum': event_datum,            # startdatum toernooi (zoals voorheen)
                                'Race_Datum': race_datum,        # NIEUW: dag waarop deze afstand gereden is
                                'Baan': event_rink,
                                'Atleet_ID': skater_id,
                                'Naam': skater_name,
                                'Categorie': categorie,
                                'Land': land,
                                'Geslacht': gender_label,
                                'Afstand': current_distance,
                                'Tijd_Raw': tijd_raw,
                                'Tijd_Sec': tijd_sec
                            })
        except Exception as e:
            print(f"[!] event {event_id} {gender_label}: {e}")

    return event_id, races


def scrape_alles_terug_naar_een(start_id=32104, eind_id=1, max_workers=6, batch_size=50):
    verwerkte_ids = set()
    if os.path.exists(FILENAME):
        try:
            bestaande_df = pd.read_csv(FILENAME, usecols=['Event_ID'])
            verwerkte_ids = set(bestaande_df['Event_ID'].unique())
            print(f"=== [Hervatten] {len(verwerkte_ids)} unieke events al aanwezig in {FILENAME} ===")
        except Exception as e:
            print(f"[!] Kon bestaand bestand niet lezen, start nieuw: {e}")

    alle_ids = [e_id for e_id in range(start_id, eind_id - 1, -1) if e_id not in verwerkte_ids]
    total_te_doen = len(alle_ids)

    print(f"=== Start met scrapen van {total_te_doen} resterende wedstrijden ===")

    buffer = []
    totaal_verwerkt = 0

    for i in range(0, total_te_doen, batch_size):
        batch = alle_ids[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(scrape_event, e_id): e_id for e_id in batch}

            for future in as_completed(future_to_id):
                e_id, resultaat = future.result()
                totaal_verwerkt += 1
                if resultaat:
                    buffer.extend(resultaat)

        if buffer:
            df_batch = pd.DataFrame(buffer)
            header_nodig = not os.path.exists(FILENAME)
            df_batch.to_csv(FILENAME, mode='a', index=False, header=header_nodig)
            buffer = []

        voortgang = (totaal_verwerkt / total_te_doen) * 100
        print(f"Voortgang: {totaal_verwerkt}/{total_te_doen} ({voortgang:.1f}%) | Opgeslagen.")


def test_een(event_id=31588):
    """Snelle controle op één toernooi (standaard de Spelen 2026) voordat je alles draait."""
    _, rijen = scrape_event(event_id)
    df = pd.DataFrame(rijen)
    if df.empty:
        print("geen rijen")
        return
    print(df.groupby(['Geslacht', 'Afstand', 'Race_Datum'], sort=False).size().to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_een(int(sys.argv[2]) if len(sys.argv) > 2 else 31588)
    else:
        scrape_alles_terug_naar_een(start_id=32104, eind_id=1, max_workers=8, batch_size=50)
