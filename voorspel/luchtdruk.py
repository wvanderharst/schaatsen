"""Luchtdruk (en temperatuur, wind, vocht) per baan en wedstrijddag, uit de historische weerdata van Open-Meteo (ERA5, vanaf 1940).

Luchtweerstand is evenredig met de luchtdichtheid rho = p / (R T). Binnen is T geregeld, dus vooral p telt; een dag met 10 hPa
lagere druk is 1% minder luchtweerstand. De druk wordt opgevraagd op de hoogte van de baan (eigen hoogte waar bekend).

Stappen (de weermap is voorspel/weer/ in de repo; ophalen draait op GitHub Actions, zie .github/workflows/luchtdruk.yml):
  python luchtdruk.py <werkmap> invoer <weermap>   # lokaal: banen.csv en baan_dagen.csv uit ritten_cat.parquet
  python luchtdruk.py <weermap> banen              # plaatsbepaling (geocoding) -> banen_locatie.csv (nakijken!)
  python luchtdruk.py <weermap> haal [N]           # weer per wedstrijddag, hoogstens N calls -> weer_dag.csv (gaat verder waar hij was)

Limieten van de gratis API: 10.000 calls per dag, 5.000 per uur, 600 per minuut; meer dan 2 weken data per locatie telt als meer
calls (per 2 weken 1). Alle wedstrijddagen samen zijn ~13.000 calls: twee runs op verschillende dagen. Banen met veel ritten van
toprijders gaan eerst. Per dag het gemiddelde over 10-20 uur lokale tijd."""
import os, sys, json, time, math, re, urllib.request, urllib.parse, urllib.error
import numpy as np, pandas as pd

GEO = 'https://geocoding-api.open-meteo.com/v1/search'
ARCH = 'https://archive-api.open-meteo.com/v1/archive'
VARS = ['surface_pressure', 'pressure_msl', 'temperature_2m', 'wind_speed_10m', 'relative_humidity_2m']
NAMEN = {'surface_pressure': 'druk', 'pressure_msl': 'druk_zee', 'temperature_2m': 'temp', 'wind_speed_10m': 'wind', 'relative_humidity_2m': 'vocht'}
LAND = {'NED': 'NL', 'GER': 'DE', 'CAN': 'CA', 'NOR': 'NO', 'JPN': 'JP', 'POL': 'PL', 'KOR': 'KR', 'KAZ': 'KZ', 'ITA': 'IT', 'CHN': 'CN',
        'USA': 'US', 'FIN': 'FI', 'AUT': 'AT', 'MGL': 'MN', 'SUI': 'CH', 'SWE': 'SE', 'CZE': 'CZ', 'BLR': 'BY', 'RUS': 'RU', 'BEL': 'BE',
        'FRA': 'FR', 'GBR': 'GB', 'HUN': 'HU', 'ROU': 'RO', 'LAT': 'LV', 'EST': 'EE', 'LTU': 'LT', 'UKR': 'UA', 'DEN': 'DK', 'ARG': 'AR',
        'AUS': 'AU', 'NZL': 'NZ', 'TPE': 'TW', 'SVK': 'SK', 'SLO': 'SI', 'ESP': 'ES', 'TUR': 'TR', 'PRK': 'KP'}
# eigen plaats en hoogte waar de geocoding de stad geeft en niet de baan (hoogte telt: 100 m is ~1,2% druk)
VAST = {
    'Salt Lake City (USA)': (40.6625, -111.9886, 1423),      # Utah Olympic Oval, Kearns
    'Calgary (CAN)': (51.0767, -114.1356, 1105),             # Olympic Oval, universiteit
    'Inzell (GER)': (47.7631, 12.7417, 690),
    'Heerenveen (NED)': (52.9573, 5.9417, 0),
    'Collalbo (ITA)': (46.5419, 11.4853, 1200),
    'Baselga di Pinè (ITA)': (46.1306, 11.2436, 1000),
    'Milano (ITA)': (45.4560, 9.2900, 110),                 # Rho / Fiera Milano (olympische baan 2026)
    'Hamar (NOR)': (60.7942, 11.1000, 127),
    'Kolomna': (55.0896, 38.7681, 120),
    'Chelyabinsk': (55.1544, 61.4026, 220), 'Chelyabinsk-Inga': (55.1544, 61.4026, 220),
    'Hachinohe - YS Arena (JPN)': (40.5050, 141.4900, 20), 'Hachinohe-Piping (JPN)': (40.5050, 141.4900, 20),
}


class Limiet(Exception):
    pass


def _json(url, pogingen=6):
    for k in range(pogingen):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            tekst = e.read().decode(errors='ignore')
            if e.code == 429 and 'aily' in tekst:            # daglimiet: stoppen, volgende run gaat verder
                raise Limiet(tekst)
            if e.code not in (429, 500, 502, 503, 504) or k == pogingen - 1:
                raise
            time.sleep(60 if e.code == 429 else 2 ** (k + 1))
        except (urllib.error.URLError, TimeoutError):
            if k == pogingen - 1:
                raise
            time.sleep(2 ** (k + 1))


def zoeknaam(baan):
    m = re.match(r'^(.*?)\s*\(([A-Z]{3})\)\s*$', baan)
    naam, land = (m.group(1), LAND.get(m.group(2))) if m else (baan, None)
    naam = re.split(r'\s+-\s+', naam)[0].strip()                                      # 'Perm - Yunost', 'Oslo - Valle Hovin'
    naam = re.sub(r'-(Inga|Boucher|Leangen|National|Piping|Baikal)$', '', naam)           # 'Chelyabinsk-Inga', 'Quebec City-Boucher'
    naam = {'Tomaszów-Mazowiecki': 'Tomaszów Mazowiecki', 'Kirovo-Chepeczk': 'Kirovo-Chepetsk'}.get(naam, naam)
    return naam, land


def invoer(w, wm):
    """Lokaal: banenlijst (met prioriteit) en de wedstrijddagen per baan, klein genoeg voor de repo."""
    import pyarrow.parquet as pq
    os.makedirs(wm, exist_ok=True)
    R = pq.read_table(os.path.join(w, 'ritten_cat.parquet'), columns=['baan', 'dag', 'skater_id']).to_pandas(); R['baan'] = R['baan'].astype(str)
    top = set(R.groupby('skater_id').size().sort_values().index[-3000:])          # rijders met de meeste ritten (wereldbeker, kampioenschappen)
    B = R.groupby('baan').agg(ritten=('dag', 'size')).join(R[R['skater_id'].isin(top)].groupby('baan').size().rename('ritten_top')).fillna(0)
    B['prio'] = B['ritten_top'] * 10 + B['ritten']
    B.sort_values('prio', ascending=False).astype(int).to_csv(os.path.join(wm, 'banen.csv'))
    R.drop_duplicates(['baan', 'dag'])[['baan', 'dag']].sort_values(['baan', 'dag']).to_csv(os.path.join(wm, 'baan_dagen.csv'), index=False)
    print(len(B), 'banen,', R.drop_duplicates(['baan', 'dag']).shape[0], 'baan-dagen')


def banen(wm):
    B = pd.read_csv(os.path.join(wm, 'banen.csv'))
    uit = os.path.join(wm, 'banen_locatie.csv')
    oud = pd.read_csv(uit).set_index('baan') if os.path.exists(uit) else pd.DataFrame(columns=['lat'])
    rij = []
    for _, x in B.iterrows():
        b = x['baan']
        if b in oud.index and np.isfinite(oud.loc[b, 'lat']):
            rij.append(oud.loc[b].to_dict() | {'baan': b}); continue
        naam, land = zoeknaam(b)
        if b in VAST:
            lat, lon, h = VAST[b]; bron = 'vast'
        else:
            q = {'name': naam, 'count': 5, 'language': 'en', 'format': 'json'}
            if land:
                q['countryCode'] = land
            res = _json(GEO + '?' + urllib.parse.urlencode(q)).get('results') or []
            if not land:                                          # zonder landcode: meestal Rusland / Wit-Rusland / Kazachstan
                res = sorted(res, key=lambda r: r.get('country_code') not in ('RU', 'BY', 'KZ', 'UA'))
            if res:
                lat, lon, h = res[0]['latitude'], res[0]['longitude'], res[0].get('elevation', np.nan); bron = f"geo:{res[0].get('name')},{res[0].get('country_code')}"
            else:
                lat = lon = h = np.nan; bron = 'niet gevonden'
            time.sleep(0.3)
        rij.append(dict(baan=b, zoeknaam=naam, land=land, lat=lat, lon=lon, hoogte=h, bron=bron, ritten=int(x['ritten']), prio=int(x['prio'])))
    L = pd.DataFrame(rij); L.to_csv(uit, index=False)
    print(L.head(30).to_string(index=False)); print('niet gevonden:', L[~np.isfinite(L['lat'].astype(float))]['baan'].tolist())


def vensters(dagen, max_len=14):
    """Wedstrijddagen -> vensters van hoogstens max_len dagen (1 call per venster)."""
    dagen = np.sort(np.unique(dagen)); uit = []; s = dagen[0]; e = dagen[0]
    for d in dagen[1:]:
        if d - s < max_len:
            e = d
        else:
            uit.append((s, e)); s = e = d
    uit.append((s, e))
    return uit


def dag_gemiddelde(j, dagen, van=10, tot=20):
    h = pd.DataFrame(j['hourly']); h['tijd'] = pd.to_datetime(h['time'])
    h = h[(h['tijd'].dt.hour >= van) & (h['tijd'].dt.hour <= tot)]                    # wedstrijduren (lokale tijd)
    h['dag'] = (h['tijd'].dt.normalize() - pd.Timestamp('1970-01-01')).dt.days
    d = h[h['dag'].isin(dagen)].groupby('dag')[VARS].mean().reset_index().rename(columns=NAMEN)
    return d


def haal(wm, budget=9000, per_uur=4500):
    L = pd.read_csv(os.path.join(wm, 'banen_locatie.csv')).sort_values('prio', ascending=False)
    D = pd.read_csv(os.path.join(wm, 'baan_dagen.csv'))
    f_dag = os.path.join(wm, 'weer_dag.csv'); f_klaar = os.path.join(wm, 'vensters_klaar.csv')
    klaar = set(map(tuple, pd.read_csv(f_klaar)[['baan', 's', 'e']].itertuples(index=False))) if os.path.exists(f_klaar) else set()
    gebruikt = 0; pauze = 3600.0 / per_uur
    try:
        for _, b in L.iterrows():
            if not np.isfinite(b['lat']):
                continue
            dagen = D.loc[D['baan'] == b['baan'], 'dag'].to_numpy()
            nieuw = []; nieuw_klaar = []
            for s, e in vensters(dagen):
                if (b['baan'], int(s), int(e)) in klaar:
                    continue
                kosten = math.ceil((e - s + 1) / 14)
                if gebruikt + kosten > budget:
                    raise Limiet(f'budget van {budget} calls op')
                q = {'latitude': b['lat'], 'longitude': b['lon'], 'start_date': str(pd.Timestamp(int(s), unit='D').date()),
                     'end_date': str(pd.Timestamp(int(e), unit='D').date()), 'hourly': ','.join(VARS), 'timezone': 'auto'}
                if np.isfinite(b['hoogte']) and b['hoogte'] > 0:
                    q['elevation'] = b['hoogte']                 # druk op de hoogte van de baan
                t0 = time.time()
                d = dag_gemiddelde(_json(ARCH + '?' + urllib.parse.urlencode(q)), dagen[(dagen >= s) & (dagen <= e)])
                d.insert(0, 'baan', b['baan']); nieuw.append(d); nieuw_klaar.append((b['baan'], int(s), int(e)))
                gebruikt += kosten
                time.sleep(max(0.0, pauze * kosten - (time.time() - t0)))
            _bewaar(nieuw, nieuw_klaar, f_dag, f_klaar)
            print(f"{b['baan']}: klaar ({gebruikt} calls)", flush=True)
        print('alles opgehaald,', gebruikt, 'calls')
    except Limiet as e:
        _bewaar(nieuw, nieuw_klaar, f_dag, f_klaar)
        print('gestopt:', str(e)[:200], f'({gebruikt} calls); volgende run gaat verder')


def _bewaar(nieuw, nieuw_klaar, f_dag, f_klaar):
    if not nieuw:
        return
    pd.concat(nieuw).round(3).to_csv(f_dag, mode='a', header=not os.path.exists(f_dag), index=False)
    pd.DataFrame(nieuw_klaar, columns=['baan', 's', 'e']).to_csv(f_klaar, mode='a', header=not os.path.exists(f_klaar), index=False)


if __name__ == '__main__':
    a = sys.argv
    if a[2] == 'invoer':
        invoer(a[1], a[3])
    elif a[2] == 'banen':
        banen(a[1])
    elif a[2] == 'haal':
        haal(a[1], int(a[3]) if len(a) > 3 else 9000)
