"""Luchtdruk (en temperatuur, wind, vocht) per baan en wedstrijddag, uit de historische weerdata van Open-Meteo (ERA5, vanaf 1940).

Luchtweerstand is evenredig met de luchtdichtheid rho = p / (R T). Binnen is T geregeld, dus vooral p telt; een dag met 10 hPa
lagere druk is 1% minder luchtweerstand. De druk wordt opgevraagd op de hoogte van de baan (eigen hoogte waar bekend).

Stappen (de weermap is voorspel/weer/ in de repo; ophalen draait op GitHub Actions, zie .github/workflows/luchtdruk.yml):
  python luchtdruk.py <datamap> invoer <weermap>   # banen.csv en baan_dagen.csv uit de brondata (data/, zie nieuwe_data.py)
  python luchtdruk.py <weermap> banen              # plaatsbepaling (geocoding) -> banen_locatie.csv (nakijken!)
  python luchtdruk.py <weermap> haal [N]           # weer per wedstrijddag, hoogstens N calls -> weer_dag.csv (gaat verder waar hij was)

Limieten van de gratis API: 10.000 calls per dag, 5.000 per uur, 600 per minuut; meer dan 2 weken data per locatie telt als meer
calls (per 2 weken 1). Alle wedstrijddagen samen zijn ~13.000 calls: twee runs op verschillende dagen. Banen met veel ritten van
toprijders gaan eerst. Per dag het gemiddelde over 10-20 uur lokale tijd."""
import os, sys, json, time, math, re, urllib.parse
import numpy as np, pandas as pd
import requests

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
    'Den Haag (NED)': (52.0705, 4.3007, 5),                  # De Uithof
    'Quebec City (CAN)': (46.7960, -71.2620, 60), 'Quebec City-Boucher (CAN)': (46.7960, -71.2620, 60),   # Centre de glace Intact Assurance
    'Båstad (NOR)': (59.5300, 11.3500, 150),                 # Båstad bij Trøgstad (Østfold)
}


class Limiet(Exception):
    pass


_S = None


def _sessie():
    """Eén blijvende verbinding (keep-alive): zonder hergebruik kostte elke nieuwe TLS-handshake soms 30 s (time-out)."""
    global _S
    if _S is None:
        _S = requests.Session(); _S.headers['User-Agent'] = 'schaatsen-luchtdruk/1.0'
        _S.mount('https://', requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2))
    return _S


def _json(url, pogingen=5):
    for k in range(pogingen):
        try:
            r = _sessie().get(url, timeout=(10, 60))
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and 'aily' in r.text:          # daglimiet: stoppen, volgende run gaat verder
                raise Limiet(r.text)
            print(f'  HTTP {r.status_code}: {r.text[:120]}', flush=True)
            if r.status_code not in (429, 500, 502, 503, 504) or k == pogingen - 1:
                r.raise_for_status()
            time.sleep(60 if r.status_code == 429 else 2 ** (k + 1))
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError):
                raise
            print(f'  verbinding: {e}', flush=True)
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


def invoer(dm, wm):
    """Banenlijst (met prioriteit) en de wedstrijddagen per baan uit de brondata, klein genoeg voor de repo. Racedag per rit uit
    nieuwe_data.racedatum; bij een toernooi zonder bekende racedagen de startdag en de 4 dagen erna (een venster tot 14 dagen kost
    1 call, dus die extra dagen kosten bijna niets)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nieuwe_data as nd
    os.makedirs(wm, exist_ok=True)
    N = nd.laad_nieuw(dm); rd, bron = nd.racedatum(N, pd.read_parquet(os.path.join(dm, 'racedatum_map.parquet')))
    dag = lambda t: ((pd.to_datetime(t) - pd.Timestamp('1970-01-01')).dt.days).astype(int)
    R = pd.DataFrame({'baan': N['Baan'].astype(str), 'dag': dag(rd), 'skater_id': N['Atleet_ID']})
    los = N.loc[bron == 'startdag', ['Baan', 'Datum']].drop_duplicates()
    extra = pd.concat([pd.DataFrame({'baan': los['Baan'].astype(str), 'dag': dag(los['Datum']) + k, 'skater_id': -1}) for k in range(5)])
    R_ = pd.concat([R, extra], ignore_index=True)
    top = set(R.groupby('skater_id').size().sort_values().index[-3000:])          # rijders met de meeste ritten (wereldbeker, kampioenschappen)
    B = R.groupby('baan').agg(ritten=('dag', 'size')).join(R[R['skater_id'].isin(top)].groupby('baan').size().rename('ritten_top')).fillna(0)
    B['prio'] = B['ritten_top'] * 10 + B['ritten']
    B.sort_values('prio', ascending=False).astype(int).to_csv(os.path.join(wm, 'banen.csv'))
    D = R_.drop_duplicates(['baan', 'dag'])[['baan', 'dag']].sort_values(['baan', 'dag'])
    D.to_csv(os.path.join(wm, 'baan_dagen.csv'), index=False)
    print(len(B), 'banen,', len(D), 'baan-dagen (waarvan', len(D) - R.drop_duplicates(['baan', 'dag']).shape[0], 'rond toernooien zonder racedagen)')


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
            try:
                res = _json(GEO + '?' + urllib.parse.urlencode(q)).get('results') or []
            except Exception as e:
                print(f'{b}: geocoding mislukt ({e})', flush=True); res = []
            if not land:                                          # zonder landcode: meestal Rusland / Wit-Rusland / Kazachstan
                res = sorted(res, key=lambda r: r.get('country_code') not in ('RU', 'BY', 'KZ', 'UA'))
            if res:
                lat, lon, h = res[0]['latitude'], res[0]['longitude'], res[0].get('elevation', np.nan); bron = f"geo:{res[0].get('name')},{res[0].get('country_code')}"
            else:
                lat = lon = h = np.nan; bron = 'niet gevonden'
            time.sleep(0.3)
        rij.append(dict(baan=b, zoeknaam=naam, land=land, lat=lat, lon=lon, hoogte=h, bron=bron, ritten=int(x['ritten']), prio=int(x['prio'])))
        print(f'{b}: {bron} ({lat}, {lon}, {h} m)', flush=True)
        if len(rij) % 20 == 0:
            pd.DataFrame(rij).to_csv(uit, index=False)
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
    # al gedaan: dagen met weer, en dagen binnen een al opgehaald venster (ook als de dag geen uitkomst gaf)
    gedaan = {}
    if os.path.exists(f_dag):
        for b_, g in pd.read_csv(f_dag, usecols=['baan', 'dag']).groupby('baan'):
            gedaan.setdefault(b_, set()).update(g['dag'].astype(int))
    if os.path.exists(f_klaar):
        for b_, s_, e_ in pd.read_csv(f_klaar)[['baan', 's', 'e']].itertuples(index=False):
            gedaan.setdefault(b_, set()).update(range(int(s_), int(e_) + 1))
    gebruikt = 0; pauze = 3600.0 / per_uur; nieuw = []; nieuw_klaar = []
    try:
        for _, b in L.iterrows():
            if not np.isfinite(b['lat']):
                continue
            dagen = D.loc[D['baan'] == b['baan'], 'dag'].to_numpy()
            dagen = np.array([d for d in dagen if int(d) not in gedaan.get(b['baan'], ())])
            nieuw = []; nieuw_klaar = []
            if len(dagen) == 0:
                continue
            for s, e in vensters(dagen):
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


def test():
    """Verbindingstest: een baan opzoeken en een dag weer ophalen."""
    print(_json(GEO + '?' + urllib.parse.urlencode({'name': 'Heerenveen', 'count': 1, 'countryCode': 'NL'}))['results'][0])
    j = _json(ARCH + '?' + urllib.parse.urlencode({'latitude': 52.957, 'longitude': 5.942, 'start_date': '2026-02-07', 'end_date': '2026-02-07',
                                                    'hourly': ','.join(VARS), 'timezone': 'auto'}))
    print(dag_gemiddelde(j, [20491]))


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'test':
        test(); sys.exit()
    if a[2] == 'invoer':
        invoer(a[1], a[3])
    elif a[2] == 'banen':
        banen(a[1])
    elif a[2] == 'haal':
        haal(a[1], int(a[3]) if len(a) > 3 else 9000)
