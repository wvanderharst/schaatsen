"""Luchtdruk (en temperatuur, wind, vocht) per baan en wedstrijddag, uit de historische weerdata van Open-Meteo (ERA5, vanaf 1940).

Luchtweerstand is evenredig met de luchtdichtheid rho = p / (R T). Binnen is T geregeld, dus vooral p telt; een dag met 10 hPa
lagere druk is 1% minder luchtweerstand. De druk wordt opgevraagd op de hoogte van de baan (eigen hoogte waar bekend).

Stappen (alles wordt bewaard in <werkmap>/weer/, een volgende run gaat verder waar hij was):
  python luchtdruk.py <werkmap> banen      # banenlijst + plaatsbepaling (geocoding) -> weer/banen_locatie.csv (nakijken!)
  python luchtdruk.py <werkmap> haal [N]   # weer per baan, alleen vensters rond de wedstrijddagen, hoogstens N calls (standaard 9000)
  python luchtdruk.py <werkmap> samen      # -> weer/weer_dag.parquet: baan, dag, druk (hPa), temperatuur, wind, vocht (10-20 uur lokaal)

Limiet van de gratis API: 10.000 calls per dag; meer dan 2 weken data per locatie telt als meer calls (per 2 weken 1).
Volgorde: banen met elitewedstrijden en veel ritten eerst."""
import os, sys, json, time, math, re, urllib.request, urllib.parse
import numpy as np, pandas as pd

GEO = 'https://geocoding-api.open-meteo.com/v1/search'
ARCH = 'https://archive-api.open-meteo.com/v1/archive'
VARS = ['surface_pressure', 'pressure_msl', 'temperature_2m', 'wind_speed_10m', 'relative_humidity_2m']
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


def _json(url, pogingen=4):
    for k in range(pogingen):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
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


def banen(w):
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(w, 'weer'), exist_ok=True)
    R = pq.read_table(os.path.join(w, 'ritten_cat.parquet'), columns=['baan', 'dag', 'skater_id']).to_pandas(); R['baan'] = R['baan'].astype(str)
    B = R.groupby('baan').agg(ritten=('dag', 'size'), eerste=('dag', 'min'), laatste=('dag', 'max')).sort_values('ritten', ascending=False)
    uit = os.path.join(w, 'weer', 'banen_locatie.csv')
    oud = pd.read_csv(uit).set_index('baan') if os.path.exists(uit) else pd.DataFrame()
    rij = []
    for b, x in B.iterrows():
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
            time.sleep(0.2)
        rij.append(dict(baan=b, zoeknaam=naam, land=land, lat=lat, lon=lon, hoogte=h, bron=bron, ritten=int(x.ritten)))
    L = pd.DataFrame(rij); L.to_csv(uit, index=False)
    print(L.head(40).to_string(index=False)); print('niet gevonden:', L[~np.isfinite(L['lat'])]['baan'].tolist())


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


def haal(w, budget=9000):
    import pyarrow.parquet as pq
    L = pd.read_csv(os.path.join(w, 'weer', 'banen_locatie.csv'))
    R = pq.read_table(os.path.join(w, 'ritten_cat.parquet'), columns=['baan', 'dag', 'skater_id']).to_pandas(); R['baan'] = R['baan'].astype(str)
    # volgorde: banen met de meeste ritten van toprijders eerst (wereldbeker / kampioenschappen), dan de rest
    top = set(R.groupby('skater_id').size().sort_values().index[-3000:])
    prio = R[R['skater_id'].isin(top)].groupby('baan').size()
    L['prio'] = L['baan'].map(prio).fillna(0) * 10 + L['ritten']
    L = L.sort_values('prio', ascending=False)
    cache = os.path.join(w, 'weer', 'uur'); os.makedirs(cache, exist_ok=True)
    gebruikt = 0
    for _, b in L.iterrows():
        if not np.isfinite(b['lat']):
            continue
        dagen = R.loc[R['baan'] == b['baan'], 'dag'].unique()
        for s, e in vensters(dagen):
            f = os.path.join(cache, f"{re.sub(r'[^A-Za-z0-9]+', '_', b['baan'])}_{int(s)}_{int(e)}.json")
            if os.path.exists(f):
                continue
            kosten = math.ceil((e - s + 1) / 14)
            if gebruikt + kosten > budget:
                print('budget op:', gebruikt, 'calls; volgende run gaat verder'); return
            q = {'latitude': b['lat'], 'longitude': b['lon'], 'start_date': str(pd.Timestamp(int(s), unit='D').date()),
                 'end_date': str(pd.Timestamp(int(e), unit='D').date()), 'hourly': ','.join(VARS), 'timezone': 'auto'}
            if np.isfinite(b['hoogte']) and b['hoogte'] > 0:
                q['elevation'] = b['hoogte']                     # druk op de hoogte van de baan
            json.dump(_json(ARCH + '?' + urllib.parse.urlencode(q)), open(f, 'w'))
            gebruikt += kosten; time.sleep(0.15)
        print(f"{b['baan']}: klaar ({gebruikt} calls)", flush=True)
    print('alles opgehaald,', gebruikt, 'calls')


def samen(w, van=10, tot=20):
    cache = os.path.join(w, 'weer', 'uur'); L = pd.read_csv(os.path.join(w, 'weer', 'banen_locatie.csv'))
    sleutel = {re.sub(r'[^A-Za-z0-9]+', '_', b): b for b in L['baan']}
    delen = []
    for f in sorted(os.listdir(cache)):
        j = json.load(open(os.path.join(cache, f))); h = pd.DataFrame(j['hourly']); h['tijd'] = pd.to_datetime(h['time'])
        h = h[(h['tijd'].dt.hour >= van) & (h['tijd'].dt.hour <= tot)]            # wedstrijduren (lokale tijd)
        h['dag'] = (h['tijd'].dt.normalize() - pd.Timestamp('1970-01-01')).dt.days
        d = h.groupby('dag')[VARS].mean().reset_index(); d['baan'] = sleutel[f.rsplit('_', 2)[0]]
        delen.append(d)
    D = pd.concat(delen).drop_duplicates(['baan', 'dag'])
    D = D.rename(columns={'surface_pressure': 'druk', 'pressure_msl': 'druk_zee', 'temperature_2m': 'temp', 'wind_speed_10m': 'wind', 'relative_humidity_2m': 'vocht'})
    D.to_parquet(os.path.join(w, 'weer', 'weer_dag.parquet')); print(len(D), 'baan-dagen;', D['baan'].nunique(), 'banen')


if __name__ == '__main__':
    w, stap = sys.argv[1], sys.argv[2]
    {'banen': lambda: banen(w), 'haal': lambda: haal(w, int(sys.argv[3]) if len(sys.argv) > 3 else 9000), 'samen': lambda: samen(w)}[stap]()
