# -*- coding: utf-8 -*-
"""
Tests voor do_analysis.py

Twee soorten tests:

A. SYNTHETISCHE DATA MET BEKENDE WAARHEID (snel, onafhankelijk van jouw CSV)
   - geen drift: geschatte niveaus blijven in de tijd bij de werkelijkheid
   - geplante piekdagen worden gevonden, gewone dagen bijna nooit
   - baaneffecten worden teruggevonden
   - veel ruizige jeugd/zwakke rijders verslechtert de schatting van het dag-effect nauwelijks

B. ECHTE DATA (langzamer: het model wordt 1x gefit en gecachet)
   - top 10 per afstand/geslacht: verwachte tijd past bij hun recente (dag-gecorrigeerde) tijden
   - geen drift in de tijd (voorspelfout per jaar), geen bias per niveaugroep, calibratie
   - T_lim onder alle waargenomen tijden
   - simulatie: winkansen tellen op, gedeeld dag-effect verandert tijden maar niet de rangorde

Draai:  python -m pytest test_model.py -s -q            (kleine testdatabase)
        python -m pytest test_model.py -s -q --volledig  (volledige database, alleen op verzoek)
"""
import json
import math
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

import do_analysis as da

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_VOL = da.data_csv(HERE) if any(os.path.exists(os.path.join(HERE, n)) for n in ('speedskating_historie_racedatum.csv', 'speedskating_historie_clean.csv')) else os.path.join(HERE, 'speedskating_historie_clean.csv')
CSV_TEST = os.path.join(HERE, 'speedskating_test.csv')        # klein: toprijders + 6% steekproef (8,7% van de rijen)
VOLLEDIG = os.environ.get('SCHAATS_VOLLEDIG') == '1'          # alleen op verzoek: python -m pytest test_model.py --volledig
CSV = os.environ.get('SCHAATS_CSV') or (CSV_VOL if VOLLEDIG else CSV_TEST)
PARAMS_JSON = os.path.join(HERE, 'params_getuned.json')
CACHE = os.path.join(HERE, '.test_model_cache_vol.pkl' if CSV == CSV_VOL else '.test_model_cache_test.pkl')


# =============================================================================
# A. SYNTHETISCHE DATA
# =============================================================================
SYN_AFSTANDEN = ['500m', '1000m', '1500m']
SYN_TLIM = {'500m': 32.0, '1000m': 63.0, '1500m': 97.0}
SYN_BASIS = {'500m': 2.3, '1000m': 3.0, '1500m': 3.4}      # gemiddeld ln(T - T_lim)
BAAN_EFFECT = {'A': -0.020, 'B': -0.010, 'C': 0.0, 'D': 0.006, 'E': 0.012, 'F': 0.020}


def maak_synthetische_data(seed=0, n_volw=250, n_events=700, extra_jeugd=0, jeugd_ruis=0.35, piek_shock=-0.02):
    """Rijders met CONSTANT niveau (dus elke drift in de schatting is een fout van het model)."""
    rng = np.random.default_rng(seed)
    dag0 = int((pd.Timestamp('2016-10-01') - pd.Timestamp('1970-01-01')).days)
    rijders = []
    for i in range(n_volw):
        a = rng.normal(0, 0.6)
        x = {d: SYN_BASIS[d] + a * (0.9 + 0.1 * k) + rng.normal(0, 0.15) for k, d in enumerate(SYN_AFSTANDEN)}
        rijders.append({'id': i, 'x': x, 'ruis': 0.10 if i % 4 else 0.16, 'groep': 'senior',
                        'gender': 'm' if i % 2 == 0 else 'v', 'land': f'L{i % 9}'})
    for i in range(n_volw, n_volw + extra_jeugd):
        a = rng.normal(0.8, 0.4)
        x = {d: SYN_BASIS[d] + a + rng.normal(0, 0.2) for d in SYN_AFSTANDEN}
        rijders.append({'id': i, 'x': x, 'ruis': jeugd_ruis, 'groep': 'jeugd',
                        'gender': 'm' if i % 2 == 0 else 'v', 'land': f'L{i % 9}'})
    volw = [r for r in rijders if r['groep'] == 'senior']
    jeugd = [r for r in rijders if r['groep'] == 'jeugd']

    dagen = rng.choice(np.arange(dag0, dag0 + 8 * 365), size=n_events, replace=False)
    banen = list(BAAN_EFFECT)
    piek_ids = set(rng.choice(n_events, size=25, replace=False))
    rng_j = np.random.default_rng(seed + 1000)          # aparte stroom: jeugd verandert de basis niet
    rows, waar = [], {}
    for e in range(n_events):
        baan = banen[rng.integers(len(banen))]
        dag_shock = rng.normal(0, 0.004) + (piek_shock if e in piek_ids else 0.0)
        n_part = int(rng.integers(45, 70)) if e in piek_ids else int(rng.integers(12, 60))
        deeln = [volw[k] for k in rng.choice(len(volw), size=min(n_part, len(volw)), replace=False)]
        if extra_jeugd:
            deeln += [jeugd[k] for k in rng_j.choice(len(jeugd), size=min(40, len(jeugd)), replace=False)]
        totaal = BAAN_EFFECT[baan] + dag_shock
        waar[e] = {'baan': baan, 'totaal': totaal, 'piek': e in piek_ids, 'dag_shock': dag_shock}
        for r in deeln:
            for d in rng.choice(SYN_AFSTANDEN, size=int(rng.integers(1, 3)), replace=False):
                afst_shock = rng.normal(0, 0.0015)
                ruis = rng.normal(0, r['ruis'])
                tijd = (SYN_TLIM[d] * (1.0 if r['gender'] == 'm' else 1.05) + np.exp(r['x'][d] + ruis)) \
                    * np.exp(totaal + afst_shock)
                rows.append({'event_id': e, 'dag': int(dagen[e]), 'skater_id': r['id'], 'skater_name': f"S{r['id']}",
                             'gender': r['gender'], 'land': r['land'], 'groep': r['groep'], 'categorie': 'X',
                             'baan': baan, 'afstand': d, 'tijd': float(tijd), 'verdacht': False,
                             'x_waar': r['x'][d] + (0.0 if r['gender'] == 'm' else np.log1p(0.0))})
    df = pd.DataFrame(rows)
    df['datum'] = pd.to_datetime(df['dag'], unit='D')
    df['jaar'] = df['datum'].dt.year
    df['seizoen'] = np.where(df['datum'].dt.month >= 7, df['jaar'], df['jaar'] - 1)
    df['afstand_idx'] = df['afstand'].map(da.NAAR_INDEX).astype(int)
    df = df.sort_values(['dag', 'event_id', 'afstand_idx', 'gender']).reset_index(drop=True)
    return df, waar


def fit_synthetisch(df, **param_kw):
    params = da.Params(piek_max_se=0.006, **param_kw)
    m = da.SchaatsModel(params).voorbereiden(df, verbose=False)
    m.fit(df, verbose=False)
    return da.kalibreer(m, df, n_iter=2, verbose=False)


@pytest.fixture(scope='module')
def syn():
    df, waar = maak_synthetische_data(seed=1)
    return fit_synthetisch(df), df, waar


def test_syn_geen_drift(syn):
    """Constant echt niveau: de 1-stap-vooruit schatting (in seconden, dus relatief) mag in de tijd niet wegdrijven."""
    m, df, waar = syn
    o = m.obs
    o = o[(o['n_tot'] >= 8) & (o['gender'] == 'm')].copy()
    tl_model = da._tl_vector(m.t_lims, o['gender'].to_numpy(), o['afstand_idx'].to_numpy())
    tl_waar = np.array([SYN_TLIM[a] for a in o['afstand']])
    o['fout'] = np.log((tl_model + np.exp(o['x_prior'].to_numpy(dtype=float))) /
                       (tl_waar + np.exp(o['x_waar'].to_numpy(dtype=float)))) * 100.0
    per_jaar = o.groupby('jaar')['fout'].median()
    per_jaar = per_jaar[o.groupby('jaar').size() >= 200]
    assert len(per_jaar) >= 5
    spreiding = per_jaar.std()
    helling = np.polyfit(per_jaar.index.values.astype(float), per_jaar.values, 1)[0]
    print("\nSYN mediane fout in verwachte tijd per jaar (%):", per_jaar.round(2).to_dict())
    assert spreiding < 0.5, f"bias varieert te veel per jaar: sd={spreiding:.2f}%"
    assert abs(helling) < 0.1, f"structurele drift: {helling:.3f} %-punt per jaar"


def test_syn_niveau_wordt_teruggevonden(syn):
    """Rijders met veel races: geschatte verwachte tijd ligt dicht bij de waarheid (absoluut, dus ook het anker klopt)."""
    m, df, waar = syn
    rel = []
    for st in m.staten.values():
        if st.gender != 'm':
            continue
        for d in SYN_AFSTANDEN:
            j = da.NAAR_INDEX[d]
            if st.n[j] < 12:
                continue
            x_waar = df[(df['skater_id'] == st.id) & (df['afstand'] == d)]['x_waar'].iloc[0]
            waar_t = SYN_TLIM[d] + np.exp(x_waar)
            schat_t = m.voorspel(st.id, d)['verwachte_tijd']
            rel.append(np.log(schat_t / waar_t))
    rel = np.array(rel)
    assert len(rel) >= 50
    robuust_sd = 1.4826 * np.median(np.abs(rel - np.median(rel)))
    print(f"\nSYN niveau: n={len(rel)} mediaan fout {np.median(rel) * 100:.2f}% robuuste sd {robuust_sd * 100:.2f}%")
    assert abs(np.median(rel)) < 0.006, f"niveau-offset {np.median(rel) * 100:.2f}%"
    assert robuust_sd < 0.013, f"te grote spreiding in niveau-schatting: {robuust_sd * 100:.2f}%"


def test_syn_piekdagen(syn):
    m, df, waar = syn
    ev = m.events.set_index('event_id')
    gepland = [e for e, w in waar.items() if w['piek']]
    normaal = [e for e, w in waar.items() if not w['piek']]
    herkend = ev.loc[gepland, 'piekdag'].mean()
    vals_pos = ev.loc[normaal, 'piekdag'].mean()
    print(f"\nSYN piekdagen: herkend {herkend * 100:.0f}% van {len(gepland)} geplante; "
          f"vals alarm {vals_pos * 100:.1f}% van {len(normaal)} gewone dagen")
    assert herkend >= 0.80
    assert vals_pos <= 0.05


def test_syn_baaneffecten(syn):
    m, df, waar = syn
    schat = {b: m.baan_effect(b, int(df['dag'].max()))[0] for b in BAAN_EFFECT}
    waarheid = pd.Series(BAAN_EFFECT)
    sch = pd.Series(schat)
    r = np.corrcoef(sch.values, waarheid.values)[0, 1]
    print("\nSYN baaneffecten geschat (%):", (sch * 100).round(2).to_dict())
    assert r > 0.9
    # helling ~ 1: de schaal klopt (niet alleen de volgorde)
    helling = np.polyfit(waarheid.values, sch.values, 1)[0]
    assert 0.6 < helling < 1.4


def test_syn_ruizige_jeugd_heeft_beperkt_effect():
    """Jouw idee: veel ruizige jeugd/zwakke rijders mogen het dag-effect nauwelijks verslechteren.
    We vergelijken met (a) een veld zonder jeugd en (b) een naieve variant die iedereen gelijk weegt."""
    df0, waar0 = maak_synthetische_data(seed=5, extra_jeugd=0)
    df1, waar1 = maak_synthetische_data(seed=5, extra_jeugd=120, jeugd_ruis=0.40)
    # de synthetische ruis is normaal verdeeld (geen slechte dagen), dus het dip-model gaat hier uit
    m0 = fit_synthetisch(df0, dip_pi=0.0)
    m1 = fit_synthetisch(df1, dip_pi=0.0)
    m2 = fit_synthetisch(df1, dip_pi=0.0, betrouwbaarheid=False)

    def rmse(m, waar):
        ev = m.events.set_index('event_id')
        wt = pd.Series({e: w['totaal'] for e, w in waar.items()}).reindex(ev.index)
        verschil = ev['theta_post'] - wt
        verschil = verschil - verschil.median()          # globale schaal is niet identificeerbaar
        return float(np.sqrt((verschil ** 2).mean()))

    r0, r1, r2 = rmse(m0, waar0), rmse(m1, waar1), rmse(m2, waar1)
    print(f"\nSYN dag-effect RMSE: geen jeugd {r0 * 100:.3f}% | 120 ruizige jeugd per event, gewogen {r1 * 100:.3f}% "
          f"| idem, iedereen gelijk gewogen {r2 * 100:.3f}%")
    assert r1 < r2, "gewichten helpen niet t.o.v. iedereen gelijk wegen"
    assert r1 <= 1.30 * r0, "ruizige jeugd verslechtert het dag-effect te veel"


# =============================================================================
# B. ECHTE DATA
# =============================================================================
def _bouw_echt_model():
    df = da.laad_dataset(CSV, verbose=False)
    params = da.Params(**json.load(open(PARAMS_JSON))) if os.path.exists(PARAMS_JSON) else da.Params()
    return da.fit_volledig(df, params, n_iter=3, verbose=False)


@pytest.fixture(scope='module')
def echt():
    if not os.path.exists(CSV):
        pytest.skip("CSV niet gevonden")
    sleutel = tuple(os.path.getmtime(p) for p in [CSV, os.path.join(HERE, 'do_analysis.py')]
                    + ([PARAMS_JSON] if os.path.exists(PARAMS_JSON) else []))
    if os.path.exists(CACHE):
        try:
            with open(CACHE, 'rb') as f:
                s, m = pickle.load(f)
            if s == sleutel:
                return m
        except Exception:
            pass
    m = _bouw_echt_model()
    with open(CACHE, 'wb') as f:
        pickle.dump((sleutel, m), f)
    return m


TOP_CASES = [(g, a) for g in ['m', 'v'] for a in ['500m', '1000m', '1500m', '3000m']]


@pytest.mark.parametrize('gender,afstand', TOP_CASES)
def test_top10_verwachte_tijd_klopt(echt, gender, afstand):
    """De 10 beste actieve rijders: hun verwachte tijd hoort dicht bij hun recente (dag-gecorrigeerde) tijden te liggen,
    niet sneller dan hun PB en binnen de bandbreedte van hun recente races."""
    t = da.top_rijders_tabel(echt, gender, afstand, n=10, min_recent=3)
    if len(t) < 10:
        pytest.skip(f"slechts {len(t)} actieve rijders met >=3 recente races (5%-steekproef)")
    kol = ['naam', 'land', 'n_races', 'n_recent', 'verwacht_s', 'sd_pct', 'recent_mediaan_dagcorr_s',
           'recent_mediaan_ruw_s', 'pb_s', 'afwijking_pct']
    print(f"\nTop 10 {gender} {afstand}\n" + t[kol].round(2).to_string(index=False))
    # een rijder in vormverandering: de verwachting mag dan bij de laatste race liggen i.p.v. de mediaan van 5
    dicht_bij_laatste = (t['verwacht_s'] / t['laatste_dagcorr_s'] - 1.0).abs() <= 0.015
    # snel verbeterende rijder: de verwachting mag tussen de recente mediaan en de laatste (dag-gecorrigeerde) race liggen
    med_, laat_, verw_ = t['recent_mediaan_dagcorr_s'], t['laatste_dagcorr_s'], t['verwacht_s']
    tussen = ((verw_ - med_) * (laat_ - med_) >= 0) & ((verw_ - med_).abs() <= (laat_ - med_).abs())
    # bewust asymmetrisch: na een snelle race tellen tragere races minder zwaar (kan niet piekken); de verwachting mag dan
    # tot 2.5% sneller zijn dan de recente mediaan (maar nooit buiten de bandbreedte van de recente races, zie `buiten`)
    slecht = t[((t['afwijking_pct'] > 1.5) | (t['afwijking_pct'] < -2.5)) & ~dicht_bij_laatste & ~tussen]
    te_snel = t[t['verwacht_s'] < 0.985 * t['pb_s']]
    buiten = t[(t['verwacht_s'] > 1.015 * t['recent_max_dagcorr_s']) | (t['verwacht_s'] < 0.985 * t['recent_min_dagcorr_s'])]
    assert slecht.empty, f"verwachte tijd wijkt >1.5% af van recente mediaan:\n{slecht[kol]}"
    assert te_snel.empty, f"verwachte tijd sneller dan PB:\n{te_snel[kol]}"
    assert buiten.empty, f"verwachte tijd buiten bandbreedte recente races:\n{buiten[kol]}"


def _fouten(echt, vanaf=2010, min_hist=1):
    score, err, var, m = da.voorspelscore(echt, vanaf)
    d = pd.DataFrame({'jaar': m['jaar'].to_numpy(), 'groep': m['groep'].to_numpy(), 'n_tot': m['n_tot'].to_numpy(),
                      'skater_id': m['skater_id'].to_numpy(), 'gender': m['gender'].to_numpy(),
                      'afstand': m['afstand'].to_numpy(), 'err': err * 100.0,
                      'zabs': np.abs(err) / np.sqrt(var)})
    return d[d['n_tot'] >= min_hist]


def test_geen_drift_per_jaar(echt):
    """Voorspelfout (mediaan, in %) per kalenderjaar voor rijders met historie: geen wegdrijven."""
    d = _fouten(echt)
    per_jaar = d.groupby('jaar')['err'].agg(['median', 'size'])
    per_jaar = per_jaar[per_jaar['size'] >= 300]
    print("\nMediane voorspelfout per jaar (%):\n" + per_jaar.round(2).T.to_string())
    assert (per_jaar['median'].abs() <= 2.0).all(), per_jaar[per_jaar['median'].abs() > 2.0]
    helling = np.polyfit(per_jaar.index.values.astype(float), per_jaar['median'].values, 1)[0]
    assert abs(helling) < 0.15, f"structurele trend in de fout: {helling:.3f} %-punt per jaar"


def test_geen_bias_per_niveaugroep(echt):
    """Ook recreatieve, jeugd- en masters-rijders: geen systematische fout en betrouwbare onzekerheid."""
    d = _fouten(echt)
    tab = d.groupby('groep').agg(n=('err', 'size'), mediaan=('err', 'median'),
                                 binnen80=('zabs', lambda s: (s <= 1.2816).mean()))
    print("\nBias en dekking per niveaugroep:\n" + tab.round(3).to_string())
    groot = tab[tab['n'] >= 500]
    assert (groot['mediaan'].abs() <= 2.0).all(), groot
    # 'onbekend' is een mengsel van onherkenbare categoriecodes en mag iets slechter gedekt zijn
    ondergrens = pd.Series(0.70, index=groot.index)
    if 'onbekend' in ondergrens.index:
        ondergrens['onbekend'] = 0.65
    assert ((groot['binnen80'] > ondergrens) & (groot['binnen80'] < 0.92)).all(), groot


def test_top10_voorspellingen_onbevooroordeeld(echt):
    """De voorspellingen VOOR hun races zijn voor de top-10 rijders gemiddeld niet te snel/te traag."""
    ids = set()
    for g, a in TOP_CASES:
        t = da.top_rijders_tabel(echt, g, a, n=10, min_recent=3)
        ids |= set(t['skater_id'])
    if len(ids) < 10:
        pytest.skip("te weinig top-rijders")
    d = _fouten(echt, vanaf=2016)
    top = d[d['skater_id'].isin(ids)]
    print(f"\nTop-rijders: {len(ids)} rijders, {len(top)} voorspellingen, mediane fout {top['err'].median():.2f}%, "
          f"gemiddeld {top['err'].mean():.2f}%")
    assert len(top) >= 50
    assert abs(top['err'].median()) <= 1.0


def test_t_lim_onder_alle_tijden(echt):
    o = echt.obs
    for g in ['m', 'v']:
        for a in da.AFSTANDEN:
            s = o[(o['gender'] == g) & (o['afstand'] == a) & (~o['verdacht'])]
            if len(s) < 30:
                continue
            tl = echt.t_lims[g][a]
            assert tl < s['tijd'].min(), (g, a, tl, s['tijd'].min())
            assert tl > 0.85 * s['tijd'].min(), (g, a, tl, s['tijd'].min())


def test_dag_tabel_consistent(echt):
    ev = echt.events
    assert ev['event_id'].is_unique
    assert not ev['theta_post'].isna().any()
    pk = ev[ev['piekdag']]
    assert (pk['z'] <= -echt.params.piek_z).all()
    assert pk['goed_gemeten'].all()


def test_simulatie(echt):
    laatste = int(echt.obs['dag'].max())
    kand = [(s.id, echt.voorspel(s.id, '1500m')['verwachte_tijd']) for s in echt.staten.values()
            if s.gender == 'm' and s.n[da.NAAR_INDEX['1500m']] >= 3 and laatste - s.laatste_dag < 730]
    kand.sort(key=lambda t: t[1])
    ids = [k[0] for k in kand[:6]]
    assert len(ids) >= 4
    a = da.simuleer_race(echt, ids, '1500m', scenario='normaal', seed=7)
    b = da.simuleer_race(echt, ids, '1500m', scenario='piekdag', seed=7)
    assert abs(a['Goud (%)'].sum() - 100.0) < 0.5
    # een gedeeld dag-effect verandert de rangorde niet: dezelfde winkansen...
    ga = a.set_index('Rijder')['Goud (%)']
    gb = b.set_index('Rijder')['Goud (%)']
    assert (ga - gb.reindex(ga.index)).abs().max() < 1e-9
    # ...maar wel de tijden: op een piekdag is de verwachte tijd sneller
    ta = a.set_index('Rijder')['Verwachte Tijd (s)']
    tb = b.set_index('Rijder')['Verwachte Tijd (s)']
    assert (tb.reindex(ta.index) < ta).all()


# =============================================================================
# C. TOESTANDSGESCHIEDENIS (tijdreizen)
# =============================================================================
@pytest.fixture(scope='module')
def hist(echt):
    df = da.laad_dataset(CSV, verbose=False)
    echt.bewaar_historie = True
    echt.fit(df, verbose=False)
    h = da.Historie(echt)
    echt.historie_ruw = None
    echt.bewaar_historie = False
    return h


def test_historie_eindtoestand_gelijk_aan_model(echt, hist):
    """Op een dag na de laatste race moet de geschiedenis exact de huidige voorspelling geven."""
    laatste = int(echt.obs['dag'].max()) + 1
    n_ok = 0
    for sid in list(hist.ids[::max(len(hist.ids)//30, 1)])[:25]:
        st = echt.staten[int(sid)]
        for a in ['500m', '1500m']:
            v1 = echt.voorspel(int(sid), a, laatste)
            v2 = hist.voorspel(int(sid), a, laatste)
            assert abs(v1['mu_log'] - v2['mu_log']) < 1e-4, (sid, a)
            assert abs(v1['sigma_log'] - v2['sigma_log']) < 1e-4, (sid, a)
            n_ok += 1
    assert n_ok >= 40


def test_historie_geen_toekomstinformatie(hist):
    """De toestand op dag D mag alleen events van vóór D bevatten, en de dagen per rijder lopen op."""
    d = hist.d
    gecontroleerd = 0
    for sid in list(hist.ids[::max(len(hist.ids)//45, 1)])[:40]:
        a, b = hist.pos[int(sid)]
        dagen = d['dag'][a:b]
        assert (np.diff(dagen) >= 0).all()
        if dagen[0] == dagen[-1]:
            continue
        mid = int(dagen[-1])
        k = hist._snap(int(sid), mid)
        assert d['dag'][k] < mid
        assert d['n'][k].sum() <= d['n'][b - 1].sum()
        gecontroleerd += 1
    assert gecontroleerd >= 10


def test_historie_trajecten_glad(hist):
    """De verwachte tijd van een top-rijder springt niet: max verandering tussen opeenvolgende events < 3%."""
    sid = int(hist.ids[np.argmax(np.diff(np.append(hist.start, len(hist.d['dag']))))])
    tr = hist.traject(sid, '500m')
    tr = tr[tr['n_races'] >= 5]
    if len(tr) > 10:
        sprong = np.abs(np.diff(np.log(tr['verwachte_tijd_s'].to_numpy())))
        assert sprong.max() < 0.03, sprong.max()


def test_simulatie_op_datum_gebruikt_alleen_verleden(echt, hist):
    dag = int((pd.Timestamp('2018-02-14') - pd.Timestamp('1970-01-01')).days)
    ids = hist.actief_op(dag, gender='v', afstand='1500m', min_races=5)
    assert len(ids) >= 20
    v = sorted(ids, key=lambda s: hist.voorspel(s, '1500m', dag)['verwachte_tijd'])[:10]
    r = da.simuleer_race(echt, v, '1500m', dag=dag, historie=hist, n_simulaties=2000, seed=0)
    assert abs(r['Goud (%)'].sum() - 100) < 0.5
    assert r['Verwachte Tijd (s)'].between(100, 130).all()


# =============================================================================
# D. BAANVERSCHILLEN CONTROLEREN MET EEN MODELVRIJE VERGELIJKING
# =============================================================================
def _paar_effect(o, baan, ref='Heerenveen (NED)'):
    """Zelfde rijder, zelfde seizoen en afstand: mediaan van het verschil in log-tijd (%) t.o.v. de referentiebaan."""
    a = o[o['baan'].isin([baan, ref])]
    g = a.groupby(['skater_id', 'seizoen', 'afstand', 'baan'])['lgt'].mean().unstack('baan').dropna()
    if len(g) < 30:
        return np.nan, np.inf
    d = (g[baan] - g[ref]).groupby(level='skater_id').mean()
    return 100.0 * d.median(), 100.0 * 1.25 * d.std() / np.sqrt(len(d))


def test_baanverschillen_kloppen_met_zelfde_rijder_vergelijking(echt):
    o = echt.obs[['skater_id', 'afstand', 'groep', 'baan', 'dag', 'tijd', 'verdacht']].copy()
    o = o[(~o['verdacht']) & o['afstand'].isin(['500m', '1000m', '1500m', '3000m'])
          & o['groep'].isin(['junior_A', 'neo', 'senior'])]
    dat = pd.to_datetime(o['dag'], unit='D')
    o['seizoen'] = np.where(dat.dt.month >= 7, dat.dt.year, dat.dt.year - 1)
    o = o[o['seizoen'] >= 2014]
    o['lgt'] = np.log(o['tijd'])
    ev = echt.events.copy()
    ev = ev[ev['dag'] >= int((pd.Timestamp('2014-07-01') - pd.Timestamp('1970-01-01')).days)]
    med = ev.groupby('baan')['theta_post'].median()
    n_ev = ev.groupby('baan').size()
    ref = 'Heerenveen (NED)'
    rijen, slecht = [], []
    for baan in ['Calgary (CAN)', 'Salt Lake City (USA)', 'Inzell (GER)', 'Erfurt (GER)', 'Enschede (NED)',
                 'Utrecht (NED)', 'Collalbo (ITA)', 'Hamar (NOR)']:
        if baan not in med.index or n_ev.get(baan, 0) < 20:
            continue
        pe, se = _paar_effect(o, baan, ref)
        me = 100.0 * (med[baan] - med[ref])
        rijen.append((baan, round(pe, 2), round(se, 2), round(me, 2)))
        if abs(pe - me) > max(0.6, 2.5 * se):
            slecht.append((baan, pe, se, me))
    print("\nBaanverschil t.o.v. Heerenveen (%): baan, zelfde-rijder, se, model\n" + "\n".join(map(str, rijen)))
    assert len(rijen) >= 5
    assert not slecht, f"model wijkt af van de zelfde-rijder vergelijking: {slecht}"
    d = {r[0]: r[3] for r in rijen}
    for hoogte in ['Calgary (CAN)', 'Salt Lake City (USA)']:
        if hoogte in d:
            assert d[hoogte] < -0.5, f"{hoogte} hoort duidelijk sneller dan Heerenveen te zijn: {d[hoogte]}"


# =============================================================================
# E. NORMALE TIJD EN PIEKTIJD
# =============================================================================
def test_normale_tijd_en_piektijd(echt):
    n = 0
    for st in list(echt.staten.values())[::max(len(echt.staten)//35, 1)][:30]:
        for j, a in enumerate(da.AFSTANDEN):
            if st.n[j] < 5:
                continue
            v = echt.voorspel(st.id, a)
            assert v['normale_tijd'] == pytest.approx(v['verwachte_tijd'])
            assert v['piek_tijd'] < v['normale_tijd']
            assert v['piek_tijd'] == pytest.approx(v['piek_vorm_tijd'])
            assert v['piek_extreem_tijd'] < v['piek_tijd']
            assert v['piek_ijs_tijd'] < v['normale_tijd']
            verhouding = v['piek_tijd'] / v['normale_tijd']
            assert 0.75 < verhouding < 0.999, (st.naam, a, verhouding)
            n += 1
    assert n >= 20


def test_trend_voorspelling_gelijk_aan_schatting(echt):
    """De correctie in `voorspel` (alleen races vóór de dag) is gelijk aan die in de log loss; een eerder resultaat in hetzelfde evenement werkt in de verwachte richting."""
    m = echt
    if not m.params.trend_corr:
        pytest.skip('trend_corr staat uit')
    f = m.trend_f
    sel = f[f['d1_k'].notna() & f['d1_0'].isna() & f['cx'].isna()].sample(300, random_state=1)
    o = m.obs.loc[sel.index]
    offline = sel[da.TREND_KOL].fillna(0).to_numpy() @ m.trend_coef
    online = np.array([da.trend_verschuiving(m, r.skater_id, r.afstand, int(r.dag)) for r in o.itertuples()])
    assert np.abs(offline - online).max() < 1e-6
    r = o.iloc[0]
    st = m.staten[int(r.skater_id)]
    sneller = da.trend_verschuiving(m, r.skater_id, '1500m' if r.afstand != '1500m' else '3000m', int(r.dag), eerder={'500m': -2.0})
    trager = da.trend_verschuiving(m, r.skater_id, '1500m' if r.afstand != '1500m' else '3000m', int(r.dag), eerder={'500m': 2.0})
    assert sneller < trager        # een snellere eerdere afstand verschuift de verwachting naar een snellere tijd


def test_niveau_sd_factor_in_voorspelling_en_score(echt):
    """De niveau-afhankelijke sd-factor werkt in de voorspelling (voorspel) en de score (voorspelscore) in dezelfde richting en maat."""
    m = echt
    if not len(m.params.niv_knopen):
        pytest.skip('niv_knopen staat uit')
    assert max(m.params.niv_sd_fac) / min(m.params.niv_sd_fac) > 1.1
    jr = int(m.obs['jaar'].max())
    o = m.obs[(m.obs['jaar'] == jr) & (~m.obs['verdacht'])]
    x = o.groupby(['gender', 'afstand_idx'])['x_prior'].median()
    # een gemiddelde rijder (mediaan x) krijgt een lagere factor dan een zwakke staart of de toppers
    g, j = x.index[0]
    f_mid = da._niv_fac(m, g, j, float(x.iloc[0]))
    ref = np.sort(o[(o['gender'] == g) & (o['afstand_idx'] == j)]['x_prior'].to_numpy())
    f_zwak = da._niv_fac(m, g, j, float(ref[-3]))
    f_top = da._niv_fac(m, g, j, float(ref[3]))
    assert f_zwak > f_mid and f_top > f_mid
    assert abs(f_mid - np.interp(50, m.params.niv_knopen, m.params.niv_sd_fac)) < 0.06


def test_loc_schaal_voorspelling_gelijk_aan_score(echt):
    """De verschuiving/schaal per afstand x niveau in `voorspel` is dezelfde tabel als in voorspelscore,
    en de tabel bevat het bekende patroon: toppers op de 500m trager t.o.v. de basisvoorspelling dan het brede veld."""
    m = echt
    if not m.params.loc_schaal or getattr(m, 'loc_schaal_tab', None) is None:
        pytest.skip('loc_schaal staat uit')
    tab = m.loc_schaal_tab
    assert tab['loc'].shape == (2, len(da.AFSTANDEN), len(da.LS_RANDEN) + 1)
    j5, j10 = da.NAAR_INDEX['500m'], da.NAAR_INDEX['1000m']
    assert tab['loc'][0, j5, -1] > tab['loc'][0, j5, 1]          # 500m: toppers trager t.o.v. de basis dan het brede veld
    assert np.abs(tab['loc']).max() > 0.1 and (tab['sc'] >= 0.7).all() and (tab['sc'] <= 1.4).all()
    o = m.obs[(m.obs['jaar'] == m.obs['jaar'].max()) & (~m.obs['verdacht'])]
    sid = int(o[o['afstand'] == '500m'].sort_values('x_prior').iloc[0]['skater_id'])
    st = m.staten[sid]
    lo, sc = da._loc_schaal_fwd(m, st.gender, j5, float(st.x[j5]))
    assert (lo, sc) == (float(tab['loc'][0 if st.gender == 'm' else 1, j5, -1]), float(tab['sc'][0 if st.gender == 'm' else 1, j5, -1]))
    v_aan = m.voorspel(sid, '500m', dag=int(o['dag'].max()) + 1)
    p0 = m.params; m.params = replace(p0, loc_schaal=False)
    try:
        v_uit = m.voorspel(sid, '500m', dag=int(o['dag'].max()) + 1)
    finally:
        m.params = p0
    assert (v_aan['normale_tijd'] > v_uit['normale_tijd']) == (lo > 0)
    assert abs(v_aan['sigma_log'] / v_uit['sigma_log'] - sc) < 1e-9


def test_vorm_andere_afstand_gelijk_aan_schatting(echt):
    """De verrassingen op andere afstanden 1-5 en 6-14 dagen ervoor worden vooraf (trend_verschuiving) hetzelfde berekend als in de schatting."""
    m = echt
    if not m.params.trend_corr or 'x5o' not in getattr(m, 'trend_f', pd.DataFrame()).columns:
        pytest.skip('trend met andere afstanden staat uit')
    f = m.trend_f
    sel = f[f['x5o'].notna() & f['d1_0'].isna() & f['cx'].isna()]
    if len(sel) < 50:
        pytest.skip('te weinig races met een andere afstand in de 5 dagen ervoor')
    sel = sel.sample(min(200, len(sel)), random_state=2)
    o = m.obs.loc[sel.index]
    offline = sel[da.TREND_KOL].fillna(0).to_numpy() @ m.trend_coef
    online = np.array([da.trend_verschuiving(m, r.skater_id, r.afstand, int(r.dag)) for r in o.itertuples()])
    assert np.abs(offline - online).max() < 1e-5
    assert m.trend_coef[da.TREND_KOL.index('x5o')] > 0          # een snelle andere afstand kort ervoor maakt de verwachting sneller


def test_piek_kansen(echt):
    """Piekkans per rijder: tellingen vooraf in de schatting en in de voorspelling zijn gelijk, kansen tellen op tot 1,
    en een hogere piekkans maakt de piektijd sneller."""
    m = echt
    if not m.params.piek_kans or getattr(m, 'piek_f', None) is None:
        pytest.skip('piek_kans staat uit')
    pf = m.piek_f.dropna()
    o = m.obs.loc[pf.index]
    s = o.sample(300, random_state=3)
    p = m.params; p0 = np.array([p.piek_q[0], p.piek_q[1] - p.piek_q[0], 1 - p.piek_q[1]])
    for r in s.itertuples():
        c = pf.loc[r.Index, ['c0', 'c1', 'c2']].to_numpy()
        verwacht = (c + p.piek_m * p0) / (c.sum() + p.piek_m)
        assert np.allclose(da.piek_kansen(m, r.skater_id, int(r.dag)), verwacht)
    alle = np.array([da.piek_kansen(m, sid) for sid in o['skater_id'].astype(int).unique()[:500]])
    assert np.allclose(alle.sum(1), 1.0) and alle[:, 0].std() > 0.005
    sid = int(o['skater_id'].iloc[0]); af = o['afstand'].iloc[0]
    v = m.voorspel(sid, af, dag=int(m.obs['dag'].max()) + 1)
    hoog = dict(v); da._piek_tijden(m, hoog, af, np.array([0.2, 0.75, 0.05]), m.anker_std)
    laag = dict(v); da._piek_tijden(m, laag, af, np.array([0.02, 0.88, 0.10]), m.anker_std)
    assert hoog['piek_tijd'] < laag['piek_tijd'] and hoog['normale_tijd'] < laag['normale_tijd']


def test_spreiding_na_eigen_uitschieter(echt):
    """Na een race die in de eigen sd van de rijder veel sneller was dan het veld, wordt de voorspel-sd groter (vooraf gelijk aan de schatting)."""
    m = echt
    p = m.params
    if not p.trend_corr or getattr(p, 'verr_c3', 0.0) == 0 or 'efs' not in m.trend_f.columns:
        pytest.skip('verr_c3 staat uit')
    f = m.trend_f
    sel = f[(f['efs'] > 0) & f['d1_0'].isna()]
    if len(sel) < 20:
        pytest.skip('te weinig eigen uitschieters')
    sel = sel.sample(min(100, len(sel)), random_state=4)
    o = m.obs.loc[sel.index]
    offline = da._verr_factor(p, sel['ef'].to_numpy(), sel['es'].to_numpy(), sel['efo'].to_numpy(), sel['efs'].to_numpy())
    online = np.array([da.trend_schaal(m, r.skater_id, r.afstand, int(r.dag)) for r in o.itertuples()])
    assert np.abs(offline - online).max() < 1e-4
    assert (online > 1.0).all()


def test_gas_vooraf_gelijk_aan_schatting(echt):
    """De GAS-toestand vooraf (gas_toestand op de racedag) is dezelfde als in de schatting (gas_f), en de normaal-equivalente z
    heeft vrijwel geen extreme waarden bij gevestigde rijders."""
    m = echt
    if not m.params.gas or getattr(m, 'gas_f', None) is None:
        pytest.skip('gas staat uit')
    gf = m.gas_f
    o = m.obs.loc[gf.index]
    s = o[(gf['m'].abs() > 1e-4).to_numpy()].sample(300, random_state=5)
    for r in s.itertuples():
        mm, ll = da.gas_toestand(m, r.skater_id, r.afstand, int(r.dag))
        assert abs(mm - float(gf.loc[r.Index, 'm'])) < 1e-4 and abs(ll - float(gf.loc[r.Index, 'l'])) < 1e-4
    z = da.normaal_z(m, 2010)
    ok = (m.obs.loc[z.index, 'n_prior'] >= 20).to_numpy() & ~(m.obs.loc[z.index, 'z'] > 6).to_numpy()
    assert np.mean(np.abs(z[ok]) > 4) < 0.002


def test_vol_asym_piek(echt):
    """Rijders met hoge volatiliteit krijgen een relatief smalle snelle kant: snel_fac < 1 voor hen, > 1 voor stabiele rijders,
    en de piektijd gebruikt die factor."""
    m = echt
    if m.params.vol_asym_bf == 0.0 or getattr(m, 'vol_ref', None) is None:
        pytest.skip('vol_asym staat uit')
    dag = int(m.obs['dag'].max()) + 1
    o = m.obs[(m.obs['jaar'] == m.obs['jaar'].max()) & (m.obs['afstand'] == '500m')]
    ids = o['skater_id'].astype(int).unique()[:300]
    v = [m.voorspel(int(s), '500m', dag) for s in ids]
    sf = np.array([x['snel_fac'] for x in v]); sd = np.array([x['sigma_log'] for x in v])
    assert sf.min() < 1.0 < sf.max()
    assert np.corrcoef(np.log(sd), np.log(sf))[0, 1] < 0
    x = dict(v[0]); y = dict(v[0]); y['snel_fac'] = 1.0
    assert (da._twee_tijden(x, m.anker_std, m.piek_ijs())['piek_tijd'] > da._twee_tijden(y, m.anker_std, m.piek_ijs())['piek_tijd']) == (x['snel_fac'] < 1.0)


def test_vormtoestand_vooraf_gelijk_aan_schatting(echt):
    """De vormtoestand vooraf (vorm_toestand op de racedag, alleen eerdere dagen) is dezelfde als in de schatting (vorm_f),
    en na snelle uitschieters wordt de snelle kant breder."""
    m = echt
    if not m.params.vorm or getattr(m, 'vorm_f', None) is None:
        pytest.skip('vorm staat uit')
    vf = m.vorm_f; o = m.obs.loc[vf.index]; q = m.params.vorm_par
    s = o[(vf['F'] > 0.5).to_numpy()].sample(200, random_state=3)
    for r in s.itertuples():
        mu, kf, ks = da.vorm_toestand(m, r.skater_id, int(r.dag), 90.0)
        x = vf.loc[r.Index]
        mu2, lf, ls = da._vorm_eff(q, float(x.Zk), float(x.F), float(x.S), float(x.Zl), float(da._vorm_w(90.0)), float(x.N))
        assert abs(mu - mu2) < 1e-3 and abs(math.log(kf) - lf) < 1e-3 and abs(math.log(ks) - ls) < 1e-3
        assert kf > 1.0


def test_piekdiepte_vooraf_gelijk_aan_schatting(echt):
    """De piekdiepte vooraf (diepte_toestand op de racedag) volgt uit dezelfde toestand als in de schatting (diepte_f)."""
    m = echt
    if not m.params.piek_diepte or getattr(m, 'diepte_f', None) is None:
        pytest.skip('piekdiepte staat uit')
    df = m.diepte_f; o = m.obs.loc[df.index]
    s = o[(df['N'] > 1).to_numpy()].sample(200, random_state=4)
    for r in s.itertuples():
        t1 = da.diepte_toestand(m, r.skater_id, int(r.dag))
        t2 = float(da._diepte_tau(m, float(df.loc[r.Index, 'E']), float(df.loc[r.Index, 'N'])))
        assert abs(t1 - t2) < 1e-3


def test_vermoeidheid_vooraf_gelijk_aan_schatting(echt):
    """De vermoeidheidsverschuiving vooraf (verm_verschuiving, meters van de dag ervoor uit de historie) is dezelfde als in de schatting
    (verm_f), en een eerder gereden 1500 m dezelfde dag maakt de verwachting op de 1000 m trager."""
    m = echt
    if not m.params.verm or getattr(m, 'verm_f', None) is None:
        pytest.skip('verm staat uit')
    vf = m.verm_f
    zelfde = vf[['vm_s', 'vm_m', 'vm_l']].sum(axis=1).to_numpy() == 0
    gis = vf[['vg_s', 'vg_m', 'vg_l']].sum(axis=1).to_numpy() > 0
    s = m.obs.loc[vf.index[zelfde & gis]].sample(200, random_state=5)
    for r in s.itertuples():
        off = float(da._verm_top(vf.loc[[r.Index], da.VERM_KOL].to_numpy(dtype=float), np.array([85.0]), m.params.verm_top)[0] @ m.verm_coef['vooraf'])
        assert abs(da.verm_verschuiving(m, r.skater_id, r.afstand, int(r.dag), niv=85.0) - off) < 1e-5
    r = s.iloc[0]
    assert da.verm_verschuiving(m, r.skater_id, '1000m', int(r.dag) + 30, eerder={'1500m': 0.0}, niv=50.0) > da.verm_verschuiving(m, r.skater_id, '1000m', int(r.dag) + 30, niv=50.0)
