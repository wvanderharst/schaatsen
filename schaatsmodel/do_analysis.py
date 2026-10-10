# -*- coding: utf-8 -*-
"""
Schaats-ratingmodel met gedeeld dag-effect (event-effect) en piekdag-detectie.

IDEE (kort)
-----------
* Skill per rijder en afstand:  x = ln(T - T_lim)   (T_lim = fysieke ondergrens per afstand/geslacht)
* Alle 8 afstanden van een rijder vormen een vector, met een covariantiematrix P die
  aangeeft hoe zeker we zijn EN hoe de afstanden samenhangen (jouw correlatiematrix).
* Na elke race: Kalman-update. Een 500m-race verschuift ook de 1000m-schatting, maar met
  de juiste regressiecoefficient (cov/var) in plaats van 'eta * corr * fout'.
  -> een nieuwe rijder (grote P) leert snel, een bekende rijder (kleine P) blijft stabiel.
  -> tussen races groeit P (procesruis * tijd): dit vervangt de aparte 'recent-average'-truc
     en voorkomt drift.
* Volatiliteit: GARCH-achtige vermenigvuldiger per rijder/afstand (variance targeting, dus
  omega volgt uit de populatievariantie in plaats van een gok).
* Gedeeld dag-effect: elke Event_ID is een baan-dag. Alle deelnemers delen dezelfde
  ijs/weer/vorm-schok theta (in % van de tijd). theta wordt per event geschat uit de
  gemiddelde afwijking van het veld t.o.v. de voorspelling, met Bayesiaanse shrinkage naar
  het historische gemiddelde van die baan. Ratings worden bijgewerkt met de dag-gecorrigeerde tijd.
* Piekdag: een dag waarop theta significant sneller is dan normaal voor die baan (z <= -2) en
  goed gemeten; 'belangrijk' als het veld sterk en internationaal is.
* Alle categorieen (jeugd, junioren, neo's, senioren, masters) doen mee; niveau-groepen
  hebben een eigen startpunt en eigen groeitrend.
"""
import os
import sys
import math
import pickle
from dataclasses import dataclass, replace, asdict, field

import numpy as np
import pandas as pd
from scipy.stats import genextreme

# =============================================================================
# 1. CONSTANTEN EN PARAMETERS
# =============================================================================
AFSTANDEN = ['100m', '300m', '500m', '1000m', '1500m', '3000m', '5000m', '10000m']
AFSTAND_METERS = {'100m': 100, '300m': 300, '500m': 500, '1000m': 1000,
                  '1500m': 1500, '3000m': 3000, '5000m': 5000, '10000m': 10000}
NAAR_INDEX = {a: i for i, a in enumerate(AFSTANDEN)}
NA = len(AFSTANDEN)

# Alleen nog gebruikt als er te weinig data is om een limiet te schatten.
FALLBACK_LIMITS = {
    'm': {'100m': 9.5, '300m': 21.5, '500m': 32.5, '1000m': 63.0, '1500m': 97.0, '3000m': 205.0, '5000m': 345.0, '10000m': 735.0},
    'v': {'100m': 10.0, '300m': 24.0, '500m': 35.5, '1000m': 68.0, '1500m': 106.0, '3000m': 220.0, '5000m': 375.0, '10000m': 800.0},
}

JEUGD_GROEPEN = ('jeugd', 'junior_C', 'junior_B')
GROEPEN = ['jeugd', 'junior_C', 'junior_B', 'junior_A', 'neo', 'senior',
           'masters_jong', 'masters_oud', 'onbekend']
DAGEN_PER_JAAR = 365.25
GROEI_KNOOP = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.6, 2.5, 4.0])   # tijd sinds laatste race (jaar)


@dataclass(frozen=True)
class Params:
    """Alle instelbare variabelen. Waarden zijn startwaarden; tune_parameters() schat ze bij."""
    # fysieke ondergrens
    t_lim_bron: str = 'record'      # 'top3': 3e beste tijd van verschillende rijders; 'record': wereldrecord (snelste bevestigde tijd)
    t_lim_marge: float = 0.01      # T_lim = referentie * (1 - marge); referentie is top3 of het record
    # rating-dynamiek
    q: float = 0.02                # procesruis per jaar (fractie van de populatievariantie)
    q_mix: float = 0.5             # correlatie van procesruis tussen afstanden (0=onafhankelijk, 1=zoals populatie)
    r_schaal: float = 0.6          # schaal op de gemeten race-ruis (na dag-correctie is die kleiner)
    p0_schaal: float = 1.0         # onzekerheid van een nieuwe rijder t.o.v. populatie
    garch_alpha: float = 0.08      # reactie op verrassing
    garch_beta: float = 0.88       # geheugen van volatiliteit
    vorm_ou_sd: float = 0.0        # kortetermijnvorm in de toestand: stationaire sd (ln-tijd) van een OU-proces gedeeld over afstanden (0 = uit)
    vorm_ou_tau: float = 10.0      # vervaltijd (dagen) van de vorm
    qvol_a: float = 0.0            # adaptieve niveau-volatiliteit per rijder: h += a * z_t * z_(t-1) (zelfde teken = echte verschuiving); 0 = uit
    qvol_lam: float = 0.95         # verval van h per race
    qvol_jaar: float = 0.5         # verval van h per jaar zonder races
    qvol_max: float = 2.5          # grenzen voor h (Q x e^h)
    qvol_min: float = -1.0
    sprong_m: float = 0.0          # eigen sprongkans per rijder: (som posterieure sprongkansen + m*pi) / (aantal races + m); 0 = voor iedereen gelijk
    sprong_pi_snel: float = 0.0    # sprongmenging in de toestand: kans dat een SNELLE verrassing een echte niveausprong is (0 = uit)
    sprong_pi_traag: float = 0.0   # idem trage kant (valpartijen tellen niet mee)
    sprong_k: float = 3.0          # sprongvariantie = sprong_k^2 x de voorspelvariantie S van die race
    dip_pi: float = 0.15           # aandeel 'slechte dag / valpartij' races (trager dan verwacht, meer spreiding); 0 = uit
    dip_a: float = 2.0             # gemiddelde vertraging van zo'n race, in standaarddeviaties van de voorspelling
    dip_b: float = 4.0             # extra variantie van zo'n race, als veelvoud van de gewone voorspelvariantie
    fr_pi: float = 0.0             # >0: eenzijdig 'frontier'-model: normale ruis + alleen-trage slechte dag (half-normaal); vervangt de dip-menging
    fr_c: float = 4.0              # schaal van de half-normale slechte dag in sd-eenheden
    fr_lam: float = 0.0            # aandeel van de update dat een slechte dag nog meeneemt (0 = valpartij/slechte dag telt helemaal niet)
    fr_snel_boost: float = 1.0     # >1: versterkt de update na een snelle tijd (e<0) in frontier-modus
    fr_lam_cap: float = 2.0        # de slechte-dag-update telt hoogstens zo veel sd mee: een valpartij van 10 sd verschuift niets extra
    fr_max_run: int = 2            # na zoveel opeenvolgende slechte dagen telt een trage tijd weer mee (echte verandering)
    fr_boost_eigen: bool = True    # boost alleen op de eigen afstand (niet op de overloop naar andere afstanden)
    dip_cap: float = 0.0           # >0: de slechte-dag-update telt hoogstens zo veel sd mee (valpartij van 10 sd verschuift niets extra)
    dip_val_z: float = 0.0         # >0: een trage race met z > dip_val_z is een valpartij/uitschieter en telt NIET mee in de update (ook niet geklemd)
    snel_boost: float = 1.0        # >1: versterkt de update na een snelle tijd (e<0) in het dip-model
    snel_run_z: float = 0.0        # >0: twee snelle uitschieters (z < -snel_run_z) kort na elkaar op dezelfde afstand = echte vormsprong:
    snel_run_dagen: int = 60       #     de tweede race telt dan volledig (geen Huber-grens) en de toestand wordt onzekerder
    snel_run_enkel: bool = False   # True: de volatiliteitsschok al bij EEN snelle uitschieter (niet pas bij de tweede)
    snel_run_modus: str = 'niveau'
    kap_snel_vol: bool = True      # False: snelle uitschieters tellen volledig mee in de volatiliteit (z^2 niet afgekapt op 9); trage wel
    kap_snel_niveau: bool = True   # False: geen Huber-grens voor snelle uitschieters in de niveau-update; trage houden hun grens # 'niveau': toestand onzeker (niveau springt); 'vol': alleen de volatiliteit (GARCH) krijgt de volle schok
    pred_sd_schaal: dict = field(default_factory=dict)   # per afstand: factor op de voorspel-sd (alleen voorspelling/simulatie/score, niet de update)
    niv_knopen: tuple = ()        # niveau-percentielen (0-100, hoog = beter) waarop sd-factor en locatie-verschuiving gelden (alleen voorspelling/score)
    niv_sd_fac: tuple = ()        # factor op de voorspel-sd bij elk knooppunt (lineair geinterpoleerd)
    niv_rel: bool = False         # True: niveau-percentiel binnen de eigen leeftijdsgroep (relatieve toppers) in plaats van over alle rijders
    baan_alias: dict = field(default_factory=dict)   # baan -> andere baan waarvan het baanverschil wordt overgenomen (bv. een nieuwe baan met te weinig wedstrijden)
    loc_schaal: bool = False      # verschuiving + schaal van de voorspelverdeling per geslacht x afstand x niveauband (alleen voorspelling)
    loc_schaal_jaren: int = 4     # geschat op de laatste zoveel jaren (de afwijkingen verschuiven langzaam, bv. 1500m rond 2021)
    loc_schaal_tot_jaar: int = 0  # >0: alleen jaren <= dit jaar gebruiken (eerlijk toetsen)
    loc_schaal_k: float = 200.0   # krimp van een niveauband naar het afstandsgemiddelde (pseudo-rijen)
    niv_rel_groepen: tuple = ()   # leeftijdsgroepen die een eigen sd-factor op het RELATIEVE niveau (binnen de groep) krijgen
    niv_rel_knopen: tuple = ()
    niv_rel_sd_fac: tuple = ()
    niv_loc: tuple = ()           # verschuiving van de voorspelling in sd-eenheden bij elk knooppunt (positief = tragere verwachting)
    snel_geh_dagen: int = 0        # >0: na een snelle race (z<-snel_geh_z) tellen tragere races zoveel dagen minder zwaar mee (kon niet piekken)
    snel_geh_z: float = 1.5
    snel_geh_demp: float = 0.5     # gain-factor voor tragere races binnen het geheugenvenster
    sim_z: str = 'normaal'         # 'empirisch': simulatie trekt de gestandaardiseerde fout uit de werkelijke (scheve, zware) verdeling
    dip_run: int = 0               # >0: na zoveel opeenvolgende slechte dagen telt een trage tijd weer gewoon mee
    fr_huber_snel: float = 6.0     # Huber-grens voor te SNELLE tijden in frontier-modus (snelle tijden zijn informatief)
    dag_c_op: float = 0.0          # >0: bovengrens (sd) voor trage residuen bij het schatten van het dag-effect; 0 = symmetrisch
    huber_c_langzaam: float = 0.0  # >0: eigen (strengere) Huber-grens voor te trage tijden; 0 = zelfde als huber_c
    ervaring_bias: bool = False    # drift per (groep, aantal eerdere races op de afstand): debuut op een afstand en leercurve
    dag_comp_c: float = 0.0        # >0: events waarvan het veld (robuust) meer dan zoveel sd's afwijkt krijgen post hoc een extra dag-effect bovenop de baancorrectie
    dag_comp_lambda: float = 0.8   # aandeel van de gemeten afwijking dat wordt gecompenseerd
    dag_comp_baan_k: float = 0.0   # >0: normale range per baan (robuuste sd van S op die baan), gekrompen naar het totaal met zoveel pseudo-events; 0 = één range voor alle banen
    dag_comp_min_rijen: int = 40   # minimaal aantal rijders (met voorgeschiedenis) in het evenement om te corrigeren
    trend_corr: bool = False       # voorspellingscorrectie op de laatste verrassingen (momentum) en de helling van de toestand
    verr_c: float = 0.0            # na een grote snelle verrassing (z<-2, zelfde of andere afstand): sd x (1 + verr_c * overschot)
    verr_c2: float = 0.0           # na een grote trage verrassing (z>+2, geen val): sd x (1 + verr_c2 * overschot)
    verr_drempel: float = 2.0      # vanaf hoeveel sd (t.o.v. het veld) een verrassing 'groot' is
    verr_c3: float = 0.0           # extra sd-factor per sd overschot als de vorige race in EIGEN sd sneller was dan -verr_eigen_drempel
    verr_eigen_drempel: float = 2.5
    verr_eigen: bool = False       # 'groot' meten in de EIGEN sd van de rijder (zonder ijs-onzekerheid): toppers hebben een kleine eigen sd
    trend_ander: bool = True       # trend: ook verrassingen op ANDERE afstanden in andere evenementen 1-5 en 6-14 dagen ervoor (vorm loopt door)
    vol_asym_bf: float = 0.0      # volatiliteit per rijder is vooral TRAAG: snelle kant schaalt met (relatieve volatiliteit)^(1+bf), bv. bf=-0,2
    ijs_dag: bool = False          # dag-relatief (t.o.v. het geschatte dag-effect): correctie (ijs - HV) x k(niveau) + hoogland x h(niveau)
    ijs_dag_par: tuple = (0.0,) * 12  # k op niveau 0/50/75/90/97/100, dan h (in % van de tijd) op dezelfde knopen
    piek_diepte: bool = False      # piekdiepte per rijder: in de piekzone wordt de afstand tot de drempel geschaald met tau_i (lang geheugen, gekrompen)
    piek_diepte_par: tuple = (1460.0, 20.0, 1.5)   # vervaltijd (dagen), krimp m (aantal piekraces), exponent kappa
    staart_q: float = 0.0          # eerlijke log loss / dag-z: snelste fractie apart als GPD-staart (body en staart apart gefit); 0 = één twee-delige t
    staart_niv: tuple = (90.0, 97.0)   # niveaugrenzen voor een eigen staart (breedte, toppers, elite)
    schaal: bool = False           # eigen schaal per rijder met lang geheugen, per kant: vervallen gemiddelde dag-z^2 van eerdere dagen (snel / traag)
    schaal_par: tuple = (0.0, 0.0, 1460.0, 30.0)   # gamma_snel, gamma_traag (exponent op de gekrompen ratio), vervaltijd (dagen), krimp m (ritten)
    staart: bool = False           # snelle staart: piekdiepte tau ook naar niveau (top) en naar de eigen frequentie van extreme ritten (lang geheugen)
    staart_par: tuple = (0.0, 0.0, 1460.0, 300.0, -3.0)   # a_top, b (exponent eigen frequentie), vervaltijd (dagen), krimp m (ritten), drempel dag-z
    verm: bool = False             # vermoeidheid: verschuiving (sd) door eerder gereden meters dezelfde dag en de dag ervoor (vooraf absoluut, dag t.o.v. het veld)
    verm_top: bool = False         # vermoeidheid ook aan de top (eigen coëfficiënten); uit = alleen de breedte, gewicht 1 - w(niveau)
    verm_dag: bool = False         # vermoeidheid ook dag-relatief (binnen de race gecentreerd); klein getoetst: geen winst, elite iets slechter
    vorm_plateau: float = 0.0      # dagen waarin de korte vorm (Zk) niet uitdooft, daarna exp(-(leeftijd - plateau) / Hk); 0 = gewone uitdooiing
    vorm: bool = False             # vormtoestand op dag-z (t.o.v. het veld): kort = vorm houdt aan (sterker bij toppers), lang = volatiliteit na uitschieters
    vorm_par: tuple = (3.0, 2.0, 240.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # Hk, c, Hl, cs, aK0, aK1, aL0, aL1, a, aS, bf, bs, bsS, bf1
    gas: bool = False              # GAS-laag (score-gedreven, scheve score) op de voorspelfout: verschuiving + log-schaal per rijder (kort) en per rijder x afstand (lang)
    gas_par: tuple = (0.2275, 0.2399, 0.0503, 0.0021, 0.0159, 0.0222, 0.9774, 0.5693, 0.0199, 0.0222, 0.0011, 0.0037)
    piek_kans: bool = False        # eigen kans per rijder op een piek (snelste 7%) en een slechte dag (traagste 7%), uit zijn eerdere races
    piek_m: float = 80.0           # krimp naar de populatie (pseudo-races)
    piek_q: tuple = (0.07, 0.93)   # grenzen van de gebieden piek | normaal | slechte dag (kwantielen van de voorspelfout)
    sd_factoren: bool = False      # schaalfactor op de voorspel-sd per (pauze, ervaring, afstandsgroep)
    split_jaar: int = 2019         # nieuwe correctietabellen worden alleen op jaren <= split_jaar geschat (eerlijke toets daarna)
    huber_c_update: float = 0.0    # >0: eigen Huber-grens voor de rating-update (1e9 = geen grens); 0 = zelfde als huber_c. Het dag-effect blijft op huber_c winsoriseren
    huber_c: float = 3.0           # grotere fouten (val, typefout) tellen minder mee
    verdacht_factor: float = 9.0   # ruisvariantie x9 voor rijen met gecorrigeerde afstand
    max_dt_jaar: float = 4.0       # maximale tijdsprong voor onzekerheidsgroei
    groei_schaal: float = 0.0      # schaal op de geschatte groeicurve (0 = geen leeftijdscorrectie; verandering wordt via q_groep opgevangen)
    q_groep: dict = field(default_factory=dict)  # vermenigvuldiger op procesruis per niveaugroep (snelheid van bijwerken)
    # dag-effect (in fractie van de tijd, 0.004 = 0.4%)
    tau_dag: float = 0.004         # normale dag-tot-dag spreiding op een baan
    tau_baan: float = 0.010        # spreiding tussen baan-gemiddelden
    tau_groep: float = 0.0025      # extra spreiding per afstand binnen een dag
    dag_gewicht_jeugd: float = 0.6 # extra gewicht-factor van jeugdgroepen bij het schatten van het dag-effect
    betrouwbaarheid: bool = True   # False = alle rijders gelijk wegen (ruisfactor per groep en gewichten uit)
    baan_halfwaarde_jaar: float = 3.0
    baan_shrink: float = 2.0
    baan_structuur: bool = True    # baanverschillen globaal schatten (zelfde rijder/seizoen op meerdere banen) en als prior gebruiken
    baan_era_seizoenen: int = 3    # de baanverschillen mogen per blok van zoveel seizoenen anders zijn
    baan_lek: float = 0.016        # extra ruis op de gemeten veld-effecten: rating-aanpassing 'lekt' baaneffect weg uit de residuen
    baan_se_floor: float = 0.004   # minimale onzekerheid (fractie) van een globaal geschat baanverschil
    niveau_shrink: float = 1e6     # precisie-equivalent waarmee het algemene niveau naar 0 getrokken wordt
    niveau_halfwaarde_dagen: float = 45.0   # geheugen van het algemene tijdsniveau (alle banen samen)
    # piekdag-detectie
    piek_z: float = 2.0
    piek_max_se: float = 0.005     # dag-effect moet met se <= 0.5% gemeten zijn
    elite_pct: float = 98.0
    min_elite: int = 4
    min_landen: int = 3


# =============================================================================
# 2. DATA INLEZEN EN OPSCHONEN
# =============================================================================
def bepaal_groep(cat):
    """Leidt een niveau/leeftijdsgroep af uit de categoriecode (bijv. MC2, LYA, MN3, MSA, M45)."""
    if cat is None or (isinstance(cat, float) and np.isnan(cat)):
        return 'onbekend'
    c = str(cat).strip().upper()
    if len(c) < 2:
        return 'onbekend'
    r = c[1:]
    if r == '?' or r == '':
        return 'onbekend'
    if r[0] == 'Y':
        return 'jeugd'
    if r[0] in 'ABC' and len(r) == 2 and r[1].isdigit():
        return 'junior_' + r[0]
    if r[0] == 'N':
        return 'neo'
    if r[0] == 'S':
        return 'senior'
    if r.isdigit():
        return 'masters_oud' if int(r) >= 55 else 'masters_jong'
    return 'onbekend'


def _normaliseer_geslacht(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {'m', 'man', 'male', 'mens', 'm.'}:
        return 'm'
    if s in {'v', 'vrouw', 'female', 'f', 'woman', 'women'}:
        return 'v'
    return np.nan


def vind_fout_velden(df, min_n=8, drempel=0.20):
    """Event/geslacht/categorie/afstand-velden waarvan de tijden niet passen bij dezelfde rijders op 500m in dat event."""
    ref = df[df['afstand'] == '500m'].groupby(['event_id', 'skater_id'], observed=True)['tijd'].min().rename('t500')
    d = df[df['afstand'].isin(['1000m', '1500m', '3000m', '5000m', '10000m'])]
    d = d.groupby(['event_id', 'skater_id', 'gender', 'categorie', 'afstand', 'groep'], observed=True)['tijd'].min().reset_index()
    d = d.join(ref, on=['event_id', 'skater_id']).dropna(subset=['t500'])
    d['lr'] = np.log(d['tijd'] / d['t500'])
    v = d.groupby(['event_id', 'gender', 'categorie', 'afstand', 'groep'], observed=True)['lr'].agg(['median', 'size']).reset_index()
    v = v[v['size'] >= min_n]
    verw = v.groupby(['gender', 'groep', 'afstand'], observed=True)['median'].transform('median')
    v['afw'] = v['median'] - verw
    return v[v['afw'].abs() > drempel]


def vind_inconsistente_resultaten(df, rij_adult=0.85, rij_jeugd=0.82, veld_drempel=0.20, min_veld=6, venster_dagen=450):
    """Resultaten die niet passen bij de eigen buurresultaten van de rijder (zelfde afstand, +-2 races, binnen ~15 maanden).
    * rij-niveau: alleen TE SNEL (een snelle fout is onmogelijk; trage fouten zoals valpartijen vangt het model zelf op)
    * veld-niveau: de mediaan van de verhoudingen in een event/geslacht/afstand wijkt meer dan veld_drempel (log) af.
    Geeft (maskers_rij, maskers_veld) als boolean arrays in de volgorde van df."""
    lang = df['afstand'].isin(['500m', '1000m', '1500m', '3000m', '5000m', '10000m']).to_numpy() & (~df['verdacht'].to_numpy())
    idx = np.flatnonzero(lang)
    d = pd.DataFrame({'i': idx, 's': df['skater_id'].to_numpy()[idx], 'a': df['afstand'].astype(str).to_numpy()[idx],
                      'dag': (df['datum'] - pd.Timestamp('1970-01-01')).dt.days.to_numpy()[idx],
                      't': df['tijd'].to_numpy()[idx], 'ev': df['event_id'].to_numpy()[idx],
                      'g': df['gender'].astype(str).to_numpy()[idx],
                      'kind': df['groep'].isin(JEUGD_GROEPEN).to_numpy()[idx]})
    d = d.sort_values(['s', 'a', 'dag', 'i'], kind='stable').reset_index(drop=True)
    grp = (d['s'].astype(np.int64) * 16 + d['a'].map(NAAR_INDEX).astype(np.int64)).to_numpy()
    cols = []
    for sh in (-2, -1, 1, 2):
        t_sh = d['t'].shift(sh).to_numpy()
        ok = (pd.Series(grp).shift(sh).to_numpy() == grp) & (np.abs(d['dag'].shift(sh).to_numpy() - d['dag'].to_numpy()) <= venster_dagen)
        cols.append(np.where(ok, t_sh, np.nan))
    nb = np.column_stack(cols)
    n_nb = np.sum(~np.isnan(nb), axis=1)
    with np.errstate(all='ignore'):
        med = np.nanmedian(np.where(n_nb[:, None] >= 3, nb, np.nan), axis=1)
    d['r'] = d['t'].to_numpy() / med
    with np.errstate(all='ignore'):
        r_min = d['t'].to_numpy() / np.nanmin(np.where(n_nb[:, None] >= 3, nb, np.nan), axis=1)    # t.o.v. de SNELSTE buur
    rij = (r_min < np.where(d['kind'], rij_jeugd, rij_adult)) & np.isfinite(r_min)   # sneller dan alle buren: andere (trage) foute buren tellen niet mee
    v = d[np.isfinite(d['r'].to_numpy())].groupby(['ev', 'a', 'g'])['r'].agg(['median', 'size']).reset_index()
    v = v[(v['size'] >= min_veld) & (np.abs(np.log(v['median'])) > veld_drempel)]
    veld_rij = d.merge(v[['ev', 'a', 'g']].assign(_f=1), how='left', on=['ev', 'a', 'g'])['_f'].notna().to_numpy()
    m_rij = np.zeros(len(df), dtype=bool); m_veld = np.zeros(len(df), dtype=bool)
    m_rij[d['i'].to_numpy()[rij]] = True
    m_veld[d['i'].to_numpy()[veld_rij]] = True
    return m_rij, m_veld


EXTRA_ATTR = ('trend_f', 'piek_f', 'gas_f', 'gas_dag', 'trend_reeks', 'piek_reeks', 'vorm_f', 'vorm_dag', 'diepte_f', 'diepte_dag', 'verm_f', 'verm_reeks',
              'staart_f', 'staart_dag', 'schaal_f', 'schaal_dag')


def bewaar_veilig(model, pad):
    """Veilig opslaan in twee bestanden: het model zelf en de grote afgeleide tabellen (pad + '.extra').
    Elk eerst naar een tijdelijk bestand en pas daarna vervangen: een afgebroken schrijfactie laat de oude versie heel."""
    import gc, pickle
    extra = {a: getattr(model, a, None) for a in EXTRA_ATTR}
    for a in EXTRA_ATTR:
        setattr(model, a, None)
    gc.collect()
    try:
        for p_, obj in ((pad + '.extra', extra), (pad, model)):
            with open(p_ + '.tmp', 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(pad + '.extra.tmp', pad + '.extra')
        os.replace(pad + '.tmp', pad)
    finally:
        for a, v in extra.items():
            setattr(model, a, v)


def laad_volledig(pad, extra=True):
    """Laadt het model en (indien aanwezig en gevraagd) de afgeleide tabellen uit pad + '.extra'."""
    import pickle
    with open(pad, 'rb') as f:
        model = pickle.load(f)
    if extra and os.path.exists(pad + '.extra'):
        with open(pad + '.extra', 'rb') as f:
            for a, v in pickle.load(f).items():
                if getattr(model, a, None) is None:
                    setattr(model, a, v)
    return model


def data_csv(map_=None):
    """Volledige dataset: de versie met racedatum per afstand als die er is, anders de oude."""
    map_ = map_ or os.path.dirname(os.path.abspath(__file__))
    for n in ('speedskating_historie_racedatum.csv', 'speedskating_historie_clean.csv'):
        p = os.path.join(map_, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError('geen speedskating_historie_racedatum.csv of speedskating_historie_clean.csv naast do_analysis.py')


def laad_dataset(bestandsnaam, verbose=True, max_rijen=None):
    """Leest de CSV (ook de volledige 4,3 miljoen rijen) geheugenzuinig, geeft uniforme kolomnamen en ruimt op.

    Er worden geen rijen weggegooid omdat ze 'traag' of 'ongebruikelijk' zijn: recreatieve
    en jeugdrijders zijn even belangrijk. Alleen fysiek onmogelijke rijen gaan eruit;
    rijen met een gecorrigeerde afstand blijven erin maar krijgen een lager gewicht.
    """
    kop = pd.read_csv(bestandsnaam, nrows=0).columns
    orig = {str(c).strip().lower(): c for c in kop}
    nodig = ['event_id', 'datum', 'race_datum', 'baan', 'atleet_id', 'naam', 'categorie', 'land', 'geslacht', 'afstand',
             'tijd_sec', 'afstand_corrigeerd']
    gebruik = [orig[k] for k in nodig if k in orig]
    cat_kol = [orig[k] for k in ['baan', 'naam', 'categorie', 'land', 'geslacht', 'afstand'] if k in orig]
    df = pd.read_csv(bestandsnaam, usecols=gebruik, dtype={c: 'category' for c in cat_kol}, nrows=max_rijen)
    df.columns = [str(c).strip().lower() for c in df.columns]
    n0 = len(df)

    df = df.rename(columns={'atleet_id': 'skater_id', 'naam': 'skater_name', 'tijd_sec': 'tijd'})
    d = pd.to_datetime(df['datum'], format='%d/%m/%Y', errors='coerce')
    if d.isna().mean() > 0.5:
        d = pd.to_datetime(df['datum'], errors='coerce', dayfirst=True)
    df['datum'] = d
    df['tijd'] = pd.to_numeric(df['tijd'], errors='coerce')
    df['afstand'] = df['afstand'].astype(str).str.strip()
    gmap = {c: _normaliseer_geslacht(c) for c in df['geslacht'].cat.categories}
    df['gender'] = df['geslacht'].map(gmap).astype(object)
    df = df.drop(columns=['geslacht'])
    df = df[df['gender'].isin(['m', 'v']) & df['datum'].notna() & df['tijd'].notna()
            & df['afstand'].isin(AFSTANDEN) & (df['tijd'] > 0)].copy()
    df = df[df['datum'].dt.year.between(1990, 2100)].copy()          # onmogelijke datums (bijv. 01/01/1500)

    # door de gebruiker afgekeurde races (vermoedelijke datafouten: onmogelijk snel t.o.v. de verwachting); bestand naast de data
    _uit = os.path.join(os.path.dirname(os.path.abspath(bestandsnaam)), 'uitgesloten_races.csv')
    if os.path.exists(_uit):
        u_ = pd.read_csv(_uit)
        sleutel = set(zip(u_['event_id'].astype(np.int64), u_['skater_id'].astype(np.int64), u_['afstand'].astype(str), u_['tijd'].round(2)))
        k_ = list(zip(df['event_id'].astype(np.int64), df['skater_id'].astype(np.int64), df['afstand'].astype(str), df['tijd'].round(2)))
        weg = np.fromiter((x in sleutel for x in k_), bool, len(k_))
        if verbose:
            print(f"  {int(weg.sum())} afgekeurde races verwijderd ({_uit})")
        df = df[~weg].copy()

    # echte racedatum per afstand (kolom Race_Datum, zie scraper_fix/backfill.py): een toernooi van meerdere dagen wordt
    # gesplitst in één event per dag, zodat vorm, dag-effect en alle 'alleen eerdere dagen'-lagen de juiste volgorde zien.
    if 'race_datum' in df.columns:
        rd = pd.to_datetime(df['race_datum'], format='%d/%m/%Y', errors='coerce')
        # de racedatum is leidend (de oude startdatum was bij toernooien over een maandgrens soms de einddatum);
        # alleen onleesbare of onmogelijke racedatums vallen terug op de toernooidatum
        rd = rd.where(rd.notna() & rd.dt.year.between(1990, 2100), df['datum'])
        off = (rd - rd.groupby(df['event_id']).transform('min')).dt.days.astype(int)
        if verbose:
            print(f"  racedatum: {100 * (off > 0).mean():.1f}% van de races ligt na de startdag van het toernooi")
        df['toernooi_id'] = df['event_id']
        df['event_id'] = df['event_id'].astype(np.int64) + 1_000_000 * off.to_numpy()      # geen botsing met bestaande ids
        df['datum'] = rd
        df = df.drop(columns=['race_datum'])

    # snelheid-check: fysiek onmogelijk (> 16 m/s) of absurd langzaam (< 1.5 m/s)
    snelheid = df['afstand'].map(AFSTAND_METERS).to_numpy(dtype=float) / df['tijd'].to_numpy()
    df = df[(snelheid <= 16.0) & (snelheid >= 1.5)].copy()

    # een rijder heeft precies 1 geslacht: neem het meest voorkomende
    vrouw = (df['gender'] == 'v').astype(float)
    aandeel_v = vrouw.groupby(df['skater_id']).transform('mean')
    df['gender'] = np.where(aandeel_v > 0.5, 'v', 'm')

    if 'afstand_corrigeerd' in df.columns:
        df['verdacht'] = df['afstand_corrigeerd'].astype(str).str.lower().isin(['true', '1']).to_numpy()
        df = df.drop(columns=['afstand_corrigeerd'])
    else:
        df['verdacht'] = False
    if 'categorie' in df.columns:
        cmap = {c: bepaal_groep(c) for c in df['categorie'].cat.categories}
        df['groep'] = df['categorie'].map(cmap).astype(object)
    else:
        df['groep'] = 'onbekend'
    # onmogelijk snelle tijden (typefouten, verkeerde afstand in een heel event):
    #  (a) >5% sneller dan de 10e snelste beste tijd van verschillende volwassen rijders (niet-jeugdgroepen)
    #  (b) jeugdgroepen (jeugd/junior C/junior B) die sneller zijn dan de 5e snelste volwassene
    onmogelijk = np.zeros(len(df), dtype=bool)
    kind = df['groep'].isin(JEUGD_GROEPEN).to_numpy()
    for (g, a_), idx in df[~df['verdacht']].groupby(['gender', 'afstand']).groups.items():
        sub_t = df.loc[idx, ['skater_id', 'tijd']]
        volw = sub_t[~kind[df.index.get_indexer(sub_t.index)]]
        beste = volw.groupby('skater_id')['tijd'].min().sort_values().values
        if len(beste) < 30:
            continue
        pos = df.index.get_indexer(sub_t.index)
        t = sub_t['tijd'].to_numpy()
        fout = t < 0.95 * beste[min(9, len(beste) - 1)]
        fout |= kind[pos] & (t < beste[min(4, len(beste) - 1)])
        onmogelijk[pos[fout]] = True
    # bekende fout-gelabelde velden (handmatig vastgesteld): hele afstand van een event weggooien
    bekend_fout = ((df['datum'] == pd.Timestamp('2024-02-10')) & df['baan'].astype(str).str.contains('Tomasz', na=False)
                   & (df['gender'] == 'v') & (df['afstand'] == '3000m')).to_numpy()
    # event-niveau: velden waarvan de tijden niet passen bij dezelfde rijders op de 500m van dat event
    veld_fout = np.zeros(len(df), dtype=bool)
    ff = vind_fout_velden(df)
    if len(ff):
        sl = ['event_id', 'gender', 'categorie', 'afstand']
        kand = np.flatnonzero(df['event_id'].isin(ff['event_id']).to_numpy())
        sub_k = df.iloc[kand][sl].astype({'categorie': str, 'afstand': str, 'gender': str})
        sub_k = sub_k.merge(ff[sl].astype({'categorie': str, 'afstand': str, 'gender': str}).assign(_f=1), how='left', on=sl)
        veld_fout[kand] = sub_k['_f'].notna().to_numpy()
    # per-rijder consistentie: te snelle resultaten t.o.v. de eigen buurresultaten + hele velden die niet passen
    inc_rij, inc_veld = vind_inconsistente_resultaten(df)
    # handmatig vastgesteld: de 3000m-reeks van Mateusz WEŁNICKI (jeugd) voor 2024-06 bevat verkeerd gelabelde afstanden
    welnicki = (df['skater_name'].astype(str).str.contains('WEŁNICKI', na=False) & (df['afstand'] == '3000m')
                & (df['datum'] < pd.Timestamp('2024-06-01'))).to_numpy() if 'skater_name' in df.columns else np.zeros(len(df), bool)
    weg = onmogelijk | bekend_fout | veld_fout | inc_rij | inc_veld | welnicki
    if verbose:
        print(f"{int(inc_rij.sum())} resultaten te snel t.o.v. eigen buurresultaten, {int(inc_veld.sum())} rijen in inconsistente velden, {int(welnicki.sum())} Wełnicki-rijen verwijderd")
        print(f"{int(onmogelijk.sum())} onmogelijk snelle tijden + {int(bekend_fout.sum())} rijen handmatig + {int(veld_fout.sum())} rijen in {len(ff)} velden met afwijkende tijdsverhouding verwijderd")
    df = df[~weg].reset_index(drop=True)
    for kol in ['baan', 'land']:
        if kol not in df.columns:
            df[kol] = ''
        df[kol] = df[kol].astype(object).where(df[kol].notna(), '').astype(str)
    if 'skater_name' not in df.columns:
        df['skater_name'] = ''
    if 'categorie' not in df.columns:
        df['categorie'] = ''
    df['jaar'] = df['datum'].dt.year.astype(np.int16)
    df['seizoen'] = np.where(df['datum'].dt.month >= 7, df['datum'].dt.year, df['datum'].dt.year - 1).astype(np.int16)
    df['dag'] = ((df['datum'] - pd.Timestamp('1970-01-01')).dt.days).astype(np.int32)
    df['afstand_idx'] = df['afstand'].map(NAAR_INDEX).astype(np.int8)
    if 'event_id' not in df.columns:
        df['event_id'] = df.groupby(['datum', 'baan']).ngroup()
    df = df.sort_values(['dag', 'event_id', 'afstand_idx', 'gender'], kind='stable').reset_index(drop=True)
    if verbose:
        print(f"Dataset: {len(df)} van {n0} rijen bruikbaar, {df['skater_id'].nunique()} rijders, "
              f"{df['event_id'].nunique()} events, {df['datum'].min().date()} .. {df['datum'].max().date()}")
    return df


def sample_rijders(df, fractie=0.12, seed=0):
    """Willekeurige selectie van HELE rijders (met hun volledige geschiedenis). Bedoeld voor snelle tuning."""
    ids = df['skater_id'].unique()
    rng = np.random.default_rng(seed)
    gekozen = set(rng.choice(ids, size=int(len(ids) * fractie), replace=False))
    return df[df['skater_id'].isin(gekozen)].reset_index(drop=True)


# =============================================================================
# 3. FYSIEKE LIMIET T_lim
# =============================================================================

STANDAARD_BAAN = 'Heerenveen (NED)'   # referentiebaan voor 'normale tijd', 'piektijd' en simulaties
_LAATSTE_RECORDS = {'m': {}, 'v': {}}


def evt_ondergrens(pb_tijden, k=100, n_boot=60, seed=1):
    """EVT (Einmahl & Magnus 2008): Generalized Pareto op de snelste k persoonlijke records.
    Geeft (eindpunt, ondergrens_CI5%, geldig_fractie); NaN als onstabiel (xi >= -0.02 of te weinig data)."""
    from scipy.stats import genpareto
    t = np.sort(np.asarray(pb_tijden, float))
    if len(t) < max(300, 3 * k):
        return np.nan, np.nan, 0.0
    def eind(x):
        x = np.sort(x); u = x[k]; y = u - x[:k]
        c, _, sc = genpareto.fit(y, floc=0)
        return u - sc / (-c) if c < -0.02 else np.nan
    e0 = eind(t)
    rng = np.random.default_rng(seed)
    bs = [eind(rng.choice(t, size=len(t), replace=True)) for _ in range(n_boot)]
    bs = np.array([b for b in bs if np.isfinite(b)])
    geldig = len(bs) / n_boot
    if not np.isfinite(e0) or geldig < 0.8:
        return e0, np.nan, geldig
    return e0, float(np.quantile(bs, 0.05)), geldig

def bereken_terminal_limits(df, afstanden=AFSTANDEN, marge=0.03, verbose=True, bron='top3'):
    """Schat T_lim per geslacht/afstand.

    * 'robuust_beste' = de 3e snelste tijd van verschillende rijders (1 typefout in de staart
      mag de limiet niet bepalen; rijen met gecorrigeerde afstand tellen niet mee).
    * GEV op seizoensminima, MET GECORRIGEERD TEKEN: voor genextreme.fit(-T) is het
      eindpunt van T gelijk aan -(loc + scale/c) (bij c>0). Het oude -(loc - scale/c) lag
      altijd bij of boven de snelste tijd, waardoor de sanity check altijd terugviel.
    * GEV wordt alleen als diagnose getoond; de gebruikte limiet is robuust_beste*(1-marge),
      omdat de GEV-fit op zulke korte staarten instabiel is. De marge wordt getuned.
    """
    limieten = {'m': {}, 'v': {}}
    global _LAATSTE_RECORDS
    _LAATSTE_RECORDS = {'m': {}, 'v': {}}
    diag = []
    for g in ['m', 'v']:
        dg = df[(df['gender'] == g) & (~df['verdacht'])]
        for dist in afstanden:
            s = dg[dg['afstand'] == dist]
            gev_eind = np.nan
            if len(s) < 30:
                t_lim = FALLBACK_LIMITS[g][dist]
                beste = np.nan
            else:
                per_rijder = s.groupby('skater_id')['tijd'].min().sort_values().values
                beste = float(per_rijder[min(2, len(per_rijder) - 1)])
                if bron in ('record', 'evt') and dist not in ('100m', '300m'):
                    # wereldrecord = snelste tijd die door een andere rijder (binnen 3%) bevestigd wordt en van een rijder
                    # met minstens 3 races op de afstand komt; een geïsoleerde snelle tijd is een typefout
                    stat = s.groupby('skater_id')['tijd'].agg(['min', 'size']).sort_values('min')
                    waar, aant = stat['min'].to_numpy(), stat['size'].to_numpy()
                    ids = stat.index.to_numpy()
                    heeft_ev = 'event_id' in s.columns
                    if heeft_ev:
                        # event van de snelste tijd per rijder
                        ix = s.loc[s.groupby('skater_id')['tijd'].idxmin()].set_index('skater_id')['event_id']
                        evs = ix.reindex(ids).to_numpy()
                    for k in range(min(30, len(waar) - 1)):
                        if aant[k] < 3:
                            continue
                        # bevestiging: een ANDERE rijder in een ANDER evenement binnen 1.5%
                        # (twee tijden uit hetzelfde evenement kunnen dezelfde fout delen)
                        ok = False
                        for j in range(len(waar)):
                            if j == k: continue
                            if waar[j] > waar[k] * 1.015: break
                            if (not heeft_ev) or evs[j] != evs[k]:
                                ok = True; break
                        if ok:
                            beste = float(waar[k]); break
                t_lim = beste * (1.0 - (max(marge, 0.03) if dist in ('100m', '300m') else marge))   # korte afstanden hebben geen record
                if bron == 'evt' and dist not in ('100m', '300m'):
                    pb = s[s['groep'].isin(['junior_A', 'neo', 'senior'])] if 'groep' in s.columns else s
                    pbt = pb[pb['tijd'] >= beste * 0.9995].groupby('skater_id')['tijd'].min().to_numpy()   # zonder onbevestigde tijden
                    e0, lo, geldig = evt_ondergrens(pbt)
                    if np.isfinite(lo) and lo < beste:
                        # stabiele EVT-schatting: ondergrens van het interval, maar nooit meer dan 5% onder het record
                        # en altijd minstens 0.3% onder het record (ln(T - T_lim) moet gedefinieerd blijven)
                        t_lim = min(max(lo, beste * 0.95), beste * (1.0 - 0.003))
                    # anders (onstabiel): terugval op record x (1 - marge)
                mins = s.groupby('seizoen')['tijd'].min()
                if len(mins) >= 8:
                    try:
                        c, loc, scale = genextreme.fit(-mins.values)
                        if c > 0.05:
                            gev_eind = -(loc + scale / c)
                    except Exception:
                        pass
            limieten[g][dist] = round(float(t_lim), 3)
            _LAATSTE_RECORDS[g][dist] = beste
            diag.append({'gender': g, 'afstand': dist, 'n': len(s), 'robuust_beste': beste,
                         'gev_eindpunt(diagnose)': gev_eind, 'T_lim': limieten[g][dist]})
    if verbose:
        print("T_lim per geslacht/afstand (marge=%.3f):" % marge)
        print(pd.DataFrame(diag).round(2).to_string(index=False))
    return limieten


# =============================================================================
# 4. POPULATIE-PRIOR (start voor nieuwe rijders) EN CORRELATIEMATRIX
# =============================================================================
def _tl_vector(t_lims, gender, afstand_idx):
    """T_lim per rij (vectorized)."""
    arr = np.array([[t_lims[g][a] for a in AFSTANDEN] for g in ['m', 'v']])
    gc = (np.asarray(gender) == 'v').astype(int)
    return arr[gc, np.asarray(afstand_idx, dtype=int)]


def _log_delta(tijd, t_lim):
    return np.log(np.maximum(np.asarray(tijd, dtype=float) - t_lim, 0.01))


def _maak_psd(cov, eps=1e-4):
    cov = (cov + cov.T) / 2.0
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def bereken_populatie_prior(df, t_lims, afstanden=AFSTANDEN, verbose=True):
    """Bepaalt uit de data (empirical Bayes):
       mu0[g][groep]  : gemiddelde ln(T-T_lim) per afstand voor een nieuwe rijder
       Sigma0[g]      : covariantie van het echte niveau tussen rijders (binnen groep)
       R0[g]          : race-ruis per afstand (binnen rijder, afstand en seizoen)
       corr[g]        : bijbehorende correlatiematrix (jouw stap 3)
    """
    d = df[~df['verdacht']].copy()
    tl = _tl_vector(t_lims, d['gender'].to_numpy(), d['afstand_idx'].to_numpy())
    d['y'] = _log_delta(d['tijd'].values, tl)

    mu0, Sigma0, R0, corr, ruis = {}, {}, {}, {}, {}
    for g in ['m', 'v']:
        dg = d[d['gender'] == g]
        # --- race-ruis: binnen rijder/afstand/seizoen
        # een waarde per rijder/afstand/DAG (herhaalde starts op dezelfde dag delen het dag-effect en
        # hebben nauwelijks spreiding; die zouden de ruis onderschatten)
        dd = dg.groupby(['skater_id', 'afstand', 'seizoen', 'dag', 'groep'], observed=True, as_index=False)['y'].mean()
        sleutel_ = ['skater_id', 'afstand', 'seizoen']
        n = dd.groupby(sleutel_)['y'].transform('size')
        w = dd[n >= 2].copy()
        w['res2'] = (w['y'] - w.groupby(sleutel_)['y'].transform('mean')) ** 2
        w['nn'] = n[n >= 2]
        w['afstand_idx'] = w['afstand'].map(NAAR_INDEX)
        R = np.full(NA, np.nan)
        for j, a in enumerate(afstanden):
            ww = w[w['afstand'] == a]
            if len(ww) > 30:
                # onbevooroordeelde variantie: som(res^2) / som(1 - 1/n)
                dof = (1.0 - 1.0 / ww['nn']).sum()
                R[j] = ww['res2'].sum() / max(dof, 1.0)
        fallback_R = np.nanmedian(R) if np.isfinite(R).any() else 0.03
        R = np.where(np.isfinite(R), R, fallback_R)
        R0[g] = np.clip(R, 0.003, 0.5)

        # --- ruisfactor per niveaugroep t.o.v. de gepoolde ruis van die afstand:
        #     jeugd/zwakkere groepen zijn onvoorspelbaarder, dus tellen hun races minder zwaar
        ruis[g] = {}
        w2 = w.copy()
        w2['j'] = w2['afstand'].map(NAAR_INDEX)
        w2['dof'] = 1.0 - 1.0 / w2['nn']
        for grp_naam in GROEPEN:
            ws = w2[w2['groep'] == grp_naam]
            logs, gew = [], []
            for j in range(NA):
                xs_ = ws[ws['j'] == j]
                if len(xs_) >= 30 and xs_['dof'].sum() > 0:
                    r_gd = xs_['res2'].sum() / xs_['dof'].sum()
                    logs.append(math.log(max(r_gd, 1e-6) / R0[g][j]))
                    gew.append(xs_['dof'].sum())
            ruis[g][grp_naam] = float(np.clip(math.exp(np.average(logs, weights=gew)), 0.3, 12.0)) if logs else 1.0

        # --- groepsgemiddelden (per rijder gemiddeld zodat frequente rijders niet domineren)
        sk = dg.groupby(['groep', 'skater_id', 'afstand'])['y'].mean().reset_index()
        overall = sk.groupby('afstand')['y'].mean()
        glob = float(sk['y'].mean()) if len(sk) else 2.0
        m = sk.groupby(['groep', 'afstand'])['y'].agg(['mean', 'count'])
        mu0[g] = {}
        for grp_naam in GROEPEN:
            vec = np.zeros(NA)
            for j, a in enumerate(afstanden):
                basis = overall.get(a, np.nan)
                if not np.isfinite(basis):
                    # afstand ontbreekt: schat via dichtstbijzijnde afstand met data
                    bekend = [k for k, b in enumerate(afstanden) if np.isfinite(overall.get(b, np.nan))]
                    basis = overall.get(afstanden[min(bekend, key=lambda k: abs(k - j))], glob) if bekend else glob
                if (grp_naam, a) in m.index:
                    mean, cnt = m.loc[(grp_naam, a)]
                    vec[j] = (cnt * mean + 15.0 * basis) / (cnt + 15.0)
                else:
                    vec[j] = basis
            mu0[g][grp_naam] = vec

        # --- niveau-covariantie tussen afstanden (binnen groep), pairwise
        eerste_groep = dg.sort_values('dag').groupby('skater_id')['groep'].first()
        sk2 = dg.groupby(['skater_id', 'afstand'])['y'].agg(['mean', 'size']).reset_index()
        sk2['groep'] = sk2['skater_id'].map(eerste_groep)
        sk2['j'] = sk2['afstand'].map(NAAR_INDEX)
        sk2['res'] = [v - mu0[g][gr][j] for v, gr, j in zip(sk2['mean'], sk2['groep'], sk2['j'])]
        piv = sk2.pivot_table(index='skater_id', columns='afstand', values='res').reindex(columns=afstanden)
        cov = piv.cov(min_periods=25).to_numpy()
        var_obs = np.array([np.nanvar(piv[a].values) if piv[a].notna().sum() > 5 else np.nan for a in afstanden])
        # ruis van een rijder-gemiddelde uit n races: R/n  -> trek gemiddelde R/n af van de diagonaal
        inv_n = sk2.groupby('afstand')['size'].apply(lambda s: float((1.0 / s).mean())).reindex(afstanden)
        latent_var = np.where(np.isfinite(var_obs),
                              np.maximum(var_obs - R0[g] * inv_n.fillna(1.0).values, 0.25 * np.nan_to_num(var_obs)),
                              np.nan)
        latent_var = np.where(np.isfinite(latent_var), latent_var, np.nanmedian(latent_var) if np.isfinite(latent_var).any() else 0.3)
        sd = np.sqrt(latent_var)
        C = np.full((NA, NA), np.nan)
        for i in range(NA):
            for j in range(NA):
                if i == j:
                    C[i, j] = 1.0
                elif np.isfinite(cov[i, j]) and np.isfinite(var_obs[i]) and np.isfinite(var_obs[j]) and var_obs[i] > 0 and var_obs[j] > 0:
                    # bereken correlatie van de ruisvrije niveaus (dempt attenuatie door ruis)
                    C[i, j] = cov[i, j] / (sd[i] * sd[j])
        # ontbrekend: val terug op afstand-nabijheid
        for i in range(NA):
            for j in range(NA):
                if not np.isfinite(C[i, j]):
                    C[i, j] = max(0.3, 0.85 - 0.12 * abs(i - j))
        C = np.clip((C + C.T) / 2.0, -0.99, 0.99)
        np.fill_diagonal(C, 1.0)
        C = _maak_psd(C)
        dd = np.sqrt(np.diag(C))
        C = C / np.outer(dd, dd)
        corr[g] = C
        Sigma0[g] = np.outer(sd, sd) * C

    if verbose:
        for g in ['m', 'v']:
            print(f"\nCorrelatiematrix niveaus ({g}):")
            print(pd.DataFrame(corr[g], index=afstanden, columns=afstanden).round(2).to_string())
            print("race-ruis sd (ln-delta) per afstand:", np.sqrt(R0[g]).round(3).tolist())
            print("ruisfactor (variantie) per groep:", {k: round(v, 2) for k, v in ruis[g].items()})
    return {'mu0': mu0, 'Sigma0': Sigma0, 'R0': R0, 'corr': corr, 'ruis': ruis}


# =============================================================================
# 5. TOESTAND PER RIJDER
# =============================================================================
Z10 = 1.2815515655446004        # 10% kwantiel van de standaardnormale verdeling


def _twee_tijden(v, anker, pk_ijs):
    """Normale wedstrijdtijd en piektijd uit een voorspelling.
    normale tijd : mediaan op een gemiddelde dag (gemiddeld recent event), normale vorm.
    piektijd     : goede vorm (10% beste van zijn eigen race-tot-race spreiding) op een gemiddelde dag.
                   Getoetst: ca. 15% van de resultaten van top-10 favorieten is sneller (ruw ~1 op 7-10).
    piek_extreem_tijd : piektijd EN piek-ijs (gemiddelde piekdag, -2.4%); dat is dubbel geteld (de vorm van het
                   veld zit al in het piekdag-effect) en wordt maar in ~0.5% van de races gehaald: alleen als
                   extreem scenario bedoeld.
    piek_vorm_tijd = piek_tijd; piek_ijs_tijd = normale vorm op een piekdag."""
    t_lim, x, sd = v['t_lim'], v['mu_log'], v['sigma_log']
    vorm = t_lim + math.exp(x - Z10 * sd * v.get('snel_fac', 1.0))         # piek = snelle kant: eigen schaal
    return {'normale_tijd': (t_lim + math.exp(x)) * math.exp(anker),
            'piek_tijd': vorm * math.exp(anker),
            'piek_vorm_tijd': vorm * math.exp(anker),
            'piek_extreem_tijd': vorm * math.exp(anker + pk_ijs),
            'piek_ijs_tijd': (t_lim + math.exp(x)) * math.exp(anker + pk_ijs)}



def schat_baan_offsets(df, params, verbose=False):
    """Globale schatting van baanverschillen: log t = a[rijder, afstand, seizoen] + b[baan, era] + ruis.
    Alleen rijders die in hetzelfde seizoen op dezelfde afstand op >= 2 banen reden dragen bij. Robuust (winsorizen),
    per era genormaliseerd op een gewogen gemiddelde van 0 (het absolute niveau wordt door het algemene tijdsniveau gedragen).
    Geeft dict {(baan, era): (b, se, n_eff)}."""
    p = params
    d = df[(~df['verdacht'].to_numpy()) & (df['baan'].to_numpy() != '') & df['afstand'].isin(AFSTANDEN[2:7]).to_numpy()]
    if len(d) == 0:
        return {}
    era = (d['seizoen'].to_numpy().astype(int) // p.baan_era_seizoenen).astype(np.int64)
    u_key = d['skater_id'].to_numpy().astype(np.int64) * 10000 + d['afstand_idx'].to_numpy().astype(np.int64) * 100 \
        + (d['seizoen'].to_numpy().astype(np.int64) - 1990)
    u, u_inv = np.unique(u_key, return_inverse=True)
    r_key = pd.factorize(d['baan'].to_numpy().astype(str))[0].astype(np.int64) * 1000 + era
    r, r_inv = np.unique(r_key, return_inverse=True)
    # groepen met minstens 2 banen
    ur = np.unique(u_inv.astype(np.int64) * (len(r) + 1) + r_inv)
    n_banen = np.bincount((ur // (len(r) + 1)), minlength=len(u))
    keep = n_banen[u_inv] >= 2
    if keep.sum() < 100:
        return {}
    u_inv, r_inv, era = u_inv[keep], r_inv[keep], era[keep]
    lt = np.log(d['tijd'].to_numpy()[keep])
    grp = d['groep'].to_numpy()[keep]
    w = np.where(np.isin(grp, ['junior_A', 'neo', 'senior', 'masters_jong']), 1.0, 0.35)
    nu, nr = int(u_inv.max()) + 1, int(r_inv.max()) + 1
    b = np.zeros(nr)
    for _ in range(40):
        a = np.bincount(u_inv, w * (lt - b[r_inv]), nu) / np.maximum(np.bincount(u_inv, w, nu), 1e-9)
        e = np.clip(lt - a[u_inv] - b[r_inv], -0.05, 0.05)
        y = a[u_inv] + b[r_inv] + e
        a = np.bincount(u_inv, w * (y - b[r_inv]), nu) / np.maximum(np.bincount(u_inv, w, nu), 1e-9)
        b = np.bincount(r_inv, w * (y - a[u_inv]), nr) / np.maximum(np.bincount(r_inv, w, nr), 1e-9)
    e = lt - a[u_inv] - b[r_inv]
    sigma = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    # effectieve aantal metingen per (baan, era): rijen minus het aandeel dat de rijder-groep zelf al verklaart
    n_grp = np.bincount(u_inv, w, nu)
    n_eff = np.bincount(r_inv, w * (1.0 - w / np.maximum(n_grp[u_inv], 1e-9)), nr)
    # normaliseren per era
    era_r = np.zeros(nr, dtype=np.int64)
    era_r[r_inv] = era
    gew_r = np.bincount(r_inv, w, nr)
    for ee in np.unique(era_r):
        sel = (era_r == ee) & (gew_r > 0)
        if sel.any():
            b[sel] -= np.average(b[sel], weights=gew_r[sel])
    fac_codes, fac_uni = pd.factorize(d['baan'].to_numpy().astype(str))
    uit = {}
    for k in range(nr):
        if n_eff[k] < 30:
            continue
        se = math.sqrt((sigma ** 2) / n_eff[k] + 0.002 ** 2)
        uit[(fac_uni[int(r[k] // 1000)], int(r[k] % 1000))] = (float(b[k]), max(se, p.baan_se_floor), float(n_eff[k]))
    if verbose:
        print(f"baanverschillen: {len(uit)} (baan, era)-combinaties, ruis {sigma * 100:.2f}%")
    return uit



class SchaatserToestand:
    __slots__ = ('id', 'naam', 'gender', 'land', 'x', 'P', 's', 'n', 'laatste_dag', 'eerste_dag',
                 'groep', 'recent', 'pb', 'prop_event', 'slow_run', 'snel_dag', 'vf', 'vpf', 'vfdag', 'qh', 'qz', 'qz_dag', 'qz_nieuw', 'sw', 'sn')

    def __init__(self, sid, naam, gender, land, groep, x0, P0, dag):
        self.id = sid
        self.naam = naam
        self.gender = gender
        self.land = land
        self.groep = groep
        self.x = x0.copy()
        self.P = P0.copy()
        self.s = np.ones(NA)
        self.n = np.zeros(NA, dtype=int)
        self.laatste_dag = dag
        self.eerste_dag = dag
        self.recent = [[] for _ in range(NA)]     # (dag, ruwe tijd, dag-gecorrigeerde tijd)
        self.pb = np.full(NA, np.inf)
        self.prop_event = -1
        self.snel_dag = np.zeros(NA)                 # dag van de laatste snelle uitschieter per afstand (0 = geen)
        self.slow_run = np.zeros(NA, dtype=int)     # aantal opeenvolgende als 'slechte dag' beoordeelde races per afstand
        self.qh = 0.0                                # log-volatiliteit van het niveau (procesruis x e^qh)
        self.qz = 0.0                                # laatste verrassing (z) van een eerdere dag
        self.qz_dag = -1
        self.qz_nieuw = None
        self.sw = 0.0                                # som van posterieure sprongkansen (eigen sprongkans)
        self.sn = 0.0
        self.vf = 0.0                                # kortetermijnvorm (ln-tijd, gedeeld over afstanden), OU-proces
        self.vpf = 0.0                               # variantie van de vorm
        self.vfdag = dag


# =============================================================================
# 6. HET MODEL
# =============================================================================
N_ERV_BIN = 6
DT_KNOPEN = np.array([0.02, 0.1, 0.3, 1.0])          # pauze in jaren: <1 week, <5 weken, <4 mnd, <1 jaar, langer
AFST_GROEP = np.array([0, 0, 0, 0, 1, 1, 2, 2])      # sprint (100-1000), midden (1500, 3000), lang (5000, 10000)


def _erv_bin(n, n_tot):
    """Ervaringsklasse: -1 = allereerste race (startcorrectie), 0 = debuut op deze afstand, 1: 1-2, 2: 3-5, 3: 6-10, 4: 11-30, 5: >30."""
    if n_tot <= 0:
        return -1
    if n <= 0:
        return 0
    return 1 if n <= 2 else 2 if n <= 5 else 3 if n <= 10 else 4 if n <= 30 else 5


def _erv_bin_vec(n, n_tot):
    b = np.select([n <= 0, n <= 2, n <= 5, n <= 10, n <= 30], [0, 1, 2, 3, 4], 5)
    return np.where(n_tot <= 0, -1, b)


def _erv_shift(model, g, groep, n, n_tot, j):
    eb = getattr(model, 'erv_bias', None)
    if eb is None or not model.params.ervaring_bias:
        return 0.0
    b = _erv_bin(n, n_tot)
    if b < 0:
        return 0.0
    return float(eb[g].get(groep, eb[g]['onbekend'])[b, j])


def _sd_fac(model, dt, n, n_tot, j):
    sf = getattr(model, 'sd_factor', None)
    if sf is None or not model.params.sd_factoren:
        return 1.0
    return float(sf[int(np.searchsorted(DT_KNOPEN, dt)), _erv_bin(n, n_tot) + 1, AFST_GROEP[j]])


class SchaatsModel:
    def __init__(self, params=None):
        self.params = params or Params()
        self.t_lims = None
        self.prior = None
        self.staten = {}
        self.baan_stats = {}
        self.obs = None
        self.groepen_tabel = None
        self.events = None
        self.groei = {gr: np.zeros(len(GROEI_KNOOP)) for gr in GROEPEN}
        self.tau_dag_empirisch = self.params.tau_dag
        self.baan_offsets = None
        self.niveau_stats = None
        self.anker = 0.0                # neutrale tijd := gemiddeld recent event (zie fit)
        self.niveau_grenzen = None      # per (geslacht, afstand): 4 grenzen van de niveau-kwintielen
        self.dag_gewicht = {}           # (groep, niveau-kwintiel) -> betrouwbaarheid voor dag-effect
        self.extra_dag = {}             # {event_id: extra dag-effect (ln-tijd)} voor statistisch afwijkende events
        self.erv_bias = None            # {g: {groep: array[N_ERV_BIN, NA]}} drift per ervaringsklasse
        self.sd_factor = None           # array[n_dt, N_ERV_BIN+1, 3] schaal op de voorspel-sd

    # -------------------------------------------------------------------------
    def voorbereiden(self, df, verbose=True, t_lim_df=None):
        """Schat T_lim (eventueel uit een andere/volledige dataset) en de populatie-prior uit de data."""
        self.t_lims = bereken_terminal_limits(t_lim_df if t_lim_df is not None else df,
                                              marge=self.params.t_lim_marge, verbose=verbose, bron=self.params.t_lim_bron)
        if self.params.t_lim_bron in ('record', 'evt'):
            # tijden sneller dan het (bevestigde) record zijn niet geloofwaardig: markeer als verdacht (9x ruis)
            m1 = self.params.t_lim_marge
            for g in ('m', 'v'):
                for a in AFSTANDEN:
                    if a in ('100m', '300m'): continue
                    rec = _LAATSTE_RECORDS[g][a]
                    if not np.isfinite(rec): continue
                    masker = ((df['gender'] == g) & (df['afstand'] == a) & (df['tijd'] < rec * 0.9995)).to_numpy()
                    if masker.any():
                        df.loc[masker, 'verdacht'] = True
                        if verbose: print('  %s %s: %d tijden sneller dan record %.2f gemarkeerd als verdacht' % (g, a, masker.sum(), rec))
        self.prior = bereken_populatie_prior(df, self.t_lims, verbose=verbose)
        return self

    def _bouw_Q(self, g):
        p = self.params
        C = self.prior['corr'][g]
        CQ = p.q_mix * C + (1.0 - p.q_mix) * np.eye(NA)
        sd = np.sqrt(np.diag(self.prior['Sigma0'][g]))
        return p.q * np.outer(sd, sd) * CQ

    # -------------------------------------------------------------------------
    # --- algemeen tijdsniveau L(dag): gedecayd gemiddelde van (event-effect - baanverschil) over alle banen
    def _niveau(self, dag):
        st = self.niveau_stats
        if st is None or st[1] <= 0:
            return 0.0, 1e-4
        laatste, sw, swt = st
        verval = 0.5 ** (max(dag - laatste, 0) / self.params.niveau_halfwaarde_dagen)
        sw *= verval
        swt *= verval
        k = self.params.niveau_shrink
        return swt / (sw + k) if (sw + k) > 0 else 0.0, 1.0 / max(sw + k, 1e-9) + 0.001 ** 2

    def _era(self, dag):
        seizoen = pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(dag))
        sz = seizoen.year if seizoen.month >= 7 else seizoen.year - 1
        return sz // self.params.baan_era_seizoenen

    def _offset(self, baan, dag):
        if not self.params.baan_structuur or not self.baan_offsets:
            return None
        baan = (self.params.baan_alias or {}).get(baan, baan)
        return self.baan_offsets.get((baan, self._era(dag)))

    def _baan_prior(self, baan, dag):
        """Prior voor het dag-effect op een baan: (gemiddelde, variantie, effectief aantal events)."""
        p = self.params
        L, varL = self._niveau(dag)
        off = self._offset(baan, dag)
        if off is not None:                      # globaal geschat baanverschil + algemeen niveau
            b, se, n_eff = off
            return L + b, p.tau_dag ** 2 + se ** 2 + varL, n_eff
        rs = self.baan_stats.get((p.baan_alias or {}).get(baan, baan))           # terugval: eigen geschiedenis van de baan (relatief aan het niveau)
        if rs is None:
            return L, p.tau_dag ** 2 + p.tau_baan ** 2 + varL, 0.0
        laatste, sw, swt = rs
        verval = 0.5 ** (max(dag - laatste, 0) / DAGEN_PER_JAAR / p.baan_halfwaarde_jaar)
        sw *= verval
        swt *= verval
        mu = L + swt / (sw + p.baan_shrink)
        var = p.tau_dag ** 2 + p.tau_baan ** 2 / (1.0 + sw) + varL
        return mu, var, sw

    def _baan_update(self, baan, dag, theta, W):
        p = self.params
        L, _ = self._niveau(dag)
        rel = theta - L
        rs = self.baan_stats.get(baan)
        if rs is None:
            sw, swt = 0.0, 0.0
        else:
            laatste, sw, swt = rs
            verval = 0.5 ** (max(dag - laatste, 0) / DAGEN_PER_JAAR / p.baan_halfwaarde_jaar)
            sw *= verval
            swt *= verval
        we = W * p.tau_dag ** 2 / (1.0 + W * p.tau_dag ** 2)
        self.baan_stats[baan] = [dag, sw + we, swt + we * rel]
        # algemeen niveau bijwerken met dit event, als het baanverschil bekend is
        off = self._offset(baan, dag)
        if off is not None:
            b, se, _ = off
            wl = 1.0 / (1.0 / max(W, 1e-9) + se ** 2 + p.tau_dag ** 2)
            st = self.niveau_stats
            if st is None:
                sw0, swt0 = 0.0, 0.0
            else:
                verval = 0.5 ** (max(dag - st[0], 0) / p.niveau_halfwaarde_dagen)
                sw0, swt0 = st[1] * verval, st[2] * verval
            self.niveau_stats = [dag, sw0 + wl, swt0 + wl * (theta - b)]

    # -------------------------------------------------------------------------
    def _nieuwe_toestand(self, sid, naam, gender, land, groep, dag):
        x0 = self.prior['mu0'][gender][groep]
        P0 = self.params.p0_schaal * self.prior['Sigma0'][gender]
        st = SchaatserToestand(sid, naam, gender, land, groep, x0, P0, dag)
        st.vpf = self.params.vorm_ou_sd ** 2
        return st

    def _vorm_prop(self, st, dag):
        """OU-vorm naar `dag`: f *= a, var = a^2 var + sd^2 (1 - a^2), a = exp(-dt/tau)."""
        p = self.params
        sd = p.vorm_ou_sd
        if sd <= 0.0:
            return 0.0, 0.0
        f = getattr(st, 'vf', 0.0); v = getattr(st, 'vpf', sd * sd); d0 = getattr(st, 'vfdag', dag)
        if dag > d0:
            a = math.exp(-(dag - d0) / p.vorm_ou_tau)
            f *= a; v = a * a * v + sd * sd * (1.0 - a * a)
        st.vf = f; st.vpf = v; st.vfdag = max(dag, d0)
        return f, v

    def _propageer(self, st, dag, groep, Q):
        p = self.params
        dt = min(max(dag - st.laatste_dag, 0) / DAGEN_PER_JAAR, p.max_dt_jaar)
        if dt > 0:
            st.x = st.x + p.groei_schaal * float(np.interp(dt, GROEI_KNOOP, self.groei[groep]))
            qh = getattr(st, 'qh', 0.0)
            if qh != 0.0:
                qh *= p.qvol_jaar ** dt; st.qh = qh
            st.P = st.P + Q * dt * p.q_groep.get(groep, 1.0) * math.exp(qh)
        st.laatste_dag = dag
        return dt

    # -------------------------------------------------------------------------
    def fit(self, df, verbose=True):
        """Chronologische pass over alle events. Vult staten, baan_stats, obs, events."""
        p = self.params
        if self.t_lims is None:
            self.voorbereiden(df, verbose=verbose)
        self.staten = {}
        self.baan_stats = {}
        self.niveau_stats = None
        if p.baan_structuur and getattr(self, 'baan_offsets', None) is None:
            self.baan_offsets = schat_baan_offsets(df, p, verbose=verbose)
        N = len(df)

        ev_id = df['event_id'].to_numpy()
        dag = df['dag'].to_numpy()
        sid_a = df['skater_id'].to_numpy()
        g_a = df['gender'].to_numpy()
        j_a = df['afstand_idx'].to_numpy()
        t_a = df['tijd'].to_numpy(dtype=float)
        verd_a = df['verdacht'].to_numpy()
        groep_a = df['groep'].to_numpy()
        baan_a = df['baan'].to_numpy()
        naam_a = df['skater_name'].to_numpy()
        land_a = df['land'].to_numpy()

        Q = {g: self._bouw_Q(g) for g in ['m', 'v']}
        tl = {g: np.array([self.t_lims[g][a] for a in AFSTANDEN]) for g in ['m', 'v']}
        R0 = {g: self.prior['R0'][g] * p.r_schaal for g in ['m', 'v']}
        ruisg = self.prior['ruis'] if p.betrouwbaarheid else {'m': {}, 'v': {}}

        # per rij te bewaren (voor tuning, diagnose, tests)
        rec = {k: np.full(N, np.nan) for k in
               ['x_prior', 'v_prior', 'n_prior', 'n_tot', 'dt', 'theta', 'mu_baan', 'var_baan', 'x_post',
                'e', 'z', 'v_theta']}
        ev_rows = []
        grp_rows = []

        bh = bool(getattr(self, 'bewaar_historie', False))
        if bh:
            H = {'skater_id': np.zeros(N, dtype=np.int64), 'event_id': np.zeros(N, dtype=np.int64),
                 'dag': np.zeros(N, dtype=np.int32), 'groep': np.zeros(N, dtype=np.int8),
                 'x': np.zeros((N, NA), dtype=np.float32), 'sd': np.zeros((N, NA), dtype=np.float32),
                 's': np.zeros((N, NA), dtype=np.float32), 'n': np.zeros((N, NA), dtype=np.int32),
                 'vf': np.zeros(N, dtype=np.float32), 'vpf': np.zeros(N, dtype=np.float32), 'qh': np.zeros(N, dtype=np.float32)}
            h_n = 0
        grenzen = np.flatnonzero(np.diff(ev_id) != 0) + 1
        starts = np.concatenate([[0], grenzen])
        eindes = np.concatenate([grenzen, [N]])
        a_h = p.garch_alpha
        b_h = p.garch_beta
        c_h = p.huber_c
        totaal = len(starts)

        for k, (i0, i1) in enumerate(zip(starts, eindes)):
            if verbose and k % 5000 == 0 and k > 0:
                print(f"  event {k}/{totaal}")
            eid = ev_id[i0]
            d_ev = int(dag[i0])
            baan = baan_a[i0]

            # ---- fase A: toestanden ophalen/propageren, voorspelling per rij
            rijen = range(i0, i1)
            prior_info = {}
            for i in rijen:
                s_id = sid_a[i]
                st = self.staten.get(s_id)
                if st is None:
                    st = self._nieuwe_toestand(s_id, naam_a[i], g_a[i], land_a[i], groep_a[i], d_ev)
                    self.staten[s_id] = st
                dt = 0.0
                if st.prop_event != eid:
                    dt = self._propageer(st, d_ev, groep_a[i], Q[st.gender])
                    st.prop_event = eid
                st.groep = groep_a[i]
                j = j_a[i]
                g = g_a[i]
                # ervaringscorrectie: alleen op de VOORSPELLING (de toestand zelf blijft ongemoeid, dus geen terugkoppeling)
                xb = st.x[j] + (_erv_shift(self, g, groep_a[i], st.n[j], st.n.sum(), j) if p.ervaring_bias else 0.0)
                R = R0[g][j] * ruisg[g].get(groep_a[i], 1.0) * st.s[j] * (p.verdacht_factor if verd_a[i] else 1.0)
                v = st.P[j, j] + R
                if p.vorm_ou_sd > 0.0:
                    f_, vf_ = self._vorm_prop(st, d_ev)
                    mul = (tl[g][j] + math.exp(xb)) / math.exp(xb)          # ln-tijd -> x = ln(T - T_lim)
                    xb = xb + f_ * mul
                    v = v + vf_ * mul * mul
                delta = math.exp(xb)
                that = tl[g][j] + delta
                # verschil in log-tijd en bijbehorende variantie (delta-methode)
                r_lt = math.log(t_a[i]) - math.log(that)
                var_lt = v * (delta / that) ** 2
                gw = 1.0
                if self.niveau_grenzen is not None and p.betrouwbaarheid:
                    niv = int(np.searchsorted(self.niveau_grenzen[g][j], st.x[j]))
                    gw = self.dag_gewicht.get((groep_a[i], niv), 1.0)
                if p.betrouwbaarheid and groep_a[i] in JEUGD_GROEPEN:
                    gw *= p.dag_gewicht_jeugd
                if verd_a[i]:
                    gw *= 0.1
                prior_info[i] = (r_lt, var_lt, R, v, dt, gw)
                rec['x_prior'][i] = xb
                rec['v_prior'][i] = v
                rec['n_prior'][i] = st.n[j]
                rec['n_tot'][i] = st.n.sum()
                rec['dt'][i] = dt

            # ---- fase B: dag-effect. Stap 1: hele event (beide geslachten, alle afstanden)
            mu_b, var_b, n_eff_b = self._baan_prior(baan, d_ev)
            groepen = {}
            for i in rijen:
                groepen.setdefault((g_a[i], j_a[i]), []).append(i)
            som_wr, som_w = 0.0, 0.0
            per_groep = {}
            for sleutel, idx in groepen.items():
                sw, swr = 0.0, 0.0
                for i in idx:
                    r_lt, var_lt, _, _, _, gw = prior_info[i]
                    sd = math.sqrt(var_lt)
                    c_op = p.dag_c_op if p.dag_c_op > 0 else c_h
                    r_w = max(-c_h * sd, min(c_op * sd, r_lt))     # winsoriseren (trage uitschieters evt. strenger)
                    w = gw / var_lt
                    sw += w
                    swr += w * r_w
                per_groep[sleutel] = (sw, swr)
                som_w += sw
                som_wr += swr
            theta_data = som_wr / som_w                   # meting zonder prior
            se_data = math.sqrt(1.0 / som_w)
            f_lek = 1.0
            if p.baan_structuur and self._offset(baan, d_ev) is not None:
                f_lek = 1.0 / (1.0 + som_w * p.baan_lek ** 2)      # de gemeten dag is minder betrouwbaar dan de baanstructuur
            prec_post = 1.0 / var_b + som_w * f_lek
            theta_ev = (mu_b / var_b + som_wr * f_lek) / prec_post + (getattr(self, 'extra_dag', None) or {}).get(int(eid), 0.0)
            var_ev = 1.0 / prec_post

            # stap 2: per (geslacht, afstand) binnen de dag
            theta_g = {}
            var_g = {}
            for sleutel, (sw, swr) in per_groep.items():
                pg = p.tau_groep ** 2 + var_ev
                prec = 1.0 / pg + sw
                theta_g[sleutel] = (theta_ev / pg + swr) / prec
                var_g[sleutel] = 1.0 / prec

            # ---- fase C: rating-updates met dag-gecorrigeerde tijd
            for sleutel, idx in groepen.items():
                th = theta_g[sleutel]
                vth = var_g[sleutel]
                g, j = sleutel
                for i in idx:
                    st = self.staten[sid_a[i]]
                    t_adj = t_a[i] * math.exp(-th)
                    y = math.log(max(t_adj - tl[g][j], 0.01))
                    fac = t_adj / max(t_adj - tl[g][j], 0.01)          # d ln(delta) / d ln(T)
                    _, _, R, _, _, _ = prior_info[i]
                    R_eff = R + vth * fac * fac
                    f_x = 0.0; pf_x = 0.0
                    if p.vorm_ou_sd > 0.0:
                        f_x = st.vf * fac; pf_x = st.vpf * fac * fac
                        R_eff = R_eff + pf_x                                 # vorm telt voor de niveau-update als extra ruis
                    Pj = st.P[:, j]
                    S = Pj[j] + R_eff
                    e = y - st.x[j] - f_x
                    z = e / math.sqrt(S)
                    z_rec = z
                    if p.qvol_a > 0.0 and not verd_a[i] and not (e > 0 and p.dip_val_z > 0 and z > p.dip_val_z):
                        if getattr(st, 'qz_dag', -1) >= 0 and st.qz_dag < d_ev:
                            zc_ = max(min(z, 4.0), -4.0)
                            st.qh = min(max(p.qvol_lam * st.qh + p.qvol_a * zc_ * st.qz, p.qvol_min), p.qvol_max)
                        if getattr(st, 'qz_dag', -1) != d_ev:
                            st.qz_nieuw = max(min(z, 4.0), -4.0)
                    pi_j = p.sprong_pi_snel if e < 0 else p.sprong_pi_traag
                    if p.sprong_m > 0.0 and pi_j > 0.0:
                        pi_j = min((getattr(st, 'sw', 0.0) + p.sprong_m * pi_j) / (getattr(st, 'sn', 0.0) + p.sprong_m), 0.5)
                    if pi_j > 0.0 and not verd_a[i] and not (e > 0 and p.dip_val_z > 0 and z > p.dip_val_z):
                        # menging van twee toestandsovergangen: gewone kleine verandering of een echte sprong (variantie k^2 S).
                        # Het gewicht van de sprong volgt uit de verrassing; de toestandsonzekerheid groeit naar rato (via de
                        # correlatie van Q ook op de andere afstanden), zodat de update verder gaat en de volatiliteit daarna hoger is.
                        qJ = p.sprong_k * p.sprong_k * S
                        SJ = S + qJ
                        l0 = (1.0 - pi_j) * math.exp(-0.5 * e * e / S) / math.sqrt(S)
                        l1 = pi_j * math.exp(-0.5 * e * e / SJ) / math.sqrt(SJ)
                        wJ = l1 / (l0 + l1) if (l0 + l1) > 0 else 1.0
                        if p.sprong_m > 0.0:
                            st.sw = getattr(st, 'sw', 0.0) + wJ; st.sn = getattr(st, 'sn', 0.0) + 1.0
                        if wJ > 1e-6:
                            Qg = Q[st.gender]; qc = Qg[:, j]
                            st.P = st.P + (wJ * qJ / Qg[j, j]) * np.outer(qc, qc)
                            Pj = st.P[:, j]
                            S = Pj[j] + R_eff
                            z = e / math.sqrt(S)
                    if st.slow_run[j] >= p.fr_max_run and z < 1.5:
                        st.slow_run[j] = 0                                   # weer een gewone race: terug naar het frontier-model
                    val = (p.dip_val_z > 0 and p.fr_pi <= 0.0 and e > 0 and z > p.dip_val_z and not verd_a[i]
                           and st.slow_run[j] < max(p.dip_run, 1))
                    frontier = p.fr_pi > 0.0 and not verd_a[i] and st.slow_run[j] < p.fr_max_run
                    regime = (p.fr_pi > 0.0 and not verd_a[i] and not frontier and z > 3.0
                              and st.slow_run[j] >= p.fr_max_run)
                    snel_regime = (p.snel_run_z > 0 and not verd_a[i] and e < 0 and z < -p.snel_run_z
                                   and (p.snel_run_enkel or (st.snel_dag[j] > 0 and 0 < d_ev - st.snel_dag[j] <= p.snel_run_dagen)))
                    if snel_regime and p.snel_run_modus == 'niveau':
                        regime = True        # tweede snelle uitschieter kort na de eerste: echte vormsprong (zelfde behandeling als het trage regime)
                    if regime:
                        # meerdere slechte dagen achter elkaar met grote verrassing: de vorm is echt veranderd (of de eerdere
                        # schatting was fout). Maak de toestand weer onzeker zodat deze race de rating volledig kan bijstellen.
                        st.P[j, j] += 0.25 * e * e
                        Pj = st.P[:, j]
                        S = Pj[j] + R_eff
                        z = e / math.sqrt(S)
                    if val:
                        c_eff = 1e9
                    elif frontier:
                        c_eff = p.fr_huber_snel if e < 0 else 1e9           # langzaam: geen Huber, de slechte-dag-component vangt het op
                    elif regime:
                        c_eff = 1e9
                    elif e < 0 and not p.kap_snel_niveau:
                        c_eff = 1e9
                    else:
                        c_eff = p.huber_c_langzaam if (e > 0 and p.huber_c_langzaam > 0) else (p.huber_c_update if p.huber_c_update > 0 else c_h)
                    if abs(z) > c_eff:                                  # robuust tegen uitschieters
                        S = e * e / (c_eff * c_eff)
                    wd = 0.0
                    x_voor = st.x[j]; pjj = Pj[j]
                    if val:
                        # valpartij / uitschieter: geen enkele update van de toestand (de race telt voor niets mee)
                        wd = 1.0
                        K = np.zeros_like(Pj)
                        st.slow_run[j] += 1
                    elif frontier:
                        # samengestelde fout (stochastic frontier): normaal + half-normaal (alleen trager). Een slechte dag
                        # (of valpartij) zegt niets over wat de schaatser kan; een snelle tijd zegt dat per definitie wel.
                        sb = p.fr_c * math.sqrt(S)
                        l0 = (1.0 - p.fr_pi) * math.exp(-0.5 * e * e / S) / math.sqrt(S)
                        l1 = p.fr_pi * 2.0 * math.exp(-0.5 * (e / sb) ** 2) / sb if e > 0 else 0.0
                        wd = l1 / (l0 + l1) if (l0 + l1) > 0 else 0.0
                        st.slow_run[j] = st.slow_run[j] + 1 if wd > 0.5 else 0
                        K0 = Pj / S
                        if e < 0 and p.fr_snel_boost != 1.0:
                            b_ = min(p.fr_snel_boost, 0.98 / max(K0[j], 1e-9))
                            if p.fr_boost_eigen:
                                K0 = K0.copy(); K0[j] *= b_
                            else:
                                K0 = K0 * b_
                        e_bad = min(e, p.fr_lam_cap * math.sqrt(S)) if e > 0 else e
                        K = (1.0 - wd * (1.0 - p.fr_lam)) * K0              # effectieve gain (voor de covariantie)
                        st.x = st.x + K0 * ((1.0 - wd) * e + wd * p.fr_lam * e_bad)
                    elif p.dip_pi > 0.0 and not verd_a[i]:
                        # menging: normale race of 'slechte dag' (trager, meer spreiding); een dip beweegt de rating minder
                        mu_d = p.dip_a * math.sqrt(S)
                        Sd = S * (1.0 + p.dip_b)
                        l0 = (1.0 - p.dip_pi) * math.exp(-0.5 * e * e / S) / math.sqrt(S)
                        l1 = p.dip_pi * math.exp(-0.5 * (e - mu_d) ** 2 / Sd) / math.sqrt(Sd)
                        wd = l1 / (l0 + l1) if (l0 + l1) > 0 else 0.0
                        if p.dip_run > 0:
                            if st.slow_run[j] >= p.dip_run:
                                wd = 0.0                                    # al meerdere slechte dagen achter elkaar: echte verandering
                                if z < 1.5:
                                    st.slow_run[j] = 0
                            else:
                                st.slow_run[j] = st.slow_run[j] + 1 if wd > 0.5 else 0
                        K0, K1 = Pj / S, Pj / Sd
                        if e < 0 and p.snel_boost != 1.0:
                            b_ = min(p.snel_boost, 0.98 / max(K0[j], 1e-9)); K0 = K0 * b_; K1 = K1 * b_
                        if p.snel_geh_dagen > 0 and e > 0 and st.snel_dag[j] > 0 and d_ev - st.snel_dag[j] <= p.snel_geh_dagen:
                            K0 = K0 * p.snel_geh_demp; K1 = K1 * p.snel_geh_demp      # waarschijnlijk niet gepiekt, geen echte achteruitgang
                        K = (1.0 - wd) * K0 + wd * K1
                        e_bad = e - mu_d
                        if p.dip_cap > 0:
                            e_bad = min(e_bad, p.dip_cap * math.sqrt(Sd))
                        st.x = st.x + (1.0 - wd) * K0 * e + wd * K1 * e_bad
                    else:
                        K = Pj / S
                        st.x = st.x + K * e
                    if pf_x > 0.0 and not val and not verd_a[i]:
                        # vorm-update: hetzelfde effectieve signaal als het niveau, gewicht var(vorm) / S (gezamenlijke Kalman-stap,
                        # kruiscovariantie verwaarloosd); uit de niveaustap dx_j = (P_jj / S) * e_eff volgt e_eff
                        e_eff = (st.x[j] - x_voor) * S / max(pjj, 1e-12) if pjj > 0 else 0.0
                        kf = pf_x / S
                        st.vf = st.vf + kf * e_eff / fac
                        st.vpf = max(st.vpf * (1.0 - kf * (1.0 - wd)), 1e-12)
                    if p.snel_geh_dagen > 0 and e < 0 and z < -p.snel_geh_z and not verd_a[i]:
                        st.snel_dag[j] = d_ev
                    if p.snel_run_z > 0 and e < 0 and not verd_a[i]:
                        z0 = e / math.sqrt(Pj[j] + R_eff) if not snel_regime else -99.0
                        if snel_regime or z0 < -p.snel_run_z:
                            st.snel_dag[j] = d_ev
                    P_new = st.P - np.outer(K, st.P[j, :])
                    st.P = (P_new + P_new.T) * 0.5
                    z2 = z * z if ((snel_regime and p.snel_run_modus == 'vol') or (e < 0 and not p.kap_snel_vol)) else min(z * z * (1.0 - wd), 9.0)
                    st.s[j] = min(max((1.0 - a_h - b_h) + a_h * z2 + b_h * st.s[j], 0.3), 6.0)
                    st.n[j] += 1
                    tr = t_a[i]
                    if not verd_a[i] and tr < st.pb[j]:
                        st.pb[j] = tr
                    if not verd_a[i]:
                        st.recent[j].append((d_ev, tr, t_adj))
                        if len(st.recent[j]) > 8:
                            st.recent[j].pop(0)
                    rec['theta'][i] = th
                    rec['v_theta'][i] = vth
                    rec['x_post'][i] = st.x[j]
                    rec['e'][i] = e
                    rec['z'][i] = z_rec
                    rec['mu_baan'][i] = mu_b
                    rec['var_baan'][i] = var_b

                sw, swr = per_groep[sleutel]
                grp_rows.append((eid, d_ev, baan, g, AFSTANDEN[j], len(idx), swr / sw, math.sqrt(1.0 / sw), th, math.sqrt(vth)))

            if p.qvol_a > 0.0:
                for s_id in dict.fromkeys(sid_a[i0:i1].tolist()):
                    st = self.staten[s_id]
                    zn = getattr(st, 'qz_nieuw', None)
                    if zn is not None:
                        st.qz = zn; st.qz_dag = d_ev; st.qz_nieuw = None
            if bh:                         # toestand van elke deelnemer NA dit event
                for s_id in dict.fromkeys(sid_a[i0:i1].tolist()):
                    st = self.staten[s_id]
                    H['skater_id'][h_n] = s_id
                    H['event_id'][h_n] = eid
                    H['dag'][h_n] = d_ev
                    H['groep'][h_n] = GROEPEN.index(st.groep) if st.groep in GROEPEN else len(GROEPEN) - 1
                    H['x'][h_n] = st.x
                    H['sd'][h_n] = np.sqrt(np.diag(st.P))
                    H['s'][h_n] = st.s
                    H['n'][h_n] = st.n
                    H['vf'][h_n] = getattr(st, 'vf', 0.0)
                    H['vpf'][h_n] = getattr(st, 'vpf', 0.0)
                    H['qh'][h_n] = getattr(st, 'qh', 0.0)
                    h_n += 1
            self._baan_update(baan, d_ev, theta_data, som_w)
            ev_rows.append((eid, d_ev, baan, i1 - i0, theta_data, se_data, mu_b, math.sqrt(var_b), theta_ev,
                            math.sqrt(var_ev), n_eff_b))

        basis_kol = [c for c in ['event_id', 'dag', 'jaar', 'skater_id', 'gender', 'afstand', 'afstand_idx', 'groep',
                                 'baan', 'land', 'tijd', 'verdacht', 'x_waar'] if c in df.columns]
        rec32 = pd.DataFrame({k: v.astype(np.float32) for k, v in rec.items()})
        self.obs = pd.concat([df[basis_kol].reset_index(drop=True), rec32], axis=1)
        self.groepen_tabel = pd.DataFrame(grp_rows, columns=[
            'event_id', 'dag', 'baan', 'gender', 'afstand', 'n', 'theta_data', 'se_data', 'theta_post', 'se_post'])
        self.events = pd.DataFrame(ev_rows, columns=[
            'event_id', 'dag', 'baan', 'n', 'theta_data', 'se_data', 'mu_baan', 'sd_prior', 'theta_post',
            'se_post', 'n_eff_baan'])
        # Het absolute niveau van 'neutraal' is niet identificeerbaar (een verschuiving in alle ratings
        # wordt exact opgevangen door de dag-effecten). We leggen het vast op het gemiddelde event
        # van de laatste 3 jaar; voorspellingen van ruwe tijden veranderen hierdoor niet.
        jr = self.obs['jaar'].max()
        rec_rows = self.obs[self.obs['jaar'] >= jr - 2]
        self.anker = float(rec_rows['theta'].mean()) if len(rec_rows) else 0.0
        self._bouw_event_tabel()
        if bh:
            self.historie_ruw = {k: v[:h_n] for k, v in H.items()}
        return self

    # -------------------------------------------------------------------------
    def _bouw_event_tabel(self):
        """Voegt veldkwaliteit, z-score en piekdag-vlaggen toe aan de event-tabel."""
        p = self.params
        o = self.obs
        ev = self.events.copy()
        ev['datum'] = pd.to_datetime(ev['dag'], unit='D')
        # veldsterkte: percentiel van de rating na de race binnen geslacht/afstand
        o = o.copy()
        # hoge x_post = trage tijd; ascending=False geeft de traagste rang 1 (laag percentiel), dus hoog percentiel = snel.
        # (Tot 1-10-2026 stond hier een extra omdraaiing waardoor 'elite' de traagste 2% was.)
        o['niveau_pct'] = o.groupby(['gender', 'afstand'])['x_post'].rank(pct=True, ascending=False) * 100.0
        o['elite'] = o['niveau_pct'] >= p.elite_pct
        agg = o.groupby('event_id').agg(n_rijders=('skater_id', 'nunique'),
                                        n_landen=('land', lambda s: s[s != ''].nunique()),
                                        n_elite=('elite', 'sum'),
                                        veldniveau=('niveau_pct', 'mean'))
        ev = ev.merge(agg, left_on='event_id', right_index=True, how='left')
        ev['dev_ruw'] = ev['theta_data'] - ev['mu_baan']
        # per baan systematisch verschil tussen gemeten en verwacht (rating-lek) eruit halen: mediaan van de vorige 30 events
        ev = ev.sort_values(['baan', 'dag', 'event_id'])
        vorig = ev.groupby('baan')['dev_ruw'].transform(lambda x: x.shift(1).rolling(30, min_periods=5).median())
        ev['dev'] = ev['dev_ruw'] - vorig.fillna(0.0)
        ev = ev.sort_values(['dag', 'event_id']).reset_index(drop=True)
        # empirische spreiding van goed gemeten dagen: robuuste sd (MAD)
        goed = ev[(ev['se_data'] <= p.piek_max_se) & (ev['n_eff_baan'] >= 1.0)]
        if len(goed) >= 30:
            mad = float(np.median(np.abs(goed['dev'] - goed['dev'].median())) * 1.4826)
            self.tau_dag_empirisch = max(mad, 1e-4)
        else:
            self.tau_dag_empirisch = p.tau_dag
        ev['z'] = ev['dev'] / np.sqrt(ev['se_data'] ** 2 + self.tau_dag_empirisch ** 2)
        ev['goed_gemeten'] = ev['se_data'] <= p.piek_max_se
        ev['belangrijk'] = (ev['n_elite'] >= p.min_elite) & (ev['n_landen'] >= p.min_landen)
        ev['snelle_dag'] = (ev['z'] <= -p.piek_z) & ev['goed_gemeten'] & (ev['n_eff_baan'] >= 1.0)
        ev['piekdag'] = ev['snelle_dag']
        ev['piek_reden'] = np.where(ev['piekdag'] & ev['belangrijk'], 'belangrijk event + gunstige omstandigheden',
                                    np.where(ev['piekdag'], 'gunstig ijs/weer', ''))
        ev['effect_pct'] = (ev['theta_post'] - self.anker) * 100.0
        ev['dev_pct'] = ev['dev'] * 100.0
        self.events = ev
        self.obs = o

    # -------------------------------------------------------------------------
    # Voorspellen
    # -------------------------------------------------------------------------
    def voorspel(self, skater_id, afstand, dag=None, eerder=None):
        """Verwachte (dag-neutrale) tijd en onzekerheid voor een rijder/afstand.
        Als `dag` (dagnummer sinds 1970-01-01) is gegeven, groeit de onzekerheid mee met de tijd sinds de laatste race."""
        st = self.staten[skater_id]
        j = NAAR_INDEX[afstand]
        x = st.x[j]
        P = st.P[j, j]
        if dag is not None:
            dt = min(max(dag - st.laatste_dag, 0) / DAGEN_PER_JAAR, self.params.max_dt_jaar)
            x = x + self.params.groei_schaal * float(np.interp(dt, GROEI_KNOOP, self.groei[st.groep]))
            P = P + self._bouw_Q(st.gender)[j, j] * self.params.q_groep.get(st.groep, 1.0) * dt * math.exp(getattr(st, 'qh', 0.0) * self.params.qvol_jaar ** dt)
        _ou_add = 0.0
        if self.params.vorm_ou_sd > 0.0:
            d0 = getattr(st, 'vfdag', st.laatste_dag)
            f_, vf_ = _ou_vooruit(self.params, getattr(st, 'vf', 0.0), getattr(st, 'vpf', self.params.vorm_ou_sd ** 2), d0, dag if dag is not None else d0)
            mul = (self.t_lims[st.gender][afstand] + math.exp(x)) / math.exp(x)
            x = x + f_ * mul; P = P + vf_ * mul * mul; _ou_add = f_ * mul
        dt_ = 0.0 if dag is None else dt
        x = x + _erv_shift(self, st.gender, st.groep, st.n[j], st.n.sum(), j)
        _xb = x - _ou_add                     # vaste rating: toestand + seizoenspauze, zonder tijdelijke vorm
        ruisf = self.prior['ruis'][st.gender].get(st.groep, 1.0) if self.params.betrouwbaarheid else 1.0
        R = self.prior['R0'][st.gender][j] * ruisf * self.params.r_schaal * st.s[j]
        sd = math.sqrt(P + R) * self.params.pred_sd_schaal.get(afstand, 1.0) * _sd_fac(self, dt_, st.n[j], st.n.sum(), j) * _niv_fac(self, st.gender, j, x, st.groep)
        t_lim = self.t_lims[st.gender][afstand]
        _ls = _loc_schaal_fwd(self, st.gender, j, x)
        if _ls != (0.0, 1.0):
            _xb = _trend_x(_xb, t_lim, sd, _ls[0])
        if dag is not None and self.params.trend_corr:
            x = _trend_x(x, t_lim, sd, trend_verschuiving(self, skater_id, afstand, dag, eerder))
            sd = sd * trend_schaal(self, skater_id, afstand, dag, eerder)
        if _ls != (0.0, 1.0):
            x = _trend_x(x, t_lim, sd, _ls[0]); sd = sd * _ls[1]
        _gm, _gl = gas_toestand(self, skater_id, afstand, dag)
        if _gm != 0.0 or _gl != 0.0:
            x = _trend_x(x, t_lim, sd, _gm); sd = sd * math.exp(_gl)
        _vv = verm_verschuiving(self, skater_id, afstand, dag, eerder, _niv_fwd(self, st.gender, j, x))
        if _vv != 0.0:
            x = _trend_x(x, t_lim, sd, _vv)
        _sf = vol_snel_factor(self, afstand, sd, t_lim, x, _niv_fwd(self, st.gender, j, x))
        _sf0 = _sf
        _vm, _kf, _ks = vorm_toestand(self, skater_id, dag, _niv_fwd(self, st.gender, j, x))
        if _vm != 0.0 or _ks != 1.0 or _kf != 1.0:
            x = _trend_x(x, t_lim, sd, _vm); sd = sd * _ks; _sf = _sf * _kf / _ks
        _qf, _qs = schaal_toestand(self, skater_id, dag)
        if _qf != 1.0 or _qs != 1.0:
            sd = sd * _qs; _sf = _sf * _qf / _qs; _ks = _ks * _qs; _kf = _kf * _qf
        v = {'piek_diepte': diepte_toestand(self, skater_id, dag) * staart_toestand(self, skater_id, dag, _niv_fwd(self, st.gender, j, x)),
             'snel_fac': _sf, 'snel_fac_vol': _sf0, 'vorm_snel': _kf, 'vorm_traag': _ks, 'vorm_mu': _vm, 'mu_log': x, 'sigma_log': sd, 'sigma_niveau': math.sqrt(P),
             't_lim': t_lim, 'verwachte_tijd': (t_lim + math.exp(x)) * math.exp(self.anker_std),
             'verwachte_tijd_intern': t_lim + math.exp(x)}
        v.update(_twee_tijden(v, self.anker_std, self.piek_ijs()))
        v['normale_basis'] = v['normale_tijd'] * (t_lim + math.exp(_xb)) / (t_lim + math.exp(v['mu_log']))
        kans = piek_kansen(self, skater_id, dag)
        v['p_piek'], v['p_slecht'] = float(kans[0]), float(kans[2])
        _piek_tijden(self, v, afstand, kans, self.anker_std)
        return v

    def piek_ijs(self):
        """Gemiddelde afwijking (fractie van de tijd) van historische piekdagen t.o.v. de eigen baan."""
        pk = getattr(self, '_piek_ijs_cache', None)
        if pk is None:
            pk = float(self.piekdag_statistiek()[0])
            self._piek_ijs_cache = pk
        return pk

    @property
    def anker_std(self):
        """Niveau van een gemiddelde dag op de standaardbaan (Heerenveen) in het interne kader."""
        try:
            return float(self.baan_effect(STANDAARD_BAAN)[0])
        except Exception:
            return self.anker

    def baan_effect(self, baan, dag=None):
        """Verwacht dag-effect (fractie van de tijd) voor een baan, met onzekerheid."""
        if dag is None:                          # laatste dag in de data (bewaard: max over alle obs-rijen is traag en wordt per voorspelling gevraagd)
            c = getattr(self, '_dag_max', None)
            if c is None or c[0] is not self.obs:
                c = self._dag_max = (self.obs, int(self.obs['dag'].max()))
            dag = c[1]
        mu, var, n_eff = self._baan_prior(baan, dag)
        return mu, math.sqrt(var), n_eff

    def piekdag_statistiek(self):
        """Gemiddelde en spreiding van de afwijking op piekdagen (fractie van de tijd)."""
        pk = self.events[self.events['piekdag']]
        if len(pk) < 3:
            return -2.5 * self.tau_dag_empirisch, self.tau_dag_empirisch
        return float(pk['dev'].mean()), float(max(pk['dev'].std(), 1e-4))


def schat_dag_gewichten(model, min_n=150):
    """Betrouwbaarheid van elke (groep, niveau-kwintiel) voor het schatten van het dag-effect.

    Rijders die vaak onverwacht presteren (jeugd, zwakkere rijders) zeggen weinig over de baan.
    Gewicht = 1 / gemiddelde gestandaardiseerde kwadratische voorspelfout, genormaliseerd op 1."""
    o = model.obs
    o = o[(~o['verdacht']) & (o['n_prior'] >= 1) & o['z'].notna()].copy()
    grenzen = {g: {} for g in ['m', 'v']}
    for (g, j), s in o.groupby(['gender', 'afstand_idx']):
        grenzen[g][j] = np.quantile(s['x_prior'].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    for g in ['m', 'v']:
        for j in range(NA):
            grenzen[g].setdefault(j, np.array([1.0, 2.0, 3.0, 4.0]))
    niv = np.zeros(len(o), dtype=int)
    for (g, j), idx in o.groupby(['gender', 'afstand_idx']).indices.items():
        niv[idx] = np.searchsorted(grenzen[g][j], o['x_prior'].to_numpy()[idx])
    o['niv'] = niv
    o['z2'] = o['z'].clip(-3, 3) ** 2
    alg = float(o['z2'].mean())
    gew = {}
    for (gr, nv), s in o.groupby(['groep', 'niv']):
        if len(s) >= min_n:
            gew[(gr, nv)] = min(max(alg / float(s['z2'].mean()), 0.1), 2.0)
    return grenzen, gew


def schat_start_correctie(model, min_n=30, shrink_n=20.0):
    """Corrigeert het startpunt (mu0) van nieuwe rijders met de gemiddelde voorspelfout op hun
    allereerste race. Zo is de prior gekalibreerd op 'eerste keer in de data' in plaats van op
    carriere-gemiddelden (die te snel zijn omdat rijders verbeteren)."""
    o = model.obs
    f = o[(o['n_tot'] == 0) & (~o['verdacht'])].copy()
    tl = _tl_vector(model.t_lims, f['gender'].to_numpy(), f['afstand_idx'].to_numpy())
    y = np.log(np.maximum(f['tijd'].to_numpy() * np.exp(-f['theta'].to_numpy()) - tl, 0.01))
    f['e0'] = y - f['x_prior'].to_numpy()
    shift = {g: {gr: np.zeros(NA) for gr in GROEPEN} for g in ['m', 'v']}
    for (g, gr), s in f.groupby(['gender', 'groep']):
        tot_n, tot_e = 0, 0.0
        per_j = {}
        for j, sj in s.groupby('afstand_idx'):
            e = np.sort(sj['e0'].to_numpy())
            k = int(len(e) * 0.05)
            gem = float(e[k:len(e) - k].mean()) if len(e) > 2 * k + 1 else float(e.mean())
            per_j[j] = (len(e), gem)
            tot_n += len(e)
            tot_e += len(e) * gem
        alg = tot_e / tot_n if tot_n >= min_n else 0.0
        for j in range(NA):
            n, gem = per_j.get(j, (0, 0.0))
            if n >= min_n:
                shift[g][gr][j] = gem * n / (n + shrink_n)
            else:
                shift[g][gr][j] = alg * tot_n / (tot_n + shrink_n)
    return shift


def pas_start_correctie_toe(model, shift):
    for g in shift:
        for gr in shift[g]:
            model.prior['mu0'][g][gr] = model.prior['mu0'][g][gr] + shift[g][gr]


def _tp_nll(par, uu):
    """NLL van een twee-delige Student-t (kans p0 op een snellere tijd, aparte schaal en vrijheidsgraden per kant)."""
    from scipy.stats import t as _t
    p0 = 1.0 / (1.0 + np.exp(-par[0])); sL, nL, sU, nU = np.exp(par[1]), 1.5 + 28.5 / (1.0 + np.exp(-par[2])), np.exp(par[3]), 1.5 + 28.5 / (1.0 + np.exp(-par[4]))     # df tussen 1,5 en 30 (anders loopt de staart naar normaal)
    f = np.where(uu < 0, p0 * 2.0 * _t.pdf(uu / sL, nL) / sL, (1.0 - p0) * 2.0 * _t.pdf(uu / sU, nU) / sU)
    return -np.log(np.maximum(f, 1e-300))


def _fit_tp(uu, n_max=150000, seed=0):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    if len(uu) > n_max:
        uu = rng.choice(uu, n_max, replace=False)
    r = minimize(lambda q: np.mean(_tp_nll(q, uu)), np.array([0.0, 0.0, 1.0, 0.0, 1.0]), method='Nelder-Mead',
                 options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-6})
    return r.x


def schat_ervaring(model, min_n=40, shrink_n=60.0):
    """Gemiddelde voorspelfout (log(T-T_lim)) per (geslacht, groep, ervaringsklasse, afstand) op jaren <= split_jaar.
    Valpartijen en verdachte rijen tellen niet mee. Geeft de bijstelling voor model.erv_bias."""
    p = model.params
    o = model.obs
    f = o[(~o['verdacht']) & (o['jaar'] <= p.split_jaar) & (o['n_tot'] > 0) & ~((o['e'] > 0) & (o['z'] > max(p.dip_val_z, 3.0)))]
    b = _erv_bin_vec(f['n_prior'].to_numpy(), f['n_tot'].to_numpy())
    f = f.assign(b=b)
    uit = {g: {gr: np.zeros((N_ERV_BIN, NA)) for gr in GROEPEN} for g in ['m', 'v']}
    for (g, gr, bb), s_ in f.groupby(['gender', 'groep', 'b']):
        if bb < 0:
            continue
        e_all = np.sort(s_['e'].to_numpy()); k = int(len(e_all) * 0.05)
        alg = float(e_all[k:len(e_all) - k].mean()) if len(e_all) > 2 * k + 1 else 0.0
        alg = alg * len(e_all) / (len(e_all) + shrink_n)
        for j in range(NA):
            e = np.sort(s_.loc[s_['afstand_idx'] == j, 'e'].to_numpy()); k = int(len(e) * 0.05)
            if len(e) >= min_n:
                gem = float(e[k:len(e) - k].mean())
                uit[g][gr][int(bb), j] = alg + (gem - alg) * len(e) / (len(e) + shrink_n)
            else:
                uit[g][gr][int(bb), j] = alg
    return uit


def schat_sd_factoren(model, shrink_n=300.0):
    """Schaalfactor op de voorspel-sd per (pauze, ervaringsklasse, afstandsgroep), uit de robuuste spreiding (MAD) van de
    gestandaardiseerde fout op jaren <= split_jaar, relatief t.o.v. de totale spreiding. Valpartijen tellen niet mee."""
    p = model.params
    oud = model.sd_factor
    model.sd_factor = None
    try:
        _, err, var, m = voorspelscore(model, 1990)
    finally:
        model.sd_factor = oud
    u = err / np.sqrt(var)
    ok = (m['jaar'].to_numpy() <= p.split_jaar) & ~((err > 0) & (m['z'].to_numpy() > max(p.dip_val_z, 3.0))) & np.isfinite(u)
    dtb = np.searchsorted(DT_KNOPEN, m['dt'].to_numpy())
    eb = _erv_bin_vec(m['n_prior'].to_numpy(), m['n_tot'].to_numpy()) + 1
    ag = AFST_GROEP[m['afstand_idx'].to_numpy()]
    F = np.ones((len(DT_KNOPEN) + 1, N_ERV_BIN + 1, 3))
    grid = np.exp(np.linspace(np.log(0.6), np.log(1.7), 45))
    for c in range(3):
        okc = ok & (ag == c)
        if okc.sum() < 1000:
            continue
        par = _fit_tp(u[okc])                               # gedeelde vorm per afstandsgroep
        for a in range(len(DT_KNOPEN) + 1):
            for b in range(N_ERV_BIN + 1):
                sel = okc & (dtb == a) & (eb == b)
                n = int(sel.sum())
                if n >= 50:
                    uu = u[sel]
                    ll = [np.mean(_tp_nll(par, uu / f)) + np.log(f) for f in grid]   # log-loss-optimale schaal
                    f_opt = grid[int(np.argmin(ll))]
                    F[a, b, c] = math.exp(math.log(f_opt) * n / (n + shrink_n))
    return F


def kalibreer(model, df, n_iter=3, demping=0.8, verbose=True):
    """Herschat afwisselend: startpunt (mu0), drift-curve en betrouwbaarheidsgewichten van het dag-effect."""
    if model.obs is None:
        model.fit(df, verbose=False)
    for it in range(n_iter):
        pas_start_correctie_toe(model, schat_start_correctie(model))
        if model.params.ervaring_bias:
            model.erv_bias = schat_ervaring(model)          # direct: obs['e'] is de fout t.o.v. de ongecorrigeerde toestand
        if model.params.dag_comp_c > 0:
            model.extra_dag = schat_dag_comp(model)
        inc = schat_groei(model)
        for k in inc:
            model.groei[k] = model.groei[k] + demping * inc[k]
        model.niveau_grenzen, model.dag_gewicht = schat_dag_gewichten(model)
        model.fit(df, verbose=False)
        if verbose:
            print(f"  kalibratie-iteratie {it + 1}: voorspelscore {voorspelscore(model)[0]:.4f}")
    if model.params.sd_factoren:
        model.sd_factor = schat_sd_factoren(model)
    if model.params.trend_corr:
        schat_trend(model)
    if model.params.loc_schaal:
        schat_loc_schaal(model)
    if model.params.piek_kans:
        schat_piek_kansen(model)
    if model.params.gas:
        schat_gas(model)
    if model.params.vol_asym_bf != 0.0:
        schat_vol_ref(model)
    if model.params.verm:
        schat_verm(model)
    if model.params.vorm:
        schat_vorm(model)
    return model


def fit_volledig(df, params=None, n_iter=3, verbose=True, prior_van=None, t_lim_df=None):
    """Voorbereiden + fit + kalibratie."""
    model = SchaatsModel(params or Params())
    if prior_van is None:
        model.voorbereiden(df, verbose=verbose, t_lim_df=t_lim_df)
    else:
        model.t_lims = prior_van.t_lims
        model.prior = {'mu0': {g: {gr: v.copy() for gr, v in prior_van.prior['mu0'][g].items()} for g in ['m', 'v']},
                       'Sigma0': prior_van.prior['Sigma0'], 'R0': prior_van.prior['R0'], 'corr': prior_van.prior['corr'],
                       'ruis': prior_van.prior['ruis']}
        model.groei = {k: v.copy() for k, v in prior_van.groei.items()}
        model.niveau_grenzen = prior_van.niveau_grenzen
        model.dag_gewicht = prior_van.dag_gewicht
        model.baan_offsets = prior_van.baan_offsets
    model.fit(df, verbose=False)
    return kalibreer(model, df, n_iter=n_iter, verbose=verbose)


# =============================================================================
# 7. GROEI PER GROEP (schatten uit de eerste pass) EN TUNING
# =============================================================================
def schat_groei(model, min_eff=60.0):
    """Schat per niveaugroep een drift-curve: hoeveel sneller/trager een rijder gemiddeld is
    dan voorspeld, als functie van de tijd sinds zijn laatste race (in ln(T-T_lim)).
    Hoedjes-kernel rond vaste knooppunten; het resultaat wordt bij de bestaande curve opgeteld."""
    o = model.obs
    m = o[(o['dt'] > 0.02) & (o['n_tot'] >= 1) & o['e'].notna() & (~o['verdacht'])]
    inc = {}
    for gr in GROEPEN:
        s = m[m['groep'] == gr]
        y = np.zeros(len(GROEI_KNOOP))
        if len(s) >= 50:
            dt = s['dt'].to_numpy()
            e = s['e'].clip(-1.5, 1.5).to_numpy()
            w = 1.0 / s['v_prior'].to_numpy()
            for k in range(1, len(GROEI_KNOOP)):
                lo = GROEI_KNOOP[k - 1]
                hi = GROEI_KNOOP[k + 1] if k + 1 < len(GROEI_KNOOP) else GROEI_KNOOP[k] + (GROEI_KNOOP[k] - lo)
                hat = np.clip(1.0 - np.abs(dt - GROEI_KNOOP[k]) / np.where(dt < GROEI_KNOOP[k], GROEI_KNOOP[k] - lo, hi - GROEI_KNOOP[k]), 0.0, None)
                wk = w * hat
                n_eff = float(hat.sum())
                if wk.sum() > 0:
                    y[k] = float((wk * e).sum() / wk.sum()) * n_eff / (n_eff + min_eff)
        inc[gr] = y
    return inc


def voorspelscore_eigen(model, vanaf_jaar=2014):
    """Gemiddelde NLL (log-tijd) onder de EIGEN voorspeldichtheid van het model: frontier (normaal + half-normaal),
    dip-menging of normaal. Geschikt om modellen met verschillende ruisaannames eerlijk te vergelijken."""
    p = model.params
    o = model.obs
    m = o[(o['jaar'] >= vanaf_jaar) & (~o['verdacht']) & (o['n_prior'] >= 0)]
    t_lim = _tl_vector(model.t_lims, m['gender'].to_numpy(), m['afstand_idx'].to_numpy())
    delta = np.exp(m['x_prior'].to_numpy())
    that = t_lim + delta
    err = np.log(m['tijd'].to_numpy()) - (np.log(that) + m['mu_baan'].to_numpy())
    var = m['v_prior'].to_numpy() * (delta / that) ** 2 + m['var_baan'].to_numpy()
    if model.params.pred_sd_schaal:
        var = var * np.array([model.params.pred_sd_schaal.get(a, 1.0) for a in AFSTANDEN])[m['afstand_idx'].to_numpy()] ** 2
    sf = getattr(model, 'sd_factor', None)
    if sf is not None and model.params.sd_factoren:
        jj = m['afstand_idx'].to_numpy()
        var = var * sf[np.searchsorted(DT_KNOPEN, m['dt'].to_numpy()), _erv_bin_vec(m['n_prior'].to_numpy(), m['n_tot'].to_numpy()) + 1, AFST_GROEP[jj]] ** 2
    sd = np.sqrt(var)
    phi = lambda u: np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    f0 = phi(err / sd) / sd
    if p.fr_pi > 0:
        sb = p.fr_c * sd
        f1 = np.where(err > 0, 2.0 * phi(err / sb) / sb, 0.0)
        f = (1 - p.fr_pi) * f0 + p.fr_pi * f1
    elif p.dip_pi > 0:
        sdd = sd * np.sqrt(1 + p.dip_b)
        f1 = phi((err - p.dip_a * sd) / sdd) / sdd
        f = (1 - p.dip_pi) * f0 + p.dip_pi * f1
    else:
        f = f0
    return float(np.mean(-np.log(np.maximum(f, 1e-300))))


def logloss_eerlijk(model, split_jaar=2019, vanaf_jaar=2010, n_fit=150000, seed=0, per_rij=False, dag=False, staart_q=None, staart_niv=None):
    """Eerlijke log loss (gemiddelde NLL in log-tijd) voor het vergelijken van modelvarianten.
    Per afstandsgroep wordt de foutverdeling u = fout / sd geschat als twee-delige Student-t
    (kans p0 op een snellere tijd dan voorspeld, schaal en vrijheidsgraden voor elke kant apart) op de jaren <= split_jaar,
    en de NLL wordt gemeten op de jaren erna. Zo winnen varianten niet doordat hun sd of staart toevallig beter past bij een vaste
    scoringsverdeling. Geeft (nll_test, nll_fit, dict met parameters per groep).
    staart_q > 0 (standaard params.staart_q): body en snelle staart apart (peaks over threshold). De snelste staart_q van de fitrijen
    per afstandsgroep x niveauband (grenzen staart_niv) krijgt een gegeneraliseerde Pareto-verdeling (GPD) op de overschrijding
    voorbij de drempel u0; daarboven de Student-t van de body, herschaald naar massa 1 - staart_q."""
    from scipy.optimize import minimize
    from scipy.stats import t as _t, genpareto as _gp
    staart_q = model.params.staart_q if staart_q is None else staart_q
    staart_niv = tuple(model.params.staart_niv if staart_niv is None else staart_niv)
    _, err, var, m = voorspelscore(model, vanaf_jaar, dag=dag)     # dag=True: t.o.v. het geschatte dag-effect (relatieve voorspelling)
    u = err / np.sqrt(var)
    zmax = np.maximum(m['z'].to_numpy(), u)          # val: te traag volgens de race zelf (z na dag-correctie) of volgens de voorspelling vooraf
    u[(err > 0) & (zmax > model.params.dip_val_z) & (model.params.dip_val_z > 0)] = np.nan       # valpartijen (z > dip_val_z, trager) tellen niet mee
    lsd = 0.5 * np.log(var)
    if getattr(model, '_vol_asym_lam', None) is not None:
        lsd = lsd + model._vol_asym_lam                     # Jacobiaan van de eigen schaal aan de snelle kant
    a = m['afstand'].to_numpy(); jr = m['jaar'].to_numpy()
    groepen = {'kort': ['100m', '300m'], 'sprint': ['500m', '1000m'], 'midden': ['1500m', '3000m'], 'lang': ['5000m', '10000m']}   # jeugdafstanden apart: hun uitschieters verstoorden de sprintfit
    pw = None
    pf = getattr(model, 'piek_f', None)
    if model.params.piek_kans and pf is not None and getattr(model, 'piek_drempel', None):
        pp = model.params; p0v = np.array([pp.piek_q[0], pp.piek_q[1] - pp.piek_q[0], 1.0 - pp.piek_q[1]])
        Cc = pf.reindex(m.index)[['c0', 'c1', 'c2']].fillna(0.0).to_numpy()
        pw = (Cc + pp.piek_m * p0v) / (Cc.sum(1, keepdims=True) + pp.piek_m)
        gidx = {'kort': 0, 'sprint': 0, 'midden': 1, 'lang': 2}
        def _cdf(par, x):
            from scipy.stats import t as _tt
            p0 = 1.0 / (1.0 + np.exp(-par[0])); sL, nL, sU, nU = np.exp(par[1]), 1.5 + 28.5 / (1.0 + np.exp(-par[2])), np.exp(par[3]), 1.5 + 28.5 / (1.0 + np.exp(-par[4]))     # df tussen 1,5 en 30 (anders loopt de staart naar normaal)
            return p0 * 2 * _tt.cdf(x / sL, nL) if x < 0 else p0 + (1 - p0) * (2 * _tt.cdf(x / sU, nU) - 1)
        def _piek_logw(par, uu, ww, gname):
            lo, hi = model.piek_drempel[gidx[gname]]
            P = np.array([_cdf(par, lo), _cdf(par, hi) - _cdf(par, lo), 1 - _cdf(par, hi)])     # massa van de gebieden onder de gefitte verdeling
            rr = np.where(uu < lo, 0, np.where(uu > hi, 2, 1))
            return np.log(ww[np.arange(len(uu)), rr] / P[rr])

    def nll_u(par, uu):
        p0 = 1.0 / (1.0 + np.exp(-par[0])); sL, nL, sU, nU = np.exp(par[1]), 1.5 + 28.5 / (1.0 + np.exp(-par[2])), np.exp(par[3]), 1.5 + 28.5 / (1.0 + np.exp(-par[4]))     # df tussen 1,5 en 30 (anders loopt de staart naar normaal)
        lo = uu < 0
        f = np.where(lo, p0 * 2.0 * _t.pdf(uu / sL, nL) / sL, (1.0 - p0) * 2.0 * _t.pdf(uu / sU, nU) / sU)
        return -np.log(np.maximum(f, 1e-300))
    def _Fb(par, x):                              # verdelingsfunctie van de body (twee-delige t)
        p0 = 1.0 / (1.0 + np.exp(-par[0])); sL, nL, sU, nU = np.exp(par[1]), 1.5 + 28.5 / (1.0 + np.exp(-par[2])), np.exp(par[3]), 1.5 + 28.5 / (1.0 + np.exp(-par[4]))
        x = np.asarray(x, float)
        return np.where(x < 0, p0 * 2 * _t.cdf(x / sL, nL), p0 + (1 - p0) * (2 * _t.cdf(x / sU, nU) - 1))
    def _gesplitst(par, gname, rij, uu_fit):
        """(nll(x), F(x)) van body + GPD-staart, staart geschat op uu_fit (fitrijen van deze groep x niveauband)."""
        u0 = float(np.quantile(uu_fit, staart_q)); y = u0 - uu_fit[uu_fit < u0]
        xi, _, sg = _gp.fit(y, floc=0.0)
        if xi < 0:                                # xi >= 0: geen harde grens aan de snelle kant (anders kans 0 voorbij het eindpunt); dan exponentiële staart
            xi, sg = 0.0, float(np.mean(y))
        xi = float(min(xi, 0.9))
        Fb0 = float(_Fb(par, u0)); q = staart_q
        def nll(x):
            x = np.asarray(x, float); f = np.where(x >= u0, (1 - q) * np.exp(-nll_u(par, x)) / max(1 - Fb0, 1e-12), q * _gp.pdf(np.maximum(u0 - x, 0), xi, 0.0, sg))
            return -np.log(np.maximum(f, 1e-300))
        def F(x):
            x = np.asarray(x, float)
            return np.where(x < u0, q * _gp.sf(np.maximum(u0 - x, 0), xi, 0.0, sg), q + (1 - q) * (_Fb(par, x) - Fb0) / max(1 - Fb0, 1e-12))
        pars[f'{gname}_staart{rij}'] = (u0, xi, sg)
        return nll, F
    nivb = np.digitize(_niv_pre(m), staart_niv) if staart_q and staart_q > 0 else np.zeros(len(m), int)
    rng = np.random.default_rng(seed)
    rij_ll = []
    tot_test, n_test, tot_fit, n_fit_tot, pars = 0.0, 0, 0.0, 0, {}
    for gname, afs in groepen.items():
        sel = np.isin(a, afs) & np.isfinite(u)
        fit_i = np.where(sel & (jr <= split_jaar))[0]; test_i = np.where(sel & (jr > split_jaar))[0]
        if len(fit_i) < 500 or len(test_i) < 100:
            continue
        sub = rng.choice(fit_i, min(n_fit, len(fit_i)), replace=False)
        x0 = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        obj = lambda p: np.mean(nll_u(p, u[sub]))
        r = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-9})
        for _ in range(3):                       # herstarten: Nelder-Mead blijft anders soms in een lokaal optimum hangen (sprintgroep)
            r2 = minimize(obj, r.x, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-9})
            if r2.fun >= r.fun - 1e-9:
                break
            r = r2
        pars[gname] = r.x
        if staart_q and staart_q > 0:
            l_te = np.zeros(len(test_i)); l_fi = np.zeros(len(fit_i))
            for k in np.unique(nivb[fit_i]):
                fk = nivb[fit_i] == k; tk = nivb[test_i] == k
                if fk.sum() < 2000:                  # te weinig voor een eigen staart: bij de lagere band voegen
                    continue
                nll_k, F_k = _gesplitst(r.x, gname, k, u[fit_i[fk]])
                hoger = (nivb[fit_i] >= k); hoger_t = (nivb[test_i] >= k)      # kleine hogere banden erven deze staart
                l_fi[hoger] = nll_k(u[fit_i[hoger]]); l_te[hoger_t] = nll_k(u[test_i[hoger_t]])
                if pw is not None:
                    lo, hi = model.piek_drempel[gidx[gname]]
                    P = np.array([float(F_k(lo)), float(F_k(hi) - F_k(lo)), float(1 - F_k(hi))])
                    uu = u[test_i[hoger_t]]; rr = np.where(uu < lo, 0, np.where(uu > hi, 2, 1))
                    l_te[hoger_t] -= np.log(pw[test_i[hoger_t]][np.arange(len(uu)), rr] / P[rr])
            l_te = l_te + lsd[test_i]; l_fi = l_fi + lsd[fit_i]
        else:
            l_te = nll_u(r.x, u[test_i]) + lsd[test_i]
            if pw is not None:
                l_te = l_te - _piek_logw(r.x, u[test_i], pw[test_i], gname)
                pars[gname + '_piek'] = True
            l_fi = nll_u(r.x, u[fit_i]) + lsd[fit_i]
        if per_rij:
            rij_ll.append(pd.Series(l_te, index=m.index[test_i]))
        tot_test += float(np.sum(l_te)); n_test += len(test_i)
        tot_fit += float(np.sum(l_fi)); n_fit_tot += len(fit_i)
    if per_rij:
        return tot_test / max(n_test, 1), tot_fit / max(n_fit_tot, 1), pars, pd.concat(rij_ll).sort_index()
    return tot_test / max(n_test, 1), tot_fit / max(n_fit_tot, 1), pars



def _niv_pre(m, rel=False):
    """Niveau-percentiel (0-100, hoog = beter) van de rijder vooraf: rang van x_prior binnen geslacht x afstand x jaar (x leeftijdsgroep als rel)."""
    r = m.groupby(['gender', 'afstand_idx', 'jaar'] + (['groep'] if rel else []))['x_prior'].rank(pct=True, ascending=False)
    return r.to_numpy() * 100.0


def _niv_aanpassing(params, niv):
    """(sd-factor, locatieverschuiving in sd) per rij uit het niveau-percentiel."""
    kn = np.asarray(params.niv_knopen, float)
    f = np.interp(niv, kn, np.asarray(params.niv_sd_fac, float)) if len(params.niv_sd_fac) else np.ones(len(niv))
    l = np.interp(niv, kn, np.asarray(params.niv_loc, float)) if len(params.niv_loc) else np.zeros(len(niv))
    return f, l


def _niv_fac(model, gender, j, x, groep=None):
    """sd-factor (en loc) voor een rijder met niveau x (ln(T-Tlim), laag = snel) op afstand j, via de verdeling van x_prior in het laatste jaar."""
    p = model.params
    if not len(p.niv_knopen) and not len(p.niv_rel_groepen) and not p.loc_schaal:
        return 1.0
    ref = getattr(model, '_niv_ref', None)
    if ref is None:
        o = model.obs
        o = o[(o['jaar'] >= o['jaar'].max() - 1) & (~o['verdacht'])]
        ref = {(g, jj): np.sort(v['x_prior'].to_numpy()) for (g, jj), v in o.groupby(['gender', 'afstand_idx'])}
        ref.update({(g, jj, gr): np.sort(v['x_prior'].to_numpy()) for (g, jj, gr), v in o.groupby(['gender', 'afstand_idx', 'groep'])})
        model._niv_ref = ref
    if groep is not None and groep in p.niv_rel_groepen:
        r = ref.get((gender, j, groep))
        if r is None or len(r) < 50:
            return 1.0
        niv = 100.0 * (len(r) - np.searchsorted(r, x, side='left')) / len(r)
        fr = float(np.interp(niv, np.asarray(p.niv_rel_knopen, float), np.asarray(p.niv_rel_sd_fac, float)))
        if not len(p.niv_knopen):
            return fr
        rr = ref.get((gender, j))
        if rr is None or len(rr) < 50:
            return fr
        na = 100.0 * (len(rr) - np.searchsorted(rr, x, side='left')) / len(rr)
        return fr * float(_niv_aanpassing(p, np.array([na]))[0][0])
    if not len(p.niv_knopen):
        return 1.0
    r = ref.get((gender, j, groep)) if (p.niv_rel and groep is not None) else ref.get((gender, j))
    if r is None or len(r) < 50:
        return 1.0
    niv = 100.0 * (len(r) - np.searchsorted(r, x, side='left')) / len(r)
    f, _ = _niv_aanpassing(p, np.array([niv]))
    return float(f[0])


LS_RANDEN = np.array([25.0, 50.0, 75.0, 90.0, 95.0, 98.0])     # niveaubanden 0-25, 25-50, 50-75, 75-90, 90-95, 95-98, 98-100
LS_GROEP = np.array([0, 0, 0, 0, 1, 1, 2, 2])                    # 100/300/500/1000 sprint, 1500/3000 midden, 5000/10000 lang


def schat_loc_schaal(model):
    """Schat per geslacht x afstand x niveauband (niveau vooraf) de mediaan (locatie) en relatieve MAD-schaal van de
    gestandaardiseerde voorspelfout u = fout/sd, op de laatste `loc_schaal_jaren` jaren, gekrompen naar het afstandsgemiddelde.
    De toepassing (u -> (u - loc)/schaal) gebeurt in voorspelscore en in de voorspelling vooraf. Geeft de tabel terug."""
    p = model.params
    model.loc_schaal_tab = None
    _, err, var, m = voorspelscore(model, 0)
    u = err / np.sqrt(var)
    zmax = np.maximum(m['z'].to_numpy(), u)
    ok = np.isfinite(u) & ~((err > 0) & (zmax > p.dip_val_z) & (p.dip_val_z > 0)) & (m['n_prior'].to_numpy() >= 3)
    jr = m['jaar'].to_numpy()
    eind = p.loc_schaal_tot_jaar if p.loc_schaal_tot_jaar > 0 else int(jr.max())
    ok &= (jr <= eind) & (jr > eind - p.loc_schaal_jaren)
    g = (m['gender'].to_numpy() == 'v').astype(int); j = m['afstand_idx'].to_numpy()
    b = np.digitize(_niv_pre(m), LS_RANDEN)
    d = pd.DataFrame({'u': u[ok], 'g': g[ok], 'j': j[ok], 'b': b[ok], 'gr': LS_GROEP[j[ok]]})
    mad = lambda s: 1.4826 * float(np.median(np.abs(s - np.median(s))))
    gm = d.groupby('gr')['u'].agg(mad)
    ga = d.groupby(['g', 'j'])['u'].agg(med='median', mad=mad, n='size')
    gab = d.groupby(['g', 'j', 'b'])['u'].agg(med='median', mad=mad, n='size')
    loc = np.zeros((2, len(AFSTANDEN), len(LS_RANDEN) + 1)); sc = np.ones_like(loc)
    for (gg, jj), r in ga.iterrows():
        if r['n'] < 200:
            continue
        ref = gm.get(LS_GROEP[jj], np.nan)
        loc[gg, jj, :] = r['med']; sc[gg, jj, :] = r['mad'] / ref
        for bb in range(len(LS_RANDEN) + 1):
            if (gg, jj, bb) in gab.index:
                q = gab.loc[(gg, jj, bb)]; w = q['n'] / (q['n'] + p.loc_schaal_k)
                loc[gg, jj, bb] = w * q['med'] + (1 - w) * r['med']
                sc[gg, jj, bb] = (w * q['mad'] + (1 - w) * r['mad']) / ref
    model.loc_schaal_tab = {'loc': np.clip(loc, -1.0, 1.0), 'sc': np.clip(np.nan_to_num(sc, nan=1.0), 0.7, 1.4),
                            'jaren': (eind - p.loc_schaal_jaren + 1, eind)}
    for a in ('_z_emp',):
        if hasattr(model, a):
            delattr(model, a)
    return model.loc_schaal_tab


def _niv_fwd(model, gender, j, x):
    """Niveau-percentiel (0-100) van een rijder met toestand x, t.o.v. de x_prior-verdeling van het laatste jaar."""
    ref = getattr(model, '_niv_ref', None)
    if ref is None:
        o = model.obs; o = o[(o['jaar'] >= o['jaar'].max() - 1) & (~o['verdacht'])]
        ref = {(g, jj): np.sort(v['x_prior'].to_numpy()) for (g, jj), v in o.groupby(['gender', 'afstand_idx'])}
        model._niv_ref = ref
    r = ref.get((gender, j))
    if r is None or len(r) < 50:
        return 50.0
    return 100.0 * (len(r) - np.searchsorted(r, x, side='left')) / len(r)


def _loc_schaal_fwd(model, gender, j, x):
    """(locatie, schaal) voor een voorspelling vooraf: niveauband uit de verdeling van x_prior in het laatste jaar."""
    tab = getattr(model, 'loc_schaal_tab', None)
    if not model.params.loc_schaal or tab is None:
        return 0.0, 1.0
    ref = getattr(model, '_niv_ref', None)
    if ref is None:
        _niv_fac(model, gender, j, x)        # bouwt _niv_ref
        ref = getattr(model, '_niv_ref', None) or {}
        if not ref:
            o = model.obs; o = o[(o['jaar'] >= o['jaar'].max() - 1) & (~o['verdacht'])]
            ref = {(g, jj): np.sort(v['x_prior'].to_numpy()) for (g, jj), v in o.groupby(['gender', 'afstand_idx'])}
            model._niv_ref = ref
    r = ref.get((gender, j))
    if r is None or len(r) < 50:
        return 0.0, 1.0
    niv = 100.0 * (len(r) - np.searchsorted(r, x, side='left')) / len(r)
    bb = int(np.digitize(niv, LS_RANDEN)); gg = 1 if gender == 'v' else 0
    return float(tab['loc'][gg, j, bb]), float(tab['sc'][gg, j, bb])


def schat_piek_kansen(model):
    """Eigen kans per rijder op een piek en een slechte dag (Bernoulli per race, drie gebieden van de voorspelfout u):
    piek = u onder het piek_q[0]-kwantiel, slechte dag = u boven het piek_q[1]-kwantiel (per afstandsgroep, jaren <= split_jaar).
    Per race worden de gebieden van ALLE eerdere evenementen van de rijder (alle afstanden) geteld; de kans is
    (telling + m * populatiekans) / (aantal + m). Valpartijen tellen niet. Zet model.piek_f (tellingen vooraf per obs-rij),
    model.piek_reeks (voor de voorspelling vooraf) en model.piek_drempel."""
    p = model.params
    _, err, var, m = voorspelscore(model, 0)
    u = err / np.sqrt(var)
    val = (err > 0) & (np.maximum(m['z'].to_numpy(), u) > p.dip_val_z) & (p.dip_val_z > 0)
    gr = LS_GROEP[m['afstand_idx'].to_numpy()]
    jr = m['jaar'].to_numpy()
    ok = np.isfinite(u) & ~val
    drempel = {}
    for g in np.unique(gr):
        sel = ok & (gr == g) & (jr <= p.split_jaar) & (m['n_prior'].to_numpy() >= 3)
        if sel.sum() < 1000:
            sel = ok & (gr == g)
        drempel[int(g)] = (float(np.quantile(u[sel], p.piek_q[0])), float(np.quantile(u[sel], p.piek_q[1])))
    lo = np.array([drempel[int(g)][0] for g in gr]); hi = np.array([drempel[int(g)][1] for g in gr])
    reg = np.where(u < lo, 0, np.where(u > hi, 2, 1))
    d = pd.DataFrame({'sk': m['skater_id'].to_numpy().astype(np.int64), 'dag': m['dag'].to_numpy(), 'ev': m['event_id'].to_numpy(),
                      'reg': reg, 'ok': ok & (m['n_prior'].to_numpy() >= 1)}, index=m.index).sort_values(['sk', 'dag', 'ev'], kind='stable')
    C = np.zeros((len(d), 3))
    for k in range(3):
        x = ((d['reg'] == k) & d['ok']).astype(float)
        cum = x.groupby(d['sk']).cumsum() - x
        zelfde = x.groupby([d['sk'], d['dag']]).cumsum() - x         # rijen van dezelfde dag (ook ander evenement) zijn vooraf nog niet bekend
        C[:, k] = (cum - zelfde).to_numpy()
    model.piek_f = pd.DataFrame(C, index=d.index, columns=['c0', 'c1', 'c2']).reindex(m.index)
    # reeks per rijder: tellingen NA elk evenement (voor voorspellen vanaf een datum)
    dd = d[d['ok']]
    oh = pd.DataFrame({'sk': dd['sk'].to_numpy(), 'dag': dd['dag'].to_numpy(), 'ev': dd['ev'].to_numpy(),
                       'r0': (dd['reg'] == 0).to_numpy().astype(int), 'r1': (dd['reg'] == 1).to_numpy().astype(int), 'r2': (dd['reg'] == 2).to_numpy().astype(int)})
    e = oh.groupby(['sk', 'dag', 'ev'], as_index=False)[['r0', 'r1', 'r2']].sum().sort_values(['sk', 'dag', 'ev'], kind='stable')
    cnt = e[['r0', 'r1', 'r2']].to_numpy()
    skk = e['sk'].to_numpy(); st = np.flatnonzero(np.r_[True, skk[1:] != skk[:-1]]) if len(e) else np.array([], int)
    ei = np.r_[st[1:], len(skk)] if len(e) else np.array([], int)
    cum = np.zeros_like(cnt)
    for a, b in zip(st, ei):
        cum[a:b] = np.cumsum(cnt[a:b], axis=0)
    model.piek_reeks = {'dag': e['dag'].to_numpy().astype(np.int32), 'cum': cum.astype(np.int32),
                        'pos': {int(skk[a]): (int(a), int(b)) for a, b in zip(st, ei)}}
    model.piek_drempel = drempel
    for a in ('_z_emp',):
        if hasattr(model, a):
            delattr(model, a)
    return drempel


def piek_kansen(model, skater_id, dag=None):
    """(kans op piek, normaal, slechte dag) voor een rijder, uit zijn races voor `dag` (None = alle races)."""
    p = model.params
    p0 = np.array([p.piek_q[0], p.piek_q[1] - p.piek_q[0], 1.0 - p.piek_q[1]])
    r = getattr(model, 'piek_reeks', None)
    if not p.piek_kans or r is None:
        return p0
    ab = r['pos'].get(int(skater_id))
    c = np.zeros(3)
    if ab is not None:
        a, b = ab
        k = (b - a) if dag is None else int(np.searchsorted(r['dag'][a:b], dag, side='left'))
        if k > 0:
            c = r['cum'][a + k - 1].astype(float)
    return (c + p.piek_m * p0) / (c.sum() + p.piek_m)


def _pool_sorted(model, afstand):
    ps = getattr(model, '_z_emp_sorted', None)
    if ps is None or getattr(model, '_z_emp', None) is None:
        ps = {a: np.sort(v) for a, v in empirische_z(model).items()}
        model._z_emp_sorted = ps
    return ps[afstand]


def _kw_met_piek(model, afstand, kans, q):
    """Kwantiel q van de voorspelfout (empirische verdeling) als de gebieden piek/normaal/slecht kansen `kans` hebben i.p.v. de populatie."""
    p = model.params
    q0, q1 = p.piek_q
    if q <= kans[0]:
        qq = q0 * q / kans[0]
    elif q <= kans[0] + kans[1]:
        qq = q0 + (q - kans[0]) / kans[1] * (q1 - q0)
    else:
        qq = q1 + (q - kans[0] - kans[1]) / kans[2] * (1 - q1)
    pool = _pool_sorted(model, afstand)
    x = min(max(qq, 0.0), 1.0) * (len(pool) - 1)              # = np.quantile (lineair) op de al gesorteerde pool, zonder opnieuw te sorteren
    i = min(int(x), len(pool) - 2)
    return float(pool[i] + (x - i) * (pool[i + 1] - pool[i]))


def _piek_tijden(model, v, afstand, kans, anker):
    """Normale en piektijd met de eigen piek-/slechte-dagkans: verschuift de mediaan en het 10%-kwantiel van de eigen vorm."""
    p = model.params
    p0 = np.array([p.piek_q[0], p.piek_q[1] - p.piek_q[0], 1.0 - p.piek_q[1]])
    if np.allclose(kans, p0):
        return
    t_lim, x, sd = v['t_lim'], v['mu_log'], v['sigma_log']
    dm = _kw_met_piek(model, afstand, kans, 0.5) - _kw_met_piek(model, afstand, p0, 0.5)
    d10 = _kw_met_piek(model, afstand, kans, 0.10) - _kw_met_piek(model, afstand, p0, 0.10)
    v['normale_tijd'] = (t_lim + math.exp(x + dm * sd)) * math.exp(anker)
    v['verwachte_tijd'] = v['normale_tijd']
    sf_ = v.get('snel_fac', 1.0)
    v['piek_tijd'] = v['piek_vorm_tijd'] = (t_lim + math.exp(x - Z10 * sd * sf_ + d10 * sd * sf_)) * math.exp(anker)


GAS_GROEP = np.array([0, 0, 1, 1, 2, 2, 3, 3])          # 100/300 | 500/1000 | 1500/3000 | 5000/10000


def _tp_cdf_v(v, x):
    from scipy.stats import t as _t
    p0, sL, nL, sU, nU = v
    return np.where(x < 0, p0 * 2 * _t.cdf(x / sL, nL), p0 + (1 - p0) * (2 * _t.cdf(x / sU, nU) - 1))


def _gas_massa(model):
    dr = model.piek_drempel; V = model.gas_vorm
    lo = np.array([dr[int(LS_GROEP[np.flatnonzero(GAS_GROEP == g)[0]])][0] for g in range(4)])
    hi = np.array([dr[int(LS_GROEP[np.flatnonzero(GAS_GROEP == g)[0]])][1] for g in range(4)])
    M = np.zeros((4, 3))
    for g in range(4):
        a, b = float(_tp_cdf_v(V[g], lo[g])), float(_tp_cdf_v(V[g], hi[g]))
        M[g] = [a, b - a, 1 - b]
    return lo, hi, M


def schat_gas(model):
    """Schat de vaste vorm van de voorspelfout (twee-delige t per afstandsgroep, jaren <= split_jaar) en draait de GAS-laag
    (parameters gas_par) over alle races: per rij de toestand vooraf (model.gas_f) en per rijder x dag de toestand erna (model.gas_dag)."""
    from scipy.optimize import minimize
    from gas_lab import gas_toestanden
    p = model.params
    model.gas_f = None
    _, err, var, m = voorspelscore(model, 0)
    u = err / np.sqrt(var)
    val = ((err > 0) & (np.maximum(m['z'].to_numpy(), u) > p.dip_val_z) & (p.dip_val_z > 0)) | ~np.isfinite(u)
    grp = GAS_GROEP[m['afstand_idx'].to_numpy()]
    jr = m['jaar'].to_numpy()
    V = np.zeros((4, 5)); rng = np.random.default_rng(0)
    for g in range(4):
        uu = u[(grp == g) & ~val & (jr <= p.split_jaar) & (jr >= 2010)]
        if len(uu) < 1000:
            uu = u[(grp == g) & ~val]
        sub = rng.choice(uu, min(200000, len(uu)), replace=False)
        f = lambda th: np.mean(_tp_nll(th, sub))
        r = minimize(f, [0, 0, 1, 0, 1], method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-9})
        r = minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-9})
        th = r.x
        V[g] = [1 / (1 + np.exp(-th[0])), np.exp(th[1]), 1.5 + 28.5 / (1 + np.exp(-th[2])), np.exp(th[3]), 1.5 + 28.5 / (1 + np.exp(-th[4]))]
    model.gas_vorm = V
    o = np.lexsort((m['afstand_idx'].to_numpy(), m['event_id'].to_numpy(), m['dag'].to_numpy(), m['skater_id'].to_numpy()))
    sk = m['skater_id'].to_numpy().astype(np.int64)[o]; dg = m['dag'].to_numpy().astype(np.int64)[o]
    ai = m['afstand_idx'].to_numpy().astype(np.int64)[o]
    uu = np.where(np.isfinite(u), u, 0.0)[o]
    mm_ = np.zeros(len(o)); ll_ = np.zeros(len(o))
    dsk = np.zeros(len(o), np.int64); ddg = np.zeros(len(o), np.int64); dst = np.zeros((len(o), 26), np.float64)
    nd = gas_toestanden(sk, dg, ai, uu, grp[o].astype(np.int64), val[o], V, np.asarray(p.gas_par, float), -1.0, 2.5,
                        mm_, ll_, dsk, ddg, dst)
    gm = np.zeros(len(o)); gl = np.zeros(len(o)); gm[o] = mm_; gl[o] = ll_
    model.gas_f = pd.DataFrame({'m': gm.astype(np.float32), 'l': gl.astype(np.float32)}, index=m.index)
    dsk, ddg, dst = dsk[:nd], ddg[:nd].astype(np.int32), dst[:nd]
    st_ = np.flatnonzero(np.r_[True, dsk[1:] != dsk[:-1]]); ei = np.r_[st_[1:], nd]
    model.gas_dag = {'dag': ddg, 'st': dst.astype(np.float32), 'pos': {int(dsk[a]): (int(a), int(b)) for a, b in zip(st_, ei)}}
    for a in ('_z_emp', '_z_emp_sorted'):
        if hasattr(model, a):
            delattr(model, a)
    return V


def gas_toestand(model, skater_id, afstand, dag):
    """(verschuiving in sd, log-schaal) van de GAS-laag voor een race op `dag`, uit alleen races van eerdere dagen."""
    p = model.params
    r = getattr(model, 'gas_dag', None)
    if not p.gas or r is None:
        return 0.0, 0.0
    ab = r['pos'].get(int(skater_id))
    if ab is None or dag is None:
        return 0.0, 0.0
    a, b = ab
    k = int(np.searchsorted(r['dag'][a:b], dag, side='left'))
    if k == 0:
        return 0.0, 0.0
    st = r['st'][a + k - 1]; gap = dag - int(r['dag'][a + k - 1])
    j = NAAR_INDEX[afstand]; q = p.gas_par; f = gap / 30.0
    d2 = st[18 + j]
    f2 = (dag - d2) / 30.0 if d2 >= 0 else 0.0                # laag 2 vervalt vanaf de laatste race op DEZE afstand
    m = st[0] * q[0] ** f + st[2 + j] * q[6] ** f2
    l = st[1] * q[1] ** f + st[10 + j] * q[7] ** f2
    return float(m), float(min(max(l, -1.0), 2.5))


def normaal_z(model, vanaf_jaar=0):
    """Normaal-equivalente z per race: Phi^-1(PIT) onder de volledige voorspelverdeling (vorm per afstandsgroep,
    GAS-laag, eigen piekkans). Bij een goed model is |z| > 4 vrijwel nooit."""
    from scipy.special import ndtri
    _, err, var, m = voorspelscore(model, vanaf_jaar)
    u = err / np.sqrt(var)
    g = GAS_GROEP[m['afstand_idx'].to_numpy()]
    V = getattr(model, 'gas_vorm', None)
    if V is None:
        return pd.Series(np.nan, index=m.index)
    lo, hi, M = _gas_massa(model)
    p = model.params; p0 = np.array([p.piek_q[0], p.piek_q[1] - p.piek_q[0], 1.0 - p.piek_q[1]])
    pf = getattr(model, 'piek_f', None)
    if p.piek_kans and pf is not None:
        C = pf.reindex(m.index)[['c0', 'c1', 'c2']].fillna(0.0).to_numpy()
        pw = (C + p.piek_m * p0) / (C.sum(1, keepdims=True) + p.piek_m)
    else:
        pw = np.tile(p0, (len(m), 1))
    pit = np.full(len(u), np.nan)
    for gg in range(4):
        s = (g == gg) & np.isfinite(u)
        F = _tp_cdf_v(V[gg], u[s]); Flo, Fhi = float(_tp_cdf_v(V[gg], lo[gg])), float(_tp_cdf_v(V[gg], hi[gg])); w = pw[s]
        pit[s] = np.where(u[s] < lo[gg], w[:, 0] * F / Flo,
                          np.where(u[s] <= hi[gg], w[:, 0] + w[:, 1] * (F - Flo) / (Fhi - Flo),
                                   w[:, 0] + w[:, 1] + w[:, 2] * (F - Fhi) / (1 - Fhi)))
    return pd.Series(ndtri(np.clip(pit, 1e-300, 1 - 1e-16)), index=m.index)


def normaal_z_dag(model, vanaf_jaar=0):
    """Normaal-equivalente z t.o.v. het veld op die dag (geschat dag-effect), met de eigen scheve verdeling.
    De vorm wordt apart geschat per afstandsgroep x niveau (toppers >= 90e percentiel apart: hun eigen spreiding is klein
    t.o.v. de onzekerheid van het dag-effect), jaren <= split_jaar. Maat voor 'hoe uitzonderlijk was deze rit'."""
    from scipy.optimize import minimize
    from scipy.special import ndtri
    p = model.params
    _, err, var, m = voorspelscore(model, 0, dag=True)
    u = err / np.sqrt(var)
    val = ((err > 0) & (m['z'].to_numpy() > p.dip_val_z) & (p.dip_val_z > 0)) | ~np.isfinite(u)
    g = GAS_GROEP[m['afstand_idx'].to_numpy()] * 2 + (_niv_pre(m) >= 90).astype(int)
    jr = m['jaar'].to_numpy()
    V = getattr(model, 'gas_vorm_dag', None)
    if V is None or len(V) != 8:
        V = np.zeros((8, 5)); rng = np.random.default_rng(1)
        for gg in range(8):
            uu = u[(g == gg) & ~val & (jr <= p.split_jaar) & (jr >= 2010)]
            sub = rng.choice(uu, min(200000, len(uu)), replace=False)
            f = lambda th: np.mean(_tp_nll(th, sub))
            r = minimize(f, [0, 0, 1, 0, 1], method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-9})
            r = minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-5, 'fatol': 1e-9})
            th = r.x
            V[gg] = [1 / (1 + np.exp(-th[0])), np.exp(th[1]), 1.5 + 28.5 / (1 + np.exp(-th[2])), np.exp(th[3]), 1.5 + 28.5 / (1 + np.exp(-th[4]))]
        model.gas_vorm_dag = V
    pit = np.full(len(u), np.nan)
    for gg in range(8):
        s = (g == gg) & np.isfinite(u)
        pit[s] = _tp_cdf_v(V[gg], u[s])
    z = pd.Series(ndtri(np.clip(pit, 1e-300, 1 - 1e-16)), index=m.index)
    return z[m['jaar'] >= vanaf_jaar]


def normaal_z_dag_kal(model, venster=730, q=0.02, niv_grenzen=(90.0, 97.0), min_n=3000, per_afstand=True, max_factor=8, erv_grenzen=(3, 10, 50), q_traag=None, xi_max=0.9, veld_grenzen=(), xi_samen=True, traag_empirisch=True):
    """Geijkte normaal-equivalente dag-z voor de kans per rit op de pagina (de vormlaag gebruikt normaal_z_dag).
    Per kalendermaand en groep (afstand x niveau vooraf (niv_grenzen) x ervaring op de afstand (n_prior, erv_grenzen) x veldgrootte van de race
    (veld_grenzen; in kleine velden is het dag-effect onzeker: minder snelle, veel meer trage uitschieters)) wordt de verdeling van u = fout / sd (t.o.v. het veld)
    geschat op de ritten vóór die maand (alleen eerdere data, valpartijen niet): het venster begint bij `venster` dagen en verdubbelt
    (tot max_factor x) tot er min_n ritten in de groep zitten, zodat ook de elite per afstand een eigen staart houdt.
    Empirisch tussen het q- en (1-q)-kwantiel, daarbuiten aan elke kant een gegeneraliseerde Pareto-staart (xi >= 0).
    Is er ook dan te weinig, dan de lagere niveauband; aan het begin van de data het eerste volle venster."""
    from scipy.stats import genpareto as _gp
    from scipy.special import ndtri
    p = model.params
    _, err, var, m = voorspelscore(model, 0, dag=True)
    u = err / np.sqrt(var)
    val = ((err > 0) & (m['z'].to_numpy() > p.dip_val_z) & (p.dip_val_z > 0)) | ~np.isfinite(u)
    dg = m['dag'].to_numpy().astype(np.int64)
    dat = pd.to_datetime(dg, unit='D'); mnd = (dat.year * 12 + dat.month - 1).to_numpy()
    nb = len(niv_grenzen) + 1; ne = len(erv_grenzen) + 1
    nveld = m.groupby(['event_id', 'gender', 'afstand_idx'])['skater_id'].transform('size').to_numpy()       # klein veld: dag-effect onzeker
    band = (np.digitize(_niv_pre(m), niv_grenzen) + nb * np.digitize(m['n_prior'].to_numpy(), erv_grenzen)     # nieuwe profs: vaker extreem traag
            + nb * ne * np.digitize(nveld, veld_grenzen))
    nbanden = nb * ne * (len(veld_grenzen) + 1)
    ag = m['afstand_idx'].to_numpy() if per_afstand else GAS_GROEP[m['afstand_idx'].to_numpy()]     # per afstand: de 1000 m heeft een veel dikkere trage staart dan de 500 m
    q_traag = q if q_traag is None else q_traag       # trage kant: incidenten (uitglijders, 5-30% trager) liggen ver in de staart
    def staart(y, xi_vast=None):
        if xi_vast is not None:                      # vorm vast (geschat op meer data), alleen de schaal per groep
            return xi_vast, float(_gp.fit(y, f0=xi_vast, floc=0.0)[2]) if xi_vast > 0 else float(np.mean(y))
        xi, _, sg = _gp.fit(y, floc=0.0)
        if xi < 0:
            xi, sg = 0.0, float(np.mean(y))
        return float(min(xi, xi_max)), float(sg)
    xi_cache = {}
    def xi_band(b, d0, kant):
        """Staartvorm per niveau x ervaring (alle afstanden samen, venster tot d0): de schatting van xi uit kleine groepen is te laag."""
        k_ = (b, d0, kant)
        if k_ not in xi_cache:
            ref_ = referentie(('alle', b), d0)
            if ref_ is None or len(ref_) < 2000:
                xi_cache[k_] = None
            else:
                ref_ = np.sort(ref_)
                if kant == 0:
                    t_ = np.quantile(ref_, q); xi_cache[k_] = staart(t_ - ref_[ref_ < t_])[0]
                else:
                    t_ = np.quantile(ref_, 1 - q_traag); xi_cache[k_] = staart(ref_[ref_ > t_] - t_)[0]
        return xi_cache[k_]
    emp_cache = {}
    def traag_staart(b, d0):
        """Empirische verdeling van de gestandaardiseerde overschrijdingen voorbij de (1-q_traag)-drempel, alle afstanden van deze band samen
        (elke afstand gedeeld door zijn eigen gemiddelde overschrijding): incidenten maken een tweede bult ver in de trage staart die een GPD mist."""
        k_ = (b, d0)
        if k_ not in emp_cache:
            ys = []
            for g_ in np.unique(ag):
                r_ = referentie((g_, b), d0)
                if r_ is None or len(r_) < 300:
                    continue
                t_ = np.quantile(r_, 1 - q_traag); e_ = r_[r_ > t_] - t_
                if len(e_) >= 5 and e_.mean() > 0:
                    ys.append(e_ / e_.mean())
            emp_cache[k_] = np.sort(np.concatenate(ys)) if ys and sum(len(y) for y in ys) >= 200 else None
        return emp_cache[k_]
    groep = {}                                       # per (afstand of samengevoegde afstanden, band): dagen en u (gesorteerd op dag)
    ags = GAS_GROEP[m['afstand_idx'].to_numpy()]
    def _maak(sleutel, s_):
        o_ = np.argsort(dg[s_], kind='stable'); groep[sleutel] = (dg[s_][o_], u[s_][o_])
    for b in range(nbanden):
        for g in np.unique(ag):
            _maak((g, b), (ag == g) & (band == b) & ~val)
        for g in np.unique(ags):
            _maak(('groep', g, b), (ags == g) & (band == b) & ~val)
        _maak(('alle', b), (band == b) & ~val)
    def referentie(sleutel, d0):
        dd, uu = groep[sleutel]
        e_ = np.searchsorted(dd, d0)
        w = venster
        while w <= venster * max_factor:
            a_ = np.searchsorted(dd, d0 - w)
            if e_ - a_ >= min_n:
                return uu[a_:e_]
            w *= 2
        return None
    pit = np.full(len(u), np.nan)
    for M in np.unique(mnd):
        d0 = int((pd.Timestamp(year=int(M // 12), month=int(M % 12) + 1, day=1) - pd.Timestamp('1970-01-01')).days)
        for g in np.unique(ag[mnd == M]):
            for b in range(nbanden - 1, -1, -1):
                doel = (mnd == M) & (ag == g) & (band == b) & np.isfinite(u)
                if not doel.any():
                    continue
                e0 = (b // nb) * nb; ref = None
                # te klein: eerst afstanden samenvoegen binnen hetzelfde niveau en dezelfde ervaring (de staart in z hangt sterk aan het
                # niveau: bij toppers is een uitglijder een veel grotere z), pas daarna een lagere niveauband
                volgorde = [k_ for bb in range(b, e0 - 1, -1) for k_ in ((g, bb), ('groep', GAS_GROEP[g] if per_afstand else g, bb), ('alle', bb))]
                for k_ in volgorde:
                    ref = referentie(k_, d0)
                    if ref is not None:
                        break
                if ref is None:                      # begin van de data: eerste ritten van de eerste groep die er genoeg heeft
                    for k_ in volgorde:
                        if len(groep[k_][1]) >= 200:
                            ref = groep[k_][1][:max(min_n, 200)]; break
                if ref is None or len(ref) < 200:
                    continue
                ref = np.sort(ref)
                lo, hi = np.quantile(ref, q), np.quantile(ref, 1 - q_traag)
                xl, sl = staart(lo - ref[ref < lo], xi_band(b, d0, 0) if xi_samen else None)
                xh, sh = staart(ref[ref > hi] - hi, xi_band(b, d0, 1) if xi_samen else None)
                x = u[doel]
                pp = np.searchsorted(ref, x, side='right') / len(ref)
                pp = np.where(x < lo, q * _gp.sf(lo - x, xl, 0.0, sl), pp)
                Y = traag_staart(b, d0) if traag_empirisch else None
                if Y is not None:
                    eh = float(np.mean(ref[ref > hi] - hi)) if (ref > hi).sum() >= 5 else sh
                    yy = (x - hi) / max(eh, 1e-9)
                    sf_emp = 1.0 - np.searchsorted(Y, yy, side='right') / len(Y)
                    ymax = Y[-1]; xg, sg_ = staart(Y[Y > np.quantile(Y, 0.9)] - np.quantile(Y, 0.9))
                    voorbij = yy > ymax          # voorbij de verste waarneming: GPD op de verste 10% van de samengevoegde overschrijdingen
                    sf_emp = np.where(voorbij, (1.0 / len(Y)) * _gp.sf(yy - ymax, xg, 0.0, sg_), np.maximum(sf_emp, 0.5 / len(Y)))
                    pp = np.where(x > hi, 1 - q_traag * sf_emp, pp)
                else:
                    pp = np.where(x > hi, 1 - q_traag * _gp.sf(x - hi, xh, 0.0, sh), pp)
                pit[doel] = np.clip(pp, 1e-12, 1 - 1e-12)
    return pd.Series(ndtri(pit), index=m.index)


KANS_GEBIEDEN = np.array([0.0, 0.02, 0.07, 0.93, 0.98, 1.0])     # snel extreem | piek | gewoon | slecht | traag extreem


def z_rijder(model, z, m_krimp=200.0, recent_n=5, fit_tot=None):
    """Kans per rit met de eigen gebiedskansen van de rijder. De geijkte kans p = Phi(z) valt in een van de vijf gebieden (KANS_GEBIEDEN).
    Lang geheugen: per rijder het aantal eerdere ritten (alle afstanden, eerdere dagen, valpartijen niet) per gebied, gekrompen naar de
    populatiekans met m_krimp ritten. Kort geheugen: het aantal ritten in elk van de twee extreme gebieden binnen de laatste recent_n
    ritten; de factor daarvoor (na het lange geheugen) wordt geschat op de jaren <= fit_tot (standaard split_jaar), apart per kant.
    Trage schokken komen in reeksen (ziekte, blessure): na een trage schok is de kans op de volgende ~3x zo groot.
    p wordt binnen zijn gebied lineair herschaald naar de eigen kansen. Geeft (z per obs-rij, factoren kort geheugen)."""
    from scipy.stats import norm
    from scipy.special import ndtri
    fit_tot = model.params.split_jaar if fit_tot is None else fit_tot
    o = model.obs.loc[z.index]
    ok = ((~o['verdacht']) & ~(o['z'] > model.params.dip_val_z) & np.isfinite(z)).to_numpy()
    p = norm.cdf(z.to_numpy()); G = KANS_GEBIEDEN; P0 = np.diff(G)
    reg = np.clip(np.searchsorted(G, p, side='right') - 1, 0, 4)
    d = pd.DataFrame({'sk': o['skater_id'].to_numpy(), 'dag': o['dag'].to_numpy(), 'reg': reg, 'ok': ok,
                      'jr': o['jaar'].to_numpy()}, index=o.index).sort_values(['sk', 'dag'], kind='stable')
    C = np.zeros((len(d), 5)); R = np.zeros((len(d), 2))
    for k in range(5):
        x = ((d['reg'] == k) & d['ok']).astype(float)
        C[:, k] = (x.groupby(d['sk']).cumsum() - x - (x.groupby([d['sk'], d['dag']]).cumsum() - x)).to_numpy()     # eerdere dagen
    # kort geheugen: extreme ritten in de laatste recent_n ritten van eerdere dagen (per rijder-dag samengevat)
    dd = d[d['ok']].groupby(['sk', 'dag']).agg(e0=('reg', lambda r: (r == 0).sum()), e4=('reg', lambda r: (r == 4).sum()), n=('reg', 'size')).reset_index()
    for c_, col in ((0, 'e0'), (1, 'e4')):
        tot = dd.groupby('sk')[col].cumsum() - dd[col]; ntot = dd.groupby('sk')['n'].cumsum() - dd['n']
        # som over de laatste recent_n ritten benaderd met de laatste rijder-dagen tot er recent_n ritten zijn
        cs = dd.groupby('sk')[col].transform(lambda x: x.shift(1).rolling(recent_n, min_periods=1).sum()).fillna(0.0)
        dd['r' + str(c_)] = cs.to_numpy()
    rr = d[['sk', 'dag']].merge(dd[['sk', 'dag', 'r0', 'r1']], on=['sk', 'dag'], how='left').fillna(0.0)
    R[:, 0] = rr['r0'].to_numpy(); R[:, 1] = rr['r1'].to_numpy()
    C = pd.DataFrame(C, index=d.index).reindex(o.index).to_numpy(); R = pd.DataFrame(R, index=d.index).reindex(o.index).to_numpy()
    W = (C + m_krimp * P0) / (C.sum(1, keepdims=True) + m_krimp)
    # factor kort geheugen per kant en aantal recente extreme ritten (0, 1, 2+), geschat op de fitjaren na het lange geheugen
    jr = o['jaar'].to_numpy(); fit = ok & (jr <= fit_tot) & (jr >= 2010)
    fac = {}
    for kant, k in ((0, 0), (1, 4)):
        rb = np.minimum(R[:, kant], 2).astype(int)
        for b in (0, 1, 2):
            s_ = fit & (rb == b)
            fac[(k, b)] = float((reg[s_] == k).mean() / max(W[s_, k].mean(), 1e-9)) if s_.sum() > 500 else 1.0
        f = np.array([fac[(k, b)] for b in (0, 1, 2)])[rb]
        nieuw = np.minimum(W[:, k] * f, 0.5)
        rest = [x for x in range(5) if x != k]
        W[:, rest] *= ((1 - nieuw) / np.maximum(1 - W[:, k], 1e-9))[:, None]; W[:, k] = nieuw
    cw = np.c_[np.zeros(len(W)), np.cumsum(W, 1)]
    i = reg; r = np.arange(len(p))
    pr = cw[r, i] + (p - G[i]) / P0[i] * W[r, i]
    return pd.Series(ndtri(np.clip(pr, 1e-12, 1 - 1e-12)), index=o.index), fac


def _ou_vooruit(p, f, v, d0, dag):
    """OU-vorm van dag d0 naar `dag` (zonder nieuwe races)."""
    if p.vorm_ou_sd <= 0.0:
        return 0.0, 0.0
    a = math.exp(-max(dag - d0, 0) / p.vorm_ou_tau)
    return f * a, a * a * v + p.vorm_ou_sd ** 2 * (1.0 - a * a)


IJS_KNOPEN = np.array([0.0, 50.0, 75.0, 90.0, 97.0, 100.0])
HOOGLAND = ('Salt Lake City (USA)', 'Calgary (CAN)')


def _ijs_dag_shift(model, m):
    """Verschuiving (ln-tijd) van de dag-relatieve fout: het dag-effect wordt voor het hele veld geschat, maar op snel ijs en vooral
    op de hooglandbanen verschilt de winst per niveau (de top en de breedte lopen verder uit t.o.v. het midden van het veld)."""
    q = np.asarray(model.params.ijs_dag_par, float); niv = _niv_pre(m)
    k = np.interp(niv, IJS_KNOPEN, q[:6]); h = np.interp(niv, IJS_KNOPEN, q[6:12])
    d = m['theta'].to_numpy() - model.anker_std
    return d * k + 0.01 * m['baan'].isin(HOOGLAND).to_numpy() * h


def _vorm_eff(q, Zk, F, S, Zl, w, N=0.0):
    """(verschuiving in dag-sd, log-factor snelle kant, log-factor trage kant) uit de vormtoestand; w = topgewicht (0 onder niveau 80, 1 aan de top).
    q[14] (staartkalibratie): extra log-factor aan de snelle kant per recente snelle uitschieter (N = vervallen aantal dagen met z < -3)."""
    aK0, aK1, aL0, aL1, a, aS, bf, bs, bsS, bf1 = q[4:14]
    g = q[14] if len(q) > 14 else 0.0
    mu = (aK0 + aK1 * w) * Zk + (aL0 + aL1 * w) * Zl + a * F + aS * S
    return mu, (bf + bf1 * w) * F + g * np.sqrt(N), bs * F + bsS * S


def _vorm_w(niv):
    return np.clip((np.asarray(niv, float) - 80.0) / 20.0, 0.0, 1.0)


def schat_vorm(model):
    """Vormtoestand per rijder uit de dag-z van EERDERE dagen (dag-z zonder deze laag), zie gas_lab.vorm_pass.
    model.vorm_f: toestand vooraf per obs-rij; model.vorm_dag: toestand na elke rijder-dag (voor voorspellingen vooruit)."""
    from gas_lab import vorm_pass
    p = model.params
    model.vorm_f = None; model.vorm_dag = None; model.diepte_f = None; model.diepte_dag = None
    z = normaal_z_dag(model, 0)
    m = model.obs.loc[z.index]
    zz = z.to_numpy().astype(float)
    val = ~np.isfinite(zz) | (zz > p.dip_val_z)
    zz = np.where(np.isfinite(zz), zz, 0.0)
    o = np.lexsort((m['afstand_idx'].to_numpy(), m['event_id'].to_numpy(), m['dag'].to_numpy(), m['skater_id'].to_numpy()))
    sk = m['skater_id'].to_numpy().astype(np.int64)[o]; dg = m['dag'].to_numpy().astype(np.int64)[o]
    q = p.vorm_par
    uit = np.zeros((len(o), 5)); dsk = np.zeros(len(o), np.int64); ddg = np.zeros(len(o), np.int64); dst = np.zeros((len(o), 5))
    nd = vorm_pass(sk, dg, zz[o], val[o], float(q[0]), float(q[1]), float(q[2]), float(q[3]), uit, dsk, ddg, dst, float(p.vorm_plateau))
    U = np.zeros((len(o), 5)); U[o] = uit
    model.vorm_f = pd.DataFrame(U.astype(np.float32), index=m.index, columns=['Zk', 'F', 'S', 'Zl', 'N'])
    dsk, ddg, dst = dsk[:nd], ddg[:nd].astype(np.int32), dst[:nd].astype(np.float32)
    st_ = np.flatnonzero(np.r_[True, dsk[1:] != dsk[:-1]]); ei = np.r_[st_[1:], nd]
    model.vorm_dag = {'dag': ddg, 'st': dst, 'pos': {int(dsk[a]): (int(a), int(b)) for a, b in zip(st_, ei)}}
    if p.vorm_plateau > 0:          # plateau: de toestand na elke rijder-dag mist de bijdragen binnen het plateau; die per rijder-dag bewaren
        model.vorm_dag['zp'] = _vorm_plateau_reeks(sk, dg, zz[o], val[o], ddg, nd)
    if p.piek_diepte:
        from gas_lab import diepte_pass
        from scipy.stats import norm as _nrm
        zlo = float(_nrm.ppf(p.piek_q[0]))
        uit2 = np.zeros((len(o), 2)); dsk2 = np.zeros(len(o), np.int64); ddg2 = np.zeros(len(o), np.int64); dst2 = np.zeros((len(o), 2))
        nd2 = diepte_pass(sk, dg, zz[o], val[o], zlo, float(p.piek_diepte_par[0]), uit2, dsk2, ddg2, dst2)
        U2 = np.zeros((len(o), 2)); U2[o] = uit2
        model.diepte_f = pd.DataFrame(U2.astype(np.float32), index=m.index, columns=['E', 'N'])
        jr_ = m['jaar'].to_numpy(); fit_ = (jr_ >= 2010) & (jr_ <= p.split_jaar) & ~val & (zz < zlo)
        model.diepte_e0 = float(np.mean(zlo - zz[fit_]))
        dsk2, ddg2, dst2 = dsk2[:nd2], ddg2[:nd2].astype(np.int32), dst2[:nd2].astype(np.float32)
        s2 = np.flatnonzero(np.r_[True, dsk2[1:] != dsk2[:-1]]); e2 = np.r_[s2[1:], nd2]
        model.diepte_dag = {'dag': ddg2, 'st': dst2, 'pos': {int(dsk2[a]): (int(a), int(b)) for a, b in zip(s2, e2)}}
    model.schaal_f = None; model.schaal_dag = None
    if p.schaal:
        schat_schaal_toestand(model, sk, dg, zz[o], val[o], o, m)
    model.staart_f = None; model.staart_dag = None
    if p.staart:
        schat_staart_toestand(model, sk, dg, zz[o], val[o], o, m)
    for a in ('_z_emp', '_z_emp_sorted'):
        if hasattr(model, a):
            delattr(model, a)
    return model.vorm_f


def _diepte_tau(model, E, N):
    H, mm, kap = model.params.piek_diepte_par
    e0 = getattr(model, 'diepte_e0', None) or 0.49
    return ((np.asarray(E, float) + mm * e0) / (np.asarray(N, float) + mm) / e0) ** kap


def diepte_toestand(model, skater_id, dag):
    """Piekdiepte tau (1 = gewoon) voor een race op `dag`, uit alleen eerdere dagen."""
    r = getattr(model, 'diepte_dag', None)
    if not model.params.piek_diepte or r is None:
        return 1.0
    ab = r['pos'].get(int(skater_id))
    if ab is None:
        return float(_diepte_tau(model, 0.0, 0.0))
    a, b = ab
    k = int(np.searchsorted(r['dag'][a:b], dag, side='left')) if dag is not None else b - a
    if k == 0:
        return float(_diepte_tau(model, 0.0, 0.0))
    st = r['st'][a + k - 1]; f = math.exp(-((dag if dag is not None else int(r['dag'][b - 1])) - int(r['dag'][a + k - 1])) / model.params.piek_diepte_par[0])
    return float(_diepte_tau(model, st[0] * f, st[1] * f))


def schat_staart_toestand(model, sk, dg, z, val, o, m):
    """Per rijder (alle afstanden) de vervallen telling van extreme snelle ritten (dag-z < drempel) en van alle ritten, alleen eerdere dagen.
    model.staart_f per obs-rij (C, R), model.staart_dag na elke rijder-dag; model.staart_r0 = frequentie op de fitjaren."""
    from gas_lab import diepte_pass
    p = model.params; H, zd = float(p.staart_par[2]), float(p.staart_par[4])
    out = []
    for zlo in (zd, 1e9):                        # 1e9: elke rit telt (som E is dan zinloos, alleen N wordt gebruikt)
        uit = np.zeros((len(sk), 2)); a_ = np.zeros(len(sk), np.int64); b_ = np.zeros(len(sk), np.int64); st = np.zeros((len(sk), 2))
        nd = diepte_pass(sk, dg, z, val, zlo, H, uit, a_, b_, st)
        out.append((uit[:, 1].copy(), a_[:nd], b_[:nd], st[:nd, 1].copy()))
    U = np.zeros((len(sk), 2)); U[o, 0] = out[0][0]; U[o, 1] = out[1][0]
    model.staart_f = pd.DataFrame(U.astype(np.float32), index=m.index, columns=['C', 'R'])
    dsk, ddg = out[0][1], out[0][2].astype(np.int32)
    s2 = np.flatnonzero(np.r_[True, dsk[1:] != dsk[:-1]]); e2 = np.r_[s2[1:], len(dsk)]
    model.staart_dag = {'dag': ddg, 'st': np.c_[out[0][3], out[1][3]].astype(np.float32),
                        'pos': {int(dsk[a]): (int(a), int(b)) for a, b in zip(s2, e2)}}
    jr = m['jaar'].to_numpy()[o]; fit = (jr >= 2010) & (jr <= p.split_jaar) & ~val
    model.staart_r0 = float(np.mean(z[fit] < zd))


def schat_schaal_toestand(model, sk, dg, z, val, o, m):
    """Per rijder (alle afstanden) vervallen sommen van dag-z^2 en aantallen, apart voor snelle (z < 0) en trage ritten, alleen eerdere dagen."""
    from gas_lab import schaal_pass
    H = float(model.params.schaal_par[2])
    uit = np.zeros((len(sk), 4)); a_ = np.zeros(len(sk), np.int64); b_ = np.zeros(len(sk), np.int64); st = np.zeros((len(sk), 4))
    nd = schaal_pass(sk, dg, np.clip(z, -5.0, 5.0), val, H, uit, a_, b_, st)
    U = np.zeros((len(sk), 4)); U[o] = uit
    model.schaal_f = pd.DataFrame(U.astype(np.float32), index=m.index, columns=['Sf', 'Nf', 'Ss', 'Ns'])
    dsk, ddg = a_[:nd], b_[:nd].astype(np.int32)
    s2 = np.flatnonzero(np.r_[True, dsk[1:] != dsk[:-1]]); e2 = np.r_[s2[1:], nd]
    model.schaal_dag = {'dag': ddg, 'st': st[:nd].astype(np.float32), 'pos': {int(dsk[a]): (int(a), int(b)) for a, b in zip(s2, e2)}}


def _schaal_fac(model, Sf, Nf, Ss, Ns):
    """(log-factor snelle kant, log-factor trage kant) op de sd: gamma * log((S + m) / (N + m)), d.w.z. de eigen gerealiseerde spreiding
    (gemiddelde normaal-equivalente z^2 aan die kant, gekrompen naar 1) tot de macht gamma."""
    gf, gs, H, mm = model.params.schaal_par
    lf = gf * np.log((np.asarray(Sf, float) + mm) / (np.asarray(Nf, float) + mm))
    ls = gs * np.log((np.asarray(Ss, float) + mm) / (np.asarray(Ns, float) + mm))
    return lf, ls


def schaal_toestand(model, skater_id, dag):
    """(factor snelle kant, factor trage kant) op de sd voor een race op `dag`, alleen eerdere dagen; (1, 1) als de laag uit staat."""
    p = model.params
    r = getattr(model, 'schaal_dag', None)
    if not p.schaal or r is None or dag is None:
        return 1.0, 1.0
    ab = r['pos'].get(int(skater_id))
    if ab is None:
        return 1.0, 1.0
    a, b = ab
    k = int(np.searchsorted(r['dag'][a:b], dag, side='left'))
    if k == 0:
        return 1.0, 1.0
    f = math.exp(-(dag - int(r['dag'][a + k - 1])) / p.schaal_par[2]); S = r['st'][a + k - 1].astype(float) * f
    lf, ls = _schaal_fac(model, S[0], S[1], S[2], S[3])
    return float(math.exp(lf)), float(math.exp(ls))


def _staart_fac(model, C, R, niv):
    """Extra factor op de piekdiepte: exp(a_top * w(niveau)) * (eigen frequentie extreme ritten / gemiddelde)^b, gekrompen met m ritten."""
    a_top, b, H, mm, zd = model.params.staart_par
    r0 = getattr(model, 'staart_r0', None) or 1e-3
    rate = (np.asarray(C, float) + mm * r0) / (np.asarray(R, float) + mm)
    return np.exp(a_top * _vorm_w(niv)) * (rate / r0) ** b


def staart_toestand(model, skater_id, dag, niv):
    """Factor van de snelle staart voor een race op `dag` (alleen eerdere dagen); 1 als de laag uit staat."""
    p = model.params
    if not p.staart or getattr(model, 'staart_dag', None) is None:
        return 1.0
    r = model.staart_dag; ab = r['pos'].get(int(skater_id)); C = R = 0.0
    if ab is not None and dag is not None:
        a, b = ab
        k = int(np.searchsorted(r['dag'][a:b], dag, side='left'))
        if k > 0:
            f = math.exp(-(dag - int(r['dag'][a + k - 1])) / p.staart_par[2]); C, R = float(r['st'][a + k - 1, 0]) * f, float(r['st'][a + k - 1, 1]) * f
    return float(_staart_fac(model, C, R, np.array([float(niv)]))[0])


PIEK_GIDX = {'100m': 0, '300m': 0, '500m': 0, '1000m': 0, '1500m': 1, '3000m': 1, '5000m': 2, '10000m': 2}


def vorm_toestand(model, skater_id, dag, niv):
    """(verschuiving in sd, factor snelle kant, factor trage kant) voor een race op `dag`, uit alleen eerdere dagen."""
    p = model.params
    r = getattr(model, 'vorm_dag', None)
    if not p.vorm or r is None or dag is None:
        return 0.0, 1.0, 1.0
    ab = r['pos'].get(int(skater_id))
    if ab is None:
        return 0.0, 1.0, 1.0
    a, b = ab
    k = int(np.searchsorted(r['dag'][a:b], dag, side='left'))
    if k == 0:
        return 0.0, 1.0, 1.0
    st = r['st'][a + k - 1]; gap = dag - int(r['dag'][a + k - 1]); q = p.vorm_par
    fk = math.exp(-gap / q[0]); fl = math.exp(-gap / q[2])
    zk = st[0] * fk
    if p.vorm_plateau > 0 and 'zp' in r:
        zk = _vorm_plateau_zk(r, a, a + k, dag, q[0], p.vorm_plateau)
    mu, lf, ls = _vorm_eff(q, zk, st[1] * fl, st[2] * fl, st[3] * fl, float(_vorm_w(niv)), (st[4] * fl if len(st) > 4 else 0.0))
    return float(mu), math.exp(lf), math.exp(ls)


def _vorm_plateau_reeks(sk, dg, z, val, ddg, nd):
    """Gemiddelde dag-z (geknipt op +-4, zonder valpartijen) per rijder-dag, in dezelfde volgorde als vorm_dag (voor het plateau vooruit)."""
    d = pd.DataFrame({'sk': sk, 'dg': dg, 'z': np.clip(z, -4.0, 4.0), 'ok': ~val})
    d = d[d['ok']].groupby(['sk', 'dg'], sort=False)['z'].mean()
    keys = pd.MultiIndex.from_arrays([np.asarray(sk)[np.r_[True, (np.diff(sk) != 0) | (np.diff(dg) != 0)]],
                                      np.asarray(dg)[np.r_[True, (np.diff(sk) != 0) | (np.diff(dg) != 0)]]])
    out = d.reindex(keys).to_numpy(dtype=float)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)[:nd]


def _vorm_plateau_zk(r, a, e, dag, Hk, P):
    """Korte vorm Zk op `dag` uit de rijder-dagen r[a:e] (alle vóór `dag`): gewicht 1 tot P dagen oud, daarna exp(-(leeftijd - P) / Hk)."""
    leeft = dag - r['dag'][a:e].astype(np.int64); zp = r['zp'][a:e].astype(float)
    w = np.exp(-np.maximum(leeft - P, 0.0) / Hk)
    ok = np.isfinite(zp) & (w > 1e-12)
    return float(np.sum(w[ok] * zp[ok]))


# =============================================================================
# VERMOEIDHEID: eerder gereden meters dezelfde dag en de dag ervoor
# =============================================================================
VERM_KOL = ['vm_s', 'vm_m', 'vm_l', 'vg_s', 'vg_m', 'vg_l']     # log(1 + km) dezelfde dag / de dag ervoor, per afstandsgroep (sprint, midden, lang)
_METERS_IDX = np.array([AFSTAND_METERS[a] for a in AFSTANDEN], float)


def _verm_X(gi, m_dag, m_gis):
    """Kenmerkmatrix (n x 6) uit afstandsgroep, eerder gereden meters dezelfde dag en meters de dag ervoor."""
    gi = np.asarray(gi); X = np.zeros((len(gi), len(VERM_KOL)))
    r = np.arange(len(gi))
    X[r, gi] = np.log1p(np.asarray(m_dag, float) / 1000.0)
    X[r, 3 + gi] = np.log1p(np.asarray(m_gis, float) / 1000.0)
    return X


def _verm_top(X, niv, top=False):
    """top: [X, X * w(niveau)] (eigen effect aan de top; w = topgewicht van de vorm, 0 onder niveau 80, 1 aan de top);
    anders X * (1 - w): alleen de breedte (volledig getoetst: aan de top maakte de vermoeidheidsterm de voorspelling na 2019 slechter)."""
    w = _vorm_w(niv)[:, None]
    return np.c_[X, X * w] if top else X * (1.0 - w)


def verm_kenmerken(model):
    """Per obs-rij: eerder gereden meters door deze rijder op dezelfde dag (kortste afstand eerst, zoals de volgorde in de data)
    en het totaal van de dag ervoor; alleen niet-verdachte races tellen als belasting. Zet ook model.verm_reeks (meters per rijder-dag)
    voor voorspellingen vooruit. Kolommen VERM_KOL plus het gemiddelde per race (`_r`, voor de dag-relatieve variant)."""
    o = model.obs
    n = len(o)
    sk = o['skater_id'].to_numpy().astype(np.int64); dg = o['dag'].to_numpy().astype(np.int64); ai = o['afstand_idx'].to_numpy().astype(np.int64)
    mm = _METERS_IDX[ai] * (~o['verdacht'].to_numpy())
    od = np.lexsort((np.arange(n), ai, dg, sk))
    s_sk, s_dg, s_mm = sk[od], dg[od], mm[od]
    nieuw = np.r_[True, (s_sk[1:] != s_sk[:-1]) | (s_dg[1:] != s_dg[:-1])]
    grp = np.cumsum(nieuw) - 1
    cs = np.cumsum(s_mm); start = np.flatnonzero(nieuw)
    voor = cs - s_mm - np.r_[0.0, cs][start][grp]               # meters eerder dezelfde dag
    tot = np.bincount(grp, weights=s_mm)                          # meters per rijder-dag
    d_sk, d_dg = s_sk[start], s_dg[start]
    sleutel = d_sk * 100000 + d_dg
    gis = pd.Series(tot, index=sleutel).reindex(s_sk * 100000 + s_dg - 1).fillna(0.0).to_numpy()
    m_dag = np.empty(n); m_dag[od] = voor
    m_gis = np.empty(n); m_gis[od] = gis
    X = _verm_X(AFST_GROEP[ai], m_dag, m_gis)
    F = pd.DataFrame(X.astype(np.float32), index=o.index, columns=VERM_KOL)
    g = F.groupby([o['event_id'].to_numpy(), o['gender'].to_numpy(), ai])
    for c in VERM_KOL:
        F[c + '_r'] = g[c].transform('mean').astype(np.float32)
    st_ = np.flatnonzero(np.r_[True, d_sk[1:] != d_sk[:-1]]); ei = np.r_[st_[1:], len(d_sk)]
    model.verm_reeks = {'dag': d_dg.astype(np.int32), 'm': tot.astype(np.float32),
                        'pos': {int(d_sk[a]): (int(a), int(b)) for a, b in zip(st_, ei)}}
    return F


def schat_verm(model):
    """Zet model.verm_f en model.verm_coef: kleinste kwadraten van de (geknipte) fout in sd vóór de vermoeidheidslaag op VERM_KOL,
    alleen jaren <= split_jaar, zonder valpartijen. 'vooraf': absolute verschuiving (met constante); 'dag': binnen de race gecentreerd
    (het dag-effect vangt de vermoeidheid van het hele veld al)."""
    p = model.params
    model.verm_coef = None
    model.verm_f = verm_kenmerken(model)
    coef = {}
    for dag in (False, True):
        model._verm_vang = True
        try:
            _, err, var, m = voorspelscore(model, 0, dag=dag)
            u = np.asarray(model._verm_u, float)
        finally:
            model._verm_vang = False; model._verm_u = None
        val = (err > 0) & (m['z'].to_numpy() > p.dip_val_z)
        ok = np.isfinite(u) & ~val & (m['jaar'].to_numpy() <= p.split_jaar)
        V = model.verm_f.reindex(m.index)
        X = V[VERM_KOL].to_numpy(dtype=float); y = np.clip(u, -4, 4)
        if dag:
            X = X - V[[c + '_r' for c in VERM_KOL]].to_numpy(dtype=float)
        X = _verm_top(X, _niv_pre(m), p.verm_top)
        if dag:
            sl = pd.Series(np.where(ok, y, np.nan)).groupby([m['event_id'].to_numpy(), m['gender'].to_numpy(), m['afstand_idx'].to_numpy()])
            y = y - sl.transform('mean').to_numpy()
            b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        else:
            b = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[ok]], y[ok], rcond=None)[0][1:]
        coef['dag' if dag else 'vooraf'] = b if (p.verm_dag or not dag) else np.zeros_like(b)
    model.verm_coef = coef
    _nm = VERM_KOL + ([c + '_top' for c in VERM_KOL] if p.verm_top else [])
    print('verm_coef vooraf', dict(zip(_nm, coef['vooraf'].round(3))), 'dag', dict(zip(_nm, coef['dag'].round(3))), flush=True)
    for a in ('_z_emp', '_z_emp_sorted'):
        if hasattr(model, a):
            delattr(model, a)
    return coef


def verm_verschuiving(model, skater_id, afstand, dag, eerder=None, niv=None):
    """Verschuiving (sd, + = trager) van de voorspelling vooraf door vermoeidheid: `eerder` = afstanden die de rijder dezelfde dag al reed,
    plus de meters van de dag ervoor uit de historie."""
    vc = getattr(model, 'verm_coef', None); r = getattr(model, 'verm_reeks', None)
    if not model.params.verm or vc is None or dag is None:
        return 0.0
    me = float(sum(AFSTAND_METERS[a] for a in eerder)) if eerder else 0.0
    mg = 0.0
    ab = r['pos'].get(int(skater_id)) if r is not None else None
    if ab is not None:
        a, b = ab
        k = a + int(np.searchsorted(r['dag'][a:b], dag - 1, side='left'))
        if k < b and int(r['dag'][k]) == dag - 1:
            mg = float(r['m'][k])
    X = _verm_top(_verm_X([AFST_GROEP[NAAR_INDEX[afstand]]], [me], [mg]), np.array([50.0 if niv is None else float(niv)]), model.params.verm_top)
    return float(X[0] @ vc['vooraf'])


VOL_BANDEN = np.array([50.0, 75.0, 90.0, 95.0, 98.0])


def schat_vol_ref(model):
    """Referentie (mediaan van log voorspel-sd) per afstand x niveauband: de volatiliteit van een rijder t.o.v. vergelijkbare rijders."""
    p = model.params
    bf = p.vol_asym_bf
    model.vol_ref = None
    _, err, var, m = voorspelscore(model, 0)
    lsd = 0.5 * np.log(var); ai = m['afstand_idx'].to_numpy(); b = np.digitize(_niv_pre(m), VOL_BANDEN)
    ref = np.zeros((len(AFSTANDEN), len(VOL_BANDEN) + 1))
    d = pd.DataFrame({'l': lsd, 'a': ai, 'b': b})[np.isfinite(lsd)]
    med = d.groupby(['a', 'b'])['l'].median()
    for (a, bb), v in med.items():
        ref[a, bb] = v
    model.vol_ref = ref
    for a in ('_z_emp', '_z_emp_sorted'):
        if hasattr(model, a):
            delattr(model, a)
    return ref


def vol_snel_factor(model, afstand, sd_x, t_lim, x, niv):
    """Factor op de sd aan de SNELLE kant voor een voorspelling vooraf (rijder met hoge volatiliteit: relatief smalle snelle kant)."""
    p = model.params
    vr = getattr(model, 'vol_ref', None)
    if p.vol_asym_bf == 0.0 or vr is None:
        return 1.0
    dl = math.exp(x); sd_t = sd_x * dl / (t_lim + dl)
    var = sd_t ** 2 + model.baan_effect(STANDAARD_BAAN)[1] ** 2
    j = NAAR_INDEX[afstand]
    rel = 0.5 * math.log(var) - vr[j, int(np.digitize(niv, VOL_BANDEN))]
    return float(math.exp(p.vol_asym_bf * rel))



def voorspelscore(model, vanaf_jaar=2014, nu=4.0, trend=True, dag=False):
    """Gemiddelde negatieve log-likelihood (Student-t) van 1-stap-vooruit voorspellingen in log-tijd.
    De voorspelling gebruikt alleen informatie van VOOR de race (rating + gemiddelde van de baan).
    dag=True: achteraf, t.o.v. het GESCHATTE dag-effect van die dag (theta, met zijn onzekerheid) i.p.v. de baan-prior:
    'hoe uitzonderlijk was deze rit t.o.v. het veld' (niet voor het toetsen van voorspellingen)."""
    o = model.obs
    m = o[(o['jaar'] >= vanaf_jaar) & (~o['verdacht']) & (o['n_prior'] >= 0)]
    t_lim = _tl_vector(model.t_lims, m['gender'].to_numpy(), m['afstand_idx'].to_numpy())
    delta = np.exp(m['x_prior'].to_numpy())
    that = t_lim + delta
    if dag:
        pred = np.log(that) + m['theta'].to_numpy()
        var = m['v_prior'].to_numpy() * (delta / that) ** 2 + m['v_theta'].to_numpy()
    else:
        pred = np.log(that) + m['mu_baan'].to_numpy()
        var = m['v_prior'].to_numpy() * (delta / that) ** 2 + m['var_baan'].to_numpy()
    if model.params.pred_sd_schaal:
        var = var * np.array([model.params.pred_sd_schaal.get(a, 1.0) for a in AFSTANDEN])[m['afstand_idx'].to_numpy()] ** 2
    sf = getattr(model, 'sd_factor', None)
    if sf is not None and model.params.sd_factoren:
        jj = m['afstand_idx'].to_numpy()
        var = var * sf[np.searchsorted(DT_KNOPEN, m['dt'].to_numpy()), _erv_bin_vec(m['n_prior'].to_numpy(), m['n_tot'].to_numpy()) + 1, AFST_GROEP[jj]] ** 2
    err = np.log(m['tijd'].to_numpy()) - pred
    if len(model.params.niv_knopen) or len(model.params.niv_rel_groepen):
        _f, _l = _niv_aanpassing(model.params, _niv_pre(m, model.params.niv_rel))
        if len(model.params.niv_rel_groepen):
            _sel = np.isin(m['groep'].to_numpy(), model.params.niv_rel_groepen)
            _fr = np.interp(_niv_pre(m, True), np.asarray(model.params.niv_rel_knopen, float), np.asarray(model.params.niv_rel_sd_fac, float))
            _f = np.where(_sel, _f * _fr, _f)       # extra factor bovenop de absolute (alleen opwaarts: relatieve toppers)
        var = var * _f ** 2
        err = err - _l * np.sqrt(var)
    tf = getattr(model, 'trend_f', None)
    if trend and model.params.trend_corr and tf is not None and getattr(model, 'trend_coef', None) is not None:
        _tk = TREND_KOL[:len(model.trend_coef)]
        _T = tf.reindex(index=m.index, columns=_tk + [c for c in ('efo', 'efs') if c in tf.columns])
        err = err - (_T[_tk].fillna(0.0).to_numpy() @ model.trend_coef) * np.sqrt(var)
        if (model.params.verr_c or model.params.verr_c2 or getattr(model.params, 'verr_c3', 0.0)) and 'ef' in _T.columns:
            var = var * _verr_factor(model.params, _T['ef'].to_numpy(), _T['es'].to_numpy(), _T['efo'].to_numpy(),
                                     _T['efs'].to_numpy() if 'efs' in _T.columns else 0.0) ** 2
    tab = getattr(model, 'loc_schaal_tab', None)
    if model.params.loc_schaal and tab is not None:
        gg = (m['gender'].to_numpy() == 'v').astype(int); jj = m['afstand_idx'].to_numpy()
        bb = np.digitize(_niv_pre(m), LS_RANDEN)
        err = err - tab['loc'][gg, jj, bb] * np.sqrt(var)
        var = var * tab['sc'][gg, jj, bb] ** 2
    gf = getattr(model, 'gas_f', None)
    if model.params.gas and gf is not None:
        G = gf.reindex(m.index).fillna(0.0)
        err = err - G['m'].to_numpy() * np.sqrt(var)
        var = var * np.exp(2.0 * G['l'].to_numpy())
    if dag and model.params.ijs_dag:
        err = err - _ijs_dag_shift(model, m)
    if getattr(model, '_verm_vang', False):
        model._verm_u = err / np.sqrt(var)               # voor schat_verm: fout vóór de vermoeidheidslaag
    vmf_ = getattr(model, 'verm_f', None); vmc_ = getattr(model, 'verm_coef', None)
    if model.params.verm and vmf_ is not None and vmc_ is not None:
        V_ = vmf_.reindex(m.index).fillna(0.0)
        X_ = V_[VERM_KOL].to_numpy(dtype=float)
        if dag:
            X_ = X_ - V_[[c + '_r' for c in VERM_KOL]].to_numpy(dtype=float)
        err = err - (_verm_top(X_, _niv_pre(m), model.params.verm_top) @ vmc_['dag' if dag else 'vooraf']) * np.sqrt(var)
    vr = getattr(model, 'vol_ref', None)
    model._vol_asym_lam = None
    lamf = lams = 0.0
    vf_ = getattr(model, 'vorm_f', None)
    if model.params.vorm and vf_ is not None:
        V_ = vf_.reindex(m.index).fillna(0.0)
        mu_, lf_, ls_ = _vorm_eff(model.params.vorm_par, V_['Zk'].to_numpy(), V_['F'].to_numpy(), V_['S'].to_numpy(), V_['Zl'].to_numpy(),
                                  _vorm_w(_niv_pre(m)), V_['N'].to_numpy() if 'N' in V_.columns else 0.0)
        vo = m['v_prior'].to_numpy() * (delta / that) ** 2; vt = m['v_theta'].to_numpy()
        r_ = np.minimum((vo + vt) / (vo + (vt if dag else m['var_baan'].to_numpy())), 1.0)      # deel van de variantie dat 'eigen + veld' is (dag-eenheden)
        var_d = var * r_
        if getattr(model, '_vorm_vang', False):      # voor het herschatten van de vormcoëfficiënten: fout vóór de vormverschuiving
            model._vorm_u = (err / np.sqrt(var_d), V_, _vorm_w(_niv_pre(m)))
        err = err - mu_ * np.sqrt(var_d)
        lamf = 0.5 * np.log(np.maximum(1.0 + r_ * np.expm1(2.0 * lf_), 1e-4)); lams = 0.5 * np.log(np.maximum(1.0 + r_ * np.expm1(2.0 * ls_), 1e-4))
    scf_ = getattr(model, 'schaal_f', None)
    if model.params.schaal and scf_ is not None:
        Q_ = scf_.reindex(m.index).fillna(0.0)
        lf2, ls2 = _schaal_fac(model, Q_['Sf'].to_numpy(), Q_['Nf'].to_numpy(), Q_['Ss'].to_numpy(), Q_['Ns'].to_numpy())
        lamf = lamf + lf2; lams = lams + ls2
    lam = np.where(err < 0, lamf, lams) + np.zeros(len(err))
    if model.params.vol_asym_bf != 0.0 and vr is not None and not dag:
        rel = 0.5 * np.log(var) - vr[m['afstand_idx'].to_numpy(), np.digitize(_niv_pre(m), VOL_BANDEN)]
        lam = lam + np.where(err < 0, model.params.vol_asym_bf * rel, 0.0)       # snelle kant: eigen schaal exp(lam)
    if np.any(lam != 0.0):
        err = err * np.exp(-lam)
        model._vol_asym_lam = lam
    df_ = getattr(model, 'diepte_f', None)
    if model.params.piek_diepte and df_ is not None and getattr(model, 'piek_drempel', None):
        Dd_ = df_.reindex(m.index).fillna(0.0)
        tau = _diepte_tau(model, Dd_['E'].to_numpy(), Dd_['N'].to_numpy())
        sf_ = getattr(model, 'staart_f', None)
        if model.params.staart and sf_ is not None:
            S_ = sf_.reindex(m.index).fillna(0.0)
            tau = tau * _staart_fac(model, S_['C'].to_numpy(), S_['R'].to_numpy(), _niv_pre(m))
        if not dag:
            vo_ = m['v_prior'].to_numpy() * (delta / that) ** 2; vt_ = m['v_theta'].to_numpy()
            r2 = np.minimum((vo_ + vt_) / (vo_ + m['var_baan'].to_numpy()), 1.0)
            tau = np.sqrt(np.maximum(1.0 + r2 * (tau ** 2 - 1.0), 1e-4))
        lo = np.array([model.piek_drempel[PIEK_GIDX.get(x, 0)][0] for x in AFSTANDEN])[m['afstand_idx'].to_numpy()]
        sd_ = np.sqrt(var); u_ = err / sd_; s_ = u_ < lo
        err = np.where(s_, (lo + (u_ - lo) / tau) * sd_, err)
        add = np.where(s_, np.log(tau), 0.0)
        model._vol_asym_lam = add if model._vol_asym_lam is None else model._vol_asym_lam + add
    z2 = err ** 2 / var
    nll = 0.5 * np.log(var) + 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
    return float(np.mean(nll)), err, var, m


def schat_dag_comp(model):
    """Post hoc extra dag-effect voor events waarvan de uitslag van het veld statistisch vreemd is (buiten de normale range van events).
    Maatstaf S = gewogen gemiddelde van de mediane z-score per race; vreemd als |S - mediaan(S)| > dag_comp_c * (MAD-sd van S).
    Het extra effect is dag_comp_lambda x de mediane ln-tijdfout van het veld, minus wat de standaard dag-correctie al opving."""
    p = model.params
    _, err, var, m = voorspelscore(model, 0, trend=False)
    z = err / np.sqrt(var)
    ok = (m['n_prior'].to_numpy() >= 3) & ~((err > 0) & (m['z'].to_numpy() > 6.0))
    d = pd.DataFrame({'ev': m['event_id'].to_numpy()[ok], 'g': m['gender'].to_numpy()[ok], 'a': m['afstand'].to_numpy()[ok],
                      'z': np.clip(z[ok], -4, 4), 'err': err[ok]})
    R = d.groupby(['ev', 'g', 'a']).agg(n=('z', 'size'), mz=('z', 'median'), me=('err', 'median')).reset_index()
    R = R[R['n'] >= 8]
    R['w'] = np.minimum(R['n'], 40)
    E = R.groupby('ev').apply(lambda g: pd.Series({'k': len(g), 'n': g['n'].sum(), 'S': np.average(g['mz'], weights=g['w']),
                                                    'D': np.average(g['me'], weights=g['w'])}))
    E = E[(E['k'] >= 3) & (E['n'] >= p.dag_comp_min_rijen)]
    if len(E) < 20:
        return {}
    med = E['S'].median(); sd = 1.4826 * (E['S'] - med).abs().median()
    ev = model.events.set_index('event_id')
    E = E[E.index.isin(ev.index)].copy()
    E['centrum'] = med; E['schaal'] = sd
    if p.dag_comp_baan_k > 0:
        k = p.dag_comp_baan_k
        E['baan'] = ev.loc[E.index, 'baan'].to_numpy()
        for b, g in E.groupby('baan'):
            n = len(g)
            mb = g['S'].median(); sb = 1.4826 * (g['S'] - mb).abs().median() if n >= 3 else sd
            E.loc[g.index, 'centrum'] = (n * mb + k * med) / (n + k)
            E.loc[g.index, 'schaal'] = math.sqrt((n * max(sb, 0.05) ** 2 + k * sd ** 2) / (n + k))
    vreemd = E[(E['S'] - E['centrum']).abs() > p.dag_comp_c * E['schaal']]
    uit = {}
    for e, r in vreemd.iterrows():
        if e not in ev.index:
            continue
        huidig = (getattr(model, 'extra_dag', None) or {}).get(int(e), 0.0)
        theta_zonder = float(ev.loc[e, 'theta_post']) - huidig              # wat de standaard correctie vastlegde
        uit[int(e)] = float(p.dag_comp_lambda * (r['D'] + float(ev.loc[e, 'mu_baan'])) - theta_zonder)
    return uit


TREND_KOL = ['d1_0', 'd1_k', 'd1_l', 'dm3', 'sl3c', 'cx', 'x5o', 'x14o', 'ef', 'es']
ANDER_VENSTERS = (('x5o', 1, 5), ('x14o', 6, 14))       # verrassingen op andere afstanden, andere evenementen, k dagen ervoor


def trend_kenmerken(model, max_gap=60, max_venster=120):
    """Kenmerken per obs-rij uit alleen VOORGAANDE races van dezelfde rijder op dezelfde afstand (valpartijen en verdachte rijen overgeslagen):
    d1 = vorige verrassing, dm3 = gemiddelde van de laatste 3 (beide in sd, gecorrigeerd voor het veld op die dag),
    sl3c = helling van de toestand (ln-eenheden per 30 dagen) over de laatste 3 races. Ontbrekend of te oud = NaN."""
    _, err, var, m = voorspelscore(model, 0, trend=False)
    z = err / np.sqrt(var)
    d = pd.DataFrame({'sk': m['skater_id'].to_numpy(), 'af': m['afstand'].to_numpy(), 'dag': m['dag'].to_numpy(),
                      'ev': m['event_id'].to_numpy(), 'gd': m['gender'].to_numpy(), 'xp': m['x_post'].to_numpy(),
                      'ai': m['afstand_idx'].to_numpy(), 'zc': np.clip(z, -4, 4), 'er': err, 'val': (err > 0) & (m['z'].to_numpy() > 6.0)}, index=m.index)
    _tl = _tl_vector(model.t_lims, m['gender'].to_numpy(), m['afstand_idx'].to_numpy()); _dl = np.exp(m['x_prior'].to_numpy())
    d['sdo'] = np.sqrt(m['v_prior'].to_numpy()) * _dl / (_tl + _dl)          # eigen sd in ln-tijd (zonder ijs-onzekerheid)
    d = d[~d['val']]
    grp = d.groupby(['ev', 'gd', 'af'])
    d['zdm'] = d['zc'] - grp['zc'].transform('median')
    d['zo'] = ((d['er'] - grp['er'].transform('median')) / d['sdo']).clip(-6, 6)     # verrassing t.o.v. het veld in eigen sd
    d['zo'] = d['zo'].astype(np.float32); d.drop(columns=['er', 'sdo'], inplace=True); del grp
    d = d.sort_values(['sk', 'af', 'dag'], kind='stable')
    g = d.groupby(['sk', 'af'], sort=False)
    d['d1'] = g['zdm'].shift(1); d['d2'] = g['zdm'].shift(2); d['d3'] = g['zdm'].shift(3)
    d['d1o'] = g['zo'].shift(1)
    dag1 = g['dag'].shift(1); dag3 = g['dag'].shift(3)
    d['gap'] = d['dag'] - dag1
    d['dm3'] = d[['d1', 'd2', 'd3']].mean(axis=1, skipna=False)
    sl = (g['xp'].shift(1) - g['xp'].shift(3)) / np.maximum(dag1 - dag3, 1) * 30.0
    d['sl3c'] = sl.clip(-0.2, 0.2)
    d.loc[(dag1 - dag3) > max_venster, 'sl3c'] = np.nan
    # de vorige verrassing in drie banden: dezelfde dag (bijv. tweede 500m), tot 2 weken, tot 2 maanden
    d['d1_0'] = np.where(d['gap'] == 0, d['d1'], np.nan)
    d['d1_k'] = np.where((d['gap'] >= 1) & (d['gap'] <= 14), d['d1'], np.nan)
    d['d1_l'] = np.where((d['gap'] > 14) & (d['gap'] <= max_gap), d['d1'], np.nan)
    d.loc[(d['gap'] == 0) | (d['gap'] > max_gap), 'dm3'] = np.nan
    d.loc[d['gap'] > max_gap, 'sl3c'] = np.nan
    # eerdere race van dezelfde rijder op een andere afstand in hetzelfde evenement (kortere afstanden eerst)
    c = d[['sk', 'ev', 'ai', 'af', 'zdm']].sort_values(['sk', 'ev', 'ai'], kind='stable')
    gc = c.groupby(['sk', 'ev'], sort=False)
    z1, a1, z2, a2 = gc['zdm'].shift(1), gc['af'].shift(1), gc['zdm'].shift(2), gc['af'].shift(2)
    cx = np.where(a1.notna() & (a1 != c['af']), z1, np.where(a2.notna() & (a2 != c['af']), z2, np.nan))
    d['cx'] = pd.Series(cx, index=c.index).reindex(d.index)
    # vorm loopt door over afstanden: gemiddelde verrassing op ANDERE afstanden in eerdere evenementen, 1-5 en 6-14 dagen ervoor
    for nm, lo, hi in ANDER_VENSTERS:
        d[nm] = np.nan
    if getattr(model.params, 'trend_ander', True):
        A = d[['sk', 'dag', 'ai', 'zdm']].groupby(['sk', 'dag', 'ai'], as_index=False)['zdm'].mean()
        key = d[['sk', 'dag', 'ai']].reset_index()
        for nm, lo, hi in ANDER_VENSTERS:
            parts = []
            for k in range(lo, hi + 1):
                B = A.copy(); B['dag'] = B['dag'] + k
                mg = key.merge(B, on=['sk', 'dag'], suffixes=('', '_p'))
                parts.append(mg[mg['ai_p'] != mg['ai']])
            P = pd.concat(parts)
            if len(P):
                d[nm] = P.groupby(P.columns[0])['zdm'].mean().reindex(d.index)
    # grote verrassingen (overschot voorbij verr_drempel sd): het filter reageert daar te zwak op
    _VERR_DREMPEL[0] = float(getattr(model.params, 'verr_drempel', 2.0)); _VERR_DREMPEL[1] = bool(getattr(model.params, 'verr_eigen', False))
    _VERR_DREMPEL[2] = float(getattr(model.params, 'verr_eigen_drempel', 2.5))
    _verr_kenmerken(d, (d['gap'] >= 1) & (d['gap'] <= max_gap))
    return d[TREND_KOL + ['efo', 'efs', 'd1', 'gap', 'zdm', 'zo', 'sk', 'ai', 'dag', 'xp']]


_VERR_DREMPEL = [2.0, False, 2.5]       # gezet vanuit params.verr_drempel / verr_eigen bij het schatten


def _verr_kenmerken(d, zelfde_ok):
    """ef/es: overschot van de vorige race op dezelfde afstand voorbij -2 / +2 sd; efo: idem voor het snelste gemiddelde op andere afstanden (1-14 dagen)."""
    d1 = d['d1o'] if _VERR_DREMPEL[1] else d['d1']; c = _VERR_DREMPEL[0]
    d['ef'] = np.where(zelfde_ok, np.minimum(0.0, d1 + c), np.nan)
    d['es'] = np.where(zelfde_ok, np.maximum(0.0, d1 - c), np.nan)
    d['efs'] = np.where(zelfde_ok, np.abs(np.minimum(0.0, d['d1o'] + _VERR_DREMPEL[2])), np.nan)
    mo = np.fmin(d['x5o'].to_numpy(dtype=float), d['x14o'].to_numpy(dtype=float))
    d['efo'] = np.where(np.isfinite(mo), np.minimum(0.0, mo + c), np.nan)


def _verr_factor(params, ef, es, efo, efs=0.0):
    return (1.0 + params.verr_c * (np.abs(np.nan_to_num(ef)) + np.abs(np.nan_to_num(efo))) + params.verr_c2 * np.nan_to_num(es)
            + getattr(params, 'verr_c3', 0.0) * np.nan_to_num(efs))


def _trend_x(x, t_lim, sd_x, shift_z):
    """Past de verwachting (in x = ln(T - T_lim)) aan met een verschuiving van `shift_z` sd's in LOG-TIJD (zoals in de schatting)."""
    if shift_z == 0.0:
        return x
    delta = math.exp(x); that = t_lim + delta
    sd_t = sd_x * delta / that                       # sd in ln(T)
    return math.log(max(that * math.exp(shift_z * sd_t) - t_lim, 1e-9))


def bouw_trend_reeks(model, f=None):
    """Bewaart per (rijder, afstand) de laatste verrassingen en toestanden (zonder valpartijen) voor de voorspelling vooraf."""
    if f is None:
        f = trend_kenmerken(model)
    g = f[['sk', 'ai', 'dag', 'zdm', 'zo', 'xp']].dropna(subset=['zdm']).sort_values(['sk', 'ai', 'dag'], kind='stable')
    sk, ai = g['sk'].to_numpy(), g['ai'].to_numpy()
    start = np.flatnonzero(np.r_[True, (sk[1:] != sk[:-1]) | (ai[1:] != ai[:-1])])
    eind = np.r_[start[1:], len(sk)]
    model.trend_reeks = {'dag': g['dag'].to_numpy().astype(np.int32), 'zdm': g['zdm'].to_numpy().astype(np.float32),
                         'zo': g['zo'].to_numpy().astype(np.float32),
                         'xp': g['xp'].to_numpy().astype(np.float32),
                         'pos': {(int(sk[a]), int(ai[a])): (int(a), int(b)) for a, b in zip(start, eind)}}
    # alle afstanden per rijder (gemiddeld per dag x afstand), voor de vorm op andere afstanden
    h = f[['sk', 'dag', 'ai', 'zdm']].dropna(subset=['zdm']).groupby(['sk', 'dag', 'ai'], as_index=False)['zdm'].mean().sort_values(['sk', 'dag'], kind='stable')
    sk2 = h['sk'].to_numpy(); st2 = np.flatnonzero(np.r_[True, sk2[1:] != sk2[:-1]]); ei2 = np.r_[st2[1:], len(sk2)]
    model.trend_reeks['alle'] = {'dag': h['dag'].to_numpy().astype(np.int32), 'ai': h['ai'].to_numpy().astype(np.int8),
                                 'zdm': h['zdm'].to_numpy().astype(np.float32),
                                 'pos': {int(sk2[a]): (int(a), int(b)) for a, b in zip(st2, ei2)}}
    return model.trend_reeks


def trend_verschuiving(model, skater_id, afstand, dag, eerder=None, max_gap=60, max_venster=120):
    f = _trend_kenm(model, skater_id, afstand, dag, eerder, max_gap, max_venster)
    if f is None:
        return 0.0
    coef = model.trend_coef
    return float(sum(c * f[n] for c, n in zip(coef, TREND_KOL)))


def trend_schaal(model, skater_id, afstand, dag, eerder=None):
    """Factor op de voorspel-sd na een grote verrassing (zelfde of andere afstand)."""
    p = model.params
    if not p.trend_corr or (p.verr_c == 0 and p.verr_c2 == 0 and getattr(p, 'verr_c3', 0.0) == 0):
        return 1.0
    f = _trend_kenm(model, skater_id, afstand, dag, eerder)
    if f is None:
        return 1.0
    return float(_verr_factor(p, f['ef'], f['es'], f['efo'], f.get('efs', 0.0)))


def _trend_kenm(model, skater_id, afstand, dag, eerder=None, max_gap=60, max_venster=120):
    """Verschuiving van de voorspelling in sd-eenheden, uit alleen races VOOR `dag` (en `eerder`).
    eerder: {afstand: verrassing (sd, veld-gecorrigeerd)} van eerdere races van deze rijder in hetzelfde evenement/dezelfde dag;
    de dichtstbijzijnde kortere afstand telt als 'eerdere afstand', dezelfde afstand als 'tweede run van de dag'."""
    r = getattr(model, 'trend_reeks', None)
    coef = getattr(model, 'trend_coef', None)
    if r is None or coef is None or not model.params.trend_corr:
        return None
    ai = NAAR_INDEX[afstand]
    f = dict.fromkeys(TREND_KOL, 0.0)
    if eerder:
        if afstand in eerder:
            f['d1_0'] = float(np.clip(eerder[afstand], -4, 4))
        ander = [a for a in eerder if NAAR_INDEX[a] != ai]
        if ander:       # de dichtstbijzijnde andere afstand (bij gelijke afstand de kortere)
            f['cx'] = float(np.clip(eerder[min(ander, key=lambda a: (abs(NAAR_INDEX[a] - ai), NAAR_INDEX[a] > ai))], -4, 4))
    aanw = set()
    ra = r.get('alle')
    if ra is not None and getattr(model.params, 'trend_ander', True):
        ab2 = ra['pos'].get(int(skater_id))
        if ab2 is not None:
            a2, b2 = ab2
            dd = ra['dag'][a2:b2]; aa = ra['ai'][a2:b2]; zz = ra['zdm'][a2:b2]
            for nm, lo, hi in ANDER_VENSTERS:
                sel = (dd >= dag - hi) & (dd <= dag - lo) & (aa != ai)
                if sel.any():
                    f[nm] = float(np.mean(zz[sel])); aanw.add(nm)
    ab = r['pos'].get((int(skater_id), ai))
    if ab is not None and f['d1_0'] == 0.0:        # met een tweede run op dezelfde dag vervallen de oudere kenmerken (zoals in de schatting)
        a, b = ab
        k = a + int(np.searchsorted(r['dag'][a:b], dag, side='left'))       # eerste race op of na `dag`
        if k - a >= 1:
            gap = int(dag - r['dag'][k - 1])
            if 1 <= gap <= max_gap:
                z1 = float(r['zdm'][k - 1])
                f['d1_k' if gap <= 14 else 'd1_l'] = z1
                _c = float(getattr(model.params, 'verr_drempel', 2.0))
                zv = float(r['zo'][k - 1]) if (getattr(model.params, 'verr_eigen', False) and 'zo' in r) else z1
                f['ef'] = min(0.0, zv + _c); f['es'] = max(0.0, zv - _c)
                if 'zo' in r:
                    f['efs'] = abs(min(0.0, float(r['zo'][k - 1]) + float(getattr(model.params, 'verr_eigen_drempel', 2.5))))
                if k - a >= 3:
                    f['dm3'] = float(np.mean(r['zdm'][k - 3:k]))
                    span = int(r['dag'][k - 1] - r['dag'][k - 3])
                    if span <= max_venster:
                        f['sl3c'] = float(np.clip((r['xp'][k - 1] - r['xp'][k - 3]) / max(span, 1) * 30.0, -0.2, 0.2))
    vo = [f[n] for n in ('x5o', 'x14o') if n in aanw]
    f['efo'] = min(0.0, min(vo) + float(getattr(model.params, 'verr_drempel', 2.0))) if vo else 0.0
    return f


def schat_trend(model):
    """Zet model.trend_f en model.trend_coef: kleinste kwadraten van de verrassing op de kenmerken, alleen jaren <= split_jaar."""
    f = trend_kenmerken(model)
    _, err, var, m = voorspelscore(model, 0, trend=False)
    z = pd.Series(np.clip(err / np.sqrt(var), -4, 4), index=m.index)
    jr = m['jaar']
    ok = (f[TREND_KOL].notna().any(axis=1)) & (jr.reindex(f.index) <= model.params.split_jaar)
    X = f.loc[ok, TREND_KOL].fillna(0.0).to_numpy(); y = f.loc[ok, 'zdm'].to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    model.trend_f = f[TREND_KOL + ['efo', 'efs']].astype(np.float32); model.trend_coef = coef; bouw_trend_reeks(model, f); print('trend_coef', dict(zip(TREND_KOL, coef.round(3))), flush=True)
    return coef


def tune_parameters(df, start=None, vanaf_jaar=2014, rondes=2, verbose=True, t_lim_df=None,
                    ckpt=None, tijdbudget=None, kern=None):
    """Coordinate search op de voorspelscore (1-stap-vooruit, alleen info van voor de race).

    Kandidaten worden 'licht' beoordeeld (een pass met de kalibratie van het huidige beste model).
    Na elke ronde wordt het beste model volledig opnieuw gekalibreerd. Een nieuwe T_lim-marge
    vereist een volledige herkalibratie, omdat alle ln(T-T_lim)-waarden verschuiven.
    ckpt: pad van een json-checkpoint (hervatbaar); tijdbudget: seconden, daarna return (klaar=False)."""
    import json, time
    t0 = time.time()
    p = start or Params()
    rooster = {
        'q': [0.005, 0.01, 0.02, 0.04, 0.08],
        'q_mix': [0.0, 0.3, 0.6, 0.9],
        'r_schaal': [0.15, 0.25, 0.4, 0.6, 0.8, 1.2],
        'p0_schaal': [0.25, 0.5, 1.0, 2.0],
        'garch_alpha': [0.03, 0.08, 0.15, 0.25],
        'tau_dag': [0.0015, 0.004, 0.008, 0.015],
        'tau_baan': [0.0025, 0.005, 0.01, 0.02],
        'dag_gewicht_jeugd': [0.25, 0.6, 1.0],
        't_lim_marge': [0.015, 0.03, 0.05, 0.08],
    }
    kandidaten = [('q_groep', gr, f) for gr in GROEPEN if gr != 'onbekend' for f in [0.5, 2.0, 4.0, 8.0]]
    kandidaten += [(n, None, w) for n, ws in rooster.items() for w in ws if n in (kern or rooster)]
    ronde0, idx0, veranderd = 0, 0, False
    if ckpt and os.path.exists(ckpt):
        st = json.load(open(ckpt))
        p = Params(**st['p'])
        ronde0, idx0, veranderd = st['ronde'], st['idx'], st['veranderd']
        if st.get('klaar'):
            return p, None, True
        if verbose:
            print(f"hervat: ronde {ronde0 + 1}, kandidaat {idx0}/{len(kandidaten)}", flush=True)
    basis = fit_volledig(df, p, n_iter=3, verbose=False, t_lim_df=t_lim_df)
    beste = voorspelscore(basis, vanaf_jaar)[0]
    if verbose:
        print(f"start (gekalibreerd) score {beste:.4f}", flush=True)

    def bewaar(ronde, idx, veranderd, klaar=False):
        if ckpt:
            json.dump({'p': asdict(p), 'ronde': ronde, 'idx': idx, 'veranderd': veranderd, 'klaar': klaar}, open(ckpt, 'w'))

    for ronde in range(ronde0, rondes):
        for idx in range(idx0 if ronde == ronde0 else 0, len(kandidaten)):
            if tijdbudget is not None and time.time() - t0 > tijdbudget:
                bewaar(ronde, idx, veranderd)
                return p, basis, False
            naam, gr, w = kandidaten[idx]
            if naam == 'q_groep':
                if p.q_groep.get(gr, 1.0) == w:
                    continue
                kand = replace(p, q_groep={**p.q_groep, gr: w})
            else:
                if getattr(p, naam) == w:
                    continue
                kand = replace(p, **{naam: w})
                if naam == 'garch_alpha':
                    kand = replace(kand, garch_beta=round(0.96 - w, 3))
            if naam == 't_lim_marge':
                m = fit_volledig(df, kand, n_iter=3, verbose=False, t_lim_df=t_lim_df)
            else:
                m = fit_volledig(df, kand, n_iter=0, verbose=False, prior_van=basis)
            sc = voorspelscore(m, vanaf_jaar)[0]
            if sc < beste - 1e-4:
                beste, p, basis, veranderd = sc, kand, m, True
                if verbose:
                    print(f"  ronde {ronde + 1}: {naam}{'[' + gr + ']' if gr else ''}={w} -> {sc:.4f}", flush=True)
            bewaar(ronde, idx + 1, veranderd)
        basis = kalibreer(basis, df, n_iter=2, verbose=False)
        beste = voorspelscore(basis, vanaf_jaar)[0]
        if verbose:
            print(f"ronde {ronde + 1} klaar, herkalibreerd score {beste:.4f}", flush=True)
        bewaar(ronde + 1, 0, False)
        if not veranderd:
            break
        veranderd = False
    bewaar(rondes, 0, False, klaar=True)
    if verbose:
        print(p)
    return p, basis, True


# =============================================================================
# 8. DIAGNOSE
# =============================================================================
def diagnose_tabel(model, vanaf_jaar=2014):
    """Bias, MAE en calibratie van de 1-stap-vooruit voorspelling, per niveaugroep en per aantal eerdere races."""
    score, err, var, m = voorspelscore(model, vanaf_jaar)
    d = pd.DataFrame({'groep': m['groep'].to_numpy(), 'n_prior': m['n_prior'].to_numpy(),
                      'err_pct': err * 100.0, 'zabs': np.abs(err) / np.sqrt(var)})
    d['n_bak'] = pd.cut(d['n_prior'], [-1, 0, 2, 5, 10, 1e9], labels=['0', '1-2', '3-5', '6-10', '11+'])
    per_groep = d.groupby('groep').agg(n=('err_pct', 'size'), bias_pct=('err_pct', 'mean'),
                                       mediaan_pct=('err_pct', 'median'),
                                       mae_pct=('err_pct', lambda s: s.abs().mean()),
                                       binnen_80=('zabs', lambda s: (s <= 1.2816).mean()))
    per_n = d.groupby('n_bak', observed=True).agg(n=('err_pct', 'size'), bias_pct=('err_pct', 'mean'),
                                                  mediaan_pct=('err_pct', 'median'),
                                                  mae_pct=('err_pct', lambda s: s.abs().mean()),
                                                  binnen_80=('zabs', lambda s: (s <= 1.2816).mean()))
    return per_groep, per_n


# =============================================================================
# 9. EXPORT
# =============================================================================
def top_rijders_tabel(model, gender, afstand, n=10, min_recent=3, max_dagen=1100, eind_dag=None):
    """De n snelste (verwachte tijd) actieve rijders op een afstand, naast hun recente werkelijke tijden.
    'afwijking_pct' = verwachte tijd t.o.v. de mediaan van de laatste 5 dag-gecorrigeerde tijden."""
    j = NAAR_INDEX[afstand]
    eind = eind_dag if eind_dag is not None else int(model.obs['dag'].max())
    rijen = []
    for st in model.staten.values():
        if st.gender != gender:
            continue
        rec = [r for r in st.recent[j] if eind - r[0] <= max_dagen]
        if len(rec) < min_recent:
            continue
        v = model.voorspel(st.id, afstand)
        raw = np.array([r[1] for r in rec][-5:])
        adj = np.array([r[2] for r in rec][-5:]) * math.exp(model.anker_std)
        delta = math.exp(v['mu_log'])
        # grove missers (verkeerde afstand/typefout: >25% van de verwachting) horen niet in de referentie
        ok = np.abs(adj / v['verwachte_tijd'] - 1.0) <= 0.25
        n_uit = int((~ok).sum())
        if ok.sum() < min_recent:
            continue
        raw, adj = raw[ok], adj[ok]
        rijen.append({'skater_id': st.id, 'naam': st.naam, 'land': st.land, 'groep': st.groep,
                      'n_races': int(st.n[j]), 'n_recent': len(rec),
                      'verwacht_s': v['verwachte_tijd'],
                      'sd_pct': 100.0 * v['sigma_log'] * delta / v['verwachte_tijd'],
                      'recent_mediaan_ruw_s': float(np.median(raw)),
                      'recent_mediaan_dagcorr_s': float(np.median(adj)),
                      'laatste_dagcorr_s': float(adj[-1]), 'n_uitschieters': n_uit,
                      'recent_min_dagcorr_s': float(adj.min()), 'recent_max_dagcorr_s': float(adj.max()),
                      'pb_s': float(st.pb[j]) if np.isfinite(st.pb[j]) else np.nan,
                      'afwijking_pct': 100.0 * (v['verwachte_tijd'] / float(np.median(adj)) - 1.0),
                      'laatste_race': pd.to_datetime(st.laatste_dag, unit='D').date()})
    out = pd.DataFrame(rijen)
    if out.empty:
        return out
    return out.sort_values('verwacht_s').head(n).reset_index(drop=True)


def exporteer_distributies_naar_csv(model, filename='skater_distributies.csv'):
    rijen = []
    for sid, st in model.staten.items():
        for j, a in enumerate(AFSTANDEN):
            if st.n[j] == 0:
                continue
            v = model.voorspel(sid, a)
            rijen.append({'skater_id': sid, 'skater_name': st.naam, 'gender': st.gender, 'land': st.land,
                          'groep': st.groep, 'afstand': a, 'n_races': int(st.n[j]),
                          'laatste_race': pd.to_datetime(st.laatste_dag, unit='D').date(),
                          'mu_log': round(v['mu_log'], 4), 'sigma_log': round(v['sigma_log'], 4),
                          'verwachte_tijd_s': round(v['verwachte_tijd'], 2),
                          'normale_tijd_s': round(v['normale_tijd'], 2), 'piek_tijd_s': round(v['piek_tijd'], 2),
                          'piek_extreem_tijd_s': round(v['piek_extreem_tijd'], 2), 'piek_ijs_tijd_s': round(v['piek_ijs_tijd'], 2),
                          'pb_s': round(float(st.pb[j]), 2) if np.isfinite(st.pb[j]) else np.nan})
    out = pd.DataFrame(rijen)
    out.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n=== Actuele vorm geëxporteerd naar '{filename}' ({len(out)} regels) ===")
    return out


def exporteer_dagen_naar_csv(model, filename='dag_effecten.csv'):
    cols = ['event_id', 'datum', 'baan', 'n', 'n_rijders', 'n_landen', 'n_elite', 'veldniveau',
            'effect_pct', 'se_data', 'mu_baan', 'dev_pct', 'z', 'goed_gemeten', 'belangrijk', 'piekdag', 'piek_reden']
    out = model.events[cols].copy()
    out['mu_baan'] = (out['mu_baan'] - model.anker) * 100.0
    out['se_data'] = out['se_data'] * 100.0
    out = out.rename(columns={'mu_baan': 'baan_gemiddelde_pct', 'se_data': 'se_pct'})
    out.round(3).to_csv(filename, index=False, encoding='utf-8')
    print(f"=== Dag-effecten geëxporteerd naar '{filename}' ({int(out['piekdag'].sum())} piekdagen) ===")
    return out


def exporteer_baancorrecties_naar_csv(model, filename='baancorrecties.csv'):
    """Dag-effect per baan, jaar en afstand (gemiddelde over events, positief = trager dan verwacht)."""
    g = model.groepen_tabel.copy()
    g['jaar'] = pd.to_datetime(g['dag'], unit='D').dt.year
    g['w'] = 1.0 / (g['se_post'] ** 2)
    g['tw'] = g['theta_post'] * g['w']
    out = g.groupby(['baan', 'jaar', 'afstand']).agg(n_events=('event_id', 'nunique'), w=('w', 'sum'), tw=('tw', 'sum')).reset_index()
    out['baan_effect_pct'] = (out['tw'] / out['w'] - model.anker) * 100.0
    out = out[['baan', 'jaar', 'afstand', 'n_events', 'baan_effect_pct']].round(3)
    out.to_csv(filename, index=False, encoding='utf-8')
    print(f"=== Baancorrecties geëxporteerd naar '{filename}' ===")
    return out


# =============================================================================
# 10. MONTE CARLO RACE SIMULATOR MET GEDEELD DAG-EFFECT
# =============================================================================
def empirische_z(model, vanaf_jaar=2014, kwantiel_niveau=0.75, max_n=40000):
    """Gestandaardiseerde 1-stap-voorspelfouten (na pred_sd_schaal) van de betere rijders, per afstand, gecentreerd op de mediaan.
    Bevat de werkelijke scheefheid: lichte snelle staart, zware trage staart (slechte dagen, valpartijen).
    Wordt gebruikt om de simulatie te laten trekken zonder normaliteitsaanname."""
    pool = getattr(model, '_z_emp', None)
    if pool is not None:
        return pool
    _, err, var, m = voorspelscore(model, vanaf_jaar)
    z = err / np.sqrt(var)
    z[(err > 0) & (m['z'].to_numpy() > model.params.dip_val_z) & (model.params.dip_val_z > 0)] = np.nan      # valpartijen niet in de simulatieverdeling
    a = m['afstand'].to_numpy()
    niv = m['niveau_pct'].to_numpy() if 'niveau_pct' in m else np.zeros(len(m))
    rng = np.random.default_rng(0)
    pool = {}
    for af in AFSTANDEN:
        sel = (a == af)
        if sel.sum() < 200:
            pool[af] = np.array([0.0]); continue
        drempel = np.quantile(niv[sel], kwantiel_niveau)
        zz = z[sel & (niv >= drempel)]
        zz = zz[np.isfinite(zz)]
        zz = np.clip(zz - np.median(zz), -6.0, 8.0)
        if len(zz) > max_n:
            zz = rng.choice(zz, max_n, replace=False)
        pool[af] = zz
    model._z_emp = pool
    return pool


def simuleer_race(model, deelnemer_ids, afstand, baan=STANDAARD_BAAN, scenario='normaal', dag=None,
                  dag_verschuiving=None, dag_sd=None, n_simulaties=10000, seed=None, historie=None, eerder=None):
    """Simuleert een race. Elke simulatie trekt EEN dag-effect dat alle rijders delen.

    baan: standaard Heerenveen; baan=None geeft een gemiddeld event over alle banen.
    scenario 'normaal'  : dag-effect ~ N(baangemiddelde, baan-onzekerheid)
    scenario 'piekdag'  : idem + gemiddelde afwijking van historische piekdagen
    Let op: een dag-effect dat alle rijders gelijk raakt verandert de winkansen niet (de rangorde
    blijft gelijk) maar wel de tijden (kans op een baanrecord, verwachte winnende tijd).
    """
    rng = np.random.default_rng(seed)
    genders = {model.staten[s].gender for s in deelnemer_ids}
    if len(genders) != 1:
        raise ValueError("Een race heeft deelnemers van een geslacht nodig.")
    n = len(deelnemer_ids)

    if historie is not None and dag is None:
        raise ValueError("met `historie` is een datum (`dag`) nodig")
    if historie is not None and baan is not None:
        mu_d, sd_d = historie.baan_effect(baan, dag)
    elif baan is not None:
        mu_d, sd_d, _ = model.baan_effect(baan, dag)
    else:
        mu_d, sd_d = model.anker, model.tau_dag_empirisch   # neutrale dag = gemiddeld recent event (zelfde kader als 'verwachte_tijd')
    if scenario == 'piekdag':
        pk_mu, pk_sd = model.piekdag_statistiek()
        mu_d += pk_mu
        sd_d = math.sqrt(sd_d ** 2 + pk_sd ** 2)
    if dag_verschuiving is not None:
        mu_d = dag_verschuiving
    if dag_sd is not None:
        sd_d = dag_sd

    theta = rng.normal(mu_d, sd_d, size=n_simulaties)              # gedeeld door alle rijders
    tijden = np.zeros((n_simulaties, n))
    eerder = eerder or {}
    voorsp = [historie.voorspel(s, afstand, dag, eerder.get(s)) if historie is not None else model.voorspel(s, afstand, dag, eerder.get(s))
              for s in deelnemer_ids]
    for i, v in enumerate(voorsp):
        if model.params.sim_z == 'empirisch' and model.params.piek_kans and 'p_piek' in v:
            pool = _pool_sorted(model, afstand); q0, q1 = model.params.piek_q
            kans = np.array([v['p_piek'], 1.0 - v['p_piek'] - v['p_slecht'], v['p_slecht']])
            regio = rng.choice(3, n_simulaties, p=kans)
            lo = np.array([0.0, q0, q1])[regio]; hi = np.array([q0, q1, 1.0])[regio]
            idx = np.minimum(((lo + (hi - lo) * rng.random(n_simulaties)) * len(pool)).astype(int), len(pool) - 1)
            z = pool[idx]
        elif model.params.sim_z == 'empirisch':
            z = rng.choice(empirische_z(model)[afstand], n_simulaties)
        else:
            z = rng.standard_normal(n_simulaties)
        z = np.where(z < 0, z * v.get('snel_fac', 1.0), z)                  # snelle kant: eigen schaal van de rijder
        tijden[:, i] = (v['t_lim'] + np.exp(v['mu_log'] + v['sigma_log'] * z)) * np.exp(theta)

    volgorde = np.argsort(tijden, axis=1)
    k = min(3, n)
    podium = [np.bincount(volgorde[:, r], minlength=n) / n_simulaties * 100.0 for r in range(k)]
    res = []
    for i, s in enumerate(deelnemer_ids):
        res.append({'Rijder': model.staten[s].naam,
                    'Normale tijd (s)': round(voorsp[i]['normale_tijd'], 2),
                    'Piektijd (s)': round(voorsp[i]['piek_tijd'], 2),
                    'Verwachte Tijd (s)': round(float(np.median(tijden[:, i])), 2),
                    'P10 (s)': round(float(np.quantile(tijden[:, i], 0.10)), 2),
                    'P90 (s)': round(float(np.quantile(tijden[:, i], 0.90)), 2),
                    'Log-Volatiliteit (σ)': round(voorsp[i]['sigma_log'], 3),
                    'Goud (%)': round(podium[0][i], 1),
                    'Zilver (%)': round(podium[1][i], 1) if k > 1 else 0.0,
                    'Brons (%)': round(podium[2][i], 1) if k > 2 else 0.0})
    return pd.DataFrame(res).sort_values('Goud (%)', ascending=False).reset_index(drop=True)



# =============================================================================
# 10b. TOESTANDS-GESCHIEDENIS (tijdreizen)
# =============================================================================
class Historie:
    """Toestand van elke rijder (niveau, onzekerheid, volatiliteit, aantal races voor alle 8 afstanden)
    na elk event waaraan hij deelnam. Hiermee kun je voor elke datum de voorspelling en simulatie doen
    zoals het model die op die dag zou hebben gemaakt (alleen info van vóór die dag)."""

    def __init__(self, model, ruw=None):
        self.model = model
        r = ruw if ruw is not None else model.historie_ruw
        o = np.lexsort((r['event_id'], r['dag'], r['skater_id']))   # rijder, dan datum, dan event
        self.d = {k: v[o] for k, v in r.items()}
        sid = self.d['skater_id']
        self.ids, self.start = np.unique(sid, return_index=True)
        self.eind = np.append(self.start[1:], len(sid))
        self.pos = {int(i): (int(a), int(b)) for i, a, b in zip(self.ids, self.start, self.eind)}

    def _snap(self, skater_id, dag):
        a, b = self.pos[int(skater_id)]
        k = a + int(np.searchsorted(self.d['dag'][a:b], dag, side='left')) - 1     # laatste event vóór `dag`
        if k < a:
            raise KeyError(f"rijder {skater_id} heeft nog geen race vóór dag {dag}")
        return k

    def voorspel(self, skater_id, afstand, dag, eerder=None):
        """Zoals SchaatsModel.voorspel, maar met de toestand zoals die vóór `dag` was."""
        m = self.model
        p = m.params
        k = self._snap(skater_id, dag)
        j = NAAR_INDEX[afstand]
        st = m.staten[int(skater_id)]
        gr = GROEPEN[int(self.d['groep'][k])]
        dt = min(max(dag - int(self.d['dag'][k]), 0) / DAGEN_PER_JAAR, p.max_dt_jaar)
        x = float(self.d['x'][k, j]) + p.groei_schaal * float(np.interp(dt, GROEI_KNOOP, m.groei[gr]))
        n_k = self.d['n'][k]
        x = x + _erv_shift(m, st.gender, gr, n_k[j], n_k.sum(), j)
        _xb = x                               # vaste rating (zonder tijdelijke vorm)
        qh_ = float(self.d['qh'][k]) * p.qvol_jaar ** dt if 'qh' in self.d else 0.0
        P = float(self.d['sd'][k, j]) ** 2 + m._bouw_Q(st.gender)[j, j] * p.q_groep.get(gr, 1.0) * dt * math.exp(qh_)
        if p.vorm_ou_sd > 0.0 and 'vf' in self.d:
            f_, vf_ = _ou_vooruit(p, float(self.d['vf'][k]), float(self.d['vpf'][k]), int(self.d['dag'][k]), dag)
            mul = (m.t_lims[st.gender][afstand] + math.exp(x)) / math.exp(x)
            x = x + f_ * mul; P = P + vf_ * mul * mul
        ruisf = m.prior['ruis'][st.gender].get(gr, 1.0) if p.betrouwbaarheid else 1.0
        R = m.prior['R0'][st.gender][j] * ruisf * p.r_schaal * float(self.d['s'][k, j])
        t_lim = m.t_lims[st.gender][afstand]
        _nf = _niv_fac(m, st.gender, j, x, gr)
        _sdx = math.sqrt(P + R) * p.pred_sd_schaal.get(AFSTANDEN[j], 1.0) * _sd_fac(m, dt, n_k[j], n_k.sum(), j) * _nf
        _ls = _loc_schaal_fwd(m, st.gender, j, x)
        if _ls != (0.0, 1.0):
            _xb = _trend_x(_xb, t_lim, _sdx, _ls[0])
        if p.trend_corr:
            x = _trend_x(x, t_lim, _sdx, trend_verschuiving(m, skater_id, afstand, dag, eerder))
            _sdx = _sdx * trend_schaal(m, skater_id, afstand, dag, eerder)
        if _ls != (0.0, 1.0):
            x = _trend_x(x, t_lim, _sdx, _ls[0]); _sdx = _sdx * _ls[1]
        _gm, _gl = gas_toestand(m, skater_id, afstand, dag)
        if _gm != 0.0 or _gl != 0.0:
            x = _trend_x(x, t_lim, _sdx, _gm); _sdx = _sdx * math.exp(_gl)
        _vv = verm_verschuiving(m, skater_id, afstand, dag, eerder, _niv_fwd(m, st.gender, j, x))
        if _vv != 0.0:
            x = _trend_x(x, t_lim, _sdx, _vv)
        _sf = vol_snel_factor(m, afstand, _sdx, t_lim, x, _niv_fwd(m, st.gender, j, x))
        _sf0 = _sf
        _vm, _kf, _ks = vorm_toestand(m, skater_id, dag, _niv_fwd(m, st.gender, j, x))
        if _vm != 0.0 or _ks != 1.0 or _kf != 1.0:
            x = _trend_x(x, t_lim, _sdx, _vm); _sdx = _sdx * _ks; _sf = _sf * _kf / _ks
        _qf, _qs = schaal_toestand(m, skater_id, dag)
        if _qf != 1.0 or _qs != 1.0:
            _sdx = _sdx * _qs; _sf = _sf * _qf / _qs; _ks = _ks * _qs; _kf = _kf * _qf
        v = {'piek_diepte': diepte_toestand(m, skater_id, dag) * staart_toestand(m, skater_id, dag, _niv_fwd(m, st.gender, j, x)),
             'snel_fac': _sf, 'snel_fac_vol': _sf0, 'vorm_snel': _kf, 'vorm_traag': _ks, 'vorm_mu': _vm, 'mu_log': x, 'sigma_log': _sdx, 'sigma_niveau': math.sqrt(P), 't_lim': t_lim,
             'verwachte_tijd': (t_lim + math.exp(x)) * math.exp(m.baan_effect(STANDAARD_BAAN, dag)[0]),
             'verwachte_tijd_intern': t_lim + math.exp(x)}
        _anker = m.baan_effect(STANDAARD_BAAN, dag)[0]
        v.update(_twee_tijden(v, _anker, m.piek_ijs()))
        v['normale_basis'] = v['normale_tijd'] * (t_lim + math.exp(_xb)) / (t_lim + math.exp(v['mu_log']))
        kans = piek_kansen(m, skater_id, dag)
        v['p_piek'], v['p_slecht'] = float(kans[0]), float(kans[2])
        _piek_tijden(m, v, afstand, kans, _anker)
        return v

    def baan_effect(self, baan, dag):
        """Baan-prior zoals het model die op `dag` had: (gemiddelde, sd) in het interne kader (bewaard per baan x dag)."""
        c = self.__dict__.setdefault('_baan_cache', {})
        k = (baan, int(dag))
        if k not in c:
            c[k] = self._baan_effect(baan, dag)
        return c[k]

    def _baan_effect(self, baan, dag):
        ev = self.model.events
        e = ev[(ev['baan'] == baan) & (ev['dag'] <= dag)]
        if e.empty:
            e = ev[ev['baan'] == baan]
        if e.empty:
            return self.model.anker, self.model.tau_dag_empirisch
        r = e.sort_values('dag').iloc[-1]
        return float(r['mu_baan']), float(r['sd_prior'])

    def actief_op(self, dag, gender=None, max_dagen=550, afstand=None, min_races=3):
        """Rijders die op `dag` actief waren (race in de laatste max_dagen dagen), evt. met min_races op een afstand."""
        uit = []
        j = NAAR_INDEX[afstand] if afstand else None
        for sid, (a, b) in self.pos.items():
            k = a + int(np.searchsorted(self.d['dag'][a:b], dag, side='left')) - 1
            if k < a or dag - int(self.d['dag'][k]) > max_dagen:
                continue
            if gender and self.model.staten[sid].gender != gender:
                continue
            if j is not None and self.d['n'][k, j] < min_races:
                continue
            uit.append(sid)
        return uit

    def traject(self, skater_id, afstand):
        """Verwachte tijd en onzekerheid van een rijder na elk van zijn events."""
        a, b = self.pos[int(skater_id)]
        j = NAAR_INDEX[afstand]
        st = self.model.staten[int(skater_id)]
        t_lim = self.model.t_lims[st.gender][afstand]
        x = self.d['x'][a:b, j].astype(float)
        t = (t_lim + np.exp(x)) * math.exp(self.model.anker_std)
        sd = self.d['sd'][a:b, j] * np.exp(x) / (t_lim + np.exp(x)) * 100.0
        return pd.DataFrame({'datum': pd.to_datetime(self.d['dag'][a:b], unit='D'), 'event_id': self.d['event_id'][a:b],
                             'verwachte_tijd_s': t, 'sd_pct': sd, 'n_races': self.d['n'][a:b, j]})

    def naar_dataframe(self):
        """Brede tabel: een rij per (rijder, event); per afstand verwachte tijd, sd (%), volatiliteit, aantal races."""
        m = self.model
        d = self.d
        gender = np.array([m.staten[int(i)].gender for i in d['skater_id']])
        out = {'skater_id': d['skater_id'], 'naam': [m.staten[int(i)].naam for i in d['skater_id']],
               'gender': gender, 'event_id': d['event_id'], 'datum': pd.to_datetime(d['dag'], unit='D'),
               'groep': [GROEPEN[int(g)] for g in d['groep']]}
        for j, a in enumerate(AFSTANDEN):
            tl = np.array([m.t_lims[g][a] for g in ['m', 'v']])[(gender == 'v').astype(int)]
            x = d['x'][:, j].astype(float)
            out[f'x_{a}'] = d['x'][:, j]
            out[f'tijd_{a}'] = ((tl + np.exp(x)) * math.exp(m.anker_std)).astype(np.float32)
            out[f'sd_pct_{a}'] = (d['sd'][:, j] * np.exp(x) / (tl + np.exp(x)) * 100.0).astype(np.float32)
            out[f'sd_x_{a}'] = d['sd'][:, j]
            out[f'vol_{a}'] = d['s'][:, j]
            out[f'n_{a}'] = d['n'][:, j]
        return pd.DataFrame(out)

    def bewaar_delen(self, prefix='skater_toestanden', n_delen=12):
        """Slim opslaan (alleen ruwe toestand; tijden en sd% zijn afleidbaar) in n_delen parquet-bestanden op rijder-id."""
        d = self.d
        grenzen = np.searchsorted(self.start, np.linspace(0, len(self.d['dag']), n_delen + 1)[1:-1])
        delen = [x for x in np.split(np.arange(len(self.ids)), grenzen) if len(x)]
        bestanden = []
        for i, sel in enumerate(delen):
            a = self.start[sel[0]]
            b = self.eind[sel[-1]]
            kol = {'skater_id': d['skater_id'][a:b], 'event_id': d['event_id'][a:b], 'dag': d['dag'][a:b],
                   'groep': d['groep'][a:b]}
            for j, af in enumerate(AFSTANDEN):
                kol[f'x_{af}'] = d['x'][a:b, j]
                kol[f'sd_{af}'] = d['sd'][a:b, j]
                kol[f'vol_{af}'] = d['s'][a:b, j]
                kol[f'n_{af}'] = d['n'][a:b, j].astype(np.int16)
            f = f'{prefix}_deel{i + 1:02d}.parquet'
            pd.DataFrame(kol).to_parquet(f, index=False, compression='zstd', compression_level=12)
            bestanden.append(f)
        return bestanden

    @classmethod
    def van_bestanden(cls, model, bestanden):
        """Bouwt de geschiedenis opnieuw uit de delen (samen met het model, dat t_lims/prior/params levert)."""
        if isinstance(bestanden, str):
            import glob
            bestanden = sorted(glob.glob(bestanden))
        t = pd.concat([pd.read_parquet(f) for f in bestanden], ignore_index=True)
        ruw = {'skater_id': t['skater_id'].to_numpy(), 'event_id': t['event_id'].to_numpy(),
               'dag': t['dag'].to_numpy(), 'groep': t['groep'].to_numpy()}
        for naam, pref in [('x', 'x_'), ('sd', 'sd_'), ('s', 'vol_'), ('n', 'n_')]:
            ruw[naam] = np.column_stack([t[f'{pref}{a}'].to_numpy() for a in AFSTANDEN])
        return cls(model, ruw)

    def bewaar(self, bestand='skater_toestanden.parquet'):
        self.naar_dataframe().to_parquet(bestand, index=False, compression='zstd')
        return bestand


# =============================================================================
# 11. OPSLAAN / LADEN
# =============================================================================
def bewaar_model(model, bestand='model.pkl'):
    with open(bestand, 'wb') as f:
        pickle.dump(model, f)


def laad_model(bestand='model.pkl'):
    with open(bestand, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# 12. HOOFDPROGRAMMA
# =============================================================================
def hoofdprogramma(bestandsnaam='speedskating_historie_clean_5pct.csv', tunen=False):
    print(f"Inlezen van {bestandsnaam}...")
    df = laad_dataset(bestandsnaam)

    if tunen:
        params, model = tune_parameters(df)
    else:
        params = Params()
        model = fit_volledig(df, params, n_iter=3)
    print("\nDrift-curve per groep (ln-delta t.o.v. voorspelling, op tijd sinds laatste race):")
    print(pd.DataFrame({k: v for k, v in model.groei.items()}, index=[f"{x:g}j" for x in GROEI_KNOOP]).round(3).T.to_string())

    per_groep, per_n = diagnose_tabel(model)
    print("\n=== Kalibratie 1-stap-vooruit (per niveaugroep) ===")
    print(per_groep.round(3).to_string())
    print("\n=== Kalibratie per aantal eerdere races op die afstand ===")
    print(per_n.round(3).to_string())

    for g_, a_ in [('m', '500m'), ('m', '1500m'), ('v', '500m'), ('v', '1500m')]:
        t_ = top_rijders_tabel(model, g_, a_)
        if len(t_):
            print(f"\n=== Top 10 {'mannen' if g_ == 'm' else 'vrouwen'} {a_} (verwacht vs recent) ===")
            print(t_[['naam', 'land', 'n_races', 'n_recent', 'verwacht_s', 'sd_pct', 'recent_mediaan_dagcorr_s',
                      'recent_mediaan_ruw_s', 'pb_s', 'afwijking_pct']].round(2).to_string(index=False))

    exporteer_distributies_naar_csv(model)
    exporteer_dagen_naar_csv(model)
    exporteer_baancorrecties_naar_csv(model)
    bewaar_model(model)

    print("\n=== Simulatie voorbeeld: 1500m mannen, top-4 actieve rijders ===")
    laatste = int(df['dag'].max())
    kand = [(s.id, model.voorspel(s.id, '1500m')['verwachte_tijd']) for s in model.staten.values()
            if s.gender == 'm' and s.n[NAAR_INDEX['1500m']] >= 3 and laatste - s.laatste_dag < 550]
    kand.sort(key=lambda t: t[1])
    ids = [k[0] for k in kand[:4]]
    if len(ids) >= 2:
        print(simuleer_race(model, ids, '1500m', scenario='normaal', seed=1).to_string(index=False))
        print("\n(zelfde start, piekdag-scenario)")
        print(simuleer_race(model, ids, '1500m', scenario='piekdag', seed=1).to_string(index=False))
    return model


if __name__ == "__main__":
    tunen_vlag = '--tune' in sys.argv
    pad = next((a for a in sys.argv[1:] if not a.startswith('--')), 'speedskating_historie_clean_5pct.csv')
    if not os.path.exists(pad):
        print(f"Fout: kan '{pad}' niet vinden. Zet de CSV in dezelfde map.")
        sys.exit(1)
    hoofdprogramma(pad, tunen=tunen_vlag)
