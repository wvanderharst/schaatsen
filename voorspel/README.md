# Nieuwe voorspeller van rittijden

Een nieuwe poging om de tijd van een rit vooraf te voorspellen. Het model houdt rekening met:
- **baanfactor**: het verwachte racepeil van die dag;
- **vorm**: kort, over alle afstanden;
- **eigen spreiding** van de rijder (historische deviatie);
- **bewezen PR** en seizoensbeste;
- **leeftijd**: groei en achteruitgang.

Alles is chronologisch: een voorspelling gebruikt alleen ritten van vóór de race. Er zijn geen groepen of buckets, alleen
functies (lineair, exponentieel). Het model is gefit op de jaren t/m 2019 en getoetst op 2020 en later. De vergelijking met het oude
model (`schaatsmodel/`, voorspelling vooraf) gaat over precies dezelfde ritten, namelijk die waarop het oude model een eerlijke
log loss heeft.

## Opbouw

1. **IJs vooraf** (`kenmerken.py`, `model.ijs_voorspelling`): het racepeil theta van een race is
   - het lopende baangemiddelde (baan x geslacht x afstand, gekrompen naar de baan),
   - plus de afwijking op eerdere dagen van hetzelfde toernooi,
   - plus de recente afwijking van de baan (snel aanpassend: verbouwing, nieuwe ijsvloer),
   - plus de algemene stand van de laatste weken.

   Fout (sd) per race op de toetsjaren: 0,73% (zonder voorspelling 2,2%).
2. **Niveau** (`kalman_mv2.py`): een Kalmanfilter per rijder over alle 8 afstanden. Het niveau wordt gemeten als ln tijd − theta − wereldtop.
   - De afstanden zijn onderling gecorreleerd (rho^|i−j|, rho ≈ 0,94).
   - De verwachte groei per leeftijd is G1·exp(−(leeftijd − 12)/TG) + G2·max(leeftijd − 35, 0)/10 per jaar.
   - Er is een eigen groeisnelheid per rijder (lokaal lineaire trend, dooft uit).
   - De onzekerheid groeit met de tijd en is groter bij jonge rijders.
   - De meetruis is de eigen spreiding van de rijder (alle afstanden samen).
   - Snelle ritten tellen zwaarder (×1,7), valpartijen niet, en er is korte vorm.

   De parameters zijn afgestemd op 2012–2019 (`kalman_tune.py`): mediane fout van alle ritten + van de toppers. Ook ritten
   die het filter zelf als valpartij ziet tellen mee, zodat het filter niet kan winnen door moeilijke ritten weg te zetten.
3. **Correctiefuncties** (`model2.py`): een robuuste regressie (Huber op gestandaardiseerde residuen), apart voor jong (t/m neo) en ouder.
   Kenmerken:
   - bewezen PR en seizoensbeste t.o.v. het niveau (ook gedeeld door wortel(aantal ritten));
   - de ruwe PR t.o.v. de verwachte tijd op deze baan;
   - de laatste fouten;
   - leeftijd (exponentiële en lineaire functies) en leeftijd x tijd sinds de vorige rit;
   - niveau x baan (hoogland, snel ijs);
   - de onzekerheid van het niveau en de seizoensstart.

   Elk kenmerk heeft ook een variant x exp(−niveau/4%), zodat de top eigen correcties krijgt.
4. **Online biascorrectie**: het uitdovende gemiddelde van de recente fouten van alle rijders (halfwaardetijd 45 dagen, alleen
   eerdere dagen), algemeen en gewogen naar de toppers. Dit vangt een algemene verschuiving, zoals sneller ijs of materiaal.
5. **Onzekerheid**:
   - de schaal per rit is een log-lineaire functie van de Kalman-onzekerheid, de eigen spreiding, de ervaring, de rustperiode, de leeftijd en het niveau, plus de onzekerheid van het ijs;
   - de foutverdeling is een twee-delige Student-t per afstandsgroep, waarvan de vijf parameters lineaire functies zijn van exp(−niveau/4%), het niveau en jong;
   - de snelle staart voorbij 2% krijgt een eigen machtsfunctie.

## Resultaat (toetsjaren 2020+, dezelfde ritten als het oude model)

| groep | ritten | mediane fout oud | mediane fout nieuw | log loss oud | log loss nieuw |
|---|---|---|---|---|---|
| alle | 1.107.298 | 1,329% | **1,298%** | −2,3143 | **−2,3251** |
| toppers (≥ 90) | 116.337 | 0,741% | **0,728%** | −2,9888 | **−3,0209** |
| elite (≥ 97) | 40.226 | 0,657% | 0,657% | −3,1123 | **−3,1293** |
| jong (t/m neo) | 822.266 | 1,340% | **1,305%** | −2,3104 | **−2,3249** |
| senior | 57.092 | 0,903% | **0,873%** | −2,7255 | **−2,7558** |

De groepen ≥ 90 en ≥ 97 zijn het niveau vooraf volgens het oude model.

Nog niet goed:
- **Extreem snel bij de top:** 1 op 1000 komt 1,7× zo vaak voor als verwacht. Een deel daarvan komt uit Kemerovo, waar
  buiten- en binnenbaan door elkaar lopen.
- **Reeksen per rijder:** er zijn te veel rijders met reeksen snelle of trage ritten. Dat zijn rijders die structureel sneller
  beter of slechter worden dan het filter volgt.

## Draaien

In een werkmap met `speedskating_historie_racedatum.csv`, `ritten_theta.parquet` (zie `elite/prep.py` en `elite/theta_merge.py`)
en voor de vergelijking `oud_vooraf.parquet`:

```
python prep_cat.py .                      # categoriecode en geboortejaar -> ritten_cat.parquet
python kenmerken.py .                     # kenmerken vooraf -> kenmerken.parquet
KALMAN=kalman_mv2 python kalman_tune.py 500 start.json kalman_par.json     # afstemmen (optioneel; ~30 min)
KALMAN=kalman_mv2 python model2.py . kalman_par.json                        # fit t/m 2019, toets 2020+ (~25 min)
KALMAN=kalman_mv2 python voorspel_nu.py . kalman_par.json nu 2026-10-21 50  # voorspellingen nu (fit op alle jaren)
```

`oud_vooraf.py` maakt de baseline van het oude model (in de map met `model_full_tuned.pkl`).


## Universele baanfactor (`baanfactor.py`, `maak_baanfactor.py`)

Eén methode voor alle onderdelen: de Kalman-meting, PR en seizoensbeste, de voorspelling vooraf en het omrekenen naar een baan.

    baanfactor(rit i in race r) = theta_r + d_r x (m(g_i) / m(gbar_r) - 1),   m(g) = 1 + A (exp(-g / TM) - exp(-G0 / TM))

- theta_r is het dagpeil zoals het veld het mat (achteraf), of zoals de baanhistorie het voorspelt (vooraf).
- d_r is hoe snel het ijs is ten opzichte van Heerenveen.
- g is het niveau van de rijder, gbar het gewogen niveau van het veld. Het gemeten peil geldt voor het veld; een rijder sterker of zwakker dan het veld krijgt meer of minder van het verschil met Heerenveen.
- Geschat op 2019 en eerder: A = 0,29, TM = 0,20. Toppers winnen op ijs dat 1,3% sneller is dan Heerenveen ongeveer 1,5%, het veld 1,3% en zwakke rijders 1,15%.

Een volledig zelfstandig dagpeil, zonder het veldpeil als invoer (`kalman_mv3.py`), drijft weg. Elke systematische fout in de niveaus wordt door het veld als ijs gelezen en versterkt zichzelf. Dat gebeurt ook met een Heerenveen-anker en zonder jeugd.

Toets (2020+, log loss; lager is beter):

| variant | alle | toppers | elite |
|---|---|---|---|
| vorige versie (lineaire baan x niveau-termen in de correcties) | −2,3251 | −3,0209 | −3,1293 |
| universele baanfactor | −2,3217 | −3,0151 | −3,1205 |
| universele baanfactor + uitschieters | −2,3246 | −3,0161 | −3,1215 |
| oud model | −2,3143 | −2,9888 | −3,1123 |

## Historische deviatie

- **Uitschieters:** het filter houdt per rijder bij hoe vaak een rit snel of traag buiten 2 sd viel. Dat gebeurt over alle afstanden, met een lang geheugen (150 ritten) en gekrompen naar de basis. Deze uitschieterkans zit in de correcties, de schaal en de vorm van de foutverdeling.
- **Week-op-week:** de ritruis van het filter is KR x CD^2 x (eigen spreiding)^BR. BR ≈ 0,1 maakt het filter zelf iets nauwkeuriger, maar het hele model slechter (log loss alle −2,3010): de eigen spreiding is nodig voor de breedte van de verdeling. Daarom blijft BR = 1.

## Talentmodel (`talent.py`)

Het model voorspelt aan het eind van elk jeugdseizoen (13–19 jaar) de kans dat de rijder op 21–24 jaar het niveau van de wereldtop-50 of top-10 haalt op de beste seniorenafstand.

- **Stand:** momentopnamen van het Kalmanfilter.
- **Kenmerken:**
  - niveau ten opzichte van de wereldtop en ten opzichte van leeftijdsgenoten;
  - groeisnelheid en verbetering over één en twee seizoenen;
  - piek (beste rit) ten opzichte van het niveau;
  - sprint- of duurprofiel;
  - uitschieters;
  - aantal ritten en jaren actief;
  - land en geslacht.

  Alles ook × leeftijd.
- **Model:** de kans dat de rijder doorgaat × de verdeling van het verwachte seniorenniveau. Die verdeling is empirisch, met een spreiding als functie van leeftijd en niveau.
- **IJking:** een kruisfit tussen de jaargangen 1992–96 en 1997–2001, met een ijkingsfunctie die kwadratisch is in de logit.

Toets (buiten de steekproef): op 16 jaar zitten van de latere top-50-profs er 50 van de 109 in de top-N van het model, tegen 43 bij rangschikken op niveau. Van de latere top-10 zijn het er 13 van de 30, tegen 8. De kansen zijn gekalibreerd (bijv. voorspeld 30% → werkelijk 30%, 62% → 65%).

De lijst van nu staat in `talenten_2026-10-21.csv`: 150 per geslacht.


## Elite op beter geprepareerd ijs (ijs/veld-term in de baanfactor)

    baanfactor = theta_r + d_r x (m(g_i) / m(gbar_r) - 1) + B x (exp(-g_i / TB) - exp(-gbar_r / TB))

Toprijders (binnen 4% van de wereldtop) rijden ten opzichte van hun niveau en het gemeten dagpeil trager naarmate hun race zwakker bezet is.
Dat geldt ook binnen dezelfde rijders: −0,03% in elite-velden en +0,15% in de zwakste velden. Het dagpeil wordt op het veld gemeten, en het
ijs en de scherpte zijn in elite-wedstrijden beter.

- **Schatting:** B en TB worden samen met A en TM geschat, met hulptermen voor een afwijking die alleen van het eigen niveau afhangt. Uitkomst: B = 1,6%, TB = 0,20, A = −0,25, TM = 0,5.
- **Bijvangst:** de term vangt ook dat het gemeten peil van een veld scheef kan zijn door de samenstelling, zoals groeiende jeugd.
- **Vooraf:** dezelfde formule werkt ook vooraf, want de startlijst is bekend.

| variant (2020+, log loss) | alle | toppers | elite | elite mediane fout |
|---|---|---|---|---|
| universele baanfactor + uitschieters | −2,3246 | −3,0161 | −3,1215 | 0,664% |
| + ijs/veld-term | **−2,3264** | **−3,0261** | **−3,1375** | **0,650%** |
| oud model | −2,3143 | −2,9888 | −3,1123 | 0,657% |

## Vormlaag (`vormlaag.py`)

Vormpieken werden niet gevangen. Na een snelle schok (kans ≤ 1/50) reed dezelfde rijder ook de weken erna sneller dan voorspeld, en na
elke schok was het onrustiger (beide staarten). Gemiddeld leken de staarten goed, maar dat was een mengsel van rustige periodes (te dunne
staarten) en periodes na een schok (veel te dikke staarten). Volgende ritten na een snelle schok (toetsjaren, `vorm_diag.py`):

| | 1–7 dagen | 8–30 dagen | 31–90 dagen |
|---|---|---|---|
| toppers, snel 1/20 vóór vormlaag | 1,7× | 1,1× | 1,2× |
| toppers, snel 1/20 met vormlaag | 0,80× | 0,98× | 1,00× |
| overige rijders, snel 1/20 vóór vormlaag | 2,5× | 1,8× | 1,5× |
| overige rijders, snel 1/20 met vormlaag | 0,83× | 0,95× | 0,92× |

**Methode.** z = Φ⁻¹(PIT) van het model; de vormlaag maakt z' = (z − mu) / sigma. Dat is een geldige verdeling voor de tijd
(Gauss-copula op de modelverdeling), en de log loss verandert exact met log φ(z) − log φ(z') + log sigma.
- mu is lineair in het uitdovende gemiddelde van de eigen z (7, 30 en 90 dagen), de snelle en trage schokken (het deel van z voorbij 2)
  en het gemiddelde rijdergeheugen van de startlijst (een circuit dat recent sneller of trager rijdt dan voorspeld).
- log sigma is lineair in de extra variantie van de laatste 7, 30 en 90 dagen (z² − 1).
- Alles ook × exp(−niveau/4%). Er zijn geen vaste termen: de fitjaren liggen voor het basismodel binnen de steekproef, en vaste termen
  leerden daar een niveau en breedte die in de toetsjaren niet kloppen (elite 0,05 sd te traag, toppers sigma 0,94).
- Alleen eerdere dagen; eigen valpartijen tellen niet mee. Geschat op 2012–2019, toppers ×3.
- De voorspelde tijd is de mediaan van de nieuwe verdeling; `voorspel_nu.py` rekent de kwantielen exact uit, met de vormlaag en de staartcorrectie.

Toets (2020+, dezelfde ritten als het oude model):

| | log loss alle | toppers | elite | mediane fout elite | kalibratiefouten toppers / elite |
|---|---|---|---|---|---|
| zonder vormlaag | −2,3303 | −3,0362 | −3,1449 | 0,655% | 7 / 4 |
| met vormlaag | **−2,3394** | **−3,0408** | **−3,1496** | **0,652%** | 9 / **1** |
| oud model | −2,3143 | −2,9888 | −3,1123 | 0,657% | |

Elite met vormlaag: snel 1/1000 0,99×, 1/100 0,92×, 1/20 1,02×; traag 1,09×, 1,01×, 0,96×. Rijders met ≥ 3, 4, 6 en 8 snelle ritten
(kans ≤ 1/50) kloppen. Nog niet goed:
- **Toppers 90–97:** de verdeling is iets te breed (z-sd 0,977): er zijn minder snelle uitschieters dan voorspeld (1/100: 0,84×). Dat zat al in de basis.
- **Trage reeksen per rijder:** te veel rijders met ≥ 5 trage ritten (toppers 83 tegen 60, elite 32 tegen 20).

## Weer (`luchtdruk.py`, `model.weer_kenmerken`)

Luchtdruk, temperatuur, wind en vocht per baan en wedstrijddag (Open-Meteo, ERA5; 10–20 uur lokale tijd), opgehaald op GitHub Actions
(`.github/workflows/luchtdruk.yml`, in deze omgeving is het internet dicht) en vastgelegd in `voorspel/weer/weer_dag.csv`.
Per baan staan plaats en hoogte in `weer/banen_locatie.csv`; waar de baan niet in de stad ligt, zijn ze met de hand gezet (Kearns 1423 m, Calgary 1105 m, ...).

Fout in het dagpeil vooraf tegen het weer van die dag (19 banen):
- **Luchtdruk:** op alle banen zo'n 0,04–0,12 × de relatieve drukafwijking (binnen ~0,1): 10 hPa lagere druk is ~0,1% sneller.
- **Wind** telt alleen op buitenbanen: Utrecht +0,34% per 10 km/u, Deventer +0,28%, Amsterdam +0,26%; binnen ~0.
- **Temperatuur:** buiten is warmer sneller (Breda −0,49%, Hoorn −0,43% per 10 graden); binnen iets trager (Hamar, Stavanger +0,2%).

In de ijsvoorspelling:
- `ijs_dp`: ln(druk) − ln(gewone druk op die baan), één coëfficiënt (0,058 met alle banen).
- `ijs_weer`: per baan de gevoeligheid voor wind en temperatuur, alleen uit eerdere wedstrijddagen op die baan (halfwaardetijd 2 jaar, krimp naar 0). Het model leert zo zelf welke banen buiten liggen en wanneer een baan overdekt werd. Wordt voor 70% overgenomen.

Toets (2020+, dezelfde ritten als het oude model):

| | log loss alle | toppers | elite | senior | mediane fout elite | kalibratiefouten toppers / elite |
|---|---|---|---|---|---|---|
| met vormlaag | −2,3394 | −3,0408 | −3,1496 | −2,7715 | 0,652% | 9 / 1 |
| + weer, 19 banen | −2,3422 | −3,0451 | **−3,1548** | −2,7755 | 0,648% | 7 / 0 |
| + weer, alle banen (98,6% van de ritten) | **−2,3434** | **−3,0457** | −3,1545 | **−2,7763** | 0,648% | 7 / **0** |

Elite met weer: snel 1/1000 0,99×, 1/100 0,92×, 1/20 1,01×; traag 1,04×, 1,03×, 0,97×.

## Nieuwe banen en snel veranderende omstandigheden (`model.startpeil_kenmerken`, `model.afwijking_kenmerken`)

Aanleiding: Milano 2026. Het model verwachtte de nieuwe olympische baan vooraf 1,2–1,5% trager dan Heerenveen; gemeten was het ongeveer even
snel (op vijf sprintafstanden sneller dan het baanrecord van Heerenveen).
- **Kleine races** (< 5 rijders met een tijd: trainingen, tijdritten) tellen niet mee in de geschiedenis van het ijs (`kenmerken._ijs`).
  In de schatting tellen ze wel, met een functie van de startlijst exp(−rijders/5): kleine races lopen trager.
- **Startpeil van een nieuwe baan:** eerst het gemiddelde van alle banen (veel trage en buitenbanen). Nu een functie van het niveau
  van het veld en van de hoogte (gewone luchtdruk), met gewicht exp(−eerdere races op de baan / 5).
- **Eerdere toernooidagen** gemeten t.o.v. de verwachting vooraf (eerst t.o.v. het eigen dagpeil, en dat is op een nieuwe baan altijd 0).
- **Snel veranderende omstandigheden:** de afwijking van eerdere races op deze baan t.o.v. de verwachting vooraf, uitdovend met
  halfwaardetijd 7 en 60 dagen, gewogen naar het aantal rijders.
- De onzekerheid van het ijs hangt ook af van de geschiedenis van de baan.

Dagpeil vooraf (sd van de fout): toetsjaren 0,730% → 0,703%; banen met < 30 eerdere races 1,84% → 1,45%. Milano, eerste olympische race:
+1,30% → +0,88% (de races van 4 februari, 13–18 rijders, waren 1,4% traag en trekken nog).

| (2020+, log loss) | alle | toppers | elite | senior | mediane fout elite |
|---|---|---|---|---|---|
| met weer | −2,3434 | −3,0457 | −3,1545 | −2,7763 | 0,648% |
| + nieuwe banen, snelle veranderingen, kleine races | **−2,3461** | **−3,0495** | **−3,1557** | **−2,7803** | 0,649% |

Snelle staart (de belangrijkste): toppers 1/1000 0,95×, 1/100 0,83×, 1/20 0,95×; elite 1,02×, 0,89×, 1,01×.

## Lopende ijking van de snelle staart (`staartijking.py`)

De snelle staart telt het zwaarst. Zijn vorm hangt niet-lineair af van het niveau en verschuift in de tijd. Een voorbeeld: toppers 90–97
hadden bij 1/100 in de fitjaren 0,95×, in 2020–22 0,84× en vanaf 2023 0,77× zoveel snelle uitschieters als voorspeld. Daarom als laatste laag
twee geneste stappen p' = Q (p/Q)^b (Q = 10%, daarna 1%). b is een gladde functie van het niveau (Gauss-kernel, bandbreedte 2%), geschat
uit eerdere dagen (halfwaardetijd 1 jaar, krimp naar 1). De kwantielen in `voorspel_nu.py` draaien de ijking terug.

| (2020+) | log loss alle | toppers | elite | toppers snel 1/1000, 1/100, 1/20 | elite snel 1/1000, 1/100, 1/20 |
|---|---|---|---|---|---|
| zonder staartijking | −2,3461 | −3,0495 | −3,1557 | 0,95 / 0,83 / 0,95 | 1,02 / 0,89 / 1,01 |
| met staartijking | **−2,3465** | **−3,0497** | −3,1557 | **1,06 / 0,95 / 0,99** | 1,14 / 1,04 / 1,05 |

Toppers met de staartijking: alle tellingen van rijders met meerdere snelle ritten kloppen (≥3: 274 tegen 289; ≥4: 142 tegen 136;
≥8: 7 tegen 3,3). Ervoor waren ≥3 en ≥4 fout (237 en 105).

## Backtest: afstand tot de internationale top (`backtest.py`)

Internationale topraces (wereldbekers, WK's, EK's, Spelen; herkend aan het veld: minstens 10 rijders binnen 5% van de wereldtop uit
minstens 5 landen): 2.777 races, 106.850 ritten, waarvan 31.057 in 2020+.
- **Maatstaf:** afstand tot de top g = ln tijd − gemiddelde ln tijd van de 5 snelsten zonder de rijder zelf. De ligging en schaal per
  rijder × afstand komen uit eerdere internationale races, met een logistische of scheef logistische verdeling. Negatieve afstanden zijn
  geen probleem, want de verdeling loopt over alle getallen. De variant met GAS-update gebruikt de score van de logistische verdeling,
  met een eigen leersnelheid voor de eerste races. Alle parameters zijn geschat op 2008–2019.
- **Onze methode met dezelfde informatie:** onze voorspelling + de gemiddelde afwijking van diezelfde 5 rijders t.o.v. onze voorspelling.

| (2020+, log loss ln tijd) | alle ritten | binnen 5% van de top | snel 1/1000, 1/100, 1/20 (binnen 5%) |
|---|---|---|---|
| maatstaf, logistisch | −2,521 | −2,779 | 1,32 / 0,91 / 0,90 |
| maatstaf GAS, logistisch | −2,549 | −2,807 | 2,59 / 1,25 / 0,97 |
| onze methode met top 5, scheef logistisch | **−3,120** | **−3,244** | 0,27 / 0,60 / 0,85 |
| onze methode vooraf (zonder top 5) | −3,037 | −3,130 | **1,14 / 1,11 / 1,02** |

Onze methode vooraf, met de onzekerheid van het ijs, is veel beter dan de maatstaf die de top 5 achteraf al kent. De afstand tot de top 5
hangt sterk af van wie er in de race zit en hoe die rijders die dag rijden. Wel scheidt de maatstaf ijs van rijder: Leerdams 1:12,35 in
Salt Lake City was t.o.v. de top 5 gewoon (maatstaf 1 op 2, onze methode met top 5 1 op 10); de 1 op 160 vooraf komt vooral van het ijs.
Lollobrigida's 3:54,28 in Milano was ook t.o.v. de top 5 uitzonderlijk (maatstaf GAS 1 op 267, onze methode met top 5 1 op 837).

## Dagpeil zonder krimp en persoonlijke hooglandaffiniteit (Leerdam, SLC 14-11-2025)

Leerdam reed 1:12,35 in SLC tegen 1:13,67 verwacht (1 op 257). T.o.v. de top 5 was de rit gewoon: het hele elite-veld reed die dag ~1%
sneller dan voorspeld, terwijl het gemeten dagpeil zei dat het ijs normaal was.
- **Krimp in het dagpeil.** Het dagpeil theta uit het oude model is een schatting die naar de verwachting voor de baan wordt getrokken
  (en extreme afwijkingen worden afgekapt). Daardoor komt maar 40–60% van een gezamenlijke afwijking van het veld als ijs in theta
  (helling 0,42 over alle races, 0,59 bij elitevelden, 0,43 in SLC en Calgary). De rest wordt de rijders toegerekend.
  `dagpeil_veld.py` maakt een dagpeil zonder krimp: de verwachting vooraf plus het mediane residu van het veld, gewogen met n/(n + 3,5).
  Met dat dagpeil was SLC op 14-11-2025 0,95–1,28% sneller dan het oude dagpeil.
- **Persoonlijke hooglandaffiniteit** (`model2.hoog_affiniteit`). Leerdam rijdt de 1000 m op hoogland gemiddeld 0,47% sneller dan voorspeld
  (13 ritten, waarvan 3 bij de snelste 5%), en wisselvalliger. Per rijder × afstand bepalen eerdere ritten op hoogland de ligging en de
  schaal op de kansschaal (b = 0,41, krimp 2,1 ritten; schaal g = 0,76, krimp 23 ritten; geschat op 2008–2019).

Geprobeerd en niet gebruikt (slechter of geen effect): een factor per baan op de gezamenlijke onzekerheid, de dag in het toernooi en
hoogland × niveau in schaal en vorm, een lopende regressie van de ijsonzekerheid, een lopende ijking per baan.

| (2020+, log loss) | alle | toppers | elite | senior | mediane fout elite | Leerdam SLC |
|---|---|---|---|---|---|---|
| met staartijking (oud dagpeil) | **−2,3465** | −3,0497 | −3,1557 | −2,7808 | 0,649% | 1 op 160 |
| dagpeil zonder krimp | −2,3443 | −3,0509 | −3,1627 | −2,7849 | 0,644% | 1 op 257 |
| + hooglandaffiniteit | −2,3446 | **−3,0526** | **−3,1662** | **−2,7868** | **0,641%** | **1 op 119** |

Snelle staart met hooglandaffiniteit: toppers 0,99 / 1,00 / 1,04×, elite 0,92 / 1,12 / 1,11×; de aantallen per rijder kloppen bij de elite.

## Dikkere staarten geprobeerd (`verdeling_lab.py`)

Proef op de opgeslagen fouten u van het huidige model: alleen de basisverdeling plus de vaste snelle-staartstap (zonder vormlaag en
affiniteit), per afstandsgroep, parameters lineair in Z, geschat op 2008–2019, getoetst op 2020+. Families: de twee-delige Student-t
(het model), een twee-delige gegeneraliseerde t (per kant los een piek p en een staart q; p = 2 is de Student-t), een t met een brede
bijmenging, en de Student-t met de vrijheidsgraden aan de snelle kant geforceerd op hoogstens 10, 6 of 4.

| (2020+, basisverdeling) | alle | toppers | elite | elite snel 1/1000, 1/100, 1/20 | elite ≥3 / ≥5 snelle ritten (verwacht 98,7 / 20,2) |
|---|---|---|---|---|---|
| twee-delige t (model) | −2,3311 | −3,0440 | **−3,1541** | 1,29 / 1,19 / 1,18 | 117 / 36 |
| gegeneraliseerde t | −2,3307 | −3,0441 | −3,1536 | 1,12 / 1,21 / 1,18 | 120 / 38 |
| t + brede bijmenging | −2,3312 | −3,0438 | −3,1537 | 1,34 / 1,20 / 1,19 | 118 / 36 |
| snelle kant ν ≤ 10 | −2,3305 | −3,0421 | −3,1522 | 1,02 / 1,04 / 1,12 | 88 / 17 |
| snelle kant ν ≤ 6 | −2,3281 | −3,0387 | −3,1487 | 0,82 / 0,78 / 1,00 | 51 / 6 |
| snelle kant ν ≤ 4 | −2,3231 | −3,0325 | −3,1420 | 0,99 / 0,54 / 0,86 | 19 / 2 |

De fitjaren willen aan de snelle kant geen dikke staart: bij de elite lopen de vrijheidsgraden tegen 30 aan (vrijwel normaal), de bijmenging
krijgt gewicht 0 en de gegeneraliseerde t kiest q ≈ 53 (sprint). Alleen de trage kant is dik (ν ≈ 5–7). Geforceerd dikker maakt de tellingen
in de verre staart beter (ν ≤ 10: elite 1,02 / 1,04×), maar kost log loss en geeft te weinig rijders met meerdere snelle ritten. Het teveel
aan snelle ritten in de toetsjaren zit al bij 1/20, dus in de schouder en niet in de verre staart: dat is een verschuiving in de tijd
(waar de vormlaag en de lopende ijkingen voor zijn), geen vorm van de verdeling. Niet gebruikt.

## Snelle staart op elk niveau en elke afstand: rust, rijder, niet-parametrische ijking (`rustlaag.py`, `staart_np.py`)

**Let op bij de toets.** `niveau_pct` van het oude model (toppers ≥90, elite ≥97) is de rating NA de race (`x_post` in `do_analysis.py`):
een snelle rit tilt de rijder in die groep. Daar zijn snelle uitschieters dus altijd oververtegenwoordigd (2020+, zelfde model: elite ≥97
snel 1/20 1,11×, elite naar het niveau vooraf k_c < 1% 0,92×). De kalibratie hieronder is op het niveau vooraf (k_c), zonder valpartijen.
De log loss-vergelijking op toppers/elite blijft op dezelfde ritten als het oude model, maar beloont een dikkere snelle staart.

Na de vorige versie lag de snelle staart op het niveau vooraf op elk niveau te dik (2020+ 1/10 en 1/20: 0,87–0,95×; per afstand 3000 m 0,80×,
500 m 0,88×), hing hij af van de rust en clusterde hij per rijder. Drie lagen na de staartijking, alle online (alleen eerdere dagen):

1. **Rust en taper** (`rustlaag.py`): z' = (z − μ) / σ met μ en σ als gladde functie van het niveau en r = ln(1 + rustdagen) (Gauss-kernels,
   halfwaardetijd 3 jaar, krimp 100 ritten). Bij de top (< 3,5%): na 4–10 dagen rust gemiddeld z −0,07 (vaker snel, taper), op dezelfde dag
   +0,07 tot +0,19, de tweede toernooidag +0,05 tot +0,12, na 1–5 maanden zonder race +0,16 tot +0,19. Bij zwakkere rijders anders.
2. **Staartmassa per rijder** (`staart_np.rijder`): λ = (waargenomen + 10) / (verwacht + 10) snelle ritten (p < 1/20) in eerdere ritten van de
   rijder (halfwaardetijd 2 jaar); p' = λ p onder 1/20, daarboven lineair. Vangt jonge, stijgende rijders (Wennemars, Stolz, Daleman, Prins)
   die jarenlang vaker snel rijden dan hun niveau zegt.
3. **Niet-parametrische ijking** (`staart_np.ijk`), laatste laag: p' = G(p), G monotoon op knopen 0,5 / 0,3 / 0,2 / 0,1 / 0,05 / 0,02 / 0,01 /
   0,005 / 0,002 / 0,001 / 0,0003 / 0,0001 aan beide kanten; per knoop de voorwaardelijke kans uit eerdere dagen (kernel in niveau, bandbreedte
   2%, halfwaardetijd 2 jaar), gekrompen naar de geijkte verhouding (30 pseudo-ritten) en per afstand naar alle afstanden samen (100).
   Log loss verandert exact met −ln G'(p). De omkering wordt per rit bewaard voor de kwantielen 0,1 / 0,5 / 0,9 (`kwantiel_u`).

De voorspelde tijd is de mediaan na alle lagen (ook voor de denkbeeldige ritten in `voorspel_nu.py` en `wat_als_heerenveen.py`).

| (2020+, log loss) | alle | toppers | elite | senior | toppers vooraf (k_c < 6%) | elite vooraf (k_c < 2%) | mediane fout elite |
|---|---|---|---|---|---|---|---|
| hooglandaffiniteit (vorige) | −2,3446 | −3,0526 | −3,1662 | −2,7868 | −3,0409 | −3,1550 | 0,641% |
| + rust, rijder, niet-parametrisch | **−2,3571** | **−3,0564** | **−3,1733** | **−2,7960** | **−3,0480** | **−3,1648** | **0,638%** |

Tellingen 2020+ (waargenomen / verwacht, zonder valpartijen), snel 1/10, 1/20, 1/100, 1/1000:

| | vorige | nu |
|---|---|---|
| niveau < 1% | 0,91 / 0,92 / 0,86 / 0,63 | 0,99 / 0,98 / 0,95 / 0,94 |
| niveau 1–2% | 0,92 / 0,92 / 0,89 / 0,86 | 1,01 / 1,03 / 1,01 / 0,93 |
| niveau 2–3,5% | 0,90 / 0,87 / 0,77 / 0,84 | 0,99 / 0,98 / 0,87 / 0,80 |
| niveau 3,5–6% | 0,91 / 0,90 / 0,82 / 0,69 | 0,99 / 1,00 / 0,92 / 0,73 |
| niveau 6–10% | 0,91 / 0,90 / 0,91 / 0,73 | 0,99 / 0,98 / 0,98 / 0,67 |
| niveau 10–18% | 0,95 / 0,95 / 0,96 / 1,08 | 1,00 / 1,00 / 1,00 / 0,98 |
| niveau > 18% | 0,90 / 0,88 / 0,84 / 0,93 | 0,99 / 0,98 / 0,96 / 0,97 |
| 500 m | 0,91 / 0,88 / 0,83 / 0,79 | 0,99 / 0,98 / 0,96 / 0,89 |
| 1000 m | 0,97 / 0,97 / 0,99 / 1,08 | 1,00 / 1,00 / 0,99 / 0,92 |
| 1500 m | 0,90 / 0,88 / 0,86 / 0,99 | 0,98 / 0,98 / 0,96 / 0,97 |
| 3000 m | 0,83 / 0,80 / 0,78 / 0,96 | 0,97 / 0,96 / 0,93 / 1,00 |
| 5000 m | 0,89 / 0,86 / 0,78 / 0,80 | 0,97 / 0,95 / 0,91 / 0,92 |
| 10000 m (5.377 ritten) | 0,98 / 1,03 / 1,41 / 2,41 | 1,11 / 1,17 / 1,56 / 2,05 |

Kalibratietoets op het niveau vooraf (zonder valpartijen): top < 2% **0 fouten** (snel 0,94 / 0,97 / 1,00×, traag 0,89 / 1,00 / 1,04×,
alle aantallen per rijder goed; vorige versie 8 fouten), top < 6% 2 fouten (snel 1/100 0,93×, traag 1/1000 0,76×; aantallen per rijder goed).
Nog niet goed: de 10 km (weinig ritten, te veel snelle uitschieters) en de top op de 3000 m (1/20 1,11×).
Leerdam 1000 m SLC 14-11-2025: 1 op 95 (was 1 op 119); Lollobrigida 3000 m Milano: 1 op 1572 (was 1 op 3497).
Proeven: `rust_staart_lab.py` (op een opgeslagen uitkomst).

## Onmogelijke ritten en diepere staart-ijking (`onmogelijk.py`, `staart_np.py`)

**Onmogelijke ritten** worden als `verdacht` gemarkeerd en tellen dan nergens mee (net als valpartijen): een hele race waarvan het veld
in de mediaan meer dan 20% sneller is dan verwacht, of een losse rit meer dan 40% sneller dan verwacht én meer dan 6 sd van het model
(het patroon van een verkeerd afstandslabel: 3000 m als 5000 m is −51%). Alleen als de rijder binnen 2 jaar daarna op die afstand nooit
binnen 15% van die tijd rijdt: zo blijven echte doorbraken staan waar de voorspelling fout zat (Brock Miron 1000 m Hamar 2007, Mike Ireland
500 m, toen nog weinig geschiedenis). Resultaat: 188 ritten (110 in 2020+), vrijwel alle jeugd, 33–70% te snel, vooral Hailar en Daqing
(bijv. Hailar 22-08-2024 '5000 m': twaalf rijders twee keer zo snel als verwacht). Lijst: `onmogelijk.csv`. Volgorde in de keten:
modelrun → `onmogelijk.py` → `kenmerken.py` → modelrun.

**Diepere staart-ijking**: de krimp in de niet-parametrische ijking neemt af met de diepte in de staart (k0 (x/0,5)^0,25 en k1 (x/0,5)^0,5
pseudo-ritten), zodat afstanden en niveaus hun eigen extreme staart sneller leren. Poisson-deviantie van de snelle aantallen (1/100, 1/1000,
1/5000) over 7 niveaus × 6 afstanden, 2020+, zonder valpartijen en verdachte ritten (~126 bij perfecte kalibratie): vorige versie 265, nu 230
(het model zonder de drie lagen: 954).

| (2020+, log loss) | alle | toppers | elite | senior | toppers vooraf | elite vooraf | mediane fout elite |
|---|---|---|---|---|---|---|---|
| rust + rijder + niet-parametrisch | −2,3571 | −3,0564 | −3,1733 | −2,7960 | −3,0480 | −3,1648 | 0,638% |
| + onmogelijke ritten eruit, diepere staart-ijking | −2,3576* | −3,0563 | −3,1731 | −2,7956 | −3,0479 | −3,1648 | 0,637% |

\* 110 ritten minder (de verdachte). Kalibratie op het niveau vooraf: top < 2% 0 fouten, top < 6% 3 fouten (snel en traag 1/1000 0,80×,
snel 1/100 0,93×). Nog niet goed: voorbij 1/5000 bij zwakkere rijders (10–18%, > 18%) en op 1500 m, 3000 m en 10 km te veel extreme
snelle ritten (alle niveaus 1/10.000: 160 tegen 110); bij de top ligt alles binnen de band.

## Lange-termijngeheugen: bewezen niveau (`plafond.py`, `rustlaag.laag`)

Diagnose (`geheugen_diag.py`, 2020+, top < 3,5%): de op een na beste gecorrigeerde rit (q = ln t − theta, achteraf gecorrigeerd voor baan,
ijs, weer en hoogte) van de laatste 2 jaar op dezelfde afstand, t.o.v. de verwachting vooraf, voorspelt de afwijking sterk: 2–3% sneller
bewezen → gemiddeld z −0,15 en 1,40× zoveel snelle ritten (1/20); 0–1% trager bewezen → z +0,16 en 0,66×. Ook de afwijking van 1–2 jaar
terug hangt nog samen met nu (helling +0,10). Het model vergat dus te snel wat een rijder al eens heeft laten zien.

Twee lagen op de kansschaal (na de rustlaag), elk z' = (z − μ) / σ met μ en σ een gladde functie van het niveau en het verschil in % tussen
de op een na beste gecorrigeerde rit in het venster en de verwachting vooraf (Gauss-kernels: niveau 2%, verschil 0,5%; halfwaardetijd 3 jaar;
krimp 30 ritten), online uit eerdere dagen: venster 1 jaar en venster 3 jaar. De op een na beste maakt het robuust tegen één meevaller.

| (2020+, log loss) | alle | toppers | elite | senior | toppers vooraf | elite vooraf | mediane fout elite |
|---|---|---|---|---|---|---|---|
| zonder bewezen niveau | −2,3576 | −3,0563 | −3,1731 | −2,7956 | −3,0479 | −3,1648 | 0,637% |
| met bewezen niveau (1 en 3 jaar) | **−2,3634** | **−3,0640** | **−3,1873** | **−2,8011** | **−3,0550** | **−3,1775** | **0,628%** |

Kalibratie op het niveau vooraf: top < 2% 0 fouten, top < 6% 3 fouten (snel 1/1000 0,76×, 1/100 0,91×, traag 1/1000 0,78×).
Lollobrigida 3000 m Milano: 1 op 1134 (was 1 op 1236): haar sterke ritten (3:58,4 in Heerenveen-termen) lagen 11–12 maanden terug; in het jaar
ervoor was haar op een na beste rit 4:00,2. Leerdam en Kok in Milano worden juist verrassender (1 op 64 en 1 op 137): hun verwachting lag al
op of boven hun bewezen niveau, en gemiddeld vallen zulke rijders iets terug. Wat ontbreekt is het belang van de wedstrijd (pieken op de
Spelen); de toernooinamen staan in de brondata (`data/`, kolom Toernooi).

## Piekmomenten: WK's en Olympische Spelen (`piek.py`)

Alleen de Spelen en de senioren-WK's (afstanden, allround, sprint) zijn piekmomenten. De brondata heeft geen toernooinamen, dus een kalender
(jaar, baan, soort) plus herkenning in de data: een wereldveld (>= 10 landen, >= 2 van USA/CAN/JPN/CHN/KOR, dus geen EK) met het
afstandspatroon van het soort toernooi (sprint 500/1000, allround zonder 1000 m, afstanden en Spelen alle zes); bij meerdere kandidaten het
breedste veld. Alle 54 toernooien 2007–2026 gevonden (`python piek.py <werkmap>` drukt de lijst af); 11.210 ritten.

Diagnose (zonder deze laag): op piekmomenten rijdt de top sneller dan voorspeld (top < 1%: z −0,24 in 2008–2019, −0,11 in 2020+) en zijn er
half zoveel trage uitschieters. Laag op de kansschaal (na het bewezen niveau): z' = (z − μ) / σ, μ en σ een gladde functie van het niveau,
alleen voor piekritten, online uit eerdere piekmomenten (krimp 30 ritten, halfwaardetijd 3 jaar).

| (2020+, log loss) | alle | toppers | elite | senior | elite vooraf | piekritten (2.815) | piekritten top < 2% |
|---|---|---|---|---|---|---|---|
| zonder piekmomenten | −2,3634 | −3,0640 | −3,1873 | −2,8011 | −3,1775 | −3,2430 | −3,2819 |
| met piekmomenten | **−2,3635** | **−3,0642** | **−3,1880** | **−2,8015** | **−3,1782** | **−3,2490** | **−3,2875** |

Op de piekritten van 2020+ schiet de laag iets door (snel 1/20 1,09 → 0,90×, traag 0,68 → 1,10×): het piekeffect was in 2008–2019 groter dan
daarna. Milano 2026: Lollobrigida 3000 m 1 op 894 (was 1 op 1134), Leerdam 1000 m 1 op 47 (was 64), Kok 1000 m 1 op 98 (was 137).

## Leeftijd (`rustlaag.laag` met x = leeftijd) en kalibratie van kampioenen (`kampioenen.py`)

**Kampioenen.** `kampioenen.py` bepaalt alle wereld- en olympisch kampioenen 2007–2026 (268 titels, 75 rijders; WK afstanden en Spelen:
snelste tijd per afstand; WK allround en sprint: eindstand in samalog-punten) en hun persoonlijke kalibratie over alle ritten, valpartijen
inbegrepen, ook zonder de titelraces zelf (die zijn op de uitkomst gekozen: gemiddeld z −0,37). Als groep goed gekalibreerd (2020+, zonder
titelraces: gem. z −0,03, snel 1/20 · 1/100 · 1/1000 = 288/257 · 55/51 · 5/5, aantallen per rijder goed). Valpartijen (0,5% van de ritten)
zitten niet in de voorspelde verdeling: met valpartijen traag 1/1000 32 tegen 5 (zonder: 5 tegen 5). Per kampioen hangt de afwijking samen
met de loopbaanfase: jonge stijgers onderschat (Jílek, Wennemars, De Boo, Kongshaug), oudere rijders overschat (Bowe, Jackson, Golikova).

**Leeftijdslaag** (na de piekmomenten): z' = (z − μ) / σ, μ en σ een gladde functie van niveau × leeftijd op de racedag (Gauss-kernels:
niveau 2%, leeftijd 1,5 jaar), online, krimp 30 ritten. Gemiddelde z van de top < 3,5% (2020+) naar leeftijd: ≤ 18 jaar −0,072 → −0,045,
18–20 −0,036 → −0,018, 28–30 +0,080 → +0,059, > 35 +0,135 → +0,067. Een laag bewezen niveau × leeftijd (in plaats van of naast niveau) was
voor de elite slechter.

| (2020+, log loss) | alle | toppers | elite | senior | elite vooraf | mediane fout elite |
|---|---|---|---|---|---|---|
| zonder leeftijdslaag | −2,3635 | −3,0642 | −3,1880 | −2,8015 | −3,1782 | 0,627% |
| met leeftijdslaag | **−2,3652** | **−3,0647** | **−3,1886** | −2,8012 | **−3,1792** | 0,628% |

Kalibratie op het niveau vooraf: top < 2% 0 fouten. De oudere kampioenen schuiven naar 0 (Bowe +0,24 → +0,16), de jonge stijgers nauwelijks
(Jílek −0,66 → −0,62, Wennemars −0,42 → −0,41, De Boo −0,42 → −0,39): zij groeien sneller dan hun leeftijdsgroep.

## Andere motoren dan het Kalman-filter (`motor_lab.py`, `gaslaag.py`)

Zelfde doel als het filter: z = ln t − theta_rijder − ref (prestatie t.o.v. het peil van de race), per race eerst voorspellen en dan bijwerken,
valpartijen niet in de update; parameters met maximale aannemelijkheid op 2008–2019 (een derde van de rijders), toets 2020+.
- **GAS** (score-gedreven, robuust): niveau per afstand (koppeling rho^|afstand|), toestandsonzekerheid die groeit met de tijd, en per rijder
  een eigen snelle en trage spreiding (log-schaal, score-gedreven). Geschat: nu 8,6, rho 0,78, terugtrekking spreiding 0,976 per rit.
- **Bootstrap + variantielaag**: gewogen mengsel van de eigen eerdere ritten (gewicht exp(−dagen/τ), andere afstanden lager), per rit een
  extra variantie, plus een brede component. Geschat: τ = 19 dagen, andere afstanden vrijwel gewicht 0, brede component 15%.

Eerlijke vergelijking (elke motor levert midden en schaal; daarop dezelfde gekalibreerde twee-delige t naar niveau; log loss van z, 2020+):

| motor | alle | top < 2% | top < 6% | > 18% |
|---|---|---|---|---|
| Kalman | −2,456 | **−3,275** | **−3,163** | −2,120 |
| GAS | −2,038 | −3,102 | −2,966 | −1,598 |
| GAS twee-delig | −2,082 | −3,106 | −2,974 | −1,661 |
| bootstrap + variantielaag | −2,302 | −3,129 | −3,000 | −1,982 |
| mengsel Kalman 0,85 + bootstrap 0,15 | **−2,471** | −3,272 | −3,161 | **−2,147** |

Het filter is de beste motor (het heeft leeftijdsafhankelijke procesruis, volledige koppeling tussen afstanden, een groeisnelheid per rijder
en een korte vormterm); de bootstrap voegt iets toe bij zwakkere rijders. **GAS als laag op de kansschaal** (`gaslaag.py`: per rijder een
verschuiving en log-spreiding van z, robuust bijgewerkt, uitdovend): afgesteld op 2020–2022 (a 0,02, halfwaardetijd 365 dagen; de spreiding
bijwerken helpt niet), toets 2023+: alle −2,3799 → −2,3808, elite vooraf −3,1708 → −3,1721; verschuiving Wennemars −0,18, Jílek −0,18.

## Soort wedstrijd: World Cups (`piek.internationaal`, `wc_kalibratie.py`)

Diagnose (`wc_kalibratie.py`): op internationale topwedstrijden (≥ 10 landen, ≥ 2 van USA/CAN/JPN/CHN/KOR, november–maart, geen WK/Spelen;
vooral World Cups) was iedereen sneller dan voorspeld en de spreiding te groot (2020+: gem. z −0,03 bij de top tot −0,18 bij > 10%, sd 1,05–1,09;
snel 1/20 1,25–1,80×, 1/100 1,35–2,6×), ook al in 2008–2019; op overige wedstrijden juist iets trager (+0,01 tot +0,04). Ontleding: vooral op
raceniveau (racegemiddelde WC −0,05 tegen overig +0,05; extra variantie tussen WC-races 0,30 tegen 0,21), nauwelijks binnen de race. Let op
selectie: rijders die later dat seizoen een WK of de Spelen rijden, rijden op World Cups gemiddeld sneller (kwalificatie volgt de World Cups).

Laag (na de piekmomenten): z' = (z − μ) / σ, μ en σ een gladde functie van het niveau, apart voor internationale en overige wedstrijden
(piekmomenten houden hun eigen laag), online, krimp 30 ritten.

| (2020+, log loss) | alle | toppers | elite | senior | toppers vooraf | elite vooraf |
|---|---|---|---|---|---|---|
| zonder laag soort wedstrijd | −2,3652 | −3,0647 | −3,1886 | −2,8012 | −3,0558 | −3,1792 |
| met laag soort wedstrijd | **−2,3656** | **−3,0652** | **−3,1891** | **−2,8016** | **−3,0569** | **−3,1804** |

World Cups 2020+ na de laag: top < 2% gem. z +0,008, sd 1,021, snel 1/20 · 1/100 = 1,04 · 1,06, traag 1,07 · 0,94; 3,5–6% en 6–10% goed;
> 10% nog te veel snelle uitschieters (1,43 · 1,82). Kalibratie top < 2% vooraf: 0 fouten. Nog open: op WK's en Spelen rijdt de top sinds 2020
iets trager dan de piekmoment-laag nu verwacht (gem. z +0,06, snel 1/20 0,80×): het piekeffect was vóór 2020 groter.

## Alleen de rijderschok (`rijderschok.py`)

Afspraak: bij ritten wordt nooit de gecombineerde schok (rijder + ijs) getoond, altijd alleen de rijderschok. Het ijs wordt achteraf
gecorrigeerd met het veld zelf (per race de mediaan van ln t − verwachting, ≥ 5 rijders zonder val); de rijderschok is de rest, gedeeld door
de eigen spreiding van de rijder; de kans komt uit de werkelijke verdeling daarvan bij alle toprijders 2008–2019. Lijst van de grootste
olympische verrassingen: `verrassingen_spelen_rijder.csv`.

## PR-snelheid in de piekmoment-laag (getest, niet opgenomen)
Toets: de piekmoment-laag als gladde functie van niveau x ln(1 + PR-afstand tot de snelste tijd in de data, alleen eerdere dagen;
`piek.pr_gat`, rooster `piek.GPR`), alleen WK's en Spelen, online geleerd (variant `piekpr` in `rust_staart_lab.py`, bandbreedte 0,15 / 0,25 / 0,4).
Resultaat 2020+ op WK en Spelen (top vooraf < 2%): log loss -3.2891 (gewone piek-laag) tegen -3.2895 / -3.2895 / -3.2896; snel 1/20 0.94 -> 1.00.
Gemiddelde z op piekritten (top vooraf < 6%, alle jaren) per PR-afstand na de huidige lagen: <0,5% +0.01 | 0,5-1,5% -0.06 | 1,5-3% -0.01 |
3-5% -0.02 | >5% -0.04: vlak. Het PR-effect uit `topper_geweest.py` zit al in de plafondlagen (beste eerdere kwaliteit t.o.v. de verwachting).
Daarom niet in model2 opgenomen.

## Eigen volatiliteit per rijder (`volatiliteit.py`, `volatlaag.py`)

Diagnose (`volatiliteit.py`, 2020+, top vooraf < 6%, zonder valpartijen, rijder-only residu: ijs achteraf via de veldmediaan): de voorspelde
spreiding per rijder (Kalman-ritruis per rijder) realiseert zich goed (kwintielen eigen spreiding: voorspeld 0,92 - 1,12, gerealiseerd 0,90 - 1,13),
maar wisselvalligheid is scheef: wisselvallige rijders zijn vooral vaker traag, nauwelijks vaker uitzonderlijk snel. Daardoor kregen ze
te veel snelle kans (eindkans snel 1/100: constantste kwintiel 1,08, wisselvalligste 0,71).
Laag (na de rijderlaag, voor de niet-parametrische ijking): eigen spreiding = sqrt(gewogen gemiddelde u^2) over eerdere dagen (u = rijder-only
residu / sd_rijder, afgekapt op +-4, gewicht halveert per 14 ritten, krimp 10 ritten naar 1); z' = z / s_snel (z < 0) of z / s_traag (z > 0),
s^2 = lopende kernel-semivariantie aan die kant, gladde functie van niveau x ln(eigen spreiding) (bw 0,02 en 0,1; halfwaardetijd 1095 dagen; k0 30).
De mediaan blijft gelijk. Een aparte staartlaag (massa en vorm onder 1/20 naar niveau x eigen spreiding) voegde niets toe.
Volledige run (2020+): alle -2.3656 -> -2.3661, toppers -3.0652 -> -3.0654, elite vooraf -3.1804 -> -3.1806; kalibratiefouten gelijk (top < 2%: 0).
Snel 1/100 per kwintiel eigen spreiding: 1,08 / 1,13 / 0,89 / 0,80 / 0,71 -> 1,00 / 1,07 / 0,90 / 0,85 / 0,83; snel 1/20 1,10 -> 1,06 en 0,82 -> 0,90.

## Top 5 vooraf op WK's en Spelen (`top5_piek.py`, uitvoer `top5_piek.log`)

Per race de vijf snelste verwachte tijden tegenover plaats 6-10 en 11+ (448 races, zonder valpartijen). Over alle WK's samen geen verschil
(gem. z modelkans alle jaren: top 5 -0,04, 6-10 -0,08, 11+ -0,03; 2020+: +0,07 / -0,04 / +0,06). Op de Spelen wel: top 5 gem. z -0,37 (se 0,06),
snel 1/20 1,69, traag 1/20 0,32; rest van het veld -0,09. Rijder-only (ijs achteraf): top 5 -0,19 tegen +0,11 voor de rest, bij alle vijf de
Spelen in dezelfde richting (verschil 0,08 / 0,62 / 0,17 / 0,14 / 0,47 sd in 2010 / 2014 / 2018 / 2022 / 2026).

## Spelen-functie: de favorieten pieken extra op de Olympische Spelen (`model2.py`, SPELEN)

Na de piekmoment-laag (WK's en Spelen samen) een laag alleen op de Spelen: z' = (z - mu) / sigma met mu en sigma een gladde functie van
ln(verwachte plaats in het veld) (rangorde van de verwachte tijd per race, vooraf; rooster ln 1 .. ln 40, bandbreedte 0,4) en het niveau
(bandbreedte 0,1), online uit eerdere Spelen (halfwaardetijd 10 jaar, krimp 30 ritten naar het effect van de piekmoment-laag).
Een functie van alleen het niveau werkte niet (log loss Spelen -3.0950 -> -3.0842): op de Spelen zijn ook de rijders buiten de top 10 toprijders,
het effect zit in de plaats in het veld.
Volledige run (2020+): toppers -3.0654 -> -3.0655, elite -3.1893 -> -3.1896, top vooraf < 2% -3.1806 -> -3.1808; kalibratiefouten elite 4 -> 3.
Spelen, alle jaren (gem. z modelkans; log loss): top 3 -0,43 -> -0,18 (-3.201 -> -3.242); 4-10 -0,32 -> -0,04 (-3.099 -> -3.170);
11+ -0,02 -> +0,08 (-3.214 -> -3.214). WK's ongewijzigd. De rest bij de top 3 komt vooral uit 2010 en 2014 (nog weinig eerdere Spelen om van te leren).

## Winnen underdogs op de Spelen zo vaak als voorspeld? (`spelen_winst.py`)

Simulatie per race van alleen de rijderschok (ijs binnen een race gelijk voor iedereen): ln t = verwachting + c x sd_rijder x trekking uit de
gestandaardiseerde rijderschokken van toprijders 2008-2019. 46 Olympische races (500 m met twee omlopen weggelaten), 133 WK-afstandenraces.
Met c = 1: winst waargenomen / verwacht, Spelen: winkans < 10% 10 / 11,4; 10-50% 25 / 24,9; > 50% 12 / 9,7; nr 1 vooraf 26 / 20,2; podium plaats
4-5 vooraf 18 / 24,3. WK afstanden: winkans < 5% 8 / 13,5; nr 1 vooraf 80 / 67,5. Underdogs winnen op de Spelen ongeveer zo vaak als voorspeld,
maar de favoriet wint vaker. Beste schaal (gem. ln winkans van de winnaar): c = 0,8 op beide (Spelen -1,487 / -1,467 / -1,462 / -1,476 bij
c = 1 / 0,9 / 0,8 / 0,7; WK -1,247 / -1,232 / -1,227 / -1,239); dan nr 1 26 / 23,6 en 80 / 76,7. De rijderschok op piekmomenten is ~20% smaller
dan de algemene referentie.
World Cups (`spelen_winst.py ... "World Cups"`: alle internationale wedstrijden buiten WK/Spelen, 2846 races): met c = 1 wint de nr 1 vooraf
1750 keer tegen 1623 verwacht, rijders met winkans 1-10% 339 tegen 413; beste schaal c = 0,9 (gem. ln winkans winnaar -1,179 / -1,173 / -1,180
bij c = 1 / 0,9 / 0,8), dan nr 1 1750 / 1711. In 2024-2025 winnen de favorieten nog vaker (nr 1 239 / 213, winkans < 10% 30 / 54 bij c = 1).

## Junioren: kalibratie in het ritmodel (`junior_trend.py`)

Gem. z van de modelkans (zonder valpartijen) naar leeftijd x niveau vooraf: voor 13-19 jaar overal binnen +-0,04; de beste junioren (< 3%)
op 16-18 jaar iets sneller dan voorspeld (-0,03); per rijder-seizoen (16-19 jaar, >= 8 ritten) bij de top -0,05. In de recente seizoenen
(2020+) zijn de beste junioren iets sneller dan voorspeld (-0,04 tot -0,11 in 2022). Snelle staart 1/20 en 1/100 per leeftijd 0,92-1,09.
Echte spreiding tussen rijder-seizoenen (boven de ruis) ~0,12-0,18 sd: groei binnen het seizoen wordt deels pas laat opgepikt.

## Welke junioren worden wereldkampioen? (`titels.py`)

Per junior-momentopname (eind van het seizoen, 15-19 jaar, Kalman-stand uit `talent.py`) het aantal in elk later seizoen tot en met 36 jaar van:
WK-titels bij de senioren (afstanden, allround, sprint), olympische titels (apart) en World Cup-zeges (internationale wedstrijd buiten WK/Spelen,
veld >= 10, mediaanleeftijd >= 20, >= 5 rijders met niveau vooraf < 3%; ~70 races per seizoen). Model: verwacht aantal in seizoen s =
te verdelen titels/races in s x exp(eta), eta glad (derdegraads) in de leeftijd in s, plus het junior-niveau ten opzichte van de seniorentop-10
van HETZELFDE seizoen (het wereldniveau werd 2006-2025 ~2,7% sneller; zonder die correctie krijgen de junioren van nu te hoge kansen), als
gladde functie ln(L + 0,05) en lineair, met interacties met de leeftijd in s en de leeftijd van de momentopname; verder verbetering, achterstand
op de beste van de eigen leeftijd, geslacht, Nederland. Poisson voor het aantal; dezelfde vorm op de seizoenen tot en met de eerste zege voor
de kans op minstens een. Alle kalenderseizoenen tellen mee, ook na stoppen. Spelen gezamenlijk gefit met de WK-afstandstitels plus een eigen
constante. Toekomstige kalender: Spelen als het kalenderjaar mod 4 = 2, WK afstanden anders, allround en sprint in even jaren.
Toets buiten de steekproef (kruisfit even/oneven geboortejaren, binnen de waargenomen seizoenen, leeftijd 17-19):
| doel | log-lik kans | AUC (alleen niveau) | top-200 voorspeld / echt | som voorspeld / echt |
|---|---|---|---|---|
| WK-titel | -0,01684 | 0,949 (0,952) | 34,3 / 31 | 96,5 / 106 |
| olympische titel | -0,00961 | 0,958 (0,941) | 21,0 / 17 | 51,9 / 55 |
| World Cup-zege | -0,05079 | 0,919 (0,941) | 73,4 / 53 | 341,8 / 371 |
Daarna een gladde ijking (kans = 1 - exp(-c S^d), aantal = c S^d, op dezelfde voorspellingen buiten de steekproef): WK kans c 1,03 d 1,00,
aantal c 0,79 d 0,88; Spelen kans c 0,81 d 0,94; World Cup kans c 0,95 d 0,98. Voorspelling over de hele carriere: geschiedenis buiten de
steekproef (wat het model op dat moment zou zeggen), de jeugd van nu gefit op alles. Lijst: `titelkansen_junioren_2026.csv`.
Bovenaan (kans op minstens een WK-titel): mannen Jílek (heeft al titels), Wiersma 73%, Taiga Sasaki 65%, Shomu Sasaki 53%; vrouwen Daleman 75%,
Shumekova 44%, Lee Na-Hyun 39%. Varianten (`TVAR`): niveau t.o.v. een vaste wereldtop gaf te hoge kansen bovenaan (top-200 40,6 / 30).

## Website (`site_data.py`, `../site/index.html`)

Alle rijders: per seizoen per afstand het niveau als tijd op een normale dag in Heerenveen, de achterstand op de top-10 van dat seizoen, hetzelfde
niveau in het niveau van nu, wereldrang, titels, World Cup-zeges en bij junioren de groei en de titelkansen; alle ritten met de tijd omgerekend
naar een normale dag in Heerenveen, plaats, soort wedstrijd en de prestatie van alleen de rijder (ijs achteraf).

## Snelheidsverbetering door baan en materiaal (`baan_materiaal.py`)

Het peil van een race (theta, uit het veld) vangt alles op wat het hele veld tegelijk sneller maakt (ijs, baan, weer, materiaal); het niveau
van de rijders hun eigen vooruitgang. Regressie van het peil per race (>= 8 rijders, 500-5000 m) op baan x geslacht x afstand + seizoen:
het gemeenschappelijke seizoenseffect daalt geleidelijk van ~+0,75% (2008) naar 0 (2025): materiaal en algemene ijsverbetering (niet te
scheiden van gezamenlijke vooruitgang van de rijders). Per baan bovenop dat effect: Heerenveen ~+0,3% tot 2014 en ~-0,5 tot -0,7% vanaf 2018
(renovatie Thialf), Inzell +2,4% -> ~0 (overkapping 2011), Tomaszów ~+3% -> -0,8% (overkapping 2017). De 10 snelste tijden per seizoen
werden 2008-2025 ~2,3% sneller (baan en materiaal plus de rijders). Samen voor een normale dag in Heerenveen (`era`, glad over 3 seizoenen,
t.o.v. 2025-26): 2008 +1,4%, 2014 +1,1%, 2018 +0,5%, 2021 +0,05%.
Website: per rit en per seizoen "Heerenveen nu" (ijs en materiaal van nu) en "Heerenveen toen" (van dat seizoen); het verschil is de
verbetering van baan en materiaal. De achterstand op de top-10 van het seizoen is los van baan en materiaal (alleen de rijders).

## Titelkansen vanaf 18 jaar (`titels_sen.py`)

Doorgetrokken voor elk seizoen van 18 tot en met 39 jaar (49.971 momentopnamen), voorspeld vanaf het volgende seizoen tot en met 40 jaar:
kans op minstens een (nog een) WK-titel / olympische titel / World Cup-zege en het verwachte aantal. Kenmerken: seniorenniveau (beste afstand
met >= 3 ritten) t.o.v. de seniorentop-10 van hetzelfde seizoen (glad), verandering t.o.v. vorig seizoen, wat de rijder al won
(ln(1 + titels / zeges), ook x leeftijd), leeftijd (kwadratisch), geslacht, Nederland. Met de titelhistorie divergeerde de Poisson-schatting
(som van de voorspellingen 202.330 tegen 1.045 titels); nu met stapverkleining en een aandeel in de te verdelen titels van hoogstens 1
(in de steekproef dan som 1.045 / 1.045). Toets buiten de steekproef (kruisfit, binnen de waargenomen seizoenen):
| doel, leeftijd | met zege daarna | kans som | AUC (alleen niveau) | top-100 voorspeld / echt |
|---|---|---|---|---|
| WK 18-21 | 161 | 152,9 | 0,973 (0,968) | 50,3 / 46 |
| WK 22-26 | 150 | 148,2 | 0,976 (0,971) | 63,7 / 61 |
| WK 27+ | 52 | 45,1 | 0,983 (0,981) | 31,0 / 34 |
| Spelen 18-21 | 83 | 86,3 | 0,973 (0,959) | 27,2 / 26 |
| Spelen 22-26 | 89 | 86,6 | 0,975 (0,964) | 40,6 / 35 |
| Spelen 27+ | 35 | 25,7 | 0,981 (0,977) | 16,2 / 22 |
| World Cup 18-21 | 548 | 536,3 | 0,955 (0,954) | 82,4 / 78 |
| World Cup 22-26 | 445 | 433,8 | 0,962 (0,961) | 89,3 / 84 |
| World Cup 27+ | 129 | 119,3 | 0,974 (0,967) | 56,6 / 55 |
IJking (kans c, d): WK 1,10 / 1,03; Spelen 0,94 / 0,97; World Cup 1,17 / 1,06. Lijst: `titelkansen_senioren_2026.csv`. Verwacht aantal
WK-titels vanaf 2026-27: mannen Stolz 18,2, Jílek 10,5, De Boo 2,9, Eitrem 2,6; vrouwen Kok 17,0, Leerdam 5,0, Beune 3,8, Wiklund 2,5.
Website: titelkansen bij elk seizoen vanaf 15 jaar (tot 18 jaar het juniorenmodel, vanaf 18 jaar dit model).
Website-indeling: zoekbalk met suggesties bovenaan (kiezen opent het rijderblad bovenaan), tabbladen Rijder, Ranglijsten (geslacht, afstand,
leeftijd van-tot, seizoen, land, rangschikken op achterstand op de top-10 van dat seizoen of op de Heerenveen-tijd nu of toen, een regel per
rijder), Talenten (titelkansen en verwachte aantallen, groei en niveau; leeftijd, seizoen of 'nu', land, sorteren op elke kans) en Vergelijken
(tot 8 rijders: loopbanen naast elkaar op leeftijd, tabel per leeftijd met niveau, kansen of groei). Lijstbestanden: `site_lijsten.py`.
Wedstrijden (`site_wedstrijden.py`): klik op een datum in de rittenlijst voor de hele uitslag (alle ritten op die baan, dag, afstand en
geslacht, gesorteerd op tijd) met achterstand op de winnaar, Heerenveen-tijd en de eigen prestatie (alleen de rijder). Simuleren: deelnemers
kiezen (zoeken, vergelijklijst, top 12 van nu, of "simuleer met deze rijders" vanuit een uitslag), afstand, niveau uit het laatste of het beste
seizoen, absoluut of omgerekend naar de top van nu (eerlijk over tijdperken); 10.000 races op een normale dag in Heerenveen: gedeelde
ijsschok (sd 0,4%) plus per rijder een rijderschok uit de eigen ritten van de laatste twee seizoenen (gewicht n / (n + 20)), anders uit de
referentie van alle toprijders 2020+ (robuuste sd 0,85% per rit, dikke staarten); spreiding x 0,9 op World Cups en x 0,8 op WK's en Spelen
(`spelen_winst.py`). Uitkomst: kans op winst, podium en top 5, verwachte plaats, mediaan tijd en 80%-interval, en een voorbeelduitslag.
Kalender vooruit (`titels.toekomst_totalen`): de WK's worden nog maar de helft van de jaren gehouden: WK afstanden in oneven jaren, allround en
sprint in even jaren, Spelen elke vier jaar. Daardoor minder te verwachten titels (Stolz nog 13,7 WK-titels in plaats van 18,2, Jílek 7,9 in
plaats van 10,5). Website: tabblad Kampioenen (alle rijders met een wereld- of olympische titel: titels en zeges tot dan, kansen en verwachte
aantallen voor de rest van de carriere; nu actief, alle kampioenen ooit of per seizoen); Talenten alleen zonder titel; in de simulatie een
leeftijdsgrens (niveau uit seizoenen tot en met die leeftijd, en de top 12 binnen die leeftijd). Kansen in de blokken: `site_kansen.py`.
Live wedstrijd (website): eigen wedstrijd aanmaken (naam, baan, geslacht, afstand, soort), deelnemers toevoegen of het veld van Simuleren
overnemen, tijden invullen. IJs van vandaag: gewogen gemiddelde van ln(tijd) - verwachting (niveau Heerenveen nu + peil van de baan) van de
gereden ritten (afgekapt op +-4%), met als voorkennis een normale dag (spreiding van het ijs per wedstrijd op die baan, Heerenveen 0,6%) en
een rijderspreiding van 0,9% per rit; daarna 4.000 simulaties met de gereden tijden vast. Bewaard per browser (localStorage). Teksten op de
website herschreven voor andere gebruikers; tabblad Uitleg met begrippen, beperkingen en bronnen.
Meerkampen in Simuleren (allround: vrouwen 500/3000 en 1500/5000, mannen 500/5000 en 1500/10.000; sprint: 2 x (500, 1000); kleine vierkamp:
500/1500 en 1000/3000), punten = samalog (tijd per 500 m opgeteld, afgekapt op 3 decimalen). Samenhang tussen ritten (`dagcorrelatie.py`,
rijderschok met het ijs eruit; racepeil t.o.v. het gewone peil van baan x geslacht x afstand x seizoen): eigen vorm zelfde dag andere afstand
Pearson 0,13 (Spearman 0,10), volgende dag 0,05; ijs zelfde dag andere afstand 0,38, volgende dag 0,15 (zelfde afstand twee keer op een dag
0,59, waarschijnlijk deels door de gedeelde ijscorrectie en dubbele registraties; niet gebruikt). In de simulatie een Gauss-copula: per rijder
een dagvorm (correlatie 0,12 binnen de dag, 0,05 tussen de dagen) die de getrokken eigen afwijking (uit de eigen ritten of de referentie)
kiest, en per dag een gedeeld ijs (0,38 binnen de dag, 0,15 tussen de dagen). Iedereen rijdt alle vier de afstanden (geen finaleafstand).

## Wekelijkse websiteupdate zonder herfit (`aanvul.py`, `wekelijks_site.sh`)

De workflow `scraper.yml` haalt elke maandag (september t/m maart) nieuwe wedstrijden op en bouwt daarna de website opnieuw (job `site`):
basiswerkmap (branch `werkmap`: modeluitkomsten tot toernooi 32104) + alle scraperdata -> `aanvul.py` -> `site_bouw.sh` -> `site_pages.sh`
(branch `gh-pages`, GitHub Pages). Parameters en lagen blijven vast; alleen de nieuwe informatie komt erbij:
- racepeil per nieuwe race uit het veld (gewogen mediaan van ln tijd - wereldtop - Kalmanniveau vooraf), dag voor dag, daarna tellen de
  ritten mee voor de niveaus; toets (ritten vanaf 2026-03-01 als nieuw): mediaan |verschil| met het peil van het volledige model 0,39%
  (races >= 15 rijders), 0,66% (5-14), 0,93% (1-4);
- het filter loopt alleen over de geschiedenis van de rijders in de nieuwe races, en alleen zij krijgen nieuwe seizoensstanden
  (oktober 2026: 6.237 van 392.000 momentopnamen); de aanvulling kost ~25 s, de hele job ~6 min;
- elke week opnieuw vanaf dezelfde basis (geen stapeling van benaderingen). Na een volledige herbouw van het model de branch `werkmap`
  vervangen. Titelkansen worden niet bijgewerkt (die horen bij afgesloten seizoenen).
