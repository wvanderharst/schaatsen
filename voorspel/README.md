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
