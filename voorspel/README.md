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
