# schaatsen
### PR-snelheid in de piekmoment-laag (getest, niet opgenomen)
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
