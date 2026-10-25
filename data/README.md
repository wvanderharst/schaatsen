# Brondata

`speedskating_historie_1_tot_32104_part1.parquet` en `_part2.parquet`: alle uitslagen van SpeedskatingResults, events 1 t/m 32104
(2006-09-23 t/m 2026-09-21). Samen 4.355.920 rijen. Kolommen: Event_ID, Toernooi, Datum, Baan, Atleet_ID, Naam, Categorie, Land,
Geslacht, Afstand, Tijd_Raw, Tijd_Sec. Dit is de echte data voor het volledige model.

Verschillen met de oude invoer (`speedskating_historie_racedatum.csv`):
- **Datum** is de startdag van het toernooi. De racedag per afstand ontbreekt. Bij de Spelen 2026 staat alles op 7 februari.
- 71.833 volledig dubbele rijen (die laat `nieuwe_data.py` vallen); daarna 4.284.087 ritten.
- 4.251.172 ritten zijn gelijk aan de oude data. Er zijn 32.915 nieuwe ritten, waarvan 152 nieuwe toernooien (vooral 2018–2026). Er zitten
  ook vreemde afstanden tussen ("1m", "2m", "7m": ploegachtervolging / massastart?) en enkele onwaarschijnlijke tijden; de
  opschoning van het model (`verdacht`) moet die vangen.
- 10.355 oude ritten staan niet in de nieuwe data.
- Toernooi is overal "SpeedskatingResults" (geen informatie).

`racedatum_map.parquet`: de racedag uit de oude data, per rit voor meerdaagse toernooien (2.236.911) en per toernooi × geslacht ×
afstand voor de rest (188.716). Gemaakt met `python voorspel/nieuwe_data.py maak_map <oude csv> data/racedatum_map.parquet`.

## Gebruik voor het volledige model

```
python voorspel/nieuwe_data.py zet_om data <werkmap>/speedskating_historie_racedatum.csv
```

Daarna de gewone keten (`elite/prep.py`, `elite/theta_merge.py`, `voorspel/prep_cat.py`, ...; zie `voorspel/README.md`). Racedagen:
4.270.583 ritten uit de oude data, 13.504 ritten in 344 toernooien hebben alleen de startdag (onder meer de nieuwe toernooien).

## Weer

`voorspel/weer/` (via `.github/workflows/luchtdruk.yml`, zie `voorspel/luchtdruk.py`): luchtdruk, temperatuur, wind en vocht per baan en
wedstrijddag, op basis van deze data (racedagen; bij toernooien zonder racedagen de startdag en de 4 dagen erna).

## Wekelijkse aanvulling

`.github/workflows/scraper.yml` draait elke maandag van september t/m maart (en met de hand) `schaatsmodel/scraper/wekelijks.py data`:
- nieuwe toernooien vanaf het hoogste bekende Event_ID (stopt na 40 nummers zonder uitslagen);
- opnieuw: toernooien die eerder nog leeg waren (tot 26 weken) en toernooien van de laatste 30 dagen (late of verbeterde uitslagen);
- uitkomst in `speedskating_nieuw.parquet` (zelfde kolommen als de historie plus `Race_Datum`, de racedag van de toernooipagina),
  `scraper_stand.json` en `scraper_log.md`. `voorspel/nieuwe_data.py` leest de aanvulling mee (nieuwere versie van een toernooi gaat voor).
Het model en de website worden daarna opnieuw gebouwd met de gewone keten (zie `voorspel/README.md`); dat gebeurt niet in de workflow
(te zwaar voor een GitHub-runner).
