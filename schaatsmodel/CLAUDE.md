# Schaatsmodel (langebaan) — werkafspraken

Lees eerst README.md (opbouw, pijplijn, stand van het onderzoek, open punten).

## Afspraken met de gebruiker (Wouter)
- Communicatie in het Nederlands. Doel van het model: **eerlijke log loss minimaliseren** (`da.logloss_eerlijk`: vorm per afstandsgroep geschat op ≤2019, getoetst op 2020+). Kalibratie (ook van de staarten / seizoensbeste) telt zwaar.
- **Heerenveen** is de standaardbaan voor normale tijd, piektijd en simulaties. Milano (ITA) gebruikt de Heerenveen-prior (`baan_alias`).
- **Valpartijen** tellen nergens mee (z > 6 aan de trage kant = val).
- Een rijder piekt 1–2 keer per jaar; snelle tijden mogen de rating meer bijwerken dan trage.
- Verkeerd gelabelde data verwijderen (`uitgesloten_races.csv`, wordt in `laad_dataset` toegepast).
- **Volledige database alleen gebruiken als de gebruiker dat zegt**; nieuwe ideeën eerst op kleine data (`speedskating_test.csv`). Tests snel houden.
- De grote parquet-historiebestanden (`skater_toestanden_deel*.parquet`) **niet** opnieuw bouwen.
- Lange jobs actief volgen (pollen), niet ongezien op de achtergrond laten lopen.

## Praktisch
- Geheugen: een volledige fit + alle lagen past nét in 8 GB. Daarom is `bouw_model.sh` opgeknipt in losse processen; laad niet twee volle modellen tegelijk. Opslaan via `da.bewaar_veilig` (model + `.extra`), laden via `da.laad_volledig`.
- Params zijn een frozen dataclass: wijzigen met `dataclasses.replace(m.params, ...)`.
- `da.data_csv()` kiest `speedskating_historie_racedatum.csv` (racedatum per afstand) als die naast de code staat, anders de oude CSV.
- Na elke modelwijziging: eerlijke log loss vóór/na (zelfde pijplijn!), uitsplitsing naar niveau (toppers ≥90, elite ≥97), en `pytest test_model.py -q`.
  `eval_ll.py` doet dit voor `model_full_tuned.pkl`; vergelijk varianten gepaard per rij (`per_rij=True`), de totale log loss alleen is te ruizig (zie README).
