# Basiswerkmap voor de wekelijkse websiteupdate

Modeluitkomsten tot en met toernooi 32104 (stand 2026-10-21), gebruikt door `voorspel/wekelijks_site.sh` (workflow `scraper.yml`, job `site`).
De wekelijkse scraperdata komt hier elke week opnieuw bovenop (`voorspel/aanvul.py`), zonder het model opnieuw te fitten.
`voorspel2_uitkomst` staat in twee delen (GitHub: bestanden < 100 MB) en alleen met de kolommen die de site en de aanvulling gebruiken;
`speedskating_historie_racedatum.csv` bevat alleen 2024-07 en later (context voor de opschoning van nieuwe ritten).
Na een volledige herbouw van het model: deze branch vervangen door de nieuwe werkmap.
