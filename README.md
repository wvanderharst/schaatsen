# schaatsen
### PR-snelheid in de piekmoment-laag (getest, niet opgenomen)
Toets: de piekmoment-laag als gladde functie van niveau x ln(1 + PR-afstand tot de snelste tijd in de data, alleen eerdere dagen;
`piek.pr_gat`, rooster `piek.GPR`), alleen WK's en Spelen, online geleerd (variant `piekpr` in `rust_staart_lab.py`, bandbreedte 0,15 / 0,25 / 0,4).
Resultaat 2020+ op WK en Spelen (top vooraf < 2%): log loss -3.2891 (gewone piek-laag) tegen -3.2895 / -3.2895 / -3.2896; snel 1/20 0.94 -> 1.00.
Gemiddelde z op piekritten (top vooraf < 6%, alle jaren) per PR-afstand na de huidige lagen: <0,5% +0.01 | 0,5-1,5% -0.06 | 1,5-3% -0.01 |
3-5% -0.02 | >5% -0.04: vlak. Het PR-effect uit `topper_geweest.py` zit al in de plafondlagen (beste eerdere kwaliteit t.o.v. de verwachting).
Daarom niet in model2 opgenomen.
