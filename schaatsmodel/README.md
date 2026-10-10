# Schaatsmodel — rating, piektijden en race-simulatie (langebaan)

## Starten
1. Zet de data naast de code: `speedskating_historie_racedatum.csv` (met `Race_Datum` per afstand; ook de oude `speedskating_historie_clean.csv` werkt).
   Uit de drie delen: `(zcat deel1.csv.gz; zcat deel2.csv.gz | tail -n +2; zcat deel3.csv.gz | tail -n +2) > speedskating_historie_racedatum.csv` (4.265.873 rijen, 2006 – 21-09-2026).
2. `pip install numpy pandas scipy numba pyarrow pytest requests beautifulsoup4`
3. Model bouwen (≈1 uur, ~6 GB): `./bouw_model.sh` → `model_full_tuned.pkl` (+ `.extra`), `normaal_z*.pkl`, `hist_nieuw.npz`; als laatste stap `eval_ll.py` → `ch_eval.log` (eerlijke log loss vooraf en dag, totaal/toppers/elite).
   De gebouwde modelbestanden zitten **niet** in deze zip (≈2 GB); ze zijn met dit script opnieuw te maken.
4. Webpagina: `python export_web.py` → `web_model.json`; daarna `python bouw_pagina.py`. `web/langebaan.html` is de laatst gepubliceerde versie.
   Let op: `VANDAAG` staat vast in `export_web.py` (nu 2026-10-10).
5. Tests: `pytest test_model.py -q` (klein, op `speedskating_test.csv`: dezelfde rijders als voorheen, nu uit de racedatum-CSV); `--volledig` voor de volledige database (alleen op verzoek).

## Opbouw van het model (`do_analysis.py`)
- **Toestand per rijder × afstand**: x = ln(T − T_lim), EKF met GARCH op de waarneming-ruis (s), procesruis per groep, correlatie tussen afstanden.
  Mengsels in de update: slechte dag (dip), **niveausprong** aan de snelle kant (sprongkans 2%, variantie 25·S) met **eigen sprongkans per rijder** (`sprong_m=50`, lang geheugen), kortetermijn-vorm als OU-proces.
- **Dag-effect** θ per event × geslacht × afstand, met baan-prior.
- **Lagen alleen op de voorspelling** (geschat in `kalibreer`, volgorde): trend/verrassingen (o.a. `cx`: vorige afstand zelfde toernooi), `loc_schaal` (per geslacht × afstand × niveau), piekkans per rijder (3 gebieden), GAS (score-gedreven, rijder kort + rijder × afstand lang, scheve score), `vol_asym` (snelle kant schaalt met relatieve volatiliteit^0,8), **vermoeidheid** (`verm`, vóór de vorm: eerder gereden km dezelfde dag en de dag ervoor, per afstandsgroep, alleen in de breedte), **vormtoestand op dag-z** (kort 7 d, uitschieters 60 d, staartkalibratie γ·√N), **ijs per niveau** (alleen dag-relatief: (ijs−HV)·k(niveau) + hoogland·h(niveau)), **piekdiepte per rijder** (τ, 4 jaar geheugen, m=20, κ=1,5).
- Normaal-equivalente z: `normaal_z` (vooraf) en `normaal_z_dag` (t.o.v. het veld) — de pagina toont de dag-z.
- `voorspel` / `Historie.voorspel` geven o.a. `normale_tijd`, `piek_tijd`, `normale_basis` (vaste rating zonder tijdelijke vorm), `snel_fac`, `piek_diepte`.

## Belangrijkste resultaten (eerlijke log loss, 2020+)
| stap | effect |
|---|---|
| loc_schaal | −0,015 (top 2%: −0,042) |
| piekkans per rijder | −0,005 |
| GAS-laag | −0,014 |
| vol_asym | −0,003 |
| piekdiepte | −0,0004 vooraf, −0,0006 dag (elite dag −0,004) |
| eigen sprongkans (m=50) | −0,0015 (elite −0,0023) |
| getest en verworpen | adaptieve procesruis per rijder (+0,0003…+0,0014), sprongen aan de trage kant, langzamere vorm-uitdoving |

Seizoensbeste onder de 5%-lijn (hoort 5%): elite 2010–19 van 15% naar 9%; vanaf 2020 nog 20% → zie open punten.

## Herbouw op racedatum + vermoeidheid (10 oktober 2026)
Volledige database (4,27 mln rijen). Vergelijking **zelfde CSV met en zonder `Race_Datum`** (zonder = elk toernooi één dag, zoals de oude CSV), zelfde pijplijn (`bouw_model.sh`), eerlijke log loss 2020+ (logs in `onderzoek/logs/`):

| | vooraf alle | toppers | elite | dag alle | toppers | elite |
|---|---|---|---|---|---|---|
| toernooidatum (oud) | −2,08564 | −2,85753 | −2,98961 | −2,18697 | −3,01079 | −3,14708 |
| racedatum (A) | −2,11526 | −2,87345 | −3,00694 | −2,22379 | −3,03978 | −3,18174 |
| **effect racedatum** | **−0,0296** | −0,0159 | −0,0173 | **−0,0368** | −0,0290 | −0,0347 |
| A + vermoeidheid + korte vorm herschat | −2,11631 | −2,87661 | −3,01240 | −2,22414 | −3,04058 | −3,18182 |
| **effect stap 2** (gepaard, se) | **−0,0011** (0,0001) | −0,0032 | −0,0055 | **−0,0003** (0,00003) | −0,0008 | −0,0002 |

Fitjaren (≤2019) gaan in stap 2 ook vooruit (vooraf −2,27457 → −2,27533, dag −2,41077 → −2,41091).

- **Racedatum** is verreweg de grootste stap tot nu toe: dag-effect per echte dag, juiste volgorde binnen meerdaagse toernooien. Bijeffect: `cx` (vorige afstand zelfde event) keek in de oude data binnen een toernooi op volgorde van afstand, dus soms naar een race van een *latere* dag (lek); met de splitsing per dag is dat weg.
- **Vermoeidheid** (`verm=True`, `schat_verm`, `verm_verschuiving`): verschuiving in sd = coef · [log(1+km eerder dezelfde dag), log(1+km de dag ervoor)] per afstandsgroep (sprint/midden/lang), kleinste kwadraten op ≤2019, toegepast vóór de vormlaag (dus ook de dag-z voor de vorm is gecorrigeerd). Volledige coëfficiënten (dezelfde dag): sprint 0,13, midden 0,13, lang 0,29 sd per log(1+km); de dag ervoor ≈ 0.
  Alleen in de breedte (gewicht 1 − w(niveau), w = topgewicht van de vorm): met een eigen topterm (`verm_top=True`) werden toppers/elite vooraf ná 2019 slechter (+0,0004 / +0,0012); aan de top krijgt een sprinter na een eerdere race dezelfde dag eerder een *snellere* tijd (topcoëfficiënt sprint −0,48). Dag-relatief (`verm_dag=True`, binnen de race gecentreerd): klein getoetst geen winst, elite iets slechter → uit.
- **Korte vorm herschat** (`vorm_par[4:6]` = aK0, aK1: −0,02175, 0,11491 → 0,00617, −0,06219): na de splitsing per dag ziet de trendlaag (d1_k, x5o) de vorige toernooidagen al, en de oude korte vorm telde dat dubbel. Herschat met kleinste kwadraten (`onderzoek/vorm_kort_ls.py`, fout vóór de vorm op Zk en Zk·w, ≤2019), gemiddelde van de vooraf- en dag-schatting; alleen deze combinatie is óók op de fitjaren beter in beide richtingen. Schalen van aK met de hand (×0,5…0) helpt de top vooraf maar maakt de dag voor het hele veld slechter.
- **Vormplateau** (`vorm_plateau`, dag-z jonger dan P dagen telt volledig): getoetst met P = 2, 3, 5 op de kleine data, geen winst t.o.v. P = 0 bij dezelfde schaal → uit (code blijft, standaard 0).
- Tests (klein, racedatum-testdata): 33 goed, 1 overgeslagen, 2 falen: de bekende historie-spreiding (0,000146 > 0,0001) en **nieuw** de baantest voor Erfurt (model 1,56%, zelfde-rijder 0,88 ± 0,18): alleen op de kleine steekproef, op beide volledige modellen slaagt hij (Erfurt 1,68 vs 1,13). De Leerdam-test slaagt nu.
- Dubbele rijen: 2.193 rijen (0,05%) komen precies dubbel voor in de CSV (vooral hele events dubbel gescrapet); `laad_dataset` haalt ze niet weg (zoals voorheen).
- Let op bij het vergelijken van log loss: `logloss_eerlijk` fit per afstandsgroep één Student-t op alle rijen; een verandering in de breedte verschuift die vorm en daarmee ook de score van de top, en de Nelder-Mead-fit valt soms in een ander optimum (één variant gaf +0,007 op dag tussen twee buren van −0,0005). Kleine verschillen dus altijd gepaard en op meerdere varianten bekijken.

## Onderzoek vorm en vermoeidheid (oktober 2026, `onderzoek/vorm_vermoeidheid*.py`)
Met de nieuwe racedatums; verrassing t.o.v. de verwachting vóór het toernooi én t.o.v. het veld in dezelfde race; zelfde dag: kortste afstand eerst.

**Vorm (helling van verrassing race 2 op race 1, zelfde toernooi):** zelfde dag 0,49 · 1 dag 0,46 · 2 dagen 0,41 · 3–4 dagen 0,37 · 5–9 dagen 0,40 (toppers/elite gelijk). Binnen meerkampen dag 1 → dag 2: sprint 0,67, klein allround 0,56, groot allround 0,59, junior allround 0,68.
→ De verrassing houdt binnen een toernooi grotendeels stand en neemt in een week maar langzaam af (≈0,49 → 0,40). Ondersteunt "vorm blijft ~5 dagen". Deels is dit blijvende ratingfout; het tijdelijke deel is ≈0,1.

**Vermoeidheid t.o.v. het veld (toppers, verrassing in sd, + = trager; basis zonder eerdere race ≈ +0,13):** 1000 m na een 1500 m dezelfde dag +0,64; 3000 m na een 1500 m +0,40; 5000 m na een kortere race +0,37…+0,43. De dag erna: klein (+0,1 op de 500 m na >5 km).

**Vermoeidheid in meerkampen (hele veld; dag-effect t.o.v. dezelfde afstand in losse races, % van de tijd):** vooral de lange slotafstand is trager: groot allround 10.000 m +0,36%, klein allround 5000 m dag 2 +0,25%, junior allround 3000 m dag 2 +0,17%. Sprint dag 2 ≈ dag 1.

## Open punten / volgende stappen
1. ~~Herbouwen op racedatum~~ en ~~vorm/vermoeidheid~~: gedaan (zie boven). Opvallend: met de nieuwe vermoeidheid levert de hele vormlaag vooraf in totaal niets meer op (`update_vorm.py --vergelijk`: met verm zonder vorm −2,11645, met vorm −2,11631); op de dag en voor de top nog wel. Nog te doen: de overige vaste parameters die op de oude, per-toernooi data getuned zijn opnieuw schatten: vooral `vorm_par` met een echte NLL-fit (zoals `vorm_lab7`, i.p.v. kleinste kwadraten), `ijs_dag_par`, `gas_par`, `tau_dag`.
2. Vermoeidheid aan de top: waarom een eerdere race dezelfde dag toppers op de sprint sneller maakt (tweede 500 m? baanindeling?) en of een term per afstandscombinatie (bv. 1000 m na 1500 m) wel standhoudt.
3. Staart sinds 2019 dikker voor toppers (seizoensbeste 2020+ te vaak extreem): vorm van de dag-verdeling op een rollend venster schatten.
4. Piektijd herdefiniëren als verwachte seizoensbeste.
5. Twee tests falen nipt op de kleine data: historie-spreiding (0,000146 > 0,0001, sinds de sprongkans per rijder) en de baantest voor Erfurt (alleen op de kleine racedatum-steekproef).
6. `web/langebaan.html` is 8 MB, waarvan 5,1 MB de trajecten (`tr`, ~330 punten per rijder: elke racedag + de dag erna + elke 30 dagen) en 2,2 MB alle uitslagen: uitdunnen of apart laden. De pagina is nog niet opnieuw gemaakt met het nieuwe model.

## Mappen
- `scraper/`: `scraper_schaatsen.py` (racedatum per afstand, slaat ploegen/massastart over, herkansingen), `ssr_racedatums.py` + `backfill.py` (bestaande CSV bijwerken).
- `onderzoek/`: labscripts van de recente stappen (nieuw: `verm_lab.py` klein, `verm_vorm_vol.py` / `vorm_kort_ls.py` / `eind_eval.py` volledig, logs in `onderzoek/logs/`) (verwijzen soms naar tussenbestanden uit de cloud-sessie, bv. `gas_d.pkl`, `zbase.pkl`; die moet je eerst opnieuw maken).
- `data_hulp/`: uitgesloten races, vermoedelijke datafouten, eventnamen.
