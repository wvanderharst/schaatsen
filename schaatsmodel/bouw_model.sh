#!/bin/bash
# Volledige herbouw van het model (ca. 1 uur, ~6 GB geheugen). Stappen apart omdat alles in één proces niet in 8 GB past.
cd "$(dirname "$0")"
set -e
python -u refit_full.py params_getuned.json model_full_tuned.pkl > ch_refit.log 2>&1   # filter + kalibratie
python -u update_full_piek.py > ch_upd.log 2>&1                                         # trend, loc_schaal, piekkansen
python -u update_gas.py > ch_gas.log 2>&1                                               # GAS-laag + vol_ref
python -u update_vorm.py > ch_vorm.log 2>&1                                             # vormtoestand + piekdiepte
python -u z_voor.py > ch_zvoor.log 2>&1                                                 # normaal-equivalente z (vooraf)
python -u z_dag.py > ch_zdag.log 2>&1                                                   # dag-z (t.o.v. het veld)
python -u bouw_historie.py > ch_hist.log 2>&1                                           # toestand per datum (hist_nieuw.npz)
python -u eval_ll.py > ch_eval.log 2>&1                                                  # eerlijke log loss vooraf/dag, per niveau
echo KLAAR > ch_klaar.log
