#!/bin/bash
# Websitedata opnieuw maken uit een werkmap (bijv. na aanvul.py), in volgorde; uitvoer voor publicatie in <sitemap>/t.
# Gebruik: bash site_bouw.sh <werkmap> <sitemap>
set -e
W=$1; S=$2; D=$(dirname "$(readlink -f "$0")")
mkdir -p "$S/t"
python3 "$D/site_data.py" "$W" "$S"
python3 "$D/site_kansen.py" "$W" "$S"
python3 "$D/site_thuisbaan.py" "$S"
python3 "$D/site_baanpeil.py" "$W" "$S"
python3 "$D/site_lijsten.py" "$S" "$S/t"
python3 "$D/site_wedstrijden.py" "$S" "$S/t"
echo "site klaar: $S/t"
