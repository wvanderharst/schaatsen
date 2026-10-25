#!/bin/bash
# Wekelijkse websiteupdate zonder herfit: basiswerkmap (branch werkmap) + alle scraperdata (data/speedskating_nieuw.parquet)
# -> aanvul.py -> site_bouw.sh -> site_pages.sh (gh-pages). Elke week opnieuw vanaf dezelfde basis, dus nooit stapeling van fouten.
# Gebruik: bash wekelijks_site.sh <basismap> <datamap> <tijdelijke map>
set -e
B=$1; DATA=$2; T=$3; D=$(dirname "$(readlink -f "$0")")
mkdir -p "$T"
if [ ! -f "$B/voorspel2_uitkomst.parquet" ]; then        # in delen opgeslagen (GitHub: bestanden < 100 MB)
  python3 -c "import pyarrow as pa, pyarrow.parquet as pq, glob, sys; pq.write_table(pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(sys.argv[1] + '/voorspel2_uitkomst_deel*.parquet'))]), sys.argv[1] + '/voorspel2_uitkomst.parquet')" "$B"
fi
python3 "$D/aanvul.py" "$B" "$DATA" "$T/werk" "$D/kalman_par.json"
[ -d "$T/werk" ] || ln -s "$(readlink -f "$B")" "$T/werk"   # geen nieuwe toernooien: de basis zelf
bash "$D/site_bouw.sh" "$T/werk" "$T/site"
bash "$D/site_pages.sh" "$T/site"
