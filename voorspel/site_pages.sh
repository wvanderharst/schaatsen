#!/bin/bash
# Website voor GitHub Pages samenstellen: de pagina (site/index.html, met een gewoon HTML-geraamte) en de data uit <sitemap>/t in <uitmap>.
# De workflow scraper.yml (job site) publiceert <uitmap> rechtstreeks op GitHub Pages (actions/deploy-pages): geen aparte branch, en de
# wekelijkse data komt niet in de geschiedenis van de repository.
# Gebruik: bash site_pages.sh <sitemap> <uitmap>
set -e
S=$1; U=$2; D=$(dirname "$(readlink -f "$0")"); REPO=$(cd "$D/.." && pwd)
mkdir -p "$U"
cp -r "$S/t/index.txt" "$S/t/schokref.json" "$S/t/l" "$S/t/r" "$S/t/w" "$U/"
{ printf '<!doctype html>\n<html lang="nl">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
  cat "$REPO/site/index.html"; printf '\n</html>\n'; } > "$U/index.html"
touch "$U/.nojekyll"
echo "website samengesteld in $U ($(du -sh "$U" | cut -f1))"
