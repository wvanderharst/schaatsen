#!/bin/bash
# Website naar GitHub Pages: de pagina (site/index.html, met een gewoon HTML-geraamte) en de data uit <sitemap>/t op de branch gh-pages.
# gh-pages heeft geen geschiedenis (elke keer één nieuwe commit, force-push), zodat de wekelijkse data de repository niet laat groeien.
# Gebruik: bash site_pages.sh <sitemap> [remote, standaard origin]
set -e
S=$1; REMOTE=${2:-origin}; D=$(dirname "$(readlink -f "$0")"); REPO=$(cd "$D/.." && pwd)
URL=$(git -C "$REPO" remote get-url "$REMOTE")
if [ -n "$GITHUB_TOKEN" ] && [ -n "$GITHUB_REPOSITORY" ]; then        # in GitHub Actions
  URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
fi
T=$(mktemp -d)
cp -r "$S/t/index.txt" "$S/t/schokref.json" "$S/t/l" "$S/t/r" "$S/t/w" "$T/"
{ printf '<!doctype html>\n<html lang="nl">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
  cat "$REPO/site/index.html"; printf '\n</html>\n'; } > "$T/index.html"
touch "$T/.nojekyll"
cd "$T"
git init -q -b gh-pages
git add -A
git -c user.name="$(git -C "$REPO" config user.name || echo schaatsen)" -c user.email="$(git -C "$REPO" config user.email || echo noreply@github.com)" \
    commit -qm "Website $(date -u +%F)"
git push -q -f "$URL" gh-pages
cd / && rm -rf "$T"
echo "gepubliceerd op gh-pages"
