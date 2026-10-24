#!/bin/bash
# Build viz21_addon.tar.gz (the three-view video add-on for the AutoDL update21 tree) from the repository:
# viz/*.py + README_viz21.md, the CAD pitcher assets (latte_imex/pitcher_geom/pitcher_kona700.{stl,json}) as viz/assets,
# scripts/autodl_viz21.sh, and MANIFEST_viz21.sha256.  The add-on only adds files to the tree.
#   bash scripts/build_viz21_addon.sh [out_dir]      -> <out_dir>/viz21_addon.tar.gz (default: current directory)
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(cd "${1:-.}" && pwd)"
ST="$(mktemp -d)"
trap 'rm -rf "$ST"' EXIT
mkdir -p "$ST/viz/assets" "$ST/scripts"
for f in scene.py raytrace.py render3.py forward_fields.py jobs21.py test_viz.py test_scene.py README_viz21.md; do cp "$REPO/viz/$f" "$ST/viz/"; done
cp "$REPO/latte_imex/pitcher_geom/pitcher_kona700.stl" "$REPO/latte_imex/pitcher_geom/pitcher_kona700.json" "$ST/viz/assets/"
cp "$REPO/scripts/autodl_viz21.sh" "$ST/scripts/autodl_viz21.sh"
cp "$REPO/viz/README_viz21.md" "$ST/README_viz21.md"
( cd "$ST" && find . -type f ! -name MANIFEST_viz21.sha256 | sort | sed 's|^\./||' | xargs sha256sum > MANIFEST_viz21.sha256 )
( cd "$ST" && tar --owner=0 --group=0 -czf "$OUT/viz21_addon.tar.gz" README_viz21.md MANIFEST_viz21.sha256 viz scripts )
sha256sum "$OUT/viz21_addon.tar.gz"
