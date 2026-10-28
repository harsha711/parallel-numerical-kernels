#!/usr/bin/env bash
# Rasterize analysis/plots/*.svg to PNG with headless Chrome. ImageMagick's
# default SVG delegate drops rotated text (the y-axis labels); Chrome does not.
set -euo pipefail
cd "$(dirname "$0")/plots"
for svg in *.svg; do
    read -r w h < <(sed -n 's/.*<svg[^>]* width="\([0-9]*\)" height="\([0-9]*\)".*/\1 \2/p' "$svg" | head -n 1)
    google-chrome --headless --no-sandbox --disable-gpu --hide-scrollbars \
        --force-device-scale-factor=2 --window-size="$w,$h" \
        --default-background-color=ffffffff \
        --screenshot="$PWD/${svg%.svg}.png" "file://$PWD/$svg" >/dev/null 2>&1
done
ls -1 *.png
