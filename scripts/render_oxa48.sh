#!/bin/bash
# Superpose the OXA-48 ESMFold folds and render the poster overlay.
set -eo pipefail
module load ChimeraX/1.10.1-1-gfbf-2024a-CUDA-12.8.0

D=/nfs/roberts/scratch/pi_skr2/mcn26/capiti/data/oracle_folds/oxa48_poster
PNG=$D/oxa48_design_overlay.png

# open order must match render_oxa48_overlay.cxc: #1 ref, #2 WT, #3-17 designs
FILES="$D/T3_WT_ref_chainA.pdb $D/folds/T3_WT.pdb"
for f in "$D"/folds/T3_mpnn_*.pdb; do FILES="$FILES $f"; done

# build a self-contained .cxc: open line + the styling/render script
SCRIPT=$(mktemp --suffix=.cxc)
echo "open $FILES" > "$SCRIPT"
sed -e "s#SAVE_PATH_y90#${PNG%.png}_y90.png#" \
    -e "s#SAVE_PATH_top#${PNG%.png}_top.png#" \
    -e "s#SAVE_PATH#$PNG#" \
    scripts/render_oxa48_overlay.cxc >> "$SCRIPT"

chimerax --offscreen --nogui --exit "$SCRIPT" 2>&1
rm -f "$SCRIPT"
echo "wrote $PNG"
