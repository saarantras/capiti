#!/bin/bash
# Render the OXA-48 overlay in 5 colour pairings.
set -eo pipefail
module load ChimeraX/1.10.1-1-gfbf-2024a-CUDA-12.8.0

D=/nfs/roberts/scratch/pi_skr2/mcn26/capiti/data/oracle_folds/oxa48_poster

# open order: #1 experimental ref, #2 ESMFold WT, #3-17 the 15 design folds
FILES="$D/T3_WT_ref_chainA.pdb $D/folds/T3_WT.pdb"
for f in "$D"/folds/T3_mpnn_*.pdb; do FILES="$FILES $f"; done

SCRIPT=$(mktemp --suffix=.cxc)
echo "open $FILES" > "$SCRIPT"
sed "s#PALDIR#$D#g" scripts/render_oxa48_palettes.cxc >> "$SCRIPT"

chimerax --offscreen --nogui --exit "$SCRIPT" 2>&1
rm -f "$SCRIPT"
echo "wrote 5 palette PNGs to $D"
