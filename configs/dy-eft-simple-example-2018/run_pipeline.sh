#!/bin/bash
# Full pipeline: spritz processing -> datacards -> workspace -> scans -> plots.
#
# Run from inside configs/dy-eft-simple-example-2018/ (or just execute this
# script directly, it cds there itself). Assumes:
#   - the `spritz` conda env exists and has spritz installed editable
#     (pip install -e . --no-deps from the spritz-fabian repo root)
#   - setup_combine.sh has already been run once (CMSSW_16_0_0/ and tools/
#     exist alongside this script)
#
# The spritz-batch step submits real HTCondor jobs; this script pauses and
# waits for you to confirm they've finished before merging, since job
# runtime isn't something to guess at in a static script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPRITZ_REPO="$(cd "$HERE/../.." && pwd)"

# ---- spritz processing ----------------------------------------------------
source /gwpool/users/gboldrini/mambaforge/etc/profile.d/conda.sh
conda activate spritz

cd "$SPRITZ_REPO"
source start.sh
cd "$HERE"

spritz-fileset
spritz-chunks
spritz-batch

echo ""
echo "[run_pipeline] HTCondor jobs submitted. Check status with 'condor_q',"
echo "[run_pipeline] and 'spritz-checkerrors' / 'spritz-resubmit' if any fail."
read -r -p "[run_pipeline] Press enter once all jobs have finished... "

spritz-merge
spritz-postproc
spritz-cov-matrix . histos.root -o covariance.root
spritz-cards

# ---- combine environment ---------------------------------------------------
cd "$HERE/CMSSW_16_0_0/src"
eval "$(scramv1 runtime -sh)"
cd "$HERE"

cd "$HERE/tools"
source env.sh
cd "$HERE"

# ---- datacard -> workspace -> scans -> plots -------------------------------
cd "$HERE/datacards/inc_mm/mll"

createJson.py --datacard datacard.txt --binname=wm1_
createCombineJson.py --datacard datacard.txt --binname=wm1_
createWS.py 2

runScans.py 2 initial
runScans.py 2 scan --doSplitPoints=10 --npoints=5000
runPlots.py 2

echo ""
echo "[run_pipeline] Done. Plots are in $HERE/datacards/inc_mm/mll (scan*.pdf/png)."
