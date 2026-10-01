#!/bin/bash
# Set up CMSSW_16_0_0 + this analysis' CombinedLimit fork (correlated MC-stat
# covariance via autoMCCorr) + AnalyticAnomalousCoupling (quadratic SMEFT
# template morphing physics model) + the eft-smp-combination tools repo
# (scan/GoF/impacts drivers), all pinned to the branches actually used to
# produce the results in this config.
#
# Usage: run from inside configs/dy-eft-simple-example-2018/
#   ./setup_combine.sh
set -euo pipefail

CMSSW_REL=CMSSW_16_0_0
export SCRAM_ARCH=el9_amd64_gcc13

COMBINE_REPO=git@github.com:GiacomoBoldrini/HiggsAnalysis-CombinedLimit.git
COMBINE_BRANCH=correlated_autoMCstat
# Known-good commit as of writing this script, for reference/reproducibility:
#   bbb8270f529d20a38658b166a8464f1c1f4dfc4d

AAC_REPO=git@github.com:amassiro/AnalyticAnomalousCoupling.git
AAC_BRANCH=template_morphing
# Known-good commit as of writing this script, for reference/reproducibility:
#   a949d8053529dbeef83070b69230b2295c48e6f1

TOOLS_REPO=ssh://git@gitlab.cern.ch:7999/eft-smp-combination/tools.git
TOOLS_BRANCH=morphing_model

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -d "$CMSSW_REL" ]; then
  echo "[setup_combine] $CMSSW_REL already exists here, skipping scram project."
else
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  scramv1 project CMSSW "$CMSSW_REL"
fi

cd "$CMSSW_REL/src"
eval "$(scramv1 runtime -sh)"

if [ -d HiggsAnalysis/CombinedLimit ]; then
  echo "[setup_combine] HiggsAnalysis/CombinedLimit already checked out, skipping clone."
else
  git clone -b "$COMBINE_BRANCH" "$COMBINE_REPO" HiggsAnalysis/CombinedLimit
fi

if [ -d HiggsAnalysis/AnalyticAnomalousCoupling ]; then
  echo "[setup_combine] HiggsAnalysis/AnalyticAnomalousCoupling already checked out, skipping clone."
else
  git clone -b "$AAC_BRANCH" "$AAC_REPO" HiggsAnalysis/AnalyticAnomalousCoupling
fi

scram b -j 8

cd "$HERE"
if [ -d tools ]; then
  echo "[setup_combine] tools/ already checked out, skipping clone."
else
  git clone -b "$TOOLS_BRANCH" "$TOOLS_REPO" tools
fi

cat << EOF

[setup_combine] Done.

To use this setup in a new shell:
  cd $HERE/$CMSSW_REL/src
  eval \`scramv1 runtime -sh\`
  cd $HERE

Datacard/scan tooling lives in $HERE/tools/combine_helpers and
$HERE/tools/plotters. If you see inconsistent results between single-core
and split/multiprocess scans, check whether --X-rtd MINIMIZER_no_analytic=1
is set in that script's secret_options -- see the tools repo's own history
for why that flag exists.
EOF
