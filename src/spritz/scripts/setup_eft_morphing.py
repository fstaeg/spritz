"""
spritz-setup-eft-morphing: set up a CMSSW + CombinedLimit + AnalyticAnomalousCoupling
+ eft-smp-combination/tools area for EFT template-morphing fits, in whatever
directory it's run from (a config directory, typically).

This is a spritz-wide generalization of configs/dy-eft-simple-example-2018's
setup_combine.sh: same repos/branches/build steps (CMSSW_16_0_0 +
HiggsAnalysis/CombinedLimit, a fork adding the correlated-MC-stat autoMCCorr
directive that spritz-cards' `covariance_file` support relies on +
HiggsAnalysis/AnalyticAnomalousCoupling, the quadratic SMEFT template-morphing
physics model + the eft-smp-combination/tools scan/GoF/impacts drivers), just
runnable as `spritz-setup-eft-morphing` from any config directory instead of a
copy of the script living next to each one. All repos/branches are overridable
via flags for whoever needs a different pin.

Usage: run from inside the config directory you want this set up in
    spritz-setup-eft-morphing
"""
import argparse
import os
import stat
import subprocess
import sys

DEFAULT_CMSSW_REL = "CMSSW_16_0_0"
DEFAULT_SCRAM_ARCH = "el9_amd64_gcc13"

DEFAULT_COMBINE_REPO = "git@github.com:GiacomoBoldrini/HiggsAnalysis-CombinedLimit.git"
DEFAULT_COMBINE_BRANCH = "correlated_autoMCstat"
# Known-good commit as of writing this script, for reference/reproducibility:
#   bbb8270f529d20a38658b166a8464f1c1f4dfc4d

DEFAULT_AAC_REPO = "git@github.com:amassiro/AnalyticAnomalousCoupling.git"
DEFAULT_AAC_BRANCH = "template_morphing"
# Known-good commit as of writing this script, for reference/reproducibility:
#   a949d8053529dbeef83070b69230b2295c48e6f1

DEFAULT_TOOLS_REPO = "ssh://git@gitlab.cern.ch:7999/eft-smp-combination/tools.git"
DEFAULT_TOOLS_BRANCH = "morphing_model"


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Set up CMSSW + this analysis' CombinedLimit fork + "
            "AnalyticAnomalousCoupling + eft-smp-combination/tools in the "
            "current directory, for EFT template-morphing fits."
        )
    )
    parser.add_argument("--cmssw-release", default=DEFAULT_CMSSW_REL)
    parser.add_argument("--scram-arch", default=DEFAULT_SCRAM_ARCH)
    parser.add_argument("--combine-repo", default=DEFAULT_COMBINE_REPO)
    parser.add_argument("--combine-branch", default=DEFAULT_COMBINE_BRANCH)
    parser.add_argument("--aac-repo", default=DEFAULT_AAC_REPO)
    parser.add_argument("--aac-branch", default=DEFAULT_AAC_BRANCH)
    parser.add_argument("--tools-repo", default=DEFAULT_TOOLS_REPO)
    parser.add_argument("--tools-branch", default=DEFAULT_TOOLS_BRANCH)
    parser.add_argument(
        "-j", "--jobs", type=int, default=8, help="scram build parallelism (default 8)"
    )
    return parser.parse_args()


def build_script(args):
    return f"""#!/bin/bash
set -euo pipefail

CMSSW_REL={args.cmssw_release}
export SCRAM_ARCH={args.scram_arch}

COMBINE_REPO={args.combine_repo}
COMBINE_BRANCH={args.combine_branch}

AAC_REPO={args.aac_repo}
AAC_BRANCH={args.aac_branch}

TOOLS_REPO={args.tools_repo}
TOOLS_BRANCH={args.tools_branch}

HERE="$(pwd)"

if [ -d "$CMSSW_REL" ]; then
  echo "[spritz-setup-eft-morphing] $CMSSW_REL already exists here, skipping scram project."
else
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  scramv1 project CMSSW "$CMSSW_REL"
fi

cd "$CMSSW_REL/src"
eval "$(scramv1 runtime -sh)"

if [ -d HiggsAnalysis/CombinedLimit ]; then
  echo "[spritz-setup-eft-morphing] HiggsAnalysis/CombinedLimit already checked out, skipping clone."
else
  git clone -b "$COMBINE_BRANCH" "$COMBINE_REPO" HiggsAnalysis/CombinedLimit
fi

if [ -d HiggsAnalysis/AnalyticAnomalousCoupling ]; then
  echo "[spritz-setup-eft-morphing] HiggsAnalysis/AnalyticAnomalousCoupling already checked out, skipping clone."
else
  git clone -b "$AAC_BRANCH" "$AAC_REPO" HiggsAnalysis/AnalyticAnomalousCoupling
fi

scram b -j {args.jobs}

cd "$HERE"
if [ -d tools ]; then
  echo "[spritz-setup-eft-morphing] tools/ already checked out, skipping clone."
else
  git clone -b "$TOOLS_BRANCH" "$TOOLS_REPO" tools
fi

cat << EOF

[spritz-setup-eft-morphing] Done.

To use this setup in a new shell:
  cd $HERE/$CMSSW_REL/src
  eval \\`scramv1 runtime -sh\\`
  cd $HERE

Datacard/scan tooling lives in $HERE/tools/combine_helpers and
$HERE/tools/plotters. If you see inconsistent results between single-core
and split/multiprocess scans, check whether --X-rtd MINIMIZER_no_analytic=1
is set in that script's secret_options -- see the tools repo's own history
for why that flag exists.
EOF
"""


def main():
    args = get_args()
    here = os.path.abspath(".")

    script_text = build_script(args)
    script_path = os.path.join(here, ".spritz_setup_eft_morphing.sh")
    with open(script_path, "w") as f:
        f.write(script_text)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

    try:
        subprocess.run([script_path], cwd=here, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    finally:
        os.remove(script_path)


if __name__ == "__main__":
    main()
