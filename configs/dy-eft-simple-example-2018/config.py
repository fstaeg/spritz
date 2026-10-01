# ruff: noqa: E501
#
# Simple DY -> mumu SMEFT EFT reweighting example (2018 only).
#
# Uses the 8 mll-binned DYMuMu_NLO_EFT_SMEFTatNLO_*_startingOne 2018 samples,
# reweighted directly into histograms (no per-event trees / bespoke scripts).
#
# The EFT points and the MC-stat covariance terms between them are handed to
# the runner through the structured per-dataset `eft_reweighting` dict
# (weight branch + {point name: column index} + covariance pairs), consumed
# by runner_3DY_eft_full_morphing_megahisto.py: every point/covariance weight
# is computed as one vectorized numpy operation on the LHEReweightingWeight
# matrix and filled into a handful of batch histograms, instead of the
# per-name eval() + fill() loop of the old `subsamples`-based
# runner_3DY_eft_reweight.py. Same physics selection, bit-for-bit the same
# histograms (checked on a Group A and a Group B chunk), much cheaper once
# the number of points grows -- see configs/dy-eft-full-smeftatnlo-2018 for
# the same setup scaled up to every operator the NLO productions probe.
# chunks.py copies every `datasets[dataset]` key straight into that dataset's
# chunk kwargs (see create_chunks()), so the reweight-point -> column index
# mapping can differ freely per dataset -- which it must, since the
# productions used two different reweight cards.
#
# cql32 and cpl2 were chosen because both mll-bin groups' reweight cards
# actually probe them (Group B's card only probes 16 of Group A's 27
# operators, so a template usable across all 8 datasets has to be built from
# operators present in both -- eft_operator_indices.py has the columns of all
# 16 operators common to the two cards, for both groups).
#
# NOTE on the reweight-card index mapping (checked against
# eft_operator_indices.py, which was extracted from
# /gwpool/users/gboldrini/spritz/configs/zmumu_EFT_trees_single_triggers_EFT_startingOne_mod50-100/config.py):
#   - mll200_400, 400_600, 600_800, 800_1000, 1500_inf ("Group A"):
#     sm=0, cql32_m1=7, cql32=8, cpl2_m1=27, cpl2=28, cql32_cpl2=139.
#   - mll50_120, 120_200, 1000_1500 ("Group B"):
#     sm=0, cql32_m1=3, cql32=4, cpl2_m1=13, cpl2=14, cql32_cpl2=52.
# Both cards follow the same layout: sm, then (wm1_<op>, w1_<op>) for every
# operator in card order, then one w11_<opi>_<opj> point per operator pair in
# itertools.combinations order. That is how the indices above (and the full
# NLO config's) can be derived instead of copied by hand.
# See scripts/dump_reweight.py / create_reweight_variables.py to (re-)derive
# them from an actual reweight_card.dat if you have one.

import json
from itertools import combinations_with_replacement

import hist
import numpy as np
from spritz.framework.framework import cmap_petroff, get_fw_path

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2018v9"
lumi = lumis[year]["tot"] / 1000  # All of 2018
plot_label = "DY EFT (simple example)"
year_label = "2018"
njobs = 300

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft_full_morphing_megahisto.py"

special_analysis_cfg = {
    "do_theory_variations": False,
}

_eft_points = ["sm", "w1_cql32", "wm1_cql32", "w1_cpl2", "wm1_cpl2", "w11_cql32_cpl2"]


# Base (point name -> LHEReweightingWeight column) per reweight-card group --
# see the module docstring above for where these indices come from.
_group_a_points = {
    "sm": 0,
    "wm1_cql32": 7,
    "w1_cql32": 8,
    "wm1_cpl2": 27,
    "w1_cpl2": 28,
    "w11_cql32_cpl2": 139,
}

_group_b_points = {
    "sm": 0,
    "wm1_cql32": 3,
    "w1_cql32": 4,
    "wm1_cpl2": 13,
    "w1_cpl2": 14,
    "w11_cql32_cpl2": 52,
}

assert set(_group_a_points) == set(_eft_points) == set(_group_b_points)


def covariance_name(name_i, name_j):
    return f"cov_{name_i}_{name_j}"


# One covariance term per unordered pair of the 6 EFT points, including the
# diagonal -- 21 terms, each Sum(events.weight**2 * rwgt_i * rwgt_j) per bin.
# This is exactly the per-bin MC-stat covariance between templates i and j:
# since both are built from the *same* underlying events, their statistical
# fluctuations are correlated, and these terms are what let a downstream
# tool assemble the full covariance matrix instead of treating each
# template's stat error as independent (which autoMCStats effectively
# assumes). The diagonal term (i == j) is redundant with template i's own
# histogram variance (hist.storage.Weight() already accumulates
# Sum(weight_i**2)), but is kept so the downstream matrix-building script can
# treat all 21 entries uniformly, and as a free cross-check (cov_i_i must
# equal histo_i's own .variances()). The runner only computes them for the
# nominal variation.
_covariance_pairs = list(combinations_with_replacement(_eft_points, 2))
assert len(_covariance_pairs) == 21


def build_eft_reweighting(points, n_weights):
    """Per-dataset `eft_reweighting` spec for one reweight-card group.
    Iterating _eft_points (not `points`' own dict order) keeps the point
    order, hence covariance_name(i, j), identical across groups and matching
    the `samples` dict construction below. `n_weights` is the width of the
    group's LHEReweightingWeight branch: the runner fails any chunk whose
    file has a different width, since the column indices above only mean
    something for that card layout."""
    return {
        "weight_branch": "LHEReweightingWeight",
        "points": {name: points[name] for name in _eft_points},
        "covariance_pairs": _covariance_pairs,
        "n_weights": n_weights,
    }


_group_a_eft_reweighting = build_eft_reweighting(_group_a_points, 406)
_group_b_eft_reweighting = build_eft_reweighting(_group_b_points, 153)

# -----------------------------
# Datasets: all 8 mll-binned 2018 startingOne SMEFTatNLO samples
# -----------------------------
datasets = {
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll50_120_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_b_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll120_200_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_b_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll200_400_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_a_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll400_600_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_a_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll600_800_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_a_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll800_1000_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_a_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll1000_1500_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_b_eft_reweighting,
    },
    "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne": {
        "files": "DYMuMu_NLO_EFT_SMEFTatNLO_mll1500_inf_Photos_startingOne",
        "task_weight": 8,
        "eft_reweighting": _group_a_eft_reweighting,
    },
}

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

# -----------------------------
# Samples: group all 8 datasets' same-named EFT point into one shape per
# point. post_process.py already sums a "samples": [...] list weighted by
# each entry's own xsec/sumw/lumi -- this is the native "sum across datasets
# into one shape" mechanism, no custom script needed. (The runner stores the
# EFT points of a dataset as one combined entry; post_process.py unpacks it
# back into f"{dataset}_{point}" before looking these names up.)
# "sm" is left as the (sole) background; the 5 reweighted points are flagged
# as signal templates used to build the EFT linear/quadratic morphing.
#
# The 21 "cov_*" entries carry the MC-stat covariance terms (see
# _covariance_pairs above). They're marked `is_variance` so post_process.py
# normalizes them quadratically (correct for a Sum(weight_i*weight_j)
# quantity) instead of linearly, and `exclude_from_datacard` so
# make_cards.py writes them into histos.root but never turns them into a
# datacard process row.
#
# `covariance_of` records which two real templates (name_i, name_j) this term
# is the covariance between -- this is what lets the generic
# spritz-cov-matrix tool (spritz.scripts.build_covariance_matrix) discover the
# matrix structure purely from `samples` dict flags, without needing to know
# this config's own naming convention (e.g. cov_<i>_<j> is ambiguous to
# reverse-parse generically, since operator names can themselves contain
# underscores -- "w1_cql32" is one example right here).
# -----------------------------
samples = {
    point: {
        "samples": [f"{dataset}_{point}" for dataset in datasets],
        **({"is_signal": True} if point != "sm" else {}),
    }
    for point in _eft_points
}

samples.update({
    covariance_name(name_i, name_j): {
        "samples": [f"{dataset}_{covariance_name(name_i, name_j)}" for dataset in datasets],
        "is_variance": True,
        "exclude_from_datacard": True,
        "covariance_of": (name_i, name_j),
    }
    for name_i, name_j in _covariance_pairs
})

colors = {name: cmap_petroff[i] for i, name in enumerate(_eft_points)}

# -----------------------------
# Regions
# -----------------------------
preselections = lambda events: (events.mll > 50)  # noqa: E731

regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"],
        "mask": 0,
    },
}

# -----------------------------
# Variables
# -----------------------------
# A flat/regular mll binning is too fine in the high-mass tail relative to
# the steeply falling DY cross section: bins above ~1 TeV end up with only a
# handful of raw MC events each, and the resulting per-bin fluctuations show
# up directly as a non-smooth likelihood scan. Instead use the same
# progressively-widening variable binning validated in the reference
# analysis (initial_mll_binning()/min_bin_width in
# /gwpool/users/gboldrini/spritz/configs/zmumu_EFT_trees_single_triggers_EFT_startingOne_mod50-100/scripts/eft_distribution_dump_wMatrix.py),
# ported verbatim.
def _initial_mll_binning(min_bin_width):
    binning = []
    for (start, stop), width in min_bin_width.items():
        edges = np.arange(start, stop, width)
        if len(binning) > 0 and edges[0] == binning[-1]:
            edges = edges[1:]  # avoid duplicates
        binning += edges.tolist()
    last_stop = list(min_bin_width.keys())[-1][1]
    if binning[-1] < last_stop:
        binning.append(last_stop)
    return np.array(binning)


_mll_bin_widths = {
    (50, 100): 2,
    (100, 200): 3,
    (200, 400): 5,
    (400, 600): 10,
    (600, 800): 18,
    (800, 1000): 27,
    (1000, 1500): 40,
    (1500, 3000): 65,
}
_mll_edges = _initial_mll_binning(_mll_bin_widths)

variables = {
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable(_mll_edges, name="mll"),
    },
}

# spritz-cards: which regions/variables to turn into datacards (overrides
# make_cards.py's hardcoded 3DY-analysis defaults).
cards_regions = ["inc_mm"]
cards_variables = ["mll"]

# Link to the MC-stat covariance matrix built by spritz-cov-matrix (see the
# `is_variance`/`covariance_of` entries in `samples` above). When set,
# make_cards.py copies the matching {region}/{variable}/covariance_matrix
# histogram into each datacard's own shapes.root, so the datacard directory
# is self-contained. Build it first with:
#   spritz-cov-matrix . histos.root -o covariance.root
covariance_file = "covariance.root"

# -----------------------------
# Nuisances / check_weights
# -----------------------------
nuisances = {}

# Shared flat lnN uncertainty (e.g. luminosity) on every process in the
# datacard. "samples" here just needs a value per row make_cards.py will
# actually emit; the 21 "cov_*" entries are excluded from the datacard
# already (see samples[...]["exclude_from_datacard"] above), so giving them
# a value too is harmless -- make_cards.py skips them before ever looking.
nuisances["lumi"] = {
    "name": "lumi",
    "type": "lnN",
    "samples": dict((skey, "1.02") for skey in samples),
}

nuisances["stat"] = {
    "type": "auto",
    "maxPoiss": "10",
    "includeSignal": "0",
    "samples": {},
}

check_weights = {}
