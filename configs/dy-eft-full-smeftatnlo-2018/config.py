# ruff: noqa: E501
#
# Full SMEFT EFT fit at NLO: every Wilson coefficient the SMEFTatNLO
# reweight cards probe in *all* mll slices, all 8 SMEFTatNLO mass slices
# (50 GeV - inf), on the triple-differential (mll, costhetastar, |y_ll|)
# observable. NLO counterpart of configs/dy-eft-full-smeftsim-propcorr-2018-megahisto
# (same observable and binning, same runner, same downstream chain) and the
# scaled-up version of configs/dy-eft-simple-example-2018 (read that
# config's docstring first for how the two reweight-card groups work).
#
# Morphing technique: identical to the SMEFTsim configs -- one template per
# LHEReweightingWeight point (SM, +1/-1 for each operator, +1/+1 for every
# operator pair = 1 + 2n + n(n-1)/2 points for n operators), plus one MC-stat
# covariance term Sum(events.weight**2 * rwgt_i * rwgt_j) for every unordered
# pair of templates (diagonal included), needed because all templates
# reweight the *same* underlying events (autoMCStats alone would treat their
# stat errors as independent).
#
# Why 16 operators and not 27: the 8 mll slices were generated with two
# different reweight cards.
#   Group A (27 operators, 406 points): mll200_400, 400_600, 600_800,
#     800_1000, 1500_inf.
#   Group B (16 operators, 153 points): mll50_120, 120_200, 1000_1500.
# Group B's operators are a strict subset of Group A's (and appear in the
# same relative order). The 11 operators only Group A probes (cqlm1, cql31,
# cqe1, cqlm3, cql33, cqe3, cpl1, cpl3, c3pl3, cpe, cpta) have no template at
# all for the three B slices -- which include the 50-200 GeV region carrying
# most of the yield -- and there is no honest way to build one: summing the
# templates only over the A slices would silently treat the missing slices as
# "no EFT effect". So the fit here uses the 16 operators common to both
# groups, every one of which has a template in all 8 slices. A 27-operator
# fit would have to be restricted to the five Group A slices (dropping
# 50-200 GeV and 1000-1500 GeV) or need those three slices reweighted again
# with the larger card.
#   16 operators -> 153 templates + 11,781 covariance terms = 11,934 names
#   per dataset x 8 datasets.
#
# Reweight-card index mapping: both cards share the same layout -- sm=0,
# then (wm1_<op>, w1_<op>) for each operator in card order, then one
# w11_<opi>_<opj> point per operator pair in itertools.combinations order --
# so the indices are computed below rather than pasted. Checked against every
# entry (153 per group) of eft_operator_indices.py in this directory, which
# was extracted from
# /gwpool/users/gboldrini/spritz/configs/zmumu_EFT_trees_single_triggers_EFT_startingOne_mod50-100/config.py.
# The A/B assignment of each dataset is also checked against the actual
# number of LHEReweightingWeight columns in the files (406 vs 153).
#
# Memory / njobs: NLO files hold 2500 events each (one chunk per file,
# 27,501 chunks). A job keeps the histograms of every chunk it processed
# until it finishes and serializes them all at the end, so its peak memory
# grows with chunks-per-job. Measured on real chunks (this config, groups A
# and B mixed): ~0.5GB + ~0.33GB per chunk in the job (6 chunks: 2.5GB,
# 12 chunks: 4.4GB), ~6s per chunk after the first (which pays ~10s to open
# the file). njobs = 4600 -> ~6 chunks/job, ~2.5GB peak: safely under a 4608MB
# condor request_memory. Don't lower it much without raising REQUEST_MEMORY in
# batch_config.json.

import itertools
import json

import hist
import numpy as np
from spritz.framework.framework import cmap_petroff, get_fw_path

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2018v9"
lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
plot_label = "DY EFT full fit -- SMEFTatNLO, 16 operators"
year_label = "2018"
njobs = 4600

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft_full_morphing_megahisto.py"

special_analysis_cfg = {
    "do_theory_variations": False,
}

# -----------------------------
# Operators, in the order each reweight card lists them.
# -----------------------------
OPERATORS_A = [
    "cqlm1", "cqlm2", "cql31", "cql32", "cqe1", "cqe2", "cqlm3", "cql33", "cqe3",
    "cll1221", "cpdc", "cpwb", "cpl1", "cpl2", "cpl3", "c3pl1", "c3pl2", "c3pl3",
    "cpe", "cpmu", "cpta", "cpqmi", "cpq3i", "cpq3", "cpqm", "cpu", "cpd",
]
OPERATORS_B = [
    "cqlm2", "cql32", "cqe2", "cll1221", "cpdc", "cpwb", "cpl2", "c3pl1", "c3pl2",
    "cpmu", "cpqmi", "cpq3i", "cpq3", "cpqm", "cpu", "cpd",
]
assert len(OPERATORS_A) == 27 and len(OPERATORS_B) == 16
# Same relative order in both cards, so a w11_<a>_<b> name means the same
# operator pair (and exists) in both.
assert [op for op in OPERATORS_A if op in OPERATORS_B] == OPERATORS_B

OPERATORS = OPERATORS_B


def _card_index(ops):
    """LHEReweightingWeight column of every point of a reweight card that
    probes `ops` (in card order)."""
    index = {"sm": 0}
    for k, op in enumerate(ops):
        index[f"wm1_{op}"] = 1 + 2 * k
        index[f"w1_{op}"] = 2 + 2 * k
    base = 1 + 2 * len(ops)
    for k, (a, b) in enumerate(itertools.combinations(ops, 2)):
        index[f"w11_{a}_{b}"] = base + k
    return index


_index_a = _card_index(OPERATORS_A)
_index_b = _card_index(OPERATORS_B)
assert len(_index_a) == 406 and len(_index_b) == 153

_eft_points = (
    ["sm"]
    + [f"w1_{op}" for op in OPERATORS]
    + [f"wm1_{op}" for op in OPERATORS]
    + [f"w11_{a}_{b}" for a, b in itertools.combinations(OPERATORS, 2)]
)
assert len(_eft_points) == 153
assert set(_eft_points) == set(_index_b) and set(_eft_points) <= set(_index_a)

_group_a_points = {name: _index_a[name] for name in _eft_points}
_group_b_points = {name: _index_b[name] for name in _eft_points}


def covariance_name(name_i, name_j):
    return f"cov_{name_i}_{name_j}"


# -----------------------------
# The structured eft_reweighting spec (see runner_3DY_eft_full_morphing_megahisto.py):
# "points" is {name: column index}, and "covariance_pairs" the (name_i, name_j)
# tuples whose covariance term is wanted. Both groups share the same names
# and pairs; only the column indices differ.
# -----------------------------
_covariance_pairs = list(itertools.combinations_with_replacement(_eft_points, 2))
assert len(_covariance_pairs) == 153 * 154 // 2  # 11,781

# "n_weights" is the exact width of each card's LHEReweightingWeight branch;
# the runner fails any chunk whose file has a different width instead of
# mapping its columns onto the wrong points. Not hypothetical: a survey of all
# 27,501 input files (scripts/survey_files.py) found 92 of them (0.34%, 230,000
# events) whose branch has fewer columns than their card -- anywhere from 0 to
# 396, no fixed truncation point -- in every dataset but mll50_120, 60 of them
# in runs of neighbouring files (i.e. neighbouring production jobs) and 32
# isolated. Listed in excluded_files.json (spritz-chunks' generic, per-config
# skip list -- see load_excluded_paths() in chunks.py), so spritz-chunks never
# builds a chunk for them in the first place, rather than relying on
# n_weights to fail them one condor job at a time. n_weights stays on as a
# backstop against any future file with the same problem. Dropping them
# leaves the normalization intact, since sumw only counts chunks
# that were processed.
_eft_reweighting_a = {
    "weight_branch": "LHEReweightingWeight",
    "points": _group_a_points,
    "covariance_pairs": _covariance_pairs,
    "n_weights": len(_index_a),
}
_eft_reweighting_b = {
    "weight_branch": "LHEReweightingWeight",
    "points": _group_b_points,
    "covariance_pairs": _covariance_pairs,
    "n_weights": len(_index_b),
}

# -----------------------------
# Datasets -- all 8 SMEFTatNLO mll slices, 2018.
# -----------------------------
_group_a_slices = ["200_400", "400_600", "600_800", "800_1000", "1500_inf"]
_group_b_slices = ["50_120", "120_200", "1000_1500"]
_mll_slices = ["50_120", "120_200", "200_400", "400_600", "600_800", "800_1000", "1000_1500", "1500_inf"]
assert sorted(_group_a_slices + _group_b_slices) == sorted(_mll_slices)

datasets = {}
for b in _mll_slices:
    name = f"DYMuMu_NLO_EFT_SMEFTatNLO_mll{b}_Photos_startingOne"
    datasets[name] = {
        "files": name,
        "task_weight": 8,
        "eft_reweighting": _eft_reweighting_a if b in _group_a_slices else _eft_reweighting_b,
    }

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

# -----------------------------
# Samples -- one shape per EFT point summed over the 8 slices (post_process.py
# renormalizes each slice by its own xsec/sumw/lumi before summing), plus the
# covariance terms, flagged so they reach histos.root and spritz-cov-matrix
# but never become datacard process rows. Same flags as the SMEFTsim configs.
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
assert len(samples) == 153 + len(_covariance_pairs)

colors = {name: cmap_petroff[i % len(cmap_petroff)] for i, name in enumerate(_eft_points)}

# -----------------------------
# Regions -- same as the SMEFTsim configs.
# -----------------------------
preselections = lambda events: (events.mll > 50)  # noqa: E731

regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"],
        "mask": 0,
    },
}

# -----------------------------
# Variables -- same triple_diff observable and binning as
# dy-eft-full-smeftsim-propcorr-2018-megahisto, so the NLO and LO fits are
# directly comparable. The last mll bin (1000-3000) collects both the
# 1000_1500 (Group B) and 1500_inf (Group A) slices; add an edge at 1500 to
# split them.
# -----------------------------
def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr / abs(nr)  # noqa: E731
    return (
        2 * get_sign((l1 + l2).pz) / (l1 + l2).mass
        * get_sign(l1.pdgId)
        * (l2.pz * l1.energy - l1.pz * l2.energy)
        / np.sqrt(((l1 + l2).mass) ** 2 + ((l1 + l2).pt) ** 2)
    )


variables = {
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    },
    "costhetastar": {
        "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
    },
    "rapll_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity),
    },
    "triple_diff": {
        "axis": [
            hist.axis.Variable([50, 120, 200, 400, 600, 800, 1000, 3000], name="mll"),
            hist.axis.Variable([-1.0, 0.0, 1.0], name="costhetastar"),
            hist.axis.Variable([0.0, 1.2, 2.4], name="rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV", "", ""],
        "xlog": True,
    },
}

cards_regions = ["inc_mm"]
cards_variables = ["triple_diff"]
covariance_file = "covariance.root"

nuisances = {}

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
