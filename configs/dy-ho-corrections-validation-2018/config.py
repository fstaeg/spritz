# Validation of the higher-order corrections applied to MiNNLO DY
# (src/spritz/modules/ho_reweight.py), same corrections as
# configs/dy-eft-2018-all-operators:
#   NNLO_N3LL_QCD: DYTurbo NNLO+N3LL / MiNNLO, vs gen Z (qT, y, mll)
#   NLO_EW:        POWHEG NLO-EW / LO-EW, vs gen Z (mll, |y|, cos theta*_CS)
#
# Both corrections are applied in the nominal weight; the runner's
# "_before" variations give the other three predictions from the same events
# (so their ratios have no statistical fluctuations):
#   bare MiNNLO               ho_before             (histo_<sample>_hoBefore)
#   MiNNLO + NLO EW           NNLO_N3LL_QCD_before  (histo_<sample>_NNLO_N3LL_QCDBefore)
#   MiNNLO + NNLO+N3LL QCD    NLO_EW_before         (histo_<sample>_NLO_EWBefore)
#   MiNNLO + both             nominal               (histo_<sample>)
# Only the HO-correction systematics are kept. Compare with
#   spritz-chunks; spritz-batch; spritz-merge; spritz-postproc; python plot_ho_validation.py

import json

import hist
import numpy as np
from spritz.framework.framework import cmap_petroff, get_fw_path
from spritz.modules.ho_reweight import cos_theta_cs, gen_boson, gen_leptons
import awkward as ak

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2018v9"
lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
plot_label = "DY higher-order corrections"
year_label = "2018"
njobs = 600

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft.py"

special_analysis_cfg = {
    "do_variations": True,
    # keep only the HO-correction variations (all other SFs stay in the nominal weight)
    "variations": ["NLO_EW_*", "NNLO_N3LL_QCD_*", "ho_before"],
    "do_theory_variations": False,
    "do_rochester_stat_variations": False,
    "do_jet_variations": False,
    "reweight_fakes": False,
}

# Higher-order corrections, as functions of the gen-level Z (GenPart last
# copy); "observable" lists the map axes in the order of the TH3 axes
nnlo_n3ll_qcd = {
    "file": "/eos/user/g/gboldrin/www/DYTurbo/run_3d_prediction_NNLO_N3LL_mll_30_3500_noCuts/kfactor_N3LO_GenPart.root",
    "object": "kfactor_N3LO_GenPart",
    "name": "NNLO_N3LL_QCD",
    "observable": "pt:rapidity:mll",  # qT, signed y_ll, m_ll
}
nlo_ew = {
    "file": "/eos/user/g/gboldrin/www/POWHEG_Z_EW/results_s23154p306000/dy_ew_kfactor_3d.root",
    "object": "ratio_correlated",  # NLO-EW / LO-EW, same-event reweighting
    "name": "NLO_EW",
    "observable": "mll:absrapidity:costhetastar",  # Collins-Soper, pre-FSR leptons
}

dy_bins = [
    "10to50", "50to100", "100to200", "200to400", "400to500", "500to700",
    "700to800", "800to1000", "1000to1500", "1500to2000", "2000toInf",
]

datasets = {
    **{
        f"DYmm_MiNNLO_M-{b}": {
            "files": f"DYJetsToMuMu_M-{b}" if b != "50to100" else "DYJetsToMuMu",
            "task_weight": 8,
            "max_weight": 1e9,
            "ho_corrections": [nnlo_n3ll_qcd, nlo_ew],
        } for b in dy_bins
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9,
        # same corrections as DYmm here, to validate both on taus
        # (dy-eft-2018-all-operators applies only NLO_EW to DYtt)
        "ho_corrections": [nnlo_n3ll_qcd, nlo_ew],
        # ~500M events of which few pass the mumu selection
        "max_chunks": 1000,
    },
}

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

samples = {
    "DYmm": {"samples": [f"DYmm_MiNNLO_M-{b}" for b in dy_bins]},
    "DYtt": {"samples": ["DYtt"]},
}

colors = {
    "DYmm": cmap_petroff[9],
    "DYtt": cmap_petroff[8],
}

# Regions
preselections = lambda events: events.mll > 50  # noqa: E731

regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events.mm,
        "mask": 0,
    },
    "bveto_mm": {
        "func": lambda events: preselections(events) & events.mm & events.bveto,
        "mask": 0,
    },
}


# Variables
def cos_theta_star(l1, l2):
    """Reco cos(theta*) as defined in the analysis configs: this is
    -cos(theta*_CS) of the negative lepton, i.e. the angle of the l+"""
    get_sign = lambda nr: nr / abs(nr)  # noqa: E731
    return (
        2 * get_sign((l1 + l2).pz) / (l1 + l2).mass
        * get_sign(l1.pdgId)
        * (l2.pz * l1.energy - l1.pz * l2.energy)
        / np.sqrt(((l1 + l2).mass) ** 2 + ((l1 + l2).pt) ** 2)
    )


def gen_Z(events):
    return gen_boson(events)[0]


def gen_cos_theta_cs(events):
    _, z_idx = gen_boson(events)
    return cos_theta_cs(*gen_leptons(events, z_idx))


mll_edges = [50, 60, 70, 80, 85, 88, 90, 92, 94, 96, 100, 110, 120, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000]
pt_edges = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600]

variables = {
    # reco
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable(mll_edges, name="mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True,
    },
    "ptll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "axis": hist.axis.Variable(pt_edges, name="ptll"),
        "label": "$p_{T}^{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True,
    },
    "rapll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity,
        "axis": hist.axis.Regular(24, -2.4, 2.4, name="rapll"),
        "label": "$y_{\\ell\\ell}$",
    },
    "costhetastar": {
        "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
        "axis": hist.axis.Regular(20, -1, 1, name="costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$ (analysis, $\\ell^{+}$)",
    },
    "rapll_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity),
    },
    # same binning as the fit observable of dy-eft-2018-all-operators
    "triple_diff": {
        "axis": [
            hist.axis.Variable([50, 70, 80, 90, 100, 110, 120, 150, 200, 300, 500], name="mll"),
            hist.axis.Variable([-1.0, -0.5, 0.0, 0.5, 1.0], name="costhetastar"),
            hist.axis.Variable([0.0, 0.48, 0.96, 1.44, 2.4], name="rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV", "", ""],
        "xlog": True,
    },
    # gen-level Z (GenPart last copy), the inputs of the corrections
    "gen_mll": {
        "func": lambda events: ak.fill_none(gen_Z(events).mass, -1.0),
        "axis": hist.axis.Variable([30, 40] + mll_edges, name="gen_mll"),
        "label": "gen $m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True,
    },
    "gen_ptll": {
        "func": lambda events: ak.fill_none(gen_Z(events).pt, -1.0),
        "axis": hist.axis.Variable(pt_edges, name="gen_ptll"),
        "label": "gen $p_{T}^{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True,
    },
    "gen_rapll": {
        "func": lambda events: ak.fill_none(gen_Z(events).rapidity, -99.0),
        "axis": hist.axis.Regular(30, -3, 3, name="gen_rapll"),
        "label": "gen $y_{\\ell\\ell}$",
    },
    "gen_costhetastar": {
        "func": lambda events: ak.fill_none(gen_cos_theta_cs(events), -99.0),
        "axis": hist.axis.Regular(20, -1, 1, name="gen_costhetastar"),
        "label": "gen $cos\\,\\theta^{\\ast}_{CS}$ ($\\ell^{-}$)",
    },
}

dy_samples = list(samples)

# Only the HO-correction systematics
nuisances = {
    "NLO EW correction": {
        "name": "NLO_EW",
        "type": "shape",
        "samples": dy_samples,
        "kind": "weight",
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "type": "shape",
        "samples": dy_samples,
        "kind": "weight",
    },
}

# "<name>_before" variations, saved as histo_<sample>_<name>Before
corrections = {
    "NLO EW correction": {
        "name": "NLO_EW",
        "samples": dy_samples,
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "samples": dy_samples,
    },
    "All HO corrections": {
        "name": "ho",
        "samples": dy_samples,
        "related_nuisances": ["NLO EW correction", "NNLO+N3LL QCD correction"],
    },
}
