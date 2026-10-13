import json
import hist
import numpy as np
from itertools import combinations_with_replacement
from spritz.framework.framework import cmap_petroff, get_fw_path, get_rw_idx, get_eft_points
from spritz.lookup_tools.eft_operator_indices import eft_operators, get_group

# dy-eft-split-2017: the SMEFT samples of this model for 2017, as the 2018
# dy-eft-split-test/smeftsim_1j with the 2017 data and samples (see
# configs/dy-2017) and all the uncertainties on: on top of 2018, Rochester
# stat. replicas, JES/JER, QCD scale and parton shower.
# the SMEFTsim1j SMEFT samples only. Data, backgrounds and
# MiNNLO DY are in ../bkg_data: process both, combine them with
# spritz-merge-configs and run spritz-postproc on the merged config.

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2017v9"
lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
year_label = "2017"
plot_label = "2017"
njobs = 500  # one dataset per job, eft_chunks_per_job chunks each

# Each job sums its EFT chunks (all of one dataset) into a single result
eft_chunks_per_job = 50

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft.py"

special_analysis_cfg = {
    "do_variations": True,
    "do_theory_variations": True, # 116 variations
    # Only the theory weight variations are computed for the SMEFT samples.
    # spritz-postproc builds every varied EFT template from MiNNLO (bkg_data)
    # through renorm_samples: EFT_point_var = MiNNLO_var * EFT_point_var / EFT_sm_var.
    # For the experimental nuisances (lepton SFs, PU, prefiring, b-tag, PU ID,
    # JES/JER, Rochester, fakes) the EFT samples have no variation, so every
    # point gets MiNNLO's relative variation. PDF, QCD scale and parton shower
    # are computed here, as they change the EFT/SM ratio itself.
    "do_rochester_stat_variations": False, # 100 variations
    "do_jet_variations": False, # 24 variations
    "variations": ["nom", "PDFWeight_*", "QCDScale_*", "PSWeight_*"],  # fnmatch patterns of the variations to keep
    "invert_one_isolation_loose": False,
    "invert_one_isolation_control": False,
    "reweight_fakes": True,
}

# SMEFTsim1j: LO 0+1 jet merged (406-wide reweight card, 27 operators)
model = "SMEFTsim1j"
file_key = "DYMuMu_LO_EFT_SMEFTsim_propcorr_1j_mll%s_Photos_startingOne"
eft_bins = ["50_120","120_200","200_400","400_600"]
operators = eft_operators["SMEFTsim"]

eft_points = get_eft_points(operators)
cov_pairs = list(combinations_with_replacement(eft_points, 2))
assert len(eft_points) == 1 + len(operators) * (len(operators) + 3) // 2

datasets = {
    f"DYmm_mll{b}_{model}": {
        "files": file_key % b,
        "task_weight": 8,
        "read_form": "mc",
        "eft_reweighting": {
            "weight_branch": "LHEReweightingWeight",
            "points": {p: get_rw_idx(file_key % b, p) for p in eft_points},
            "n_weights": 406,  # reweight card width, guards against a mismatched file
            # covariance terms are booked only for cards_regions x cards_variables
            # (the only ones spritz-cov-matrix reads); override with
            # the top-level covariance_regions / covariance_variables
            "covariance_pairs": cov_pairs,
        }
    } for b in eft_bins
}

# sample names carry the model, so that several models can be merged; the
# datacard process names (sm, w1_<op>, ...) do not
point_samples = [f"DYmm_{model}_{point}" for point in eft_points]

samples = {
    **{
        f"DYmm_{model}_{point}": {
            "samples": [f"{dataset}_{point}" for dataset in datasets],
            "name": point,  # datacard process name
            "is_smeft": True,
            "is_signal": point != "sm",
            "stack": point == "sm",  # spritz-plot: the DY in the stack, instead of DYmm_MiNNLO
            # no histogram errors on the nominal templates: their MC stat is the autoMCCorr covariance matrix
            "noStat": True,
        } for point in eft_points
    },
    **{
        f"DYmm_{model}_cov_{a}_{b}": {
            "samples": [f"{dataset}_cov_{a}_{b}" for dataset in datasets],
            "covariance_of": (f"DYmm_{model}_{a}", f"DYmm_{model}_{b}"),
            "is_variance": True,
            "exclude_from_datacard": True,
        } for a, b in cov_pairs
    },
}

colors = {f"DYmm_{model}_sm": cmap_petroff[9]}

# Renormalize the templates from "reference" (this model's SM) to "target"
# (MiNNLO, from bkg_data): spritz-postproc does it on the merged config
renorm_samples = {
    "target": "DYmm_MiNNLO",
    "reference": f"DYmm_{model}_sm",
    "samples": point_samples + [f"DYmm_{model}_cov_{a}_{b}" for a, b in cov_pairs],
}

# Nuisances and corrections: the definitions of bkg_data, for the EFT
# templates (once merged, each applies to the samples of both configs)
all_samples = [s for s in samples if not samples[s].get("is_variance")]
mc_samples = [s for s in all_samples if not samples[s].get("is_data")]

# Single top datasets with an unusable LHEPdfWeight, see the PDF nuisances (not
# in this config, kept so that the definitions stay the same as in bkg_data)
st_datasets_without_pdf = ["ST_s-channel", "ST_tW_top_noHad", "ST_tW_antitop_noHad"]

# JES uncertainty sources of Full2017v9 (data/Full2017v9/cfg.json, "jes"):
# the runner varies the jets by each of them (do_jet_variations)
jes_sources = [
    "Absolute", "Absolute_2017", "BBEC1", "BBEC1_2017", "EC2", "EC2_2017",
    "FlavorQCD", "HF", "HF_2017", "RelativeBal", "RelativeSample_2017",
]

# QCD scale: LHEScaleWeight index of the 6 (muR, muF) points. The SMEFT
# samples have 8 weights (no nominal: [0] muF=0.5 muR=0.5, [1] 1 0.5, [2] 2 0.5,
# [3] 0.5 1, [4] 2 1, [5] 0.5 2, [6] 1 2, [7] 2 2). bkg_data maps its samples in
# the same nuisance: spritz-merge-configs keeps the indices of all of them
qcd_scale_points = [(0.5, 0.5), (0.5, 1), (1, 0.5), (1, 2), (2, 1), (2, 2)]
qcd_scale_indices = {"8 weights": [0, 1, 3, 4, 6, 7]}
qcd_scale_layout = {s: "8 weights" for s in point_samples}
# samples with QCD scale and parton shower (PSWeight, 4 weights) variations
theory_samples = point_samples

nuisances = {
    "lumi": {
        "name": "lumi",
        "type": "lnN",
        "samples": {s: str(lumi_unc) for s in mc_samples}
    },
    ## Use the following if you want to apply the automatic combine MC stat nuisances
    "stat": {
        "type": "auto",
        "maxPoiss": "10",
        "includeSignal": "0",
        "samples": {}
    },
    "Pile-up corr.": {
        "name": "PU",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "L1 pre-firing corr.": {
        "name": "prefireWeight",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    #############
    # Leptons
    #############
    "Trigger SF": {
        "name": "mu_trig",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Reconstruction SF": {
        "name": "mu_reco",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon ID SF": {
        "name": "mu_id",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Isolation SF": {
        "name": "mu_iso",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Rochester corr. (stat)": {
        "name": "rochester_stat",
        "type": "shape",
        "kinematic": True,
        "kind": "stdev",
        "samples": all_samples,
        "variations": [
            {"label": f"Rochester stat. repl. {i}", "tag": f"rochester_stat{i}"} for i in range(100)
        ]
    },
    "Rochester corr. (syst)": {
        "name": "rochester_syst",
        "type": "shape",
        "kinematic": True,
        "kind": "square",
        "samples": all_samples,
        "variations": [
            {"label": "Rochester corr. set2", "tag": "rochester_set2"},
            {"label": "Rochester corr. set3", "tag": "rochester_set3"},
            {"label": "Rochester corr. set4", "tag": "rochester_set4"}
        ]
    },
    #############
    # Theory
    #############
    "NLO EW correction": {
        "name": "NLO_EW",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    #############
    # QCD scale and parton shower
    #############
    "QCD scale": {
        "name": "QCDScale",
        "type": "shape",
        "kinematic": False,
        "kind": "envelope",
        "samples": theory_samples,
        "variations": [
            {
                "label": f"$\\mu_{{R}}={mur:g}, \\mu_{{F}}={muf:g}$",
                "tag": {s: f"QCDScale_{qcd_scale_indices[layout][i]}" for s, layout in qcd_scale_layout.items()},
            } for i, (mur, muf) in enumerate(qcd_scale_points)
        ],
    },
    "Parton shower": {
        "name": "PSWeight",
        "type": "shape",
        "kinematic": False,
        "kind": "envelope",
        "samples": theory_samples,
        "variations": [
            {"label": "ISR=2, FSR=1", "tag": "PSWeight_0"},
            {"label": "ISR=1, FSR=2", "tag": "PSWeight_1"},
            {"label": "ISR=0.5, FSR=1", "tag": "PSWeight_2"},
            {"label": "ISR=1, FSR=0.5", "tag": "PSWeight_3"},
        ]
    },
    #############
    # PDF
    #############
    # LHEPdfWeight is NNPDF3.1 NNLO, LHA IDs 325300-325402
    # (NNPDF31_nnlo_as_0118_mc_hessian_pdfas): member 0 is the central value,
    # 1-100 are symmetric Hessian eigenvectors (one nuisance each, Down
    # mirrored around nominal), 101-102 are alpha_s variations.
    # Since st channels are buggy we just take the nominal in postprocessing 
    # So that the whole variation only comes from the non buggy t-ch sample.
    # If nominals where not added the variations would not be centered around nominal 
    # Which is "SIngle Top" as a whole.
    **{
        f"PDF eigenvector {i}": {
            "name": f"pdf_eig{i}",
            "type": "shape",
            "kinematic": False,
            "kind": "hessian",
            "tag": f"PDFWeight_{i}",
            "samples": mc_samples,
            "use_nominal_for_datasets": st_datasets_without_pdf,
        } for i in range(1, 101)
    },
    "PDF $\\alpha_{s}$": {
        "name": "pdf_alphas",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PDFWeight_102",  # alpha_s(m_Z) = 0.120
        "tag_down": "PDFWeight_101",  # alpha_s(m_Z) = 0.116
        "samples": mc_samples,
        "use_nominal_for_datasets": st_datasets_without_pdf,
    },
    #############
    # Jets: resolution, and scale (one nuisance per JES source of the year,
    # see jes_sources)
    #############
    "JER": {
        "name": "JER",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    **{
        f"JES_{source}": {
            "name": f"JES_{source}",
            "type": "shape",
            "kinematic": True,
            "samples": mc_samples,
            "kind": "weight"
        } for source in jes_sources
    },
    #############
    # b-tagging
    #############
    "puidSF": {
        "name": "puidSF",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "btagSF_sf": {
        "name": "btagSF_sf",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "btagSF_eff": {
        "name": "btagSF_eff",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    #############
    # Fakes
    #############
    "Fakes transfer factor: Fit": {
        "name": "fakes_param",
        "type": "shape",
        "kinematic": False,
        "kind": "weight",
        "samples": all_samples,
    },
    "Fakes transfer factor: Model": {
        "name": "fakes_model",
        "type": "shape",
        "kinematic": False,
        "kind": "envelope",
        "samples": all_samples,
        "variations": [
            {"label": "fakes_model", "tag": "fakes_model"}
        ],
    },
}

corrections = {
    "Pile-up corr.": { 
        "name": "PU",
        "kinematic": False,
        "samples": mc_samples 
    },
    "L1 pre-firing corr.": {
        "name": "prefireWeight",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Trigger SF": { 
        "name": "mu_trig",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Reconstruction SF": { 
        "name": "mu_reco",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon ID SF": { 
        "name": "mu_id",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Isolation SF": { 
        "name": "mu_iso",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Rochester corr.": { 
        "name": "rochester",
        "kinematic": True,
        "samples": all_samples, 
        "related_nuisances": ["Rochester corr. (stat)", "Rochester corr. (syst)"]
    },
    "NLO EW correction": { 
        "name": "NLO_EW",
        "kinematic": False,
        "samples": mc_samples 
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "kinematic": False,
        "samples": mc_samples
    },
    "puidSF": { 
        "name": "puidSF",
        "kinematic": False,
        "samples": mc_samples 
    },
    "btagSF": { 
        "name": "btagSF",
        "kinematic": False,
        "samples": mc_samples,
        "related_nuisances": ["btagSF_sf", "btagSF_eff"] 
    },
    "JES+JER": {
        "name": "JES_JER",
        "kinematic": True,
        "samples": all_samples,
        "related_nuisances": ["JER"] + [f"JES_{source}" for source in jes_sources],
    },
    "Fakes transfer factor": { 
        "name": "fakes",
        "kinematic": False,
        "samples": all_samples,
        "related_nuisances": ["Fakes transfer factor: Fit", "Fakes transfer factor: Model"] 
    },
}

# Define regions
preselections = lambda events: (50 < events.mll) & (events.mll < 500)

regions = {
    "bveto_mm": {
        "func": lambda events: preselections(events) & events.mm & events.bveto,
        "mask": 0,
    },
    "bveto_mm_ss": {
        "func": lambda events: preselections(events) & events.mm_ss & events.bveto,
        "mask": 0,
    },
}

# Define variables and histograms
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
        "axis": hist.axis.Regular(60, 50, 200, name="mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV"
    },
    "mll_medium": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable([50,55,60,65,70,75,80,85,90,95,100,105,110,
            115,120,130,140,150,160,170,180,190,200,220,240,260,280,300,325,350,375,
            400,450,500], name="mll_medium"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "costhetastar": {
        "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
        "axis": hist.axis.Regular(50, -1, 1, name="costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$"
    },
    "rapll_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity),
        "axis": hist.axis.Regular(48, 0, 2.4, name="rapll_abs"),
        "label": "$|y_{\\ell\\ell}|$"
    },
    "triple_diff": {
        "axis": [
            hist.axis.Variable([50,70,80,90,100,110,120,150,200,300,500], name="mll"),
            hist.axis.Variable([-1.0,-0.5,0.0,0.5,1.0], name="costhetastar"),
            hist.axis.Variable([0.0,0.48,0.96,1.44,2.4], name="rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV", "", ""],
        "xlog": True,
    },
}

cards_regions = ["bveto_mm"]
cards_variables = ["triple_diff"]
