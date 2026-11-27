# Written by configs/make_dy_eft_ext_configs.py from configs/dy-2016preVFP/config.py: edit that
# script (or the benchmark) and run it again rather than this file.
#
# dy-eft-ext-2016preVFP/smeftsim_1j: the SMEFTsim1j SMEFT samples only, the templates of
# every reweighting point and their covariance terms, over the full mll range
# of the samples and all their extensions. Data, backgrounds and MiNNLO DY are
# in the benchmark dy-2016preVFP: process both, merge them (the benchmark first)
# and run spritz-postproc on the merged config:
#     spritz-merge-configs --configs dy-2016preVFP dy-eft-ext-2016preVFP/smeftsim_1j -o dy-eft-ext-2016preVFP/merged_1j
import json
import awkward as ak
import hist
import numpy as np
from itertools import combinations_with_replacement
from spritz.framework.framework import cmap_petroff, get_fw_path, get_rw_idx, get_eft_points
from spritz.lookup_tools.eft_operator_indices import eft_operators, get_group

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2016v9HIPM"
lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
lumi_syst = lumis[year].get("syst", {})
year_label = "2016"
plot_label = "2016preVFP"

# an upper limit: the jobs left without chunks are dropped
njobs = 3000

# Each job sums its EFT chunks (all of one dataset) into a single result
eft_chunks_per_job = 50

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft.py"

special_analysis_cfg = {
    "do_variations": True,
    "do_theory_variations": True,
    # Only the theory weight variations are computed for the SMEFT samples:
    # PDF, alpha_s, QCD scale and parton shower change the EFT/SM ratio itself.
    # spritz-postproc builds every other varied EFT template from MiNNLO
    # (renorm_samples): EFT_point_var = MiNNLO_var * EFT_point_var / EFT_sm_var,
    # i.e. the relative variation of MiNNLO for the experimental nuisances
    # (lepton SFs, PU, prefiring, b-tag, PU ID, JES/JER, muon momentum, fakes)
    # and the higher-order corrections
    "do_rochester_stat_variations": False,
    "do_jet_variations": False,
    "variations": ["nom", "PDFWeight_*", "QCDScale_*", "PSWeight_*"],  # fnmatch patterns of the variations to keep
    "invert_one_isolation_loose": False,
    "invert_one_isolation_control": False,
    "reweight_fakes": True,
}

# SMEFTsim, LO 0+1 jet merged (406-wide reweight card, 27 operators)
model = "SMEFTsim1j"
file_key = "DYMuMu_LO_EFT_SMEFTsim_propcorr_1j_mll%s_Photos_startingOne"
eft_bins = ['50_120', '120_200', '200_400', '400_600', '600_800', '800_1000', '1000_3000']
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

# MiNNLO DY bins of the benchmark (its datasets are processed there)
dy_bins = ["10to50","50to100","100to200","200to400","400to500","500to700","700to800",
    "800to1000","1000to1500","1500to2000","2000toInf"]

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
    # The benchmark's DYmm_MiNNLO (same datasets), out of the datacard and of the
    # plot stack: the SM template of this model, renormalized to it, replaces
    # it. Merged after the benchmark, this definition replaces the benchmark's
    "DYmm_MiNNLO": {
        "samples": [f"DYmm_MiNNLO_M-{b}" for b in dy_bins],
        "is_signal": True,
        "exclude_from_datacard": True,
        "stack": False,
    },
}

colors = {f"DYmm_{model}_sm": cmap_petroff[9]}

# Renormalize the templates from "reference" (this model's SM) to "target"
# (MiNNLO, from the benchmark): spritz-postproc does it on the merged config
renorm_samples = {
    "target": "DYmm_MiNNLO",
    "reference": f"DYmm_{model}_sm",
    "samples": point_samples + [f"DYmm_{model}_cov_{a}_{b}" for a, b in cov_pairs],
}

# Define regions
preselections = lambda events: (60 < events.mll)

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
        "axis": hist.axis.Variable([60,65,70,75,80,85,90,95,100,105,110,115,120,130,
            140,150,160,170,180,190,200,220,240,260,280,300,325,350,375,400,425,450,
            475,500,550,600,700,800,900,1000,1200,1400,1600,2000,3000], name="mll"),
        "blind_data": True,
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "mll_low": {
        "func": lambda events: events.mll,
        "axis": hist.axis.Regular(18, 70, 115, name="mll_low"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
    },
    "mll_medium": {
        "func": lambda events: events.mll,
        "axis": hist.axis.Variable([60,65,70,75,80,85,90,95,100,105,110,
            115,120,130,140,150,160,170,180,190,200,220,240,260,280,300,325,350,375,
            400,450,500], name="mll_medium"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "axis": hist.axis.Variable([0,5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100,110,
            120,140,160,180,200,225,250,275,300], name="ptll"),
        "label": "$p_{T}^{\\ell\\ell$",
        "unit": "GeV",
        "xlog": True
    },
    "ptl1": {
        "func": lambda events: events.Lepton[:, 0].pt,
        "axis": hist.axis.Variable([29,40,50,60,70,80,90,100,120,140,160,180,200,
            225,250,300,350,400,500,700,1000], name="ptj1"),
        "label": "$p_{T}^{j_{1}}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptl2": {
        "func": lambda events: events.Lepton[:, 1].pt,
        "axis": hist.axis.Variable([15,20,25,30,35,40,50,60,70,80,90,100,120,160,200,
            250,300,400,500,700], name="ptl2"),
        "label": "$p_{T}^{\\ell_{2}}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptj1": {
        "func": lambda events: ak.fill_none(ak.pad_none(events.Jet, 1)[:, 0].pt, -1),
        "axis": hist.axis.Variable([20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,55,
            60,65,70,80,90,100,120,140,170,200], name="ptj1"),
        "label": "$p_{T}^{j_{1}}$",
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
            hist.axis.Variable([60,80,90,100,120,140,160,200,300,500,750,1000,1500,3000], name="mll"),
            hist.axis.Variable([-1.0,-0.5,0.0,0.5,1.0], name="costhetastar"),
            hist.axis.Variable([0.0,0.48,0.96,1.44,2.4], name="rapll_abs"),
        ],
        "blind_data": True,
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV", "", ""],
        "xlog": True,
    },
}

cards_regions = ["bveto_mm"]
cards_variables = ["triple_diff"]
covariance_file = "covariance.root"

# Nuisances and corrections: those of the benchmark that apply to DYmm_MiNNLO,
# for the EFT templates (once merged, each applies to the samples of both)
eft_samples = point_samples

lumi_nuis = {
    k: {
        "name": k,
        "type": "lnN",
        "samples": {s: str(v) for s in eft_samples}
    } for k, v in lumi_syst.items()
}

nuisances = lumi_nuis | {
    "Pile-up corr.": {
        "name": "PU",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "L1 pre-firing corr. Muon (stat, 2016preVFP)": {
        "name": "prefireWeightMuon_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "prefireWeightMuon_stat_up",
        "tag_down": "prefireWeightMuon_stat_down",
    },
    "L1 pre-firing corr. Muon (syst)": {
        "name": "prefireWeightMuon_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "L1 pre-firing corr. ECAL": {
        "name": "prefireWeightECAL",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "H2ErratumFix_2016preVFP": {
        "name": "h2err",
        "type": "shape",
        "kinematic": False,
        "kind": "weight",
        "samples": eft_samples,
    },
    #############
    # Leptons
    #############
    "Trigger SF high pT (stat, 2016preVFP)": {
        "name": "mu_trig_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_trig_highpt_stat_up",
        "tag_down": "mu_trig_highpt_stat_down",
    },
    "Trigger SF high pT (syst, 2016)": {
        "name": "mu_trig_highpt_syst_2016",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_trig_highpt_syst_up",
        "tag_down": "mu_trig_highpt_syst_down",
    },
    "Trigger SF medium pT (stat, 2016preVFP)": {
        "name": "mu_trig_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_trig_lowpt_stat_up",
        "tag_down": "mu_trig_lowpt_stat_down",
    },
    "Trigger SF medium pT (syst, 2016)": {
        "name": "mu_trig_lowpt_syst_2016",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_trig_lowpt_syst_up",
        "tag_down": "mu_trig_lowpt_syst_down",
    },
    "Muon Reco SF high pT (stat, 2016preVFP)": {
        "name": "mu_reco_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_reco_highpt_stat_up",
        "tag_down": "mu_reco_highpt_stat_down",
    },
    "Muon Reco SF high pT (syst)": {
        "name": "mu_reco_highpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "Muon Reco SF medium pT (stat, 2016preVFP)": {
        "name": "mu_reco_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_reco_lowpt_stat_up",
        "tag_down": "mu_reco_lowpt_stat_down",
    },
    "Muon Reco SF medium pT (syst)": {
        "name": "mu_reco_lowpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "Muon ID SF high pT (stat, 2016preVFP)": {
        "name": "mu_id_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_id_highpt_stat_up",
        "tag_down": "mu_id_highpt_stat_down",
    },
    "Muon ID SF high pT (syst)": {
        "name": "mu_id_highpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "Muon ID SF medium pT (stat, 2016preVFP)": {
        "name": "mu_id_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_id_lowpt_stat_up",
        "tag_down": "mu_id_lowpt_stat_down",
    },
    "Muon ID SF medium pT (syst)": {
        "name": "mu_id_lowpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "Muon Iso SF high pT (stat, 2016preVFP)": {
        "name": "mu_iso_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_iso_highpt_stat_up",
        "tag_down": "mu_iso_highpt_stat_down",
    },
    "Muon Iso SF high pT (syst)": {
        "name": "mu_iso_highpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "Muon Iso SF medium pT (stat, 2016preVFP)": {
        "name": "mu_iso_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "updown",
        "tag_up": "mu_iso_lowpt_stat_up",
        "tag_down": "mu_iso_lowpt_stat_down",
    },
    "Muon Iso SF medium pT (syst)": {
        "name": "mu_iso_lowpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "Muon Scale Corrections high pT": {
        "name": "mu_scale_highpt",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "hessian",
        "tag": "mu_scale_highpt_nominal"
    },
    "Muon Resolution Corrections high pT": {
        "name": "mu_res_highpt",
        "type": "shape",
        "value": 3.,
        "kinematic": True,
        "samples": eft_samples,
        "kind": "hessian",
        "tag": "mu_res_highpt_syst"
    },
    "Rochester Resolution Corrections": {
        "name": "rochester_res",
        "type": "shape",
        "value": 3.,
        "kinematic": True,
        "samples": eft_samples,
        "kind": "hessian",
        "tag": "rochester_res"
    },
    "Rochester Scale Corrections": {
        "name": "rochester_syst",
        "type": "shape",
        "kinematic": True,
        "kind": "square",
        "samples": eft_samples,
        "variations": [
            {"label": f"rochester_set{i}", "tag": f"rochester_set{i}"} for i in range(2,5)
        ]
    },
    #############
    # Theory
    #############
    "NLO EW correction": {
        "name": "NLO_EW",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    #############
    # PDF
    #############
    **{
        f"PDF eigenvector {i}": {
            "name": f"pdf_eig{i}",
            "type": "shape",
            "kinematic": False,
            "kind": "hessian",
            "tag": f"PDFWeight_{i}",
            "samples": eft_samples,
            "plot": False
        } for i in range(1, 101)
    },
    "PDF $\\alpha_{s}$": {
        "name": "pdf_alphas",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PDFWeight_102",  # alpha_s(m_Z) = 0.120
        "tag_down": "PDFWeight_101",  # alpha_s(m_Z) = 0.116
        "samples": eft_samples,
    },
    #############
    # QCD scale
    #############
    "QCDScale_muR_DY": {
        "name": "QCDScale_muR_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muR_up",
        "tag_down": "QCDScale_muR_down",
        "samples": eft_samples,
    },
    "QCDScale_muF_DY": {
        "name": "QCDScale_muF_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muF_up",
        "tag_down": "QCDScale_muF_down",
        "samples": eft_samples,
    },
    "QCDScale_muRmuF_DY": {
        "name": "QCDScale_muRmuF_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muRmuF_up",
        "tag_down": "QCDScale_muRmuF_down",
        "samples": eft_samples,
    },
    #############
    # PS Weight
    #############
    "PSWeight_ISR_DY": {
        "name": "PSWeight_ISR_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_ISR_up",
        "tag_down": "PSWeight_ISR_down",
        "samples": eft_samples,
    },
    "PSWeight_FSR_EWK": {
        "name": "PSWeight_FSR_EWK",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_FSR_up",
        "tag_down": "PSWeight_FSR_down",
        "samples": eft_samples,
    },
    #############
    # b-tagging
    #############
    "puidSF": {
        "name": "puidSF",
        "type": "shape",
        "kinematic": False,
        "samples": eft_samples,
        "kind": "weight"
    },
    "btagSF_bc_correlated": {
        "name": "btagSF_bc_correlated",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_bc_correlated_up",
        "tag_down": "btagSF_bc_correlated_down",
        "samples": eft_samples,
    },
    "btagSF_bc_2016preVFP": {
        "name": "btagSF_bc_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_bc_uncorrelated_up",
        "tag_down": "btagSF_bc_uncorrelated_down",
        "samples": eft_samples,
    },
    "btagSF_light_correlated": {
        "name": "btagSF_light_correlated",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_light_correlated_up",
        "tag_down": "btagSF_light_correlated_down",
        "samples": eft_samples,
    },
    "btagSF_light_2016preVFP": {
        "name": "btagSF_light_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_light_uncorrelated_up",
        "tag_down": "btagSF_light_uncorrelated_down",
        "samples": eft_samples,
    },
    #############
    # Jets
    #############
    "JER": {
        "name": "JER",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_Absolute_2016": {
        "name": "JES_Absolute_2016",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_Absolute": {
        "name": "JES_Absolute",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_BBEC1_2016": {
        "name": "JES_BBEC1_2016",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_BBEC1": {
        "name": "JES_BBEC1",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_EC2_2016": {
        "name": "JES_EC2_2016",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_EC2": {
        "name": "JES_EC2",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_FlavorQCD": {
        "name": "JES_FlavorQCD",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_HF_2016": {
        "name": "JES_HF_2016",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_HF": {
        "name": "JES_HF",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_RelativeBal": {
        "name": "JES_RelativeBal",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
        "kind": "weight"
    },
    "JES_RelativeSample_2016": {
        "name": "JES_RelativeSample_2016",
        "type": "shape",
        "kinematic": True,
        "samples": eft_samples,
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
        "samples": eft_samples,
    },
    "Fakes transfer factor: Model": {
        "name": "fakes_model",
        "type": "shape",
        "kinematic": False,
        "kind": "envelope",
        "samples": eft_samples,
        "variations": [
            {"label": "fakes_model", "tag": "fakes_model"}
        ],
    },
}

corrections = {
    "Pile-up corr.": { 
        "name": "PU",
        "kinematic": False,
        "samples": eft_samples 
    },
    "H2ErratumFix_2016preVFP": {
        "name": "h2err",
        "kinematic": False,
        "samples": eft_samples,
    },
    "L1 pre-firing corr. Muon": {
        "name": "prefireWeightMuon",
        "kinematic": False,
        "samples": eft_samples,
        "related_nuisances": ["L1 pre-firing corr. Muon (stat, 2016preVFP)", "L1 pre-firing corr. Muon (syst)"]
    },
    "L1 pre-firing corr. ECAL": {
        "name": "prefireWeightECAL",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Trigger SF high pT": { 
        "name": "mu_trig_highpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Trigger SF medium pT": { 
        "name": "mu_trig_lowpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Trigger SF combined": { 
        "name": "mu_trig_combined",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon Reco SF high pT": { 
        "name": "mu_reco_highpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon Reco SF medium pT": { 
        "name": "mu_reco_lowpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon Reco SF combined": { 
        "name": "mu_reco_combined",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon ID SF high pT": { 
        "name": "mu_id_highpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon ID SF medium pT": { 
        "name": "mu_id_lowpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon ID SF combined": { 
        "name": "mu_id_combined",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon Iso SF high pT": { 
        "name": "mu_iso_highpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon Iso SF medium pT": { 
        "name": "mu_iso_lowpt",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Muon Iso SF combined": { 
        "name": "mu_iso_combined",
        "kinematic": False,
        "samples": eft_samples 
    },
    "Rochester corr.": { 
        "name": "rochester",
        "kinematic": True,
        "samples": eft_samples, 
        "related_nuisances": ["Rochester Scale Corrections"]
    },
    "NLO EW correction": { 
        "name": "NLO_EW",
        "kinematic": False,
        "samples": eft_samples 
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "kinematic": False,
        "samples": eft_samples
    },
    "All HO corrections": {
        "name": "ho",
        "kinematic": False,
        "samples": eft_samples,
        "related_nuisances": ["NLO EW correction", "NNLO+N3LL QCD correction"],
    },
    "puidSF": { 
        "name": "puidSF",
        "kinematic": False,
        "samples": eft_samples 
    },
    "btagSF": { 
        "name": "btagSF",
        "kinematic": False,
        "samples": eft_samples,
        "related_nuisances": ["btagSF_bc_correlated", "btagSF_bc_2016preVFP",
            "btagSF_light_correlated", "btagSF_light_2016preVFP"] 
    },
    "JES+JER": {
        "name": "JES_JER",
        "kinematic": True,
        "samples": eft_samples,
        "related_nuisances": ["JER", "JES_Absolute", "JES_Absolute_2016", "JES_BBEC1", "JES_BBEC1_2016",
            "JES_EC2", "JES_EC2_2016", "JES_FlavorQCD", "JES_HF", "JES_HF_2016", "JES_RelativeBal", 
            "JES_RelativeSample_2016"]
    },
    "Fakes transfer factor": { 
        "name": "fakes",
        "kinematic": False,
        "samples": eft_samples,
        "related_nuisances": ["Fakes transfer factor: Fit", "Fakes transfer factor: Model"] 
    },
}
