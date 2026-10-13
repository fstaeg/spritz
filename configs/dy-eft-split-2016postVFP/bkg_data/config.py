import json
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path

# dy-eft-split-2016postVFP: data, backgrounds and MiNNLO DY for 2016postVFP, as the 2018
# dy-eft-split-test/bkg_data with the 2016postVFP data and samples (see
# configs/dy-2016postVFP) and all the uncertainties on: on top of 2018, Rochester
# stat. replicas, JES/JER, QCD scale and parton shower.
# The SMEFT samples are in the sibling configs (smeftsim_1j, smeftsim_0j,
# smeftatnlo): process each, then combine with spritz-merge-configs and run
# spritz-postproc on the merged config.

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2016v9noHIPM"
lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
year_label = "2016"
plot_label = "2016postVFP"
njobs = 1000

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft.py"

special_analysis_cfg = {
    "do_variations": True,
    "do_theory_variations": True, # 116 variations
    "do_rochester_stat_variations": True, # 100 variations
    "do_jet_variations": True, # 24 variations
    "invert_one_isolation_loose": False,
    "invert_one_isolation_control": False,
    "reweight_fakes": True,
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

# SM samples
# I am not processing every mll bin since it is blind above 500 GeV (events.mll < 500)
# To full scale we should uncomment the remaining bins 
dy_bins = ["10to50","50to100","100to200","200to400","400to500","500to700"]#,"700to800",
    #"800to1000","1000to1500","1500to2000","2000toInf"]
ggll_bins = ["10to30","30to50","50to200","200to1500"]#,"1500toInf"]

datasets = { 
    **{
        f"DYmm_MiNNLO_M-{b}": {
            "files": f"DYJetsToMuMu_M-{b}" if b!="50to100" else "DYJetsToMuMu",
            "task_weight": 8,
            "max_weight": 1e9,
            "ho_corrections": [nnlo_n3ll_qcd, nlo_ew],
            "exclude_from_datacard": True,
        } for b in dy_bins 
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": [nnlo_n3ll_qcd, nlo_ew] 
    },
    "ST_s-channel": {
        "files": "ST_s-channel",
        "task_weight": 8,
    },
    "ST_t-channel_top_5f": {
        "files": "ST_t-channel_top_5f",
        "task_weight": 8,
    },
    "ST_t-channel_antitop_5f": {
        "files": "ST_t-channel_antitop_5f",
        "task_weight": 8,
    },
    "ST_tW_top_noHad": {
        "files": "ST_tW_top_noHad",
        "task_weight": 8,
    },
    "ST_tW_antitop_noHad": {
        "files": "ST_tW_antitop_noHad",
        "task_weight": 8,
    },
    "TTTo2L2Nu": {
        "files": "TTTo2L2Nu",
        "task_weight": 8,
        "top_pt_rwgt": True,
    },
    "TTToSemiLeptonic": {
        "files": "TTToSemiLeptonic",
        "task_weight": 8,
        "top_pt_rwgt": True,
        "genmatching_nlep": 1,
    },
    "WWTo2L2Nu": {
        "files": "WWTo2L2Nu",
        "task_weight": 8,
    },
    "WZTo3LNu": {
        "files": "WZTo3LNu",
        "task_weight": 8,
    },
    "WZTo2Q2L": {
        "files": "WZTo2Q2L",
        "task_weight": 8,
    },
    "ZZTo4L": {
        "files": "ZZTo4L",
        "task_weight": 8,
    },
    "ZZTo2L2Nu": {
        "files": "ZZTo2L2Nu",
        "task_weight": 8,
    },
    "ZZTo2Q2L": {
        "files": "ZZTo2Q2L",
        "task_weight": 8,
    },
    **{
        f"GGToMuMu_M-{b}_{case}": {
            "files": f"GGToMuMu_M-{b}_{case}",
            "task_weight": 8 
        } for b in ggll_bins 
        for case in ["El-El", "Inel-El_El-Inel", "Inel-Inel"] 
    }
}

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

# Data
samples_data = []
for era in ["F", "G", "H"]:
    datasets[f"SingleMuon_{era}"] = {
        "files": f"SingleMuon_Run{year_label}{era}-UL{year_label}-v1",
        "trigger_sel": "events.SingleMu",
        "read_form": "data",
        "is_data": True,
        "era": f"UL{year_label}{era}"
    }
    samples_data.append(f"SingleMuon_{era}")

# Merge samples
samples = {
    "Data": {
        "samples": samples_data,
        "is_data": True,
    },
    "GGToLL": { 
        "samples": [
            f"GGToMuMu_M-{b}_{case}" for b in ggll_bins 
            for case in ["El-El", "Inel-El_El-Inel", "Inel-Inel"]
        ] 
    },
    "Single_Top": {
        "samples": [
            "ST_s-channel",
            "ST_t-channel_top_5f",
            "ST_t-channel_antitop_5f",
            "ST_tW_top_noHad",
            "ST_tW_antitop_noHad",
        ]
    },
    "TT": {
        "samples": [
            "TTTo2L2Nu",
            "TTToSemiLeptonic"
        ]
    },
    "VV": {
        "samples": [
            "WWTo2L2Nu",
            "WZTo3LNu",
            "WZTo2Q2L",
            "ZZTo4L",
            "ZZTo2L2Nu",
            "ZZTo2Q2L"
        ]
    },
    "DYtt": {
        "samples": [
            "DYtt"
        ]
    },
    "DYmm_MiNNLO": {
        "samples": [
            f"DYmm_MiNNLO_M-{b}" for b in dy_bins
        ],
        "exclude_from_datacard": True,
        # spritz-plot: the DY in the stack is the SM of an EFT model (renormalized
        # to this one by its renorm_samples), see the EFT configs
        "stack": False
    }
}

colors = {
    "Fakes": cmap_petroff[0],
    "GGToLL": cmap_petroff[1],
    "Single_Top": cmap_petroff[2],
    "TT": cmap_petroff[3],
    "VV": cmap_petroff[4],
    "DYtt": cmap_petroff[8],
    "DYmm_MiNNLO": cmap_petroff[9],
    "DYmm": cmap_petroff[9]
}

# spritz-plot, ratio panel: on top of the full uncertainty band, draw the
# cumulative quadrature sums of these components, in list order (stat, then
# stat+PDF, ...). Per entry: "name" is the legend text, "nuisances" lists
# nuisance keys or "name"s, glob patterns allowed (default: [name]), any other
# key goes to matplotlib's stairs (color, alpha, hatch, fill, ...).
# Remove it to get the single band only.
split_uncertainties = [
    {"name": "stat", "color": "tab:red", "alpha": 0.6},
    {"name": "PDF", "nuisances": ["PDF eigenvector *"], "color": "tab:blue", "alpha": 0.4},
]

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
covariance_file = "covariance.root"

all_samples = [s for s in samples if not samples[s].get("is_variance")]
mc_samples = [s for s in all_samples if not samples[s].get("is_data")]

# Single top datasets whose LHEPdfWeight is unusable (see the PDF nuisances
# below for why). They stay in the Single_Top process, but get NO PDF
# variation: in the PDF Up/Down histograms they enter with their nominal.
st_datasets_without_pdf = ["ST_s-channel", "ST_tW_top_noHad", "ST_tW_antitop_noHad"]

# Fakes
fakes_dict = {
    "regions": [
        {"target": "bveto_mm", "source": "bveto_mm_ss"}
    ],
    "subtract_mc": [
        s for s in mc_samples if not (samples[s].get("is_smeft"))
    ],
    "nuisances": [
        "Fakes transfer factor: Fit", "Fakes transfer factor: Model"
    ]
}

# JES uncertainty sources of Full2016v9noHIPM (data/Full2016v9noHIPM/cfg.json, "jes"):
# the runner varies the jets by each of them (do_jet_variations)
jes_sources = [
    "Absolute", "Absolute_2016", "BBEC1", "BBEC1_2016", "EC2", "EC2_2016",
    "FlavorQCD", "HF", "HF_2016", "RelativeBal", "RelativeSample_2016",
]

# QCD scale: LHEScaleWeight index of the 6 (muR, muF) points in each sample
# (as in dy-2017). MiNNLO has 18 weights, most samples 9 (with the nominal),
# the VV amc@NLO ones 8 (without). The EFT configs add their samples to the
# same nuisance: spritz-merge-configs keeps the indices of all of them
qcd_scale_points = [(0.5, 0.5), (0.5, 1), (1, 0.5), (1, 2), (2, 1), (2, 2)]
qcd_scale_indices = {
    "18 weights": [0, 2, 6, 10, 14, 16],
    "9 weights": [0, 1, 3, 5, 7, 8],
    "8 weights": [0, 1, 3, 4, 6, 7],
}
# sample (or dataset, for VV whose datasets differ) -> its LHEScaleWeight layout
qcd_scale_layout = {
    "DYmm_MiNNLO": "18 weights",
    "DYtt": "18 weights",
    "Single_Top": "9 weights",
    "TT": "9 weights",
    "WWTo2L2Nu": "9 weights",
    "WZTo3LNu": "9 weights",
    "ZZTo4L": "9 weights",
    "ZZTo2L2Nu": "9 weights",
    "WZTo2Q2L": "8 weights",
    "ZZTo2Q2L": "8 weights",
}
# samples with QCD scale and parton shower (PSWeight) variations
theory_samples = ["DYmm_MiNNLO", "DYtt", "Single_Top", "TT", "VV"]

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
        "samples": ["DYmm_MiNNLO", "DYtt"],
        "kind": "weight"
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "type": "shape",
        "kinematic": False,
        "samples": ["DYmm_MiNNLO"],
        "kind": "weight"
    },
    "Top $p_{T}$ corr.": {
        "name": "tt_ptrw",
        "type": "shape",
        "kinematic": False,
        "samples": ["TT"],
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
        "samples": ["DYmm_MiNNLO", "DYtt"] 
    },
    "NNLO+N3LL QCD correction": {
        "name": "NNLO_N3LL_QCD",
        "kinematic": False,
        "samples": ["DYmm_MiNNLO"]
    },
    # Alternative MiNNLO predictions from the runner's "<name>_before"
    # variations, saved as histo_<sample>_<name>Before:
    #   bare MiNNLO              histo_DYmm_MiNNLO_hoBefore
    #   MiNNLO + NLO EW          histo_DYmm_MiNNLO_NNLO_N3LL_QCDBefore
    #   MiNNLO + NNLO+N3LL QCD   histo_DYmm_MiNNLO_NLO_EWBefore
    "All HO corrections": {
        "name": "ho",
        "kinematic": False,
        "samples": ["DYmm_MiNNLO", "DYtt"],
        "related_nuisances": ["NLO EW correction", "NNLO+N3LL QCD correction"],
    },
    "Top $p_{T}$ corr.": {
        "name": "tt_ptrw",
        "kinematic": False,
        "samples": ["TT"] 
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
