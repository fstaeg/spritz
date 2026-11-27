import json
import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path

fw_path = get_fw_path()
with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

year = "Full2016v9HIPM"
lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
lumi_syst = lumis[year].get("syst", {})
year_label = "2016"
plot_label = "2016preVFP"
njobs = 1000

eft_chunks_per_job = 50

runner = f"{fw_path}/src/spritz/runners/runner_3DY_eft.py"

special_analysis_cfg = {
    "do_variations": True,
    "do_theory_variations": True, # 116 variations
    "do_rochester_stat_variations": False, # 100 variations
    "do_jet_variations": False, # 24 variations
    "invert_one_isolation_loose": True,
    "invert_one_isolation_control": False,
    "reweight_fakes": False,
}

nnlo_n3ll_qcd = {
    "file": f"{fw_path}/data/common/kfactor_N3LO_GenPart.root",
    "object": "kfactor_N3LO_GenPart",
    "name": "NNLO_N3LL_QCD",
    "observable": "pt:rapidity:mll",  # qT, signed y_ll, m_ll
}
nlo_ew = {
    "file": f"{fw_path}/data/common/dy_ew_kfactor_3d.root",
    "object": "ratio_correlated",  # NLO-EW / LO-EW, same-event reweighting
    "name": "NLO_EW",
    "observable": "mll:absrapidity:costhetastar",  # Collins-Soper, pre-FSR leptons
}

# SM samples
dy_bins = ["10to50","50to100","100to200","200to400","400to500","500to700","700to800",
    "800to1000","1000to1500","1500to2000","2000toInf"]
ggll_bins = ["10to30","30to50","50to200","200to1500","1500toInf"]

datasets = { 
    **{
        f"DYmm_MiNNLO_M-{b}": {
            "files": f"DYJetsToMuMu_M-{b}" if b!="50to100" else "DYJetsToMuMu",
            "task_weight": 8,
            "max_weight": 1e9,
            "ho_corrections": [nnlo_n3ll_qcd, nlo_ew],
        } for b in dy_bins 
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": [nnlo_n3ll_qcd, nlo_ew],
        "h2erratum_rwgt": True
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
for era in ["B", "C", "D", "E", "F"]:
    datasets[f"SingleMuon_{era}"] = {
        "files": f"SingleMuon_Run{year_label}{era}{"-ver2" if era=="B" else ""}-HIPM-UL{year_label}-v2",
        "trigger_sel": "events.SingleMu | events.HighPtMu",
        "read_form": "data",
        "is_data": True,
        "era": f"UL{year_label}_preVFP{era}"
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
        "is_signal": True
    },
}

colors = {
    "Fakes": cmap_petroff[0],
    "GGToLL": cmap_petroff[1],
    "Single_Top": cmap_petroff[2],
    "TT": cmap_petroff[3],
    "VV": cmap_petroff[4],
    "DYtt": cmap_petroff[8],
    "DYmm_MiNNLO": cmap_petroff[9],
    "DYmm_sm": cmap_petroff[9],
    "DYmm": cmap_petroff[9]
}

# Define regions
preselections = lambda events: (40 < events.mll)

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
        "axis": hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,
            120,140,160,180,200,225,250,300,350,500,1000], name="mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "mll_medium": {
        "func": lambda events: events.mll,
        "axis": hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,
            120,140,160,180,200,225,250,300,350,500], name="mll_medium"),
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
            hist.axis.Variable([40,60,80,100,120,140,160,200,300,500,750,1000,2000], name="mll"),
            hist.axis.Variable([-1.0,-0.5,0.0,0.5,1.0], name="costhetastar"),
            hist.axis.Variable([0.0,0.48,0.96,1.44,2.4], name="rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV", "", ""],
        "xlog": True,
    },
}

all_samples = [s for s in samples if not samples[s].get("is_variance")]
mc_samples = [skey for skey in samples if not samples[skey].get("is_data",False)]

# LHA ID 306000 (NNPDF31_nnlo_hessian_pdfas)
datasets_pdf_lhaid_306000 = [
    *[f"DYmm_MiNNLO_M-{b}" for b in dy_bins], 
    "DYtt", "TTTo2L2Nu", "TTToSemiLeptonic", "ZZTo4L"
]

# LHA ID 325300 (NNPDF31_nnlo_as_0118_mc_hessian_pdfas)
datasets_pdf_lhaid_325300 = ["WZTo3LNu", "WZTo2Q2L", "ZZTo2Q2L"]

# LHA ID 320900 (NNPDF31_nnlo_as_0118_nf_4) -> Replicas!
datasets_pdf_lhaid_320900 = ["ZZTo2L2Nu", "WWTo2L2Nu"]

lumi_nuis = {
    k: {
        "name": k,
        "type": "lnN",
        "samples": {s: str(v) for s in mc_samples}
    } for k, v in lumi_syst.items()
}

nuisances = lumi_nuis | {
    "stat": {
        "type": "auto",
        "maxPoiss": "10",
        "includeSignal": "0",
        "samples": {}
    },
    "H2ErratumFix_2016preVFP": {
        "name": "h2err",
        "type": "shape",
        "kinematic": False,
        "kind": "weight",
        "samples": ["DYtt"],
    },
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
    "PDF $\\alpha_{s}$": {
        "name": "pdf_alphas",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PDFWeight_102",  # alpha_s(m_Z) = 0.120
        "tag_down": "PDFWeight_101",  # alpha_s(m_Z) = 0.116
        "samples": ["DYmm_MiNNLO", "DYtt", "TT", "VV"],
        "use_nominal_for_datasets": datasets_pdf_lhaid_320900,
    },
    "QCDScale_muR_DY": {
        "name": "QCDScale_muR_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muR_up",
        "tag_down": "QCDScale_muR_down",
        "samples": ["DYmm_MiNNLO", "DYtt"],
    },
    "QCDScale_muF_DY": {
        "name": "QCDScale_muF_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muF_up",
        "tag_down": "QCDScale_muF_down",
        "samples": ["DYmm_MiNNLO", "DYtt"],
    },
    "QCDScale_muR_TT": {
        "name": "QCDScale_muR_TT",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muR_up",
        "tag_down": "QCDScale_muR_down",
        "samples": ["TT"],
    },
    "QCDScale_muF_TT": {
        "name": "QCDScale_muF_TT",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muF_up",
        "tag_down": "QCDScale_muF_down",
        "samples": ["TT"],
    },
    "QCDScale_muR_ST": {
        "name": "QCDScale_muR_ST",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muR_up",
        "tag_down": "QCDScale_muR_down",
        "samples": ["Single_Top"],
    },
    "QCDScale_muF_ST": {
        "name": "QCDScale_muF_ST",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muF_up",
        "tag_down": "QCDScale_muF_down",
        "samples": ["Single_Top"],
    },
    "QCDScale_muR_VV": {
        "name": "QCDScale_muR_VV",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muR_up",
        "tag_down": "QCDScale_muR_down",
        "samples": ["VV"],
    },
    "QCDScale_muF_VV": {
        "name": "QCDScale_muF_VV",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "QCDScale_muF_up",
        "tag_down": "QCDScale_muF_down",
        "samples": ["VV"],
    },
    "btagSF_bc_correlated": {
        "name": "btagSF_bc_correlated",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_bc_correlated_up",
        "tag_down": "btagSF_bc_correlated_down",
        "samples": mc_samples,
    },
    "btagSF_bc_2016preVFP": {
        "name": "btagSF_bc_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_bc_uncorrelated_up",
        "tag_down": "btagSF_bc_uncorrelated_down",
        "samples": mc_samples,
    },
    "btagSF_light_correlated": {
        "name": "btagSF_light_correlated",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_light_correlated_up",
        "tag_down": "btagSF_light_correlated_down",
        "samples": mc_samples,
    },
    "btagSF_light_2016preVFP": {
        "name": "btagSF_light_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "btagSF_light_uncorrelated_up",
        "tag_down": "btagSF_light_uncorrelated_down",
        "samples": mc_samples,
    },
}

fakes_dict = {
    "regions": [
        {"target": "bveto_mm_ss", "source": "bveto_mm_ss"},
        {"target": "bveto_mm", "source": "bveto_mm"}
    ],
    "subtract_mc": [s for s in samples if not (samples[s].get("is_data"))],
    "nuisances": [n for n in nuisances if nuisances[n].get("type","") == "shape"]
}

corrections = {
}

