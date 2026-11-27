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
    "do_jet_variations": True, # 24 variations
    "invert_one_isolation_loose": False,
    "invert_one_isolation_control": False,
    "reweight_fakes": True,
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
            "exclude_from_datacard": True,
        } for b in dy_bins 
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": [nnlo_n3ll_qcd, nlo_ew] ,
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
        # "exclude_from_datacard": True,
        #"stack": False
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

# split_uncertainties = [
#     {"name": "stat", "color": "tab:red", "alpha": 0.6},
#     {"name": "PDF", "nuisances": ["PDF eigenvector *"], "color": "tab:blue", "alpha": 0.4},
# ]

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

all_samples = [s for s in samples if not samples[s].get("is_variance")]
mc_samples = [s for s in all_samples if not samples[s].get("is_data")]

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
    "Pile-up corr.": {
        "name": "PU",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "L1 pre-firing corr. Combined": {
        "name": "prefireWeight_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["L1 pre-firing corr. Muon (stat, 2016preVFP)", "L1 pre-firing corr. Muon (syst)", "L1 pre-firing corr. ECAL"]
    },
    "L1 pre-firing corr. Muon (stat, 2016preVFP)": {
        "name": "prefireWeightMuon_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "prefireWeightMuon_stat_up",
        "tag_down": "prefireWeightMuon_stat_down",
    },
    "L1 pre-firing corr. Muon (syst)": {
        "name": "prefireWeightMuon_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "L1 pre-firing corr. ECAL": {
        "name": "prefireWeightECAL",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "H2ErratumFix_2016preVFP": {
        "name": "h2err",
        "type": "shape",
        "kinematic": False,
        "kind": "weight",
        "samples": ["DYtt"],
    },
    #############
    # Leptons
    #############
    "Trigger SF Combined": {
        "name": "mu_trig_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["Trigger SF medium pT (stat, 2016preVFP)", "Trigger SF medium pT (syst, 2016)", "Trigger SF high pT (stat, 2016preVFP)", "Trigger SF high pT (syst, 2016)"]
    },
    "Trigger SF high pT (stat, 2016preVFP)": {
        "name": "mu_trig_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_trig_highpt_stat_up",
        "tag_down": "mu_trig_highpt_stat_down",
    },
    "Trigger SF high pT (syst, 2016)": {
        "name": "mu_trig_highpt_syst_2016",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_trig_highpt_syst_up",
        "tag_down": "mu_trig_highpt_syst_down",
    },
    "Trigger SF medium pT (stat, 2016preVFP)": {
        "name": "mu_trig_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_trig_lowpt_stat_up",
        "tag_down": "mu_trig_lowpt_stat_down",
    },
    "Trigger SF medium pT (syst, 2016)": {
        "name": "mu_trig_lowpt_syst_2016",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_trig_lowpt_syst_up",
        "tag_down": "mu_trig_lowpt_syst_down",
    },
    "Muon Reco SF Combined": {
        "name": "mu_reco_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["Muon Reco SF medium pT (stat, 2016preVFP)", "Muon Reco SF medium pT (syst)", "Muon Reco SF high pT (stat, 2016preVFP)", "Muon Reco SF high pT (syst)"]
    },
    "Muon Reco SF high pT (stat, 2016preVFP)": {
        "name": "mu_reco_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_reco_highpt_stat_up",
        "tag_down": "mu_reco_highpt_stat_down",
    },
    "Muon Reco SF high pT (syst)": {
        "name": "mu_reco_highpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Reco SF medium pT (stat, 2016preVFP)": {
        "name": "mu_reco_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_reco_lowpt_stat_up",
        "tag_down": "mu_reco_lowpt_stat_down",
    },
    "Muon Reco SF medium pT (syst)": {
        "name": "mu_reco_lowpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon ID SF Combined": {
        "name": "mu_id_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["Muon ID SF medium pT (stat, 2016preVFP)", "Muon ID SF medium pT (syst)", "Muon ID SF high pT (stat, 2016preVFP)", "Muon ID SF high pT (syst)"]
    },
    "Muon ID SF high pT (stat, 2016preVFP)": {
        "name": "mu_id_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_id_highpt_stat_up",
        "tag_down": "mu_id_highpt_stat_down",
    },
    "Muon ID SF high pT (syst)": {
        "name": "mu_id_highpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon ID SF medium pT (stat, 2016preVFP)": {
        "name": "mu_id_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_id_lowpt_stat_up",
        "tag_down": "mu_id_lowpt_stat_down",
    },
    "Muon ID SF medium pT (syst)": {
        "name": "mu_id_lowpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Iso SF Combined": {
        "name": "mu_iso_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["Muon Iso SF medium pT (stat, 2016preVFP)", "Muon Iso SF medium pT (syst)", "Muon Iso SF high pT (stat, 2016preVFP)", "Muon Iso SF high pT (syst)"]
    },
    "Muon Iso SF high pT (stat, 2016preVFP)": {
        "name": "mu_iso_highpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_iso_highpt_stat_up",
        "tag_down": "mu_iso_highpt_stat_down",
    },
    "Muon Iso SF high pT (syst)": {
        "name": "mu_iso_highpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Iso SF medium pT (stat, 2016preVFP)": {
        "name": "mu_iso_lowpt_stat_2016preVFP",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "updown",
        "tag_up": "mu_iso_lowpt_stat_up",
        "tag_down": "mu_iso_lowpt_stat_down",
    },
    "Muon Iso SF medium pT (syst)": {
        "name": "mu_iso_lowpt_syst",
        "type": "shape",
        "kinematic": False,
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Scale and Resolution": {
        "name": "mu_scale_res",
        "type": "group",
        "samples": [],
        "nuisances": ["Muon Scale Corrections high pT", "Muon Resolution Corrections high pT", 
            "Rochester Resolution Corrections", "Rochester Scale Corrections"]
    },
    "Muon Scale Corrections high pT": {
        "name": "mu_scale_highpt",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "hessian",
        "tag": "mu_scale_highpt_nominal"
    },
    "Muon Resolution Corrections high pT": {
        "name": "mu_res_highpt",
        "type": "shape",
        "value": 3.,
        "kinematic": True,
        "samples": mc_samples,
        "kind": "hessian",
        "tag": "mu_res_highpt_syst"
    },
    "Rochester Resolution Corrections": {
        "name": "rochester_res",
        "type": "shape",
        "value": 3.,
        "kinematic": True,
        "samples": mc_samples,
        "kind": "hessian",
        "tag": "rochester_res"
    },
    "Rochester Scale Corrections": {
        "name": "rochester_syst",
        "type": "shape",
        "kinematic": True,
        "kind": "square",
        "samples": all_samples,
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
    # PDF
    #############
    **{
        f"PDF eigenvector {i}": {
            "name": f"pdf_eig{i}",
            "type": "shape",
            "kinematic": False,
            "kind": "hessian",
            "tag": f"PDFWeight_{i}",
            "samples": ["DYmm_MiNNLO", "DYtt", "TT", "VV"],
            "use_nominal_for_datasets": [d for d in datasets if not d in datasets_pdf_lhaid_306000],
            "plot": False
        } for i in range(1, 101)
    },
    # "PDF_combined": {
    #     "name": "PDF_combined",
    #     "type": "group",
    #     "samples": [],
    #     "nuisances": ["PDF Hessian (LHA ID 306000)", "PDF Hessian (LHA ID 325300)", "PDF Replicas (LHA ID 320900)"]
    # },
    # "PDF Hessian (LHA ID 306000)": {
    #     "name": "pdf_hess_lhaid30600",
    #     "type": "shape",
    #     "kinematic": False,
    #     "kind": "square",
    #     "samples": ["DYmm_MiNNLO", "DYtt", "TT", "VV"],
    #     "use_nominal_for_datasets": [d for d in datasets if not d in datasets_pdf_lhaid_306000],
    #     "variations": [
    #         {"label": f"pdf_hess_lhaid30600_{i}", "tag": f"PDFWeight_{i}"} for i in range(1,101)
    #     ]
    # },
    "PDF Hessian (LHA ID 325300)": {
        "name": "pdf_hess",
        "type": "shape",
        "kinematic": False,
        "kind": "square",
        "samples": ["VV"],
        "use_nominal_for_datasets": [d for d in datasets if not d in datasets_pdf_lhaid_325300],
        "variations": [
            {"label": f"pdf_hess_{i}", "tag": f"PDFWeight_{i}"} for i in range(1,101)
        ]
    },
    "PDF Replicas (LHA ID 320900)": {
        "name": "pdf_rep",
        "type": "shape",
        "kinematic": False,
        "kind": "stdev",
        "samples": ["VV"],
        "use_nominal_for_datasets": [d for d in datasets if not d in datasets_pdf_lhaid_320900],
        "variations": [
            {"label": f"pdf_rep_{i}", "tag": f"PDFWeight_{i}"} for i in range(1,101)
        ]
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
    #############
    # QCD scale
    #############
    "QCDScale_muR_combined": {
        "name": "QCDScale_muR_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["QCDScale_muR_DY", "QCDScale_muR_TT", "QCDScale_muR_ST", "QCDScale_muR_VV"]
    },
    "QCDScale_muF_combined": {
        "name": "QCDScale_muF_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["QCDScale_muF_DY", "QCDScale_muF_TT", "QCDScale_muF_ST", "QCDScale_muF_VV"]
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
    #############
    # PS Weight
    #############
    "PSWeight_ISR_combined": {
        "name": "PSWeight_ISR_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["PSWeight_ISR_DY", "PSWeight_ISR_VV", "PSWeight_ISR_TT", "PSWeight_ISR_ST"]
    },
    "PSWeight_FSR_combined": {
        "name": "PSWeight_FSR_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["PSWeight_FSR_EWK", "PSWeight_FSR_Top"]
    },
    "PSWeight_ISR_DY": {
        "name": "PSWeight_ISR_DY",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_ISR_up",
        "tag_down": "PSWeight_ISR_down",
        "samples": ["DYmm_MiNNLO", "DYtt"],
    },
    "PSWeight_ISR_VV": {
        "name": "PSWeight_ISR_VV",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_ISR_up",
        "tag_down": "PSWeight_ISR_down",
        "samples": ["VV"],
    },
    "PSWeight_FSR_EWK": {
        "name": "PSWeight_FSR_EWK",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_FSR_up",
        "tag_down": "PSWeight_FSR_down",
        "samples": ["DYmm_MiNNLO", "DYtt", "VV"],
    },
    "PSWeight_ISR_TT": {
        "name": "PSWeight_ISR_TT",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_ISR_up",
        "tag_down": "PSWeight_ISR_down",
        "samples": ["TT"],
    },
    "PSWeight_ISR_ST": {
        "name": "PSWeight_ISR_ST",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_ISR_up",
        "tag_down": "PSWeight_ISR_down",
        "samples": ["Single_Top"],
    },
    "PSWeight_FSR_Top": {
        "name": "PSWeight_FSR_Top",
        "type": "shape",
        "kinematic": False,
        "kind": "updown",
        "tag_up": "PSWeight_FSR_up",
        "tag_down": "PSWeight_FSR_down",
        "samples": ["TT", "Single_Top"],
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
    "btagSF_combined": {
        "name": "btagSF_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["btagSF_bc_correlated", "btagSF_bc_2016preVFP", "btagSF_light_correlated", "btagSF_light_2016preVFP"]
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
    #############
    # Jets
    #############
    "JES_JER_combined": {
        "name": "JES_JER_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["JER", "JES_Absolute_2016", "JES_Absolute", "JES_BBEC1_2016", "JES_BBEC1", 
            "JES_EC2_2016", "JES_EC2", "JES_FlavorQCD", "JES_HF_2016", "JES_HF", "JES_RelativeBal", 
            "JES_RelativeSample_2016"]
    },
    "JER": {
        "name": "JER",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_Absolute_2016": {
        "name": "JES_Absolute_2016",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_Absolute": {
        "name": "JES_Absolute",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_BBEC1_2016": {
        "name": "JES_BBEC1_2016",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_BBEC1": {
        "name": "JES_BBEC1",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_EC2_2016": {
        "name": "JES_EC2_2016",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_EC2": {
        "name": "JES_EC2",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_FlavorQCD": {
        "name": "JES_FlavorQCD",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_HF_2016": {
        "name": "JES_HF_2016",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_HF": {
        "name": "JES_HF",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_RelativeBal": {
        "name": "JES_RelativeBal",
        "type": "shape",
        "kinematic": True,
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_RelativeSample_2016": {
        "name": "JES_RelativeSample_2016",
        "type": "shape",
        "kinematic": True,
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
    "H2ErratumFix_2016preVFP": {
        "name": "h2err",
        "kinematic": False,
        "samples": ["DYtt"],
    },
    "L1 pre-firing corr. Combined": {
        "name": "prefireWeight_combined",
        "type": "group",
        "samples": [],
        "nuisances": ["L1 pre-firing corr. Muon (stat, 2016preVFP)", "L1 pre-firing corr. Muon (syst)", "L1 pre-firing corr. ECAL"]
    },
    "L1 pre-firing corr. Muon": {
        "name": "prefireWeightMuon",
        "kinematic": False,
        "samples": mc_samples,
        "related_nuisances": ["L1 pre-firing corr. Muon (stat, 2016preVFP)", "L1 pre-firing corr. Muon (syst)"]
    },
    "L1 pre-firing corr. ECAL": {
        "name": "prefireWeightECAL",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Trigger SF high pT": { 
        "name": "mu_trig_highpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Trigger SF medium pT": { 
        "name": "mu_trig_lowpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Trigger SF combined": { 
        "name": "mu_trig_combined",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Reco SF high pT": { 
        "name": "mu_reco_highpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Reco SF medium pT": { 
        "name": "mu_reco_lowpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Reco SF combined": { 
        "name": "mu_reco_combined",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon ID SF high pT": { 
        "name": "mu_id_highpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon ID SF medium pT": { 
        "name": "mu_id_lowpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon ID SF combined": { 
        "name": "mu_id_combined",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Iso SF high pT": { 
        "name": "mu_iso_highpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Iso SF medium pT": { 
        "name": "mu_iso_lowpt",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Muon Iso SF combined": { 
        "name": "mu_iso_combined",
        "kinematic": False,
        "samples": mc_samples 
    },
    "Rochester corr.": { 
        "name": "rochester",
        "kinematic": True,
        "samples": all_samples, 
        "related_nuisances": ["Rochester Scale Corrections"]
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
        "related_nuisances": ["btagSF_bc_correlated", "btagSF_bc_2016preVFP",
            "btagSF_light_correlated", "btagSF_light_2016preVFP"] 
    },
    "JES+JER": {
        "name": "JES_JER",
        "kinematic": True,
        "samples": all_samples,
        "related_nuisances": ["JER", "JES_Absolute", "JES_Absolute_2016", "JES_BBEC1", "JES_BBEC1_2016",
            "JES_EC2", "JES_EC2_2016", "JES_FlavorQCD", "JES_HF", "JES_HF_2016", "JES_RelativeBal", 
            "JES_RelativeSample_2016"]
    },
    "Fakes transfer factor": { 
        "name": "fakes",
        "kinematic": False,
        "samples": all_samples,
        "related_nuisances": ["Fakes transfer factor: Fit", "Fakes transfer factor: Model"] 
    },
}
