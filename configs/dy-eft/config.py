import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path

fw_path = get_fw_path()

year = "Full2018v9"
runner = f"{fw_path}/src/spritz/runners/runner_3DY.py"

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
plot_label = "DY"
year_label = "2018"
njobs = 500

special_analysis_cfg = {
    "do_variations": True,
    "do_theory_variations": True, # 116 variations
    "do_rochester_stat_variations": True, # 100 variations
    "do_jet_variations": True, # 24 variations
    "invert_one_isolation_loose": False,
    "invert_one_isolation_control": False,
    "reweight_fakes": True,
}

bins = {
    "mll": np.linspace(40, 200, 64),
}

datasets = {
    "DYmm_M-10to50": {
        "files": "DYJetsToMuMu_M-10to50",
        "task_weight": 8,
        "max_weight": 1e9, # filter MC events with extremely large weights
        "nlo_ew_rwgt": True
    },
    "DYmm_M-50to100": {
        "files": "DYJetsToMuMu",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-100to200": {
        "files": "DYJetsToMuMu_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-200to400": {
        "files": "DYJetsToMuMu_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-400to500": {
        "files": "DYJetsToMuMu_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-500to700": {
        "files": "DYJetsToMuMu_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-700to800": {
        "files": "DYJetsToMuMu_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-800to1000": {
        "files": "DYJetsToMuMu_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-1000to1500": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-1500to2000": {
        "files": "DYJetsToMuMu_M-1500to2000",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYmm_M-2000toInf": {
        "files": "DYJetsToMuMu_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9,
        "nlo_ew_rwgt": True
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
    "GGToMuMu_M-10to30_El-El": {
        "files": "GGToMuMu_M-10to30_El-El",
        "task_weight": 8,
    },
    "GGToMuMu_M-10to30_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-10to30_Inel-El_El-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-10to30_Inel-Inel": {
        "files": "GGToMuMu_M-10to30_Inel-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-30to50_El-El": {
        "files": "GGToMuMu_M-30to50_El-El",
        "task_weight": 8,
    },
    "GGToMuMu_M-30to50_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-30to50_Inel-El_El-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-30to50_Inel-Inel": {
        "files": "GGToMuMu_M-30to50_Inel-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-50to200_El-El": {
        "files": "GGToMuMu_M-50to200_El-El",
        "task_weight": 8,
    },
    "GGToMuMu_M-50to200_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-50to200_Inel-El_El-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-50to200_Inel-Inel": {
        "files": "GGToMuMu_M-50to200_Inel-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-200to1500_El-El": {
        "files": "GGToMuMu_M-200to1500_El-El",
        "task_weight": 8,
    },
    "GGToMuMu_M-200to1500_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-200to1500_Inel-El_El-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-200to1500_Inel-Inel": {
        "files": "GGToMuMu_M-200to1500_Inel-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-1500toInf_El-El": {
        "files": "GGToMuMu_M-1500toInf_El-El",
        "task_weight": 8,
    },
    "GGToMuMu_M-1500toInf_Inel-El_El-Inel": {
        "files": "GGToMuMu_M-1500toInf_Inel-El_El-Inel",
        "task_weight": 8,
    },
    "GGToMuMu_M-1500toInf_Inel-Inel": {
        "files": "GGToMuMu_M-1500toInf_Inel-Inel",
        "task_weight": 8,
    },
    # "WJetsToLNu_0J": {
    #     "files": "WJetsToLNu_0J",
    #     "task_weight": 8,
    #     "skip_genmatching": True,
    # },
    # "WJetsToLNu_1J": {
    #     "files": "WJetsToLNu_1J",
    #     "task_weight": 8,
    #     "skip_genmatching": True,
    # },
    # "WJetsToLNu_2J": {
    #     "files": "WJetsToLNu_2J",
    #     "task_weight": 8,
    #     "skip_genmatching": True,
    # },
}


for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


samples_data = []
for era in ["A", "B", "C", "D"]:
    datasets[f"SingleMuon_{era}"] = {
        "files": f"SingleMuon_Run{year_label}{era}-UL{year_label}-GT36",
        "trigger_sel": "events.SingleMu",
        "read_form": "data",
        "is_data": True,
        "era": f"UL{year_label}{era}"
    }
    samples_data.append(f"SingleMuon_{era}")


samples = {
    "Data": {
        "samples": samples_data,
        "is_data": True,
    },
    # "W+Jets": {
    #     "samples": [
    #         "WJetsToLNu_0J",
    #         "WJetsToLNu_1J",
    #         "WJetsToLNu_2J",
    #    ]
    # },
    "GGToLL": { 
        "samples": [
            "GGToMuMu_M-10to30_El-El",
            "GGToMuMu_M-10to30_Inel-El_El-Inel",
            "GGToMuMu_M-10to30_Inel-Inel",
            "GGToMuMu_M-30to50_El-El",
            "GGToMuMu_M-30to50_Inel-El_El-Inel",
            "GGToMuMu_M-30to50_Inel-Inel",
            "GGToMuMu_M-50to200_El-El",
            "GGToMuMu_M-50to200_Inel-El_El-Inel",
            "GGToMuMu_M-50to200_Inel-Inel",
            "GGToMuMu_M-200to1500_El-El",
            "GGToMuMu_M-200to1500_Inel-El_El-Inel",
            "GGToMuMu_M-200to1500_Inel-Inel",
            "GGToMuMu_M-1500toInf_El-El",
            "GGToMuMu_M-1500toInf_Inel-El_El-Inel",
            "GGToMuMu_M-1500toInf_Inel-Inel",
        ] 
    },
    "Single Top": {
        "samples": [
            "ST_s-channel",
            "ST_t-channel_top_5f",
            "ST_t-channel_antitop_5f",
            "ST_tW_top_noHad",
            "ST_tW_antitop_noHad",
        ]
    },
    "TTTo2L2Nu": {
        "samples": [
            "TTTo2L2Nu",
        ]
    },
    "TTToSemiLeptonic": {
        "samples": [
            "TTToSemiLeptonic"
        ]
    },
    "WW": {
        "samples": [
            "WWTo2L2Nu"
        ]
    },
    "WZ": {
        "samples": [
            "WZTo3LNu",
            "WZTo2Q2L"
        ]
    },
    "ZZ": {
        "samples": [
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
    "DYll": {
        "samples": [
            "DYmm_M-10to50",
            "DYmm_M-50to100",
            "DYmm_M-100to200",
            "DYmm_M-200to400",
            "DYmm_M-400to500",
            "DYmm_M-500to700",
            "DYmm_M-700to800",
            "DYmm_M-800to1000",
            "DYmm_M-1000to1500",
            "DYmm_M-1500to2000",
            "DYmm_M-2000toInf",
        ],
        "is_signal": True
    },
}

colors = {}
colors["W+Jets"] = cmap_pastel[0]
colors["Fakes"] = cmap_petroff[0]
colors["GGToLL"] = cmap_petroff[1]
colors["Single Top"] = cmap_petroff[2]
colors["TTTo2L2Nu"] = cmap_petroff[3]
colors["TTToSemiLeptonic"] = cmap_petroff[4]
colors["WW"] = cmap_petroff[5]
colors["WZ"] = cmap_petroff[6]
colors["ZZ"] = cmap_petroff[7]
colors["DYtt"] = cmap_petroff[8]
colors["DYll"] = cmap_petroff[9]

# regions

preselections = lambda events: (events.mll > 40) & (events.mll < 500)

regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events.mm,
        "mask": 0
    },
    "inc_mm_ss": {
        "func": lambda events: preselections(events) & events.mm_ss,
        "mask": 0
    },
    "bveto_mm": {
        "func": lambda events: preselections(events) & events.mm & events.bveto,
        "mask": 0
    },
    "bveto_mm_ss": {
        "func": lambda events: preselections(events) & events.mm_ss & events.bveto,
        "mask": 0
    },
    # "btag_mm": {
    #     "func": lambda events: preselections(events) & events.mm & events.btag,
    #     "mask": 0
    # },
    # "btag_mm_ss": {
    #     "func": lambda events: preselections(events) & events.mm_ss & events.btag,
    #     "mask": 0
    # },
}

def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr/abs(nr)
    return 2*get_sign((l1+l2).pz)/(l1+l2).mass * get_sign(l1.pdgId)*(l2.pz*l1.energy-l1.pz*l2.energy)/np.sqrt(((l1+l2).mass)**2+((l1+l2).pt)**2)

def transverse_mass(l, nu):
    return np.sqrt(2*l.pt*nu.pt*(1-np.cos(l.phi-nu.phi)))

def iso_transverse_mass(l1, l2, nu):
    return ak.where(
        l1.pfRelIso04_all < l2.pfRelIso04_all,
        transverse_mass(l1, nu),
        transverse_mass(l2, nu)
    )

variables = {
    "nPVs": {
        "func": lambda events: events.PV.npvs,
        "axis": hist.axis.Regular(80, 0, 80, name="nPVs"),
        "label": "$N_{PVs}$",
    },
    #############
    # Dilepton
    #############
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(64, 40, 200, name="mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV"
    },
    "mll_medium": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,
            130,140,150,160,170,180,190,200,210,220,230,240,255,270,285,300,325,350,
            375,400,450,500], name="mll_medium"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "axis": hist.axis.Regular(40, 0, 400, name="ptll"),
        "label": "$p_{T}^{\\ell\\ell}$",
        "unit": "GeV"
    },
    "costhetastar": {
        "func": lambda events: cos_theta_star(events.Lepton[:, 0], events.Lepton[:, 1]),
        "axis": hist.axis.Regular(50, -1, 1, name="costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$"
    },
    "rapll_abs": {
        "func": lambda events: abs((events.Lepton[:, 0] + events.Lepton[:, 1]).rapidity),
        "axis": hist.axis.Regular(50, 0, 2.5, name="rapll_abs"),
        "label": "$|y_{\\ell\\ell}|$"
    },
    #############
    # Single lepton
    #############
    "ptl1": {
        "func": lambda events: events.Lepton[:, 0].pt,
        "axis": hist.axis.Regular(50, 30, 280, name="ptl1"),
        "label": "$p_{T}^{\\ell_{1}}$",
        "unit": "GeV"
    },
    "etal1": {
        "func": lambda events: events.Lepton[:, 0].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal1"),
        "label": "$\\eta_{\\ell_{1}}$"
    },
    "ptl2": {
        "func": lambda events: events.Lepton[:, 1].pt,
        "axis": hist.axis.Regular(50, 15, 165, name="ptl2"),
        "label": "$p_{T}^{\\ell_{2}}$",
        "unit": "GeV"
    },
    "etal2": {
        "func": lambda events: events.Lepton[:, 1].eta,
        "axis": hist.axis.Regular(50, -2.5, 2.5, name="etal2"),
        "label": "$\\eta_{\\ell_{2}}$"
    },
    #############
    # Jets
    #############
    "max_btag": {
        "func": lambda events: events.btagDeepFlavB_max,
        "axis": hist.axis.Regular(20, 0, 1, name="max_btag"),
        "label": "max_btag",
    },
    "has_btag": {
        "func": lambda events: ak.num(events.BJet) >= 1,
        "axis": hist.axis.Regular(2, 0, 2, name="has_btag"),
        "label": "has_btag",
    },
    "nbtag": {
        "func": lambda events: ak.num(events.BJet),
        "axis": hist.axis.Regular(4, 0, 4, name="nbtag"),
        "label": "nbtag",
    },
    #############
    # Multi-differential
    #############
    "triple_diff": {
        "axis": [
            hist.axis.Variable([40,60,80,100,120,140,180,220,270,350,500], name="mll"),
            hist.axis.Variable([-1.0,-0.6,-0.2,0.2,0.6,1.0], name="costhetastar"),
            hist.axis.Variable([0.0,0.48,0.96,1.44,2.4], name="rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV","",""],
        "xlog": True
    },
}

mc_samples = [skey for skey in samples if not samples[skey].get('is_data',False)]

nuisances = {
    "lumi": {
        "name": "lumi",
        "type": "lnN",
        "samples": dict((skey, lumi_unc) for skey in mc_samples)
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
        "samples": mc_samples,
        "kind": "weight"
    },
    "L1 pre-firing corr.": {
        "name": "prefireWeight",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    #############
    # Leptons
    #############
    "Trigger SF": {
        "name": "mu_trig",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Reconstruction SF": {
        "name": "mu_reco",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon ID SF": {
        "name": "mu_id",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "Muon Isolation SF": {
        "name": "mu_iso",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "Rochester corr. (stat)": {
        "name": "rochester_stat",
        "type": "shape",
        "kind": "stdev",
        "samples": samples,
        "variations": [
            {"label": f"Rochester stat. repl. {i}", "tag": f"rochester_stat{i}"} for i in range(100)
        ]
    },
    "Rochester corr. (syst)": {
        "name": "rochester_syst",
        "type": "shape",
        "kind": "square",
        "samples": samples,
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
        "name": "nlo",
        "type": "shape",
        "samples": ["DYll", "DYtt"],
        "kind": "weight"
    },
    "Top $p_{T}$ corr.": {
        "name": "tt_ptrw",
        "type": "shape",
        "samples": ["TTTo2L2Nu", "TTToSemiLeptonic"],
        "kind": "weight"
    },
    "QCD scale": {
        "name": "QCDScale",
        "type": "shape",
        "kind": "envelope",
        "samples": ["DYll", "DYtt", "Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZ", "ZZ"],
        "variations": [
            {   "label": "$\\mu_{R}=0.5, \\mu_{F}=0.5$", 
                "tag": {
                    k: "QCDScale_0" for k in ["Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZTo3LNu", "ZZTo4L", "ZZTo2L2Nu"]} | {
                    k: "QCDScale_0" for k in ["WZTo2Q2L", "ZZTo2Q2L"] } | {
                    k: "QCDScale_0" for k in ["DYll", "DYtt"] }},
            {   "label": "$\\mu_{R}=0.5, \\mu_{F}=1$",
                "tag": {
                    k: "QCDScale_1" for k in ["Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZTo3LNu", "ZZTo4L", "ZZTo2L2Nu"]} | {
                    k: "QCDScale_1" for k in ["WZTo2Q2L", "ZZTo2Q2L"] } | {
                    k: "QCDScale_2" for k in ["DYll", "DYtt"] }},
            {   "label": "$\\mu_{R}=1, \\mu_{F}=0.5$",
                "tag": {
                    k: "QCDScale_3" for k in ["Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZTo3LNu", "ZZTo4L", "ZZTo2L2Nu"]} | {
                    k: "QCDScale_3" for k in ["WZTo2Q2L", "ZZTo2Q2L"] } | {
                    k: "QCDScale_6" for k in ["DYll", "DYtt"] }},
            {   "label": "$\\mu_{R}=1, \\mu_{F}=2$",
                "tag": {
                    k: "QCDScale_5" for k in ["Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZTo3LNu", "ZZTo4L", "ZZTo2L2Nu"]} | {
                    k: "QCDScale_4" for k in ["WZTo2Q2L", "ZZTo2Q2L"]} | {
                    k: "QCDScale_10" for k in ["DYll", "DYtt"] }},
            {   "label": "$\\mu_{R}=2, \\mu_{F}=1$",
                "tag": {
                    k: "QCDScale_7" for k in ["Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZTo3LNu", "ZZTo4L", "ZZTo2L2Nu"]} | {
                    k: "QCDScale_6" for k in ["WZTo2Q2L", "ZZTo2Q2L"] } | {
                    k: "QCDScale_14" for k in ["DYll", "DYtt"] }},
            {   "label": "$\\mu_{R}=2, \\mu_{F}=2$",
                "tag": {
                    k: "QCDScale_8" for k in ["Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZTo3LNu", "ZZTo4L", "ZZTo2L2Nu"]} | {
                    k: "QCDScale_7" for k in ["WZTo2Q2L", "ZZTo2Q2L"] } | {
                    k: "QCDScale_16" for k in ["DYll", "DYtt"] }},
        ]
    },
    "PDF": {
        "name": "PDFWeight",
        "type": "shape",
        "kind": "square",
        "samples": ["DYll", "DYtt", "Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZ", "ZZ"],
        "variations": [
            {"label": f"PDF Hessian set {i}", "tag": f"PDFWeight_{i}"} for i in range(1,101)
        ]
    },
    "$\\alpha_{S}$": {
        "name": "alphaS",
        "type": "shape",
        "kind": "envelope",
        "samples": ["DYll", "DYtt", "TTTo2L2Nu", "TTToSemiLeptonic", "ZZ", "WZ"],
        "variations": [
            {   "label": "$\\alpha_{S} = 0.116$",
                "tag": {k: "PDFWeight_101" for k in ["DYll", "DYtt", "TTTo2L2Nu", "TTToSemiLeptonic", "ZZTo4L", "ZZTo2Q2L", "WZ"]} },
            {   "label": "$\\alpha_{S} = 0.120$",
                "tag": {k: "PDFWeight_102" for k in ["DYll", "DYtt", "TTTo2L2Nu", "TTToSemiLeptonic", "ZZTo4L", "ZZTo2Q2L", "WZ"]} }
        ]
    },
    "Parton shower": {
        "name": "PSWeight",
        "type": "shape",
        "kind": "envelope",
        "samples": ["DYll", "DYtt", "Single Top", "TTTo2L2Nu", "TTToSemiLeptonic", "WW", "WZ", "ZZ"],
        "variations": [
            {"label": "ISR=2, FSR=1", "tag": "PSWeight_0"},
            {"label": "ISR=1, FSR=2", "tag": "PSWeight_1"},
            {"label": "ISR=0.5, FSR=1", "tag": "PSWeight_2"},
            {"label": "ISR=1, FSR=0.5", "tag": "PSWeight_3"} 
        ]
    },
    #############
    # Jets
    #############
    "JER": {
        "name": "JER",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_Absolute_2018": {
        "name": "JES_Absolute_2018",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_Absolute": {
        "name": "JES_Absolute",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_BBEC1_2018": {
        "name": "JES_BBEC1_2018",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_BBEC1": {
        "name": "JES_BBEC1",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_EC2_2018": {
        "name": "JES_EC2_2018",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_EC2": {
        "name": "JES_EC2",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_FlavorQCD": {
        "name": "JES_FlavorQCD",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_HF_2018": {
        "name": "JES_HF_2018",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_HF": {
        "name": "JES_HF",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_RelativeBal": {
        "name": "JES_RelativeBal",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "JES_RelativeSample_2018": {
        "name": "JES_RelativeSample_2018",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    #############
    # b-tagging
    #############
    "btag_SF": {
        "name": "btagSF_sf",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "btag_Eff": {
        "name": "btagSF_eff",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    #############
    # Fakes
    #############
    "fakes_param": {
        "name": "fakes_param",
        "type": "shape",
        "kind": "weight",
        "samples": samples,
    },
    "fakes_model": {
        "name": "fakes_model",
        "type": "shape",
        "kind": "envelope",
        "samples": samples,
        "variations": [
            {"label": "fakes_model", "tag": "fakes_model"}
        ],
    },
}

corrections = {
    "Pile-up corr.": { 
        "name": "PU",
        "samples": mc_samples 
    },
    "L1 pre-firing corr.": {
        "name": "prefireWeight",
        "samples": mc_samples 
    },
    "Trigger SF": { 
        "name": "mu_trig",
        "samples": mc_samples 
    },
    "Muon Reconstruction SF": { 
        "name": "mu_reco",
        "samples": mc_samples 
    },
    "Muon ID SF": { 
        "name": "mu_id",
        "samples": mc_samples 
    },
    "Muon Isolation SF": { 
        "name": "mu_iso",
        "samples": mc_samples 
    },
    "Rochester corr.": { 
        "name": "rochester",
        "samples": [skey for skey in samples], 
        "related_nuisances": ["Rochester corr. (stat)", "Rochester corr. (syst)"] 
    },
    "NLO EW correction": { 
        "name": "nlo",
        "samples": ["DYll", "DYtt"] 
    },
    "Top $p_{T}$ corr.": { 
        "name": "tt_ptrw",
        "samples": ["TTTo2L2Nu", "TTToSemiLeptonic"] 
    },
    "btag SF": { 
        "name": "btagSF",
        "samples": mc_samples 
    },
}
