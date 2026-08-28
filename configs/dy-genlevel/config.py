# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path, interpolate_colors

fw_path = get_fw_path()

year = "Full2018v9"
runner = f"{fw_path}/src/spritz/runners/runner_3DY_btag_method1a.py"

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

lumi = lumis[year]["tot"] / 1000  # All of 2018
plot_label = "DY"
year_label = "2018"
njobs = 100

# ── Method 1a b-tag SF workflow ─────────────────────────────────────────────
# Pass 1: run with collect_btag_eff=True (variations off for speed).
#         Then: python extract_btag_eff.py  →  writes btag_eff_maps/*.npz
# Pass 2: switch to the block below (collect_btag_eff=False, btag_eff_maps set).

# Pass 1 — efficiency collection (uncomment to re-collect)
# special_analysis_cfg = {
#     "do_theory_variations": False,
#     "do_rochester_variations": False,
#     "do_variations": False,
#     "invert_one_isolation_loose": False,
#     "invert_one_isolation_control": False,
#     "skip_genmatching": False,
#     "reweight_fakes": True,
#     "collect_btag_eff": True,
# }

# Pass 2 — apply Method 1a
special_analysis_cfg = {
    "do_theory_variations": False,
    "do_rochester_variations": False,
    "do_variations": False,
    "invert_one_isolation_loose": False,
    "invert_one_isolation_control": False,
    "skip_genmatching": False,
    "reweight_fakes": ["inc_mm_bveto", "inc_mm"],
    "collect_btag_eff": False,
    "btag_eff_maps": {
        "DYll_b":           "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/DYll_b.npz",
        "DYll_c":           "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/DYll_c.npz",
        "DYll_light":       "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/DYll_light.npz",
        "DYtt_b":           "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/DYtt_b.npz",
        "DYtt_c":           "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/DYtt_c.npz",
        "DYtt_light":       "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/DYtt_light.npz",
        "GGToLL":           "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/GGToLL.npz",
        "Single Top":       "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/Single Top.npz",
        "TT":               "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/TT.npz",
        "TTToSemiLeptonic": "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/TTToSemiLeptonic.npz",
        "WW":               "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/WW.npz",
        "WZ":               "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/WZ.npz",
        "ZZ":               "/gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-btag-method1a_mujets_invertedIso/btag_eff_maps/ZZ.npz",
    },
}

bins = {
    "mll": np.linspace(40, 200, 64),
}
# ,{
#         "module": "gendressed_ho_reweight",
#         "file": "/eos/user/g/gboldrin/www/prova/plots_no_AlphaQED/ratio_rebin3.root",
#         "object": "m_ll/NLO_EW_7_no_AlphaQED",
#         "observable": "mll",
#         "name": "NLOEW",
#     }     

hocorr = [{
        "module": "gendressed_ho_reweight",
        "file": "/eos/user/g/gboldrin/www/DYTurbo/kfactor_ewscheme3_3D_N3LO_N3LL_NNLO_NNLL.root",
        "object": "ratio_N3LO+N3LL_over_NNLO+NNLL",
        "observable": "mll",
        "name": "N3LO",
    },{
        "module": "lhe_ho_reweight",
        "file": "/eos/user/g/gboldrin/www/POWHEG_Z_EW/powheg_ew_ratio.root",
        "object": "h_ratio",
        "observable": "mll",
        "name": "NLOEW",
    }       
]

hocorr_dytt = [hocorr[1]]

subsamples_jetflavour = {
    "b":     "ak.any(events.Jet.hadronFlavour == 5, axis=1)",
    "c":     "~ak.any(events.Jet.hadronFlavour == 5, axis=1) & ak.any(events.Jet.hadronFlavour == 4, axis=1)",
    "light": "~ak.any(events.Jet.hadronFlavour == 5, axis=1) & ~ak.any(events.Jet.hadronFlavour == 4, axis=1)",
}

datasets = {
    "DYmm_M-10to50": {
        "files": "DYJetsToMuMu_M-10to50",
        "task_weight": 8,
        "max_weight": 1e9, # filter MC events with extremely large weights
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-50to100": {
        "files": "DYJetsToMuMu",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-100to200": {
        "files": "DYJetsToMuMu_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-200to400": {
        "files": "DYJetsToMuMu_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-400to500": {
        "files": "DYJetsToMuMu_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-500to700": {
        "files": "DYJetsToMuMu_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-700to800": {
        "files": "DYJetsToMuMu_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-800to1000": {
        "files": "DYJetsToMuMu_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-1000to1500": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-1500to2000": {
        "files": "DYJetsToMuMu_M-1500to2000",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYmm_M-2000toInf": {
        "files": "DYJetsToMuMu_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr,
        "subsamples": subsamples_jetflavour,
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9,
        "ho_corrections": hocorr_dytt,
        "subsamples": subsamples_jetflavour,
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
    },
    "WWTo2L2Nu": {
        "files": "WWTo2L2Nu",
        "task_weight": 8,
    },
    "WZ": {
        "files": "WZ_TuneCP5_13TeV-pythia8",
        "task_weight": 8,
    },
    "ZZ": {
        "files": "ZZ_TuneCP5_13TeV-pythia8",
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
    # },
    # "WJetsToLNu_1J": {
    #     "files": "WJetsToLNu_1J",
    #     "task_weight": 8,
    # },
    # "WJetsToLNu_2J": {
    #     "files": "WJetsToLNu_2J",
    #     "task_weight": 8,
    # },
}

# datasets = {
#     "DYmm_M-10to50": {
#         "files": "DYJetsToMuMu_M-10to50",
#         "task_weight": 8,
#         "max_weight": 1e9, # filter MC events with extremely large weights
#         "ho_corrections": [{
#             "module": "lhe_ho_reweight",
#             "file": "/eos/user/g/gboldrin/www/DYTurbo/kfactor.root",
#             "object": "ratio_N3LO+N3LL_over_NNLO",
#             "observable": "mll",
#             "name": "N3LO",
#         }]
#     }
# }


for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


DataRun = [
    ["A", "Run2018A-UL2018-v1"],
    ["B", "Run2018B-UL2018-v1"],
    ["C", "Run2018C-UL2018-v1"],
    ["D", "Run2018D-UL2018-v1"],
]

DataSets = ["SingleMuon"]

DataTrig = {
    "SingleMuon": "events.SingleMu",
}


samples_data = []
for era, sd in DataRun:
    for pd in DataSets:
        tag = pd + "_" + sd

        if "Run2018" in sd and "Muon" in pd:
            tag = tag.replace("v1","GT36")

        datasets[f"{pd}_{era}"] = {
            "files": tag,
            "trigger_sel": DataTrig[pd],
            "read_form": "data",
            "is_data": True,
            "era": f"UL2018{era}",
        }
        samples_data.append(f"{pd}_{era}")


# datasets = {k:v for k,v in datasets.items() if k == "TTToSemiLeptonic"}

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
    "TT": {
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
            "WWTo2L2Nu",
        ]
    },
    "WZ": {
        "samples": [
            "WZ",
        ]
    },
    "ZZ": {
        "samples": [
            "ZZ"
        ]
    },
    "DYtt_b":     {"samples": ["DYtt_b"]},
    "DYtt_c":     {"samples": ["DYtt_c"]},
    "DYtt_light": {"samples": ["DYtt_light"]},
    "DYll_b": {
        "samples": [
            "DYmm_M-10to50_b",
            "DYmm_M-50to100_b",
            "DYmm_M-100to200_b",
            "DYmm_M-200to400_b",
            "DYmm_M-400to500_b",
            "DYmm_M-500to700_b",
            "DYmm_M-700to800_b",
            "DYmm_M-800to1000_b",
            "DYmm_M-1000to1500_b",
            "DYmm_M-1500to2000_b",
            "DYmm_M-2000toInf_b",
        ],
        "is_signal": True
    },
    "DYll_c": {
        "samples": [
            "DYmm_M-10to50_c",
            "DYmm_M-50to100_c",
            "DYmm_M-100to200_c",
            "DYmm_M-200to400_c",
            "DYmm_M-400to500_c",
            "DYmm_M-500to700_c",
            "DYmm_M-700to800_c",
            "DYmm_M-800to1000_c",
            "DYmm_M-1000to1500_c",
            "DYmm_M-1500to2000_c",
            "DYmm_M-2000toInf_c",
        ],
        "is_signal": True
    },
    "DYll_light": {
        "samples": [
            "DYmm_M-10to50_light",
            "DYmm_M-50to100_light",
            "DYmm_M-100to200_light",
            "DYmm_M-200to400_light",
            "DYmm_M-400to500_light",
            "DYmm_M-500to700_light",
            "DYmm_M-700to800_light",
            "DYmm_M-800to1000_light",
            "DYmm_M-1000to1500_light",
            "DYmm_M-1500to2000_light",
            "DYmm_M-2000toInf_light",
        ],
        "is_signal": True
    },
}

# samples = {k:v for k,v in samples.items() if k == "TTToSemiLeptonic"}

colors = {}

extended_petroff = interpolate_colors(cmap_petroff, 11)[::-1]
#colors["W+Jets"] = cmap_petroff[0]
colors["Fakes"] = extended_petroff[0]
colors["GGToLL"] = extended_petroff[1]
colors["Single Top"] = extended_petroff[2]
colors["TT"] = extended_petroff[3]
colors["WW"] = extended_petroff[4]
colors["WZ"] = extended_petroff[5]
colors["ZZ"] = extended_petroff[6]
colors["DYtt_light"] = extended_petroff[7]
colors["DYtt_c"] = cmap_pastel[0]
colors["DYtt_b"] = cmap_pastel[1]
colors["DYll_light"] = extended_petroff[9]
colors["DYll_c"] = cmap_pastel[2]
colors["DYll_b"] = cmap_pastel[3]
colors["TTToSemiLeptonic"] = extended_petroff[10]

# regions

preselections = lambda events: (events.mll > 40)

regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"] & (events.mll < 500),
        "mask": 0
    },
    "inc_mm_bveto": {
        "func": lambda events: preselections(events) & events["mm"] & (events.mll < 500) & events["bVeto"],
        "mask": 0
    },
    "inc_mm_ss": {
        "func": lambda events: preselections(events) & (events.mll < 500) & events["mm_ss"] ,
        "mask": 0
    },
    "inc_mm_bveto_ss": {
        "func": lambda events: preselections(events) & (events.mll < 500) & events["mm_ss"] & events["bVeto"],
        "mask": 0
    }
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
    #############
    # Dilepton
    #############
    "mll": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(64, 40, 200, name="mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV"
    },
    "mll_dense": {
        "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
        "axis": hist.axis.Regular(200, 40, 640, name="mll_dense"),
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
    "double_diff_mll_theta": {
        "axis": [
            hist.axis.Variable([40,60,80,100,120,140,180,220,270,350,500], name="mll"),
            hist.axis.Regular(20,-1, 1, name="costhetastar"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$cos\\,\\theta^{\\ast}$"],
        "unit": ["GeV",""],
        "xlog": True
    },
    "double_diff_mll_yll": {
        "axis": [
            hist.axis.Variable([40,60,80,100,120,140,180,220,270,350,500], name="mll"),
            hist.axis.Regular(15, 0.0, 2.4, name="rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV",""],
        "xlog": True
    },
}

mc_samples = [skey for skey in samples if not samples[skey].get('is_data',False)]

nuisances = {
    "lumi": {
        "name": "lumi",
        "type": "lnN",
        "samples": dict((skey, "1.0084") for skey in mc_samples)
    },
    ## Use the following if you want to apply the automatic combine MC stat nuisances
    "stat": {
        "type": "auto",
        "maxPoiss": "10",
        "includeSignal": "0",
        "samples": {}
    },
    "mu_reco": {
        "name": "mu_reco",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "mu_idiso": {
        "name": "mu_idiso",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "mu_trig": {
        "name": "mu_trig",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "PU": {
        "name": "PU",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "prefireWeight": {
        "name": "prefireWeight",
        "type": "shape",
        "samples": mc_samples,
        "kind": "weight"
    },
    "tt_ptrw": {
        "name": "tt_ptrw",
        "type": "shape",
        "samples": ['TT'],
        "kind": "weight"
    },
    "rochester_stat": {
        "name": "rochester_stat",
        "type": "shape",
        "kind": "stdev",
        "samples": { k: [f"rochester_stat{i}" for i in range(100)] for k in samples},
    },
    "rochester_syst": {
        "name": "rochester_syst",
        "type": "shape",
        "kind": "square",
        "samples": { k: [f"rochester_{set_i}" for set_i in ["set2","set3","set4"]] for k in samples},
    },
    "QCDscale": {
        "name": "QCDScale",
        "type": "shape",
        "kind": "envelope",
        "samples": ({ k: [f"QCDScale_{i}" for i in [0,1,3,4,5,7,8]] for k in ['Single Top', 'TT', 'WW'] }
            | { k: [(f"QCDScale_{2*i}", f"QCDScale_{i}") for i in [0,1,3,4,5,7,8]] for k in ['DYll', 'DYtt'] }),
        "is_theory_unc": True
    },
    "PDFweight": {
        "name": "PDFweight",
        "type": "shape",
        "kind": "square",
        "samples": { k: [f"PDFWeight_{i}" for i in range(101)] for k in ['DYll', 'DYtt', 'Single Top', 'TT', 'WW'] },
        "is_theory_unc": True
    },
    "alphaS": {
        "name": "alphaS",
        "type": "shape",
        "kind": "envelope",
        "samples": { k: [f"PDFWeight_{i}" for i in [101,102]] for k in ['DYll', 'DYtt'] },
        "is_theory_unc": True
    },
    "PSWeight": {
        "name": "PSWeight",
        "type": "shape",
        "kind": "envelope",
        "samples": { k: [f"PSWeight_{i}" for i in range(4)] for k in ['DYll', 'DYtt', 'Single Top', 'TT', 'WW', 'WZ', 'ZZ'] },
        "is_theory_unc": True
    },
}

nuisances = {
    "lumi": {
        "name": "lumi",
        "type": "lnN",
        "samples": dict((skey, "1.0084") for skey in mc_samples)
    }
}
