import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path

fw_path = get_fw_path()

year = "Full2018v9"
runner = f"{fw_path}/src/spritz/runners/runner_3DY_btageff.py"

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

lumi = lumis[year]["tot"] / 1000  # All of 2018
plot_label = "2018"
year_label = "2018"
njobs = 600

special_analysis_cfg = {
}

datasets = {
    "DYmm_M-10to50": {
        "files": "DYJetsToMuMu_M-10to50",
        "task_weight": 8,
        "max_weight": 1e9 # filter MC events with extremely large weights
    },
    "DYmm_M-50to100": {
        "files": "DYJetsToMuMu",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-100to200": {
        "files": "DYJetsToMuMu_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-200to400": {
        "files": "DYJetsToMuMu_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-400to500": {
        "files": "DYJetsToMuMu_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-500to700": {
        "files": "DYJetsToMuMu_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-700to800": {
        "files": "DYJetsToMuMu_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-800to1000": {
        "files": "DYJetsToMuMu_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-1000to1500": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-1500to2000": {
        "files": "DYJetsToMuMu_M-1500to2000",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYmm_M-2000toInf": {
        "files": "DYJetsToMuMu_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e9
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
    },
    "TTToSemiLeptonic": {
        "files": "TTToSemiLeptonic",
        "task_weight": 8,
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
    }
}


for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


samples = {
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
        ]
    },
}

samples["Inclusive"] = {
    "samples": [s for process in samples for s in samples[process]["samples"]]
}

# regions

preselections = lambda events: (40 < events.mll) & (events.mll < 500)

regions = {
    "inc_mm": {
        "func": lambda events: preselections(events) & events["mm"],
        "mask": 0
    },
}

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
    #############
    # Light Jets
    #############
    "ptlj": {
        "func": lambda events: events.Ljet.pt,
        "axis": hist.axis.Variable([50,70,100,140,200,300,600,1000], name="ptlj"),
        "label": "$p_{T}^{light jet}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptlj_btagloose": {
        "func": lambda events: events.LjetBtagLoose.pt,
        "axis": hist.axis.Variable([50,70,100,140,200,300,600,1000], name="ptlj_btagloose"),
        "label": "$p_{T}^{light jet(looseBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptlj_btagmedium": {
        "func": lambda events: events.LjetBtagMedium.pt,
        "axis": hist.axis.Variable([50,70,100,140,200,300,600,1000], name="ptlj_btagmedium"),
        "label": "$p_{T}^{light jet(mediumBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptlj_btagtight": {
        "func": lambda events: events.LjetBtagTight.pt,
        "axis": hist.axis.Variable([50,70,100,140,200,300,600,1000], name="ptlj_btagtight"),
        "label": "$p_{T}^{light jet(tightBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    #############
    # C Jets
    #############
    "ptcj": {
        "func": lambda events: events.Cjet.pt,
        "axis": hist.axis.Variable([50,70,100,200,1000], name="ptcj"),
        "label": "$p_{T}^{c jet}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptcj_btagloose": {
        "func": lambda events: events.CjetBtagLoose.pt,
        "axis": hist.axis.Variable([50,70,100,200,1000], name="ptcj_btagloose"),
        "label": "$p_{T}^{c jet(looseBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptcj_btagmedium": {
        "func": lambda events: events.CjetBtagMedium.pt,
        "axis": hist.axis.Variable([50,70,100,200,1000], name="ptcj_btagmedium"),
        "label": "$p_{T}^{c jet(mediumBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptcj_btagtight": {
        "func": lambda events: events.CjetBtagTight.pt,
        "axis": hist.axis.Variable([50,70,100,200,1000], name="ptcj_btagtight"),
        "label": "$p_{T}^{c jet(tightBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    #############
    # B Jets
    #############
    "ptbj": {
        "func": lambda events: events.Bjet.pt,
        "axis": hist.axis.Variable([50,70,100,200,600,1000], name="ptbj"),
        "label": "$p_{T}^{b jet}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptbj_btagloose": {
        "func": lambda events: events.BjetBtagLoose.pt,
        "axis": hist.axis.Variable([50,70,100,200,600,1000], name="ptbj_btagloose"),
        "label": "$p_{T}^{b jet(looseBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptbj_btagmedium": {
        "func": lambda events: events.BjetBtagMedium.pt,
        "axis": hist.axis.Variable([50,70,100,200,600,1000], name="ptbj_btagmedium"),
        "label": "$p_{T}^{b jet(mediumBtag)}$",
        "unit": "GeV",
        "xlog": True
    },
    "ptbj_btagtight": {
        "func": lambda events: events.BjetBtagTight.pt,
        "axis": hist.axis.Variable([50,70,100,200,600,1000], name="ptbj_btagtight"),
        "label": "$p_{T}^{b jet(tightBtag)}$",
        "unit": "GeV",
        "xlog": True 
    }
}

mc_samples = [skey for skey in samples if not samples[skey].get('is_data',False)]

nuisances = {
    ## Use the following if you want to apply the automatic combine MC stat nuisances
    "stat": {
        "type": "auto",
        "maxPoiss": "10",
        "includeSignal": "0",
        "samples": {}
    },
}

corrections = {
}
