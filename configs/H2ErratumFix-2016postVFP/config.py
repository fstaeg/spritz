import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path

fw_path = get_fw_path()

year = "Full2016v9noHIPM"
runner = f"{fw_path}/src/spritz/runners/runner_H2ErratumFix.py"

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

lumi = lumis[year]["tot"] / 1000
lumi_unc = lumis[year]["rel_unc"]
plot_label = "2016postVFP"
year_label = "2016"
njobs = 200

special_analysis_cfg = {
    "do_variations": False,
}

datasets = {
    "DYmm": {
        "files": "DYJetsToMuMu",
        "task_weight": 8,
        "max_weight": 1e7,
    },
    "DYmm_noH2ErratumFix": {
        "files": "DYJetsToMuMu_noH2ErratumFix",
        "task_weight": 8,
        "max_weight": 1e7,
    },
    "DYtt": {
        "files": "DYJetsToTauTau",
        "task_weight": 8,
        "max_weight": 1e7,
    },
    "DYtt_noH2ErratumFix": {
        "files": "DYJetsToTauTau_noH2ErratumFix",
        "task_weight": 8,
        "max_weight": 1e7,
    },
}

for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"

samples = {
    "DYtt": {
        "samples": [
            "DYtt"
        ],
    },
    "DYtt_noH2ErratumFix": {
        "samples": [
            "DYtt_noH2ErratumFix"
        ]
    },
    "DYll": {
        "samples": [
            "DYmm",
        ],
    },
    "DYll_noH2ErratumFix": {
        "samples": [
            "DYmm_noH2ErratumFix",
        ],
    },
}

colors = {}
colors["DYtt"] = "cyan"
colors["DYll"] = "magenta"
colors["DYtt_noH2ErratumFix"] = "blue"
colors["DYll_noH2ErratumFix"] = "red"


# regions

regions = {
    "inc": {
        "func": lambda events: (50 < events.mll),
        "mask": 0
    },
}

def cos_theta_star(l1, l2):
    get_sign = lambda nr: nr/abs(nr)
    return 2*get_sign((l1+l2).pz)/(l1+l2).mass * get_sign(l1.pdgId)*(l2.pz*l1.energy-l1.pz*l2.energy)/np.sqrt(((l1+l2).mass)**2+((l1+l2).pt)**2)


variables = {
    #############
    # LHE dileptons
    #############
    "mll": {
        "func": lambda events: (events.LHELepton[:, 0] + events.LHELepton[:, 1]).mass,
        "axis": hist.axis.Regular(10, 50, 100, name="mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV"
    },
    "ptll": {
        "func": lambda events: (events.LHELepton[:, 0] + events.LHELepton[:, 1]).pt,
        "axis": hist.axis.Variable([0,1,2,3,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,60,
            80,100,125,150,200,300,1000], name="ptll"),
        "label": "$p_{T}^{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "costhetastar": {
        "func": lambda events: cos_theta_star(events.LHELepton[:, 0], events.LHELepton[:, 1]),
        "axis": hist.axis.Regular(50, -1, 1, name="costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$"
    },
    "rapll_abs": {
        "func": lambda events: abs((events.LHELepton[:, 0] + events.LHELepton[:, 1]).rapidity),
        "axis": hist.axis.Regular(50, 0, 2.5, name="rapll_abs"),
        "label": "$|y_{\\ell\\ell}|$"
    },
}

nuisances = {
}

corrections = {
}
