# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path, interpolate_colors

fw_path = get_fw_path()

year = "Full2018v9"
runner = f"{fw_path}/src/spritz/runners/runner_3DY_genZ.py"

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

lumi = 1.0
plot_label = "DY GenPart"
year_label = "2018"
njobs = 100

# ── DYTurbo 3D kfactor axes (exact binning for direct comparison) ───────────
# Axis 0 (qT):  [0,10,20,...,100,150,200,300,400] GeV  → 14 bins
# Axis 1 (yll): [-4,-1,-0.5,0,0.5,1,4]               → 6 bins (signed)
# Axis 2 (mll): [30,50,80,90,100,130,200,300,500,800,1000,1500,3000] GeV → 12 bins

_edges_qT  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,75,100,150,200,250,300,400,500,600] 
_edges_yll = [ -3.0, -2.0, -1.0 , 0.0, 1.0, 2.0, 3.0 ]
_edges_mll = [ 30,120,200,600,1000,2000,3500 ]

# arne 

_edges_qT  = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 65, 70, 80, 90, 100]
_edges_yll = [ -5.0, -4.0, -3.5, -3.25, -3.0, -2.75, -2.5, -2.0, -1.8, -1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0 ]

_edges_mll = [ 60, 120 ]




_edges_qT  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,75,100,150,200,250,300,400,500,600] 
_edges_yll = [ -3.0, -2.0, -1.0 , 0.0, 1.0, 2.0, 3.0 ]
_edges_mll = [30, 50, 120, 200, 600, 1000, 2000, 3500 ]

datasets = {
    "DYmm_M-10to50": {
        "files": "DYJetsToMuMu_M-10to50",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-50to100": {
        "files": "DYJetsToMuMu",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-100to200": {
        "files": "DYJetsToMuMu_M-100to200",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-200to400": {
        "files": "DYJetsToMuMu_M-200to400",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-400to500": {
        "files": "DYJetsToMuMu_M-400to500",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-500to700": {
        "files": "DYJetsToMuMu_M-500to700",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-700to800": {
        "files": "DYJetsToMuMu_M-700to800",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-800to1000": {
        "files": "DYJetsToMuMu_M-800to1000",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-1000to1500": {
        "files": "DYJetsToMuMu_M-1000to1500",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-1500to2000": {
        "files": "DYJetsToMuMu_M-1500to2000",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
    "DYmm_M-2000toInf": {
        "files": "DYJetsToMuMu_M-2000toInf",
        "task_weight": 8,
        "max_weight": 1e9,
        "read_form": "mc",
    },
}

samples = {
    "DYll": {
        "samples": list(datasets.keys()),
        "is_signal": True,
    },
}

colors = {
    "DYll": cmap_petroff[0],
}

# Single inclusive region at gen level: require mll > 30 GeV
# (matching the lowest edge of the DYTurbo kfactor histogram)

# regions = {
#     "gen_inclusive": {
#         "func": lambda events: (events.gen_mll > 60) & (events.gen_mll < 120) & (events.gen_qT < 100) & (events.gen_yll < 5) & (events.gen_yll > -5),
#         "mask": 0,
#     },
# }

regions = {
    "gen_inclusive": {
        "func": lambda events: (events.gen_mll > 30) & (events.gen_mll < 3500) & (events.gen_qT < 600) & (events.gen_yll < 3) & (events.gen_yll > -3),
        "mask": 0,
    },
}

variables = {
    # ── 1D projections ──────────────────────────────────────────────────────
    "gen_qT": {
        "axis": hist.axis.Variable(_edges_qT, name="gen_qT"),
        "label": "$q_{T}^{\\mathrm{gen}}$",
        "unit": "GeV",
    },
    "gen_yll": {
        "axis": hist.axis.Variable(_edges_yll, name="gen_yll"),
        "label": "$y_{\\ell\\ell}^{\\mathrm{gen}}$",
    },
    "gen_mll": {
        "axis": hist.axis.Variable(_edges_mll, name="gen_mll"),
        "label": "$m_{\\ell\\ell}^{\\mathrm{gen}}$",
        "unit": "GeV",
    },
    # ── 3D distribution matching DYTurbo kfactor binning ────────────────────
    "gen_3D": {
        "axis": [
            hist.axis.Variable(_edges_qT,  name="gen_qT"),
            hist.axis.Variable(_edges_yll, name="gen_yll"),
            hist.axis.Variable(_edges_mll, name="gen_mll"),
        ],
        "label": ["$q_{T}$", "$y_{\\ell\\ell}$", "$m_{\\ell\\ell}$"],
        "unit": ["GeV", "", "GeV"],
    },
}

nuisances = {}
