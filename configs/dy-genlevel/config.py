# ruff: noqa: E501

import json

import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import cmap_pastel, cmap_petroff, get_fw_path

fw_path = get_fw_path()

year = "Full2018v9"
runner = f"{fw_path}/src/spritz/runners/runner_3DY_genlevel.py"

with open(f"{fw_path}/data/common/lumi.json") as file:
    lumis = json.load(file)

lumi = 1. 
plot_label = "DY"
year_label = "2018"
njobs = 500

special_analysis_cfg = {
    "do_theory_variations": True,
    "do_variations": True,
}

bins = {
    "mll": np.linspace(40, 200, 64),
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
}


for dataset in datasets:
    datasets[dataset]["read_form"] = "mc"


samples = {
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
colors["DYll"] = cmap_petroff[9]

# regions
regions = {
    "inc_mm": {
        "func": lambda events: (events["dressed"] | events["lhe"] | events["gen"]),
        "mask": 0
    },
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
    "dressedlep_mll": {
        "func": lambda events: ak.where(events.dressed,
            (events.GenDressedLepton[:, 0] + events.GenDressedLepton[:, 1]).mass,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,
            130,140,150,175,200,225,250,300,350,400,450,500,600,700,800,900,1000,1250,1500,
            1750,2000,2500,3000,4000], name="dressedlep_mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "genlep_mll": {
        "func": lambda events: ak.where(events.gen,
            (events.GenLepton[:, 0] + events.GenLepton[:, 1]).mass,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,
            130,140,150,175,200,225,250,300,350,400,450,500,600,700,800,900,1000,1250,1500,
            1750,2000,2500,3000,4000], name="genlep_mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "lhelep_mll": {
        "func": lambda events: ak.where(events.lhe,
            (events.LHELepton[:, 0] + events.LHELepton[:, 1]).mass,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,
            130,140,150,175,200,225,250,300,350,400,450,500,600,700,800,900,1000,1250,1500,
            1750,2000,2500,3000,4000], name="lhelep_mll"),
        "label": "$m_{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "dressedlep_ptll": {
        "func": lambda events: ak.where(events.dressed,
            (events.GenDressedLepton[:, 0] + events.GenDressedLepton[:, 1]).pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,
            250,300,400,500,700], name="dressedlep_ptll"),
        "label": "$p_{T}^{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "genlep_ptll": {
        "func": lambda events: ak.where(events.gen,
            (events.GenLepton[:, 0] + events.GenLepton[:, 1]).pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,
            250,300,400,500,700], name="genlep_ptll"),
        "label": "$p_{T}^{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "lhelep_ptll": {
        "func": lambda events: ak.where(events.lhe,
            (events.LHELepton[:, 0] + events.LHELepton[:, 1]).pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,
            250,300,400,500,700], name="lhelep_ptll"),
        "label": "$p_{T}^{\\ell\\ell}$",
        "unit": "GeV",
        "xlog": True
    },
    "dressedlep_rapll_abs": {
        "func": lambda events: ak.where(events.dressed,
            abs((events.GenDressedLepton[:, 0] + events.GenDressedLepton[:, 1]).rapidity),
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([.0,.25,.5,.75,1.,1.25,1.5,1.75,2.,2.4], name="dressedlep_rapll_abs"),
        "label": "$|y_{\\ell\\ell}|$"
    },
    "genlep_rapll_abs": {
        "func": lambda events: ak.where(events.gen,
            abs((events.GenLepton[:, 0] + events.GenLepton[:, 1]).rapidity),
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([.0,.25,.5,.75,1.,1.25,1.5,1.75,2.,2.4], name="genlep_rapll_abs"),
        "label": "$|y_{\\ell\\ell}|$"
    },
    "lhelep_rapll_abs": {
        "func": lambda events: ak.where(events.lhe,
            abs((events.LHELepton[:, 0] + events.LHELepton[:, 1]).rapidity),
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([.0,.25,.5,.75,1.,1.25,1.5,1.75,2.,2.4], name="lhelep_rapll_abs"),
        "label": "$|y_{\\ell\\ell}|$"
    },
    "dressedlep_costhetastar": {
        "func": lambda events: ak.where(events.dressed,
            cos_theta_star(events.GenDressedLepton[:, 0], events.GenDressedLepton[:, 1]),
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Regular(50, -1, 1, name="dressedlep_costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$"
    },
    "genlep_costhetastar": {
        "func": lambda events: ak.where(events.gen,
            cos_theta_star(events.GenLepton[:, 0], events.GenLepton[:, 1]),
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Regular(50, -1, 1, name="genlep_costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$"
    },
    "lhelep_costhetastar": {
        "func": lambda events: ak.where(events.lhe,
            cos_theta_star(events.LHELepton[:, 0], events.LHELepton[:, 1]),
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Regular(50, -1, 1, name="lhelep_costhetastar"),
        "label": "$cos\\,\\theta^{\\ast}$"
    },
    #############
    # Single lepton
    #############
    "dressedlep_ptl1": {
        "func": lambda events: ak.where(events.dressed,
            events.GenDressedLepton[:, 0].pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([30,35,40,50,60,80,100,125,150,175,200,250,300,
            400,500,700], name="dressedlep_ptl1"),
        "label": "$p_{T}^{\\ell_{1}}$",
        "unit": "GeV",
        "xlog": True
    },
    "genlep_ptl1": {
        "func": lambda events: ak.where(events.gen,
            events.GenLepton[:, 0].pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([30,35,40,50,60,80,100,125,150,175,200,250,300,
            400,500,700], name="genlep_ptl1"),
        "label": "$p_{T}^{\\ell_{1}}$",
        "unit": "GeV",
        "xlog": True
    },
    "lhelep_ptl1": {
        "func": lambda events: ak.where(events.lhe,
            events.LHELepton[:, 0].pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([30,35,40,50,60,80,100,125,150,175,200,250,300,
            400,500,700], name="lhelep_ptl1"),
        "label": "$p_{T}^{\\ell_{1}}$",
        "unit": "GeV",
        "xlog": True
    },
    "dressedlep_ptl2": {
        "func": lambda events: ak.where(events.dressed,
            events.GenDressedLepton[:, 1].pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([15,20,25,30,35,40,50,60,80,100,125,150,175,200,
            250,300,400,500,700], name="dressedlep_ptl2"),
        "label": "$p_{T}^{\\ell_{2}}$",
        "unit": "GeV",
        "xlog": True
    },
    "genlep_ptl2": {
        "func": lambda events: ak.where(events.gen,
            events.GenLepton[:, 1].pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([15,20,25,30,35,40,50,60,80,100,125,150,175,200,
            250,300,400,500,700], name="genlep_ptl2"),
        "label": "$p_{T}^{\\ell_{2}}$",
        "unit": "GeV",
        "xlog": True
    },
    "lhelep_ptl2": {
        "func": lambda events: ak.where(events.lhe,
            events.LHELepton[:, 1].pt,
            -99*ak.ones_like(events.weight)),
        "axis": hist.axis.Variable([15,20,25,30,35,40,50,60,80,100,125,150,175,200,
            250,300,400,500,700], name="lhelep_ptl2"),
        "label": "$p_{T}^{\\ell_{2}}$",
        "unit": "GeV",
        "xlog": True
    },
    "dressedlep_nLeptons": {
        "func": lambda events: ak.num(events.GenDressedLepton),
        "axis": hist.axis.Regular(8, 0, 8, name="dressedlep_nLeptons"),
        "label": "$N_{leptons}$",
    },
    "genlep_nLeptons": {
        "func": lambda events: ak.num(events.GenLepton),
        "axis": hist.axis.Regular(8, 0, 8, name="genlep_nLeptons"),
        "label": "$N_{leptons}$",
    },
    "lhelep_nLeptons": {
        "func": lambda events: ak.num(events.LHELepton),
        "axis": hist.axis.Regular(8, 0, 8, name="lhelep_nLeptons"),
        "label": "$N_{leptons}$",
    },
    #############
    # Multi-differential
    #############
    "dressedlep_double_diff": {
        "axis": [
            hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,
                130,140,150,175,200,225,250,300,350,400,450,500,600,700,800,900,1000,
                1250,1500,1750,2000,2500,3000,4000], name="dressedlep_mll"),
            hist.axis.Variable([0.0,0.3,0.6,1.2,2.4], name="dressedlep_rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV",""],
        "xlog": True
    },
    "dressedlep_double_diff_coarse": {
        "axis": [
            hist.axis.Variable([40,50,60,70,80,90,100,110,120,130,140,150,175,200,250,
                300,350,400,450,500,600,700,800,1000,1500,2000,2500,4000], name="dressedlep_mll"),
            hist.axis.Variable([0.0,0.3,0.6,1.2,2.4], name="dressedlep_rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV",""],
        "xlog": True
    },
    "genlep_double_diff": {
        "axis": [
            hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,
                130,140,150,175,200,225,250,300,350,400,450,500,600,700,800,900,1000,
                1250,1500,1750,2000,2500,3000,4000], name="genlep_mll"),
            hist.axis.Variable([0.0,0.3,0.6,1.2,2.4], name="genlep_rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV","",""],
        "xlog": True
    },
    "genlep_double_diff_coarse": {
        "axis": [
            hist.axis.Variable([40,50,60,70,80,90,100,110,120,130,140,150,175,200,250,
                300,350,400,450,500,600,700,800,1000,1500,2000,2500,4000], name="genlep_mll"),
            hist.axis.Variable([0.0,0.3,0.6,1.2,2.4], name="genlep_rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV","",""],
        "xlog": True
    },
    "lhelep_double_diff": {
        "axis": [
            hist.axis.Variable([40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,120,
                130,140,150,175,200,225,250,300,350,400,450,500,600,700,800,900,1000,
                1250,1500,1750,2000,2500,3000,4000], name="lhelep_mll"),
            hist.axis.Variable([0.0,0.3,0.6,1.2,2.4], name="lhelep_rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV",""],
        "xlog": True
    },
    "lhelep_double_diff_coarse": {
        "axis": [
            hist.axis.Variable([40,50,60,70,80,90,100,110,120,130,140,150,175,200,250,
                300,350,400,450,500,600,700,800,1000,1500,2000,2500,4000], name="lhelep_mll"),
            hist.axis.Variable([0.0,0.3,0.6,1.2,2.4], name="lhelep_rapll_abs"),
        ],
        "label": ["$m_{\\ell\\ell}$", "$|y_{\\ell\\ell}|$"],
        "unit": ["GeV",""],
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
    "QCDscale": {
        "name": "QCDScale",
        "type": "shape",
        "kind": "envelope",
        "samples": { 'DYll': [f"QCDScale_{2*i}" for i in [0,1,3,4,5,7,8]] },
        "is_theory_unc": True
    },
    "PDFweight": {
        "name": "PDFweight",
        "type": "shape",
        "kind": "square",
        "samples": { 'DYll': [f"PDFWeight_{i}" for i in range(101)] },
        "is_theory_unc": True
    },
    "alphaS": {
        "name": "alphaS",
        "type": "shape",
        "kind": "envelope",
        "samples": { 'DYll': [f"PDFWeight_{i}" for i in [101,102]] },
        "is_theory_unc": True
    },
}
