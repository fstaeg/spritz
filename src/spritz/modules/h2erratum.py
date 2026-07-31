import json
import awkward as ak
import numpy as np
from scipy.interpolate import UnivariateSpline

def h2erratum_reweight(events, variations, cfg, dataset):
    # LHELeptons
    ele_mask = (abs(events.LHEPart.pdgId) == 11)
    mu_mask = (abs(events.LHEPart.pdgId) == 13)
    tau_mask = (abs(events.LHEPart.pdgId) == 15)
    lep_mask = (ele_mask | mu_mask | tau_mask)

    leptons = events.LHEPart[lep_mask]
    ptll = (leptons[:, 0] + leptons[:, 1]).pt

    # Reweight
    with open(cfg["h2erratumRW"], "r") as f:
        h2erratumRW = json.load(f)

    process = "DYtt" if "DYtt" in dataset else "DYmm"
    
    edges = np.array(h2erratumRW[process]["edges"])
    nominal = np.array(h2erratumRW[process]["nominal"])
    err = np.array(h2erratumRW[process]["err"])

    ptll_bins = np.digitize(ptll, edges) - 1
    ptll_bins = np.clip(ptll_bins, 0, len(nominal)-1)

    weights = nominal[ptll_bins]
    weights_err = err[ptll_bins]

    events["H2ErratumWeight"] = weights
    events["H2ErratumWeight_h2err_up"] = weights+weights_err
    events["H2ErratumWeight_h2err_down"] = weights-weights_err
    events["H2ErratumWeight_h2err_before"] = ak.ones_like(events.weight)

    variations.register_variation(["H2ErratumWeight"], "h2err_up")
    variations.register_variation(["H2ErratumWeight"], "h2err_down")
    variations.register_variation(["H2ErratumWeight"], "h2err_before")

    return events, variations
