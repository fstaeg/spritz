import json
import awkward as ak
import numpy as np
from scipy.interpolate import UnivariateSpline

def nlo_ew_reweight(events, variations, cfg):
    # LHELeptons
    lhe_ele_mask = (abs(events.LHEPart.pdgId) == 11)
    lhe_mu_mask = (abs(events.LHEPart.pdgId) == 13)
    lhe_tau_mask = (abs(events.LHEPart.pdgId) == 15)
    lhe_lep_mask = (lhe_ele_mask | lhe_mu_mask | lhe_tau_mask)

    lhe_leptons = events.LHEPart[lhe_lep_mask]
    lhe_mll = (lhe_leptons[:, 0] + lhe_leptons[:, 1]).mass

    # GenDressedLeptons
    gen_ele_mask = (abs(events.GenDressedLepton.pdgId) == 11)
    gen_mu_mask = (abs(events.GenDressedLepton.pdgId) == 13)
    gen_tau_mask = (abs(events.GenDressedLepton.pdgId) == 15)
    gen_lep_mask = (gen_ele_mask | gen_mu_mask | gen_tau_mask)
    
    gen_leptons = events.GenDressedLepton[gen_lep_mask]
    gen_leptons = ak.pad_none(gen_leptons, 2)
    gen_mll = (gen_leptons[:, 0] + gen_leptons[:, 1]).mass

    # Is there pair of GenDressedLeptons with same flavour as the LHELeptons?
    elepair_mask = (
        (ak.num(events.GenDressedLepton[gen_ele_mask])==2) &
        (ak.num(events.LHEPart[lhe_ele_mask])==2)
    )
    mupair_mask = (
        (ak.num(events.GenDressedLepton[gen_mu_mask])==2) &
        (ak.num(events.LHEPart[lhe_mu_mask])==2)
    )
    taupair_mask = (
        (ak.num(events.GenDressedLepton[gen_tau_mask])==2) &
        (ak.num(events.LHEPart[lhe_tau_mask])==2)
    )
    genpair_mask = (elepair_mask | mupair_mask | taupair_mask)
    
    mll = ak.where(genpair_mask, gen_mll, lhe_mll)

    # Reweight
    with open(cfg["nloewRW"], "r") as f:
        nloew_rw = json.load(f)
        x = np.array(nloew_rw["x"])
        y = np.array(nloew_rw["y"])
        yerr = np.array(nloew_rw["yerr"])

    spline_nom = UnivariateSpline(x, y, s=0)
    spline_up = UnivariateSpline(x, y+yerr, s=0)
    spline_down = UnivariateSpline(x, y-yerr, s=0)

    events["ewNloWeight"] = spline_nom(mll)
    events["ewNloWeight_nlo_up"] = spline_up(mll)
    events["ewNloWeight_nlo_down"] = spline_down(mll)

    variations.register_variation(["ewNloWeight"], "nlo_up")
    variations.register_variation(["ewNloWeight"], "nlo_down")

    events["ewNloWeight_nlo_before"] = ak.ones_like(events.weight)
    variations.register_variation(["ewNloWeight"], "nlo_before")

    return events, variations
