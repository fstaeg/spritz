import awkward as ak
import numpy as np
from spritz.framework.variation import Variation
from spritz.framework.framework import correctionlib_wrapper
from spritz.modules.rochester import rochester_corrections


def vary_rochester(events, is_data, rochester, s_nom=5, do_stat=False, scale_only=False):
    corr = lambda s, m: rochester_corrections(events, is_data, rochester, s, m, scale_only)
    pt_rochester = {}
    
    sfs = {f"set{s}": corr(s, 0) for s in [0, 2, 3, 4, 5]}
    if do_stat:
        sfs["set1"] = [corr(1, i) for i in range(100)]

    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    mu_mask = abs(events.Lepton.pdgId) == 13
    
    for s in [0, 2, 3, 4, 5]:
        if s == s_nom: continue
        muSF = sfs[f"set{s}"] / sfs["set0"] * sfs[f"set{s_nom}"]
        pt_rochester[f"set{s}"] = (muSF * events.Muon.pt)[mu_idx]

    if do_stat:
        for i in range(100):
            muSF = sfs["set1"][i] / sfs["set0"] * sfs[f"set{s_nom}"]
            pt_rochester[f"stat{i}"] = (muSF * events.Muon.pt)[mu_idx]

    return pt_rochester


def correct_rochester(events, is_data, rochester, s=5, m=0, scale_only=False):
    pt_rochester = {}

    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    muSF = rochester_corrections(events, is_data, rochester, s, m, scale_only=scale_only)
    pt_rochester["nominal"] = (muSF * events.Muon.pt)[mu_idx]

    if scale_only and not is_data:
        muSF_resolution = rochester_corrections(events, is_data, rochester, s, m, scale_only=False)
        pt_rochester["resolution"] = (muSF_resolution * events.Muon.pt)[mu_idx]

    return pt_rochester


def correct_highpt(events, clib_wrap_scale, clib_wrap_resolution):
    leptons = ak.copy(events.Lepton)
    mu_mask = abs(leptons.pdgId) == 13

    ### Scale correction (General Endpoint method)
    kappa = {
        "nom": 1e-3 * ak.where(mu_mask, clib_wrap_scale(leptons.phi, leptons.eta, "nominal"), 0.),
        "syst": 1e-3 * ak.where(mu_mask, clib_wrap_scale(leptons.phi, leptons.eta, "syst"), 0.)
    }

    # calculate corrected curvature
    charge = leptons.pdgId / (-abs(leptons.pdgId))
    curvature = charge / leptons.pt
    curvature_corr = curvature + kappa["nom"]
    curvature_up = curvature + kappa["syst"]
    curvature_down = curvature - kappa["syst"]

    ### Resolution correction
    sigma = {
        "nom": ak.where(mu_mask, clib_wrap_resolution(leptons.p, leptons.eta, "nominal"), 0.),
        "syst": ak.where(mu_mask, clib_wrap_resolution(leptons.p, leptons.eta, "syst"), 0.)
    }

    # random number
    counts = ak.num(leptons.pt)
    rng = np.random.default_rng(seed=0)
    psmear = {
        "nom": ak.unflatten(rng.normal(1., ak.flatten(sigma["nom"]), size=ak.sum(counts)), counts),
        "syst": ak.unflatten(rng.normal(1., ak.flatten(sigma["syst"]), size=ak.sum(counts)), counts)
    }

    # calculate corrected and smeared pt
    pt_scale_and_resolution = {
        "scale_nominal": 1.0 / abs(curvature_corr),
        "scale_up": 1.0 / abs(curvature_up), 
        "scale_down": 1.0 / abs(curvature_down),
        "resolution_nominal": leptons.pt * psmear["nom"],
        "resolution_syst": leptons.pt * psmear["syst"]
    }

    return pt_scale_and_resolution


def mu_scale_and_res(events, variations, is_data, ceval, rochester, do_stat=False, scale_only=True, s=5):
    clib_wrap_scale = correctionlib_wrapper(ceval["muon_highpt_scale"])
    clib_wrap_resolution = correctionlib_wrapper(ceval["muon_highpt_resolution"])

    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    mu_mask = abs(events.Lepton.pdgId) == 13
    mediumpt_mask = ~events.Lepton.isHighPtMuon
    highpt_mask = events.Lepton.isHighPtMuon
    pt_before = ak.copy(events.Lepton.pt)

    # Get Rochester corrections for medium-pt muons
    pt_rochester_variations = vary_rochester(events, is_data, rochester, s_nom=s, do_stat=do_stat, scale_only=scale_only)
    pt_rochester = correct_rochester(events, is_data, rochester, s=s, m=0, scale_only=scale_only)

    # Get scale and resolution corrections for high-pt muons
    if not is_data:
        pt_highpt = correct_highpt(events, clib_wrap_scale, clib_wrap_resolution)

    # Save Rochester nominal
    events[("Lepton", "pt")] = ak.where(
        mu_mask & mediumpt_mask, pt_rochester["nominal"], events.Lepton.pt
    )
    
    # Save Rochester systematic and statistical variations
    for var_name, var_pt in pt_rochester_variations.items():
        vcol = Variation.format_varied_column(("Lepton","pt"), f"rochester_{var_name}")
        events[vcol] = ak.where(
            mu_mask & mediumpt_mask, var_pt, events.Lepton.pt
        )
        variations.register_variation([("Lepton","pt")], f"rochester_{var_name}")

    # Save Rochester resolution as a nuisance
    if scale_only and not is_data:
        vcol = Variation.format_varied_column(("Lepton","pt"), "rochester_res")
        events[vcol] = ak.where(
            mu_mask & mediumpt_mask, pt_rochester["resolution"], events.Lepton.pt
        )
        variations.register_variation([("Lepton","pt")], "rochester_res")

    # Save before variation
    vcol = Variation.format_varied_column(("Lepton","pt"), "rochester_before")
    events[vcol] = pt_before
    variations.register_variation([("Lepton", "pt")], "rochester_before")

    # Save high-pt scale and resolution variations
    if not is_data:
        for variation in ["nominal", "up", "down"]:
            vcol = Variation.format_varied_column(("Lepton","pt"), f"mu_scale_highpt_{variation}")
            events[vcol] = ak.where(
                mu_mask & highpt_mask, pt_highpt[f"scale_{variation}"], events.Lepton.pt
            )
            variations.register_variation([("Lepton", "pt")], f"mu_scale_highpt_{variation}")

        vcol = Variation.format_varied_column(("Lepton","pt"), "mu_res_highpt_nominal")
        events[vcol] = ak.where(
            mu_mask & highpt_mask, pt_highpt["resolution_nominal"], events.Lepton.pt
        )
        variations.register_variation([("Lepton", "pt")], "mu_res_highpt_nominal")

        vcol = Variation.format_varied_column(("Lepton","pt"), "mu_res_highpt_syst")
        events[vcol] = ak.where(
            mu_mask & highpt_mask, pt_highpt["resolution_syst"], events.Lepton.pt
        )
        variations.register_variation([("Lepton", "pt")], "mu_res_highpt_syst")

    return events, variations


