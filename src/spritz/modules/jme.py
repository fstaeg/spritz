import awkward as ak
import correctionlib
import numpy as np
import spritz.framework.variation as variation_module
from data.common.TrigMaker_cfg import Trigger


def jet_veto(events, cfg):
    cset = correctionlib.CorrectionSet.from_file(cfg["jetvetomaps"])
    veto_tag = cfg["jme"]["jet_veto_tag"]
    
    jet_phi = events.Jet.phi
    jet_eta = events.Jet.eta
    
    jet_veto = cset[veto_tag].evaluate("jetvetomap", jet_eta, jet_phi)
    jet_veto = ak.Array(jet_veto)
    jet_veto = ak.from_regular(jet_veto)
    jet_veto = ak.values_astype(jet_veto, bool)
    
    events["Jet"] = events.Jet[~jet_veto]
    
    return events


def HEM_issue(events, cfg):
    year = cfg["era"]
    events["HEM_issue"] = ak.zeros_like(events.weight) == 1

    # only events from Run2018(B),C,D have HEM issue
    for era in Trigger[year]:
        if Trigger[year][era].get("HEMIssue", False):
            events["HEM_issue"] = events.HEM_issue | (events.run_period == era)

    return events


def remove_jets_HEM_issue(events, cfg):
    events = HEM_issue(events, cfg)

    jets = ak.copy(events.Jet)
    HEM_jets_mask = (
        (-1.57 < jets.phi) & (jets.phi < -0.87) &
        (-3.2 < jets.eta) & (jets.eta < -1.3)
    )
    
    events["Jet"] = ak.where(events.HEM_issue, jets[~HEM_jets_mask], jets)
    return events


def remove_events_HEM_issue(events, cfg):
    events = HEM_issue(events, cfg)

    jets = ak.copy(events.Jet)
    HEM_jets_mask = (
        (-1.57 < jets.phi) & (jets.phi < -0.87) &
        (-3.2 < jets.eta) & (jets.eta < -1.3)
    )
    HEM_jets = jets[HEM_jets_mask]
    
    events = events[~events.HEM_issue | (ak.num(HEM_jets)==0)]
    return events


def get_random_seed(events):
    runnum = events.run << 20
    luminum = events.luminosityBlock << 10
    evtnum = events.event
    
    jet0eta = events.Jet.eta
    jet0eta = ak.Array([jet0eta]) if jet0eta.ndim==1 else jet0eta
    jet0eta = ak.pad_none(jet0eta / 0.01, 1, clip=True)
    jet0eta = ak.fill_none(jet0eta, 0.0)[:, 0]
    jet0eta = ak.values_astype(jet0eta, int)
    
    event_random_seed = 1 + runnum + luminum + evtnum + jet0eta

    return event_random_seed


# CMSJME in awkward

def correct_jets_mc(
    events, variations: variation_module.Variation, cfg, run_variations=False
):
    cset_jerc = correctionlib.CorrectionSet.from_file(cfg["jet_jerc"])
    cset_jersmear = correctionlib.CorrectionSet.from_file(cfg["jer_smear"])
    
    jme_cfg = cfg["jme"]
    jec_tag = jme_cfg["jec_tag"]["mc"] # e.g. Summer19UL18_V5_MC
    jer_tag = jme_cfg["jer_tag"] # e.g. Summer19UL18_JRV2_MC
    jet_algo = jme_cfg["jet_algo"] # e.g. AK4PFchs
    jes = jme_cfg["jes"]
    
    cset_jec = cset_jerc.compound[f"{jec_tag}_L1L2L3Res_{jet_algo}"]
    cset_jer = cset_jerc[f"{jer_tag}_ScaleFactor_{jet_algo}"]
    cset_jer_ptres = cset_jerc[f"{jer_tag}_PtResolution_{jet_algo}"]
    cset_jersmear = cset_jersmear["JERSmear"]

    events_jme = ak.copy(events)
    jets = ak.copy(events_jme.Jet)
    
    rho = ak.broadcast_arrays(events_jme.fixedGridRhoFastjetAll, jets.pt)[0]
    EventID = ak.broadcast_arrays(get_random_seed(events_jme), jets.pt)[0]

    # matched GenJet pt, or -1 if no match
    trueGenJetMask = (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet))
    trueGenJetIdx = ak.mask(jets.genJetIdx, trueGenJetMask)
    gen_pt = ak.fill_none(events.GenJet[trueGenJetIdx].pt, -1.0)

    # Raw pt and mass (before JEC): rawFactor = 1-p_raw/p_old
    jets["pt_raw"] = jets.pt * (1.0 - jets.rawFactor)
    jets["mass_raw"] = jets.mass * (1.0 - jets.rawFactor)

    # Apply JEC
    # sf = p_new/p_raw
    sf_jec = ak.Array(
        cset_jec.evaluate(jets.area, jets.eta, jets.pt_raw, rho)
    )

    # newc = p_new/p_old
    newc = (1.0 - jets.rawFactor) * sf_jec
    jets["pt"] = ak.where(newc > 0.0, jets.pt_raw * sf_jec, jets.pt)
    jets["mass"] = ak.where(newc > 0.0, jets.mass_raw * sf_jec, jets.mass)

    # Apply JER smearing
    sf_jer = {
        "nom": ak.Array(cset_jer.evaluate(jets.eta, "nom")),
        "up": ak.Array(cset_jer.evaluate(jets.eta, "up")),
        "down": ak.Array(cset_jer.evaluate(jets.eta, "down"))
    }
    sf_jer_ptres = ak.Array(cset_jer_ptres.evaluate(jets.eta, jets.pt, rho))
    sf_jers = {}

    for tag in ["nom", "up", "down"]:
        sf_jers[tag] = ak.Array(cset_jersmear.evaluate(
            jets.pt, jets.eta, gen_pt, rho, EventID, sf_jer_ptres, sf_jer[tag]
        ))

    if run_variations:
        for tag in ["up", "down"]:
            events[("Jet", f"pt_JER_{tag}")] = jets.pt * sf_jers[tag]
            events[("Jet", f"mass_JER_{tag}")] = jets.mass * sf_jers[tag]
            variations.register_variation(
                columns=[("Jet","pt"), ("Jet","mass")], variation_name=f"JER_{tag}"
            )

    jets["pt"] = jets.pt * sf_jers["nom"]
    jets["mass"] = jets.mass * sf_jers["nom"]

    # Apply JES variations
    if run_variations:
        for unc in jes:
            cset_jes = cset_jerc[f"{jec_tag}_Regrouped_{unc}_{jet_algo}"]
            delta_jes = ak.Array(cset_jes.evaluate(jets.eta, jets.pt))

            for sign,tag in zip([+1,-1],["up","down"]):
                events[("Jet", f"pt_JES_{unc}_{tag}")] = jets.pt * (1 + sign*delta_jes)
                events[("Jet", f"mass_JES_{unc}_{tag}")] = jets.mass * (1 + sign*delta_jes)
                variations.register_variation(
                    columns=[("Jet","pt"), ("Jet","mass")], variation_name=f"JES_{unc}_{tag}"
                )
    
    # 'before' variation
    events[("Jet", "pt_JES_JER_before")] = ak.copy(events.Jet.pt)
    events[("Jet", "mass_JES_JER_before")] = ak.copy(events.Jet.mass)
    variations.register_variation(
        columns=[("Jet","pt"), ("Jet","mass")], variation_name="JES_JER_before"
    )

    events[("Jet", "pt")] = jets.pt
    events[("Jet", "mass")] = jets.mass
    
    return events, variations


def correct_jets_data(events, variations, cfg, era):
    cset_jerc = correctionlib.CorrectionSet.from_file(cfg["jet_jerc"])
    
    jme_cfg = cfg["jme"]
    jec_tag = jme_cfg["jec_tag"]["data"][era]
    jet_algo = jme_cfg["jet_algo"]
    
    cset_jec = cset_jerc.compound[f"{jec_tag}_L1L2L3Res_{jet_algo}"]

    events_jme = ak.copy(events)
    jets = ak.copy(events_jme.Jet)

    rho = ak.broadcast_arrays(events_jme.fixedGridRhoFastjetAll, jets.pt)[0]

    # Raw pt and mass (before JEC): rawFactor = 1-p_raw/p_old
    jets["pt_raw"] = jets.pt * (1.0 - jets.rawFactor)
    jets["mass_raw"] = jets.mass * (1.0 - jets.rawFactor)

    # Apply JEC
    sf_jec = ak.Array(
        cset_jec.evaluate(jets.area, jets.eta, jets.pt_raw, rho)
    )

    newc = (1.0 - jets.rawFactor) * sf_jec
    jets["pt"] = ak.where(newc > 0.0, jets.pt_raw * sf_jec, jets.pt)
    jets["mass"] = ak.where(newc > 0.0, jets.mass_raw * sf_jec, jets.mass)

    # 'before' variation
    events[("Jet", "pt_JES_JER_before")] = ak.copy(events.Jet.pt)
    events[("Jet", "mass_JES_JER_before")] = ak.copy(events.Jet.mass)
    variations.register_variation(
        columns=[("Jet","pt"), ("Jet","mass")], variation_name="JES_JER_before"
    )

    events[("Jet", "pt")] = jets.pt
    events[("Jet", "mass")] = jets.mass
    
    return events, variations


def correct_met(events, variations, ceval, is_data):
    cset_met = correctionlib.CorrectionSet.from_file(cfg["met"])
    cset_key = "%s_metphicorr_%s_data" if is_data else "%s_metphicorr_%s_mc"
    csets = {
        "PuppiMET": { var: cset_met[cset_key % (var, "puppimet")] for var in ["pt", "phi"] },
        "MET": { var: cset_met[cset_key % (var, "pfmet")] for var in ["pt", "phi"] },
    }

    for coll in ["PuppiMET", "MET"]:
        pt = ak.mask(events[coll].pt, events[coll].pt < 6500)
        phi = ak.mask(events[coll].phi, events[coll].pt < 6500)
        
        for var in ["pt", "phi"]:
            raw = events[coll][var]
            events[(coll, f"{var}_before")] = raw
            variations.register_variation(
                columns=[(coll, var)], variation_name=f"{var}_before"
            )
            
            corrected = csets[coll][var].evaluate(pt, phi, events.PV.npvs, events.run)
            events[(coll, var)] = ak.fill_none(corrected, raw)

    return events, variations

