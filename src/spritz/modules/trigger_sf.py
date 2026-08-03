import awkward as ak
import numpy as np
import spritz.framework.variation as variation_module
from spritz.framework.framework import correctionlib_wrapper

format_varied_column = variation_module.Variation.format_varied_column

def none_like(arr): 
    return ak.mask(arr, ak.full_like(arr, False, dtype=bool))

def broadcast_arrays(arr1, arr2): # fix a weird behaviour of ak.broadcast_arrays()
    return ak.where(
        ak.is_none(arr1),
        none_like(arr2),
        ak.broadcast_arrays(arr1,arr2)[0]
    )

def match_trigger_object(events, cfg):
    muWP = cfg["leptonsWP"]["muWP"]
    year = cfg["year"]
    dRmax = 0.1
    
    events[("TrigObj", "mass")] = ak.zeros_like(events.TrigObj.pt)
    events[("Lepton", "isTrigMatched")] = ak.full_like(events.Lepton.pt, False, dtype=bool)

    # Filter TrigObj related to single muon triggers
    # Each year has different triggers
    # 2016: IsoMu24|IsoTkMu24; 2017: IsoMu27; 2018: IsoMu24
    trig_mumask = (events.TrigObj.id == 13)
    
    if year == "2017":
        trig_ptmask = (events.TrigObj.pt > 27.)
    else:
        trig_ptmask = (events.TrigObj.pt > 24.)
    
    if year == "2016":
        trig_filterbitmask = (
            ((events.TrigObj.filterBits & (1<<1))!=0) # Iso
            | ((events.TrigObj.filterBits & (1<<3))!=0) # IsoTkMu
        )
    else:
        trig_filterbitmask = (
            ((events.TrigObj.filterBits & (1<<1))!=0) # Iso
            & ((events.TrigObj.filterBits & (1<<3))!=0) # 1mu
        )

    trigobjs = events.TrigObj[trig_mumask & trig_ptmask & trig_filterbitmask]
    trigobj_indices = ak.local_index(trigobjs)

    # Filter Muons with tight ID and Iso
    mu_idmask = events.Lepton[f"isTightMuon_{muWP}"]
    mu_isomask = events.Lepton["isTightMuon_RelIso"]
    
    if year == "2017":
        mu_ptmask = (events.Lepton.pt > 29.)
    else:
        mu_ptmask = (events.Lepton.pt > 26.)
    
    leptons = ak.mask(events.Lepton, mu_idmask & mu_isomask & mu_ptmask)
    lepton_indices = ak.local_index(leptons)
    
    while ak.count(leptons) > 0:
        pair_lep,pair_trig = ak.unzip(ak.cartesian((leptons,trigobjs), axis=1, nested=True))
        dR = pair_lep.deltaR(pair_trig)

        dR_min_trigobj = ak.min(dR,axis=-2)
        closest_trigobj = ak.argmin(dR_min_trigobj, axis=-1)
        closest_trigobj_broadcasted = broadcast_arrays(closest_trigobj, trigobj_indices)
        trigobjs = ak.mask(trigobjs, trigobj_indices!=closest_trigobj_broadcasted)

        dR_min_lep = ak.min(dR, axis=-1)
        closest_lep = ak.argmin(dR_min_lep, axis=-1)
        closest_lep_broadcasted = broadcast_arrays(closest_lep, lepton_indices)
        leptons = ak.mask(leptons, lepton_indices!=closest_lep_broadcasted)
        
        lep_ismatched = ak.fill_none(
            ak.mask(dR_min_lep, lepton_indices==closest_lep_broadcasted) < dRmax, 
            False
        )
        
        events[("Lepton", "isTrigMatched")] = events.Lepton.isTrigMatched | lep_ismatched

    events["nTrigMatched"] = ak.sum(events.Lepton.isTrigMatched, axis=1)
    return events


def trigger_sf(events, variations, ceval_lepton_sf, cfg):
    trigsf_key = cfg["muTrigSfKey"]
    year = cfg["year"]

    events["TriggerSF"] = ak.ones_like(events.weight)
    events["TriggerSF_err"] = ak.zeros_like(events.weight)
    
    mu_mask = abs(events.Lepton.pdgId) == 13
    trigmatched_mask = ak.values_astype(events.Lepton.isTrigMatched, bool)

    eta = ak.mask(events.Lepton.eta, mu_mask)
    pt = ak.mask(events.Lepton.pt, mu_mask)

    maxeta = 2.3999
    eta = ak.where(eta < -maxeta, -maxeta, eta)
    eta = ak.where(eta > maxeta, maxeta, eta)

    minpt = 29.0001 if year=="2017" else 26.0001
    pt = ak.where(pt < minpt, minpt, pt)

    # load SF
    clib_wrap = correctionlib_wrapper(ceval_lepton_sf[trigsf_key])
    sf_nominal = ak.where(mu_mask & trigmatched_mask, clib_wrap(eta, pt, "nominal"), 1.)
    sf_stat = ak.where(mu_mask & trigmatched_mask, clib_wrap(eta, pt, "stat"), 0.)
    sf_syst = ak.where(mu_mask & trigmatched_mask, clib_wrap(eta, pt, "syst"), 0.)

    # save per-lepton scale factor and variation
    events[("Lepton", "TriggerSF")] = sf_nominal
    events[("Lepton", "TriggerSF_err")] = np.sqrt( sf_stat**2 + sf_syst**2 )
    
    # compute per-event scale factor
    ones = ak.ones_like(events.weight)
    matched_lep = ak.pad_none(events.Lepton[trigmatched_mask], 2)
    l1_sf = matched_lep[:,0].TriggerSF
    l2_sf = matched_lep[:,1].TriggerSF

    events["TriggerSF"] = ak.where(
        events.nTrigMatched > 1, 
        ones-(ones-l1_sf)*(ones-l2_sf), 
        events["TriggerSF"]
    )
    events["TriggerSF"] = ak.where(
        events.nTrigMatched == 1, 
        l1_sf, 
        events["TriggerSF"]
    )

    # save before variation
    var_name = "mu_trig_before"
    varied_col = format_varied_column("TriggerSF", var_name)
    events[varied_col] = ak.ones_like(events["TriggerSF"])
    variations.register_variation(["TriggerSF"], var_name)

    # compute per-event variation
    l1_sferr = matched_lep[:,0].TriggerSF_err
    l2_sferr = matched_lep[:,1].TriggerSF_err

    events["TriggerSF_err"] = ak.where(
        events.nTrigMatched > 1,
        np.sqrt( ((ones-l2_sf)*l1_sferr)**2 + ((ones-l1_sf)*l2_sferr)**2 ),
        events["TriggerSF_err"]
    )
    events["TriggerSF_err"] = ak.where(
        events.nTrigMatched == 1,
        l1_sferr,
        events["TriggerSF_err"]
    )

    # save up and down variations
    for sign,variation in zip([+1,-1], ["up","down"]):
        var_name = f"mu_trig_{variation}"
        varied_col = format_varied_column("TriggerSF", var_name)
        events[varied_col] = events["TriggerSF"] + sign*events["TriggerSF_err"]
        variations.register_variation(["TriggerSF"], var_name)
    
    return events, variations
