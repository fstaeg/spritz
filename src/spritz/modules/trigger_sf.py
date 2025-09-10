import awkward as ak
import numpy as np
import spritz.framework.variation as variation_module
from spritz.framework.framework import correctionlib_wrapper

def none_like(arr): 
    return ak.mask(arr, ak.full_like(arr, False, dtype=bool))

def broadcast_arrays(arr1, arr2): # fix a weird behaviour of ak.broadcast_arrays()
    return ak.where(
        ak.is_none(arr1),
        none_like(arr2),
        ak.broadcast_arrays(arr1,arr2)[0]
    )

def match_trigger_object(events, cfg, dRmax=0.1):
    events[("TrigObj", "mass")] = ak.zeros_like(events.TrigObj.pt)
    trigobjs = events.TrigObj[(
        ((events.TrigObj.id==11) & (events.TrigObj.pt>32.) & ((events.TrigObj.filterBits & (1<<1))!=0)) 
        | ((events.TrigObj.id==13) & (events.TrigObj.pt>24.) & ((events.TrigObj.filterBits & (1<<1))!=0) & ((events.TrigObj.filterBits & (1<<3))!=0))
    )]
    trigobj_indices = ak.local_index(trigobjs)

    events[("Lepton", "isTrigMatched")] = ak.full_like(events.Lepton.pt, False, dtype=bool)
    events[("Lepton", "dRmatchedTrig")] = none_like(events.Lepton.pt)
    events[("Lepton", "dRnextTrig")] = none_like(events.Lepton.pt)
    
    tight_mask = (
        (events.Lepton["isTightElectron_" + cfg["leptonsWP"]["eleWP"]] & (events.Lepton.pt>32.)) 
        | (events.Lepton["isTightMuon_" + cfg["leptonsWP"]["muWP"]]  & (events.Lepton.pt>24.))
    )
    leptons = ak.mask(events.Lepton, tight_mask)
    lepton_indices = ak.local_index(leptons)
    
    while ak.count(leptons) > 0:
        pair_lep,pair_trig = ak.unzip(ak.cartesian((leptons,trigobjs), axis=1, nested=True))
        dR = pair_lep.deltaR(pair_trig)
        #lep_nmatches = ak.sum(dR<dRmax, axis=-1)
        #dR = ak.where(ak.any(lep_nmatches==1,axis=-1), ak.mask(dR, lep_nmatches==1), dR)

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
        
        events[("Lepton", "dRnextTrig")] = ak.where(
            ak.is_none(events.Lepton.dRnextTrig, axis=-1),
            dR_min_lep,
            events.Lepton.dRnextTrig
        )
        events[("Lepton", "dRmatchedTrig")] = ak.where(
            lep_ismatched,
            dR_min_lep,
            events.Lepton.dRmatchedTrig
        )

    events[("Lepton", "dRnextTrig")] = ak.fill_none(events.Lepton.dRnextTrig, -1)
    events[("Lepton", "dRmatchedTrig")] = ak.fill_none(events.Lepton.dRmatchedTrig, -1)
    events["nTrigMatched"] = ak.sum(events.Lepton.isTrigMatched, axis=1)
    return events


def trigger_sf(events, variations, ceval_lepton_sf, cfg):
    minpt_mu = 26.0001
    mineta_mu = -2.3999
    maxeta_mu = 2.3999

    minpt_ele = 32.0001
    maxpt_ele = 499.9999
    mineta_ele = -2.4999
    maxeta_ele = 2.4999

    mu_mask = abs(events.Lepton.pdgId) == 13
    ele_mask = abs(events.Lepton.pdgId) == 11

    pt = ak.copy(events.Lepton.pt)
    eta = ak.copy(events.Lepton.eta)

    pt = ak.where(mu_mask & (pt < minpt_mu), minpt_mu, pt)
    pt = ak.where(ele_mask & (pt < minpt_ele), minpt_ele, pt)
    pt = ak.where(ele_mask & (pt > maxpt_ele), maxpt_ele, pt)
    eta = ak.where(mu_mask & (eta < mineta_mu), mineta_mu, eta)
    eta = ak.where(mu_mask & (eta > maxeta_mu), maxeta_mu, eta)
    eta = ak.where(ele_mask & (eta < mineta_ele), mineta_ele, eta)
    eta = ak.where(ele_mask & (eta > maxeta_ele), maxeta_ele, eta)

    sfs_dict = {
        "mu_trig_sf": {
            "wrap": correctionlib_wrapper(ceval_lepton_sf["Muon_TriggerSF_tightId"]),
            "mask": mu_mask,
            "output": "trig_sf"
        },
        "ele_trig_sf": {
            "wrap": correctionlib_wrapper(ceval_lepton_sf["Electron_TriggerSF_Ele32_WP90"]),
            "mask": ele_mask,
            "output": "trig_sf"
        },
    }
    lepton_trigger_sf = {'nominal': ak.ones_like(pt)}

    for reco_sf in ["mu_trig_sf","ele_trig_sf"]:
        mask = sfs_dict[reco_sf]["mask"]
        _eta = ak.mask(eta, mask)
        _pt = ak.mask(pt, mask)
        
        lepton_trigger_sf['nominal'] = ak.where(
            mask & ak.values_astype(events.Lepton.isTrigMatched, bool),
            sfs_dict[reco_sf]['wrap'](_eta, _pt, 'nominal'),
            lepton_trigger_sf['nominal']
        )

    events[('Lepton','TriggerSF')] = lepton_trigger_sf['nominal']
    matched_lep = ak.pad_none(events.Lepton[ak.values_astype(events.Lepton.isTrigMatched, bool)], 2)

    ones = ak.ones_like(events.weight)
    TriggerSFweight_2l = ak.where(
        events.nTrigMatched>1,
        ones-(ones-matched_lep[:,0].TriggerSF)*(ones-matched_lep[:,1].TriggerSF),
        ones
    )
    TriggerSFweight_2l = ak.where(
        events.nTrigMatched==1,
        matched_lep[:,0].TriggerSF,
        TriggerSFweight_2l
    )
    events['TriggerSFweight_2l'] = TriggerSFweight_2l
    
    return events, variations
