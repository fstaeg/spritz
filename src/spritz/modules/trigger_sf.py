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
    muWP_lowpt = cfg["leptonsWP"]["muWP"]
    muWP_highpt = cfg["leptonsWP"]["muWP_highpt"]
    year = cfg["year"]
    dRmax = 0.1
    
    events[("TrigObj", "mass")] = ak.zeros_like(events.TrigObj.pt)
    events[("Lepton", "isTrigMatched")] = ak.full_like(events.Lepton.pt, False, dtype=bool)

    # Filter TrigObj related to single muon triggers
    # Each year has different triggers
    # 2016: IsoMu24|IsoTkMu24|Mu50; 2017: IsoMu27|Mu50|TkMu100|OldMu100; 2018: IsoMu24|Mu50|TkMu100|OldMu100
    trig_muonmask = (events.TrigObj.id == 13)
    
    if year == "2016":
        isomu_bit = (events.TrigObj.filterBits & (1<<1)) != 0 # Iso
        tkisomu_bit = (events.TrigObj.filterBits & (1<<3)) != 0 # IsoTkMu
        mu50_bit = (events.TrigObj.filterBits & (1<<4)) != 0 # 1mu (Mu50)

        trig_mask = (
            ((events.TrigObj.pt > 24.) & (isomu_bit | tkisomu_bit)) 
            | ((events.TrigObj.pt > 50.) & mu50_bit)
        )
    
    else:
        isomu_minpt = 27. if year == "2017" else 24.
        isomu_bit = (
            ((events.TrigObj.filterBits & (1<<1)) != 0) # Iso
            & ((events.TrigObj.filterBits & (1<<3)) != 0) # 1mu
        )
        mu50_bit = (events.TrigObj.filterBits & (1<<10)) != 0 # 1mu (Mu50)
        mu100_bit = (events.TrigObj.filterBits & (1<<11)) != 0 # 1mu (Mu100)

        trig_mask = (
            ((events.TrigObj.pt > isomu_minpt) & isomu_bit)
            | ((events.TrigObj.pt > 50.) & mu50_bit)
            | ((events.TrigObj.pt > 100.) & mu100_bit)
        )

    trigobjs = events.TrigObj[trig_muonmask & trig_mask]
    trigobj_indices = ak.local_index(trigobjs)

    # Filter Muons with tight ID and Iso
    mu_idmask = ak.where(
        events.Lepton.isHighPtMuon, events.Lepton[f"isTightMuon_{muWP_highpt}"], events.Lepton[f"isTightMuon_{muWP_lowpt}"]
    )
    mu_isomask = ak.where(
        events.Lepton.isHighPtMuon, events.Lepton.isTightMuon_TkRelIso, events.Lepton.isTightMuon_RelIso
    )
    
    mu_minpt = 29. if year == "2017" else 26.
    mu_ptmask = (events.Lepton.pt > mu_minpt)
    
    leptons = ak.mask(events.Lepton, mu_idmask & mu_isomask & mu_ptmask)
    lepton_indices = ak.local_index(leptons)
    
    while ak.count(leptons) > 0:
        pair_lep, pair_trig = ak.unzip(ak.cartesian((leptons,trigobjs), axis=1, nested=True))
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


def trigger_sf(events, variations, ceval_lowpt, ceval_highpt, cfg):
    trigsf_key_lowpt = cfg["muTrigSfKey"] #"NUM_IsoMu24_or_Mu50_DEN_CutBasedIdTight_and_PFIsoTight"
    trigsf_key_highpt = cfg["muTrigSfKey_highpt"]#"NUM_HLT_DEN_HighPtTightRelIsoProbes"
    year = cfg["year"]        
    
    trigmatched_mask = (
        (abs(events.Lepton.pdgId) == 13) & ak.values_astype(events.Lepton.isTrigMatched, bool)
    )
    lowpt_mask = ~events.Lepton.isHighPtMuon
    highpt_mask = events.Lepton.isHighPtMuon

    eta = ak.mask(events.Lepton.eta, trigmatched_mask)
    pt = ak.mask(events.Lepton.pt, trigmatched_mask)
    p = ak.mask(events.Lepton.p, trigmatched_mask)

    maxeta = 2.3999
    eta = ak.where(eta < -maxeta, -maxeta, eta)
    eta = ak.where(eta > maxeta, maxeta, eta)

    minpt = 29.0001 if year == "2017" else 26.0001
    maxpt = 199.9999
    pt = ak.where(pt < minpt, minpt, pt)
    pt = ak.where(pt > maxpt, maxpt, pt)

    minp = 200.0001
    p = ak.where(p < minp, minp, p)

    # load SF
    clib_wrap_lowpt = correctionlib_wrapper(ceval_lowpt[trigsf_key_lowpt])
    clib_wrap_highpt = correctionlib_wrapper(ceval_highpt[trigsf_key_highpt])
    sfs = {
        "lowpt_nominal": ak.where(trigmatched_mask & lowpt_mask, clib_wrap_lowpt(abs(eta), pt, "value"), 1.),
        "lowpt_stat": ak.where(trigmatched_mask & lowpt_mask, clib_wrap_lowpt(abs(eta), pt, "stat"), 0.),
        "lowpt_syst": ak.where(trigmatched_mask & lowpt_mask, clib_wrap_lowpt(abs(eta), pt, "syst"), 0.),
        "highpt_nominal": ak.where(trigmatched_mask & highpt_mask, clib_wrap_highpt(eta, p, "nominal"), 1.),
        "highpt_stat": ak.where(trigmatched_mask & highpt_mask, clib_wrap_highpt(eta, p, "stat"), 0.),
        "highpt_syst": ak.where(trigmatched_mask & highpt_mask, clib_wrap_highpt(eta, p, "syst"), 0.),
    }

    # save per-lepton scale factor and variations
    events[("Lepton", "TriggerSF")] = ak.where(
        events.Lepton.isHighPtMuon, sfs["highpt_nominal"], sfs["lowpt_nominal"]
    )
    
    events[("Lepton", "TriggerSF_highpt_before")] = ak.where(
        events.Lepton.isHighPtMuon, ak.ones_like(sfs["highpt_nominal"]), sfs["lowpt_nominal"]
    )
    events[("Lepton", "TriggerSF_lowpt_before")] = ak.where(
        events.Lepton.isHighPtMuon, sfs["highpt_nominal"], ak.ones_like(sfs["lowpt_nominal"])
    )

    for uncertainty in ["stat", "syst"]:
        for sign, variation in zip([1,-1], ["up", "down"]):
            events[("Lepton", f"TriggerSF_lowpt_{uncertainty}_{variation}")] = ak.where(
                events.Lepton.isHighPtMuon, 
                sfs["highpt_nominal"], 
                sfs["lowpt_nominal"] + sign*sfs[f"lowpt_{uncertainty}"]
            )
            events[("Lepton", f"TriggerSF_highpt_{uncertainty}_{variation}")] = ak.where(
                events.Lepton.isHighPtMuon, 
                sfs["highpt_nominal"] + sign*sfs[f"highpt_{uncertainty}"], 
                sfs["lowpt_nominal"]
            )
    
    # compute per-event scale factor
    matched_lep = ak.pad_none(events.Lepton[trigmatched_mask], 2)
    l1_sf = matched_lep[:,0].TriggerSF
    l2_sf = matched_lep[:,1].TriggerSF

    events["TriggerSF"] = ak.where(
        events.nTrigMatched > 1, 
        l1_sf + l2_sf - l1_sf*l2_sf,
        ak.ones_like(events.weight)
    )
    events["TriggerSF"] = ak.where(
        events.nTrigMatched == 1, 
        l1_sf, 
        events.TriggerSF
    )

    # compute per-event variations
    for ptrange in ["lowpt", "highpt"]:
        for variation in ["before", "stat_up", "stat_down", "syst_up", "syst_down"]:
            var_name = f"mu_trig_{ptrange}_{variation}"
            varied_col = format_varied_column("TriggerSF", var_name)

            l1_sf = matched_lep[:,0][f"TriggerSF_{ptrange}_{variation}"]
            l2_sf = matched_lep[:,1][f"TriggerSF_{ptrange}_{variation}"]

            events[varied_col] = ak.where(
                events.nTrigMatched > 1,
                l1_sf + l2_sf - l1_sf*l2_sf,
                ak.ones_like(events.weight)
            )
            events[varied_col] = ak.where(
                events.nTrigMatched == 1,
                l1_sf,
                events[varied_col]
            )
            variations.register_variation(["TriggerSF"], var_name)

    var_name = "mu_trig_combined_before"
    varied_col = format_varied_column("TriggerSF", var_name)
    events[varied_col] = ak.ones_like(events.TriggerSF)
    variations.register_variation(["TriggerSF"], var_name)
    
    return events, variations
