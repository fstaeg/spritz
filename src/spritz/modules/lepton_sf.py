import awkward as ak
import numpy as np
import spritz.framework.variation as variation_module
from spritz.framework.framework import correctionlib_wrapper

format_varied_column = variation_module.Variation.format_varied_column

def lepton_sf(events, variations, ceval_lowpt, ceval_highpt, cfg, components=None):
    # components: e.g. {"iso": ["stat", "AltSig", "massBin", "massRange", "tagIso"]}
    components = components or {}
    
    muWP_lowpt = cfg["leptonsWP"]["muWP"]
    recosf_key_lowpt = cfg["muRecoSfKey"]
    idsf_key_lowpt = cfg["muIdSfKey"]
    isosf_key_lowpt = cfg["muIsoSfKey"]

    muWP_highpt = cfg["leptonsWP"]["muWP_highpt"]
    recosf_key_highpt = cfg["muRecoSfKey_highpt"]
    idsf_key_highpt = cfg["muIdSfKey_highpt"]
    isosf_key_highpt = cfg["muIsoSfKey_highpt"]

    leptons = ak.copy(events.Lepton)
    mu_mask = abs(leptons.pdgId) == 13
    lowpt_mask = ~leptons.isHighPtMuon
    highpt_mask = leptons.isHighPtMuon

    sfs_dict = {
        "reco": {
            "column": ("Lepton", "RecoSF"),
            "wrap_lowpt": correctionlib_wrapper(ceval_lowpt[recosf_key_lowpt]),
            "wrap_highpt": correctionlib_wrapper(ceval_highpt[recosf_key_highpt]),
            "mask_lowpt": mu_mask & lowpt_mask,
            "mask_highpt": mu_mask & highpt_mask,
            "minpt": 40.0001,
            "minp": 50.0001,
            "maxeta": 2.3999
        },
        "id": {
            "column": ("Lepton", "IdSF"),
            "wrap_lowpt": correctionlib_wrapper(ceval_lowpt[idsf_key_lowpt]),
            "wrap_highpt": correctionlib_wrapper(ceval_highpt[idsf_key_highpt]),
            "mask_lowpt": mu_mask & lowpt_mask & leptons[f"isTightMuon_{muWP_lowpt}"],
            "mask_highpt": mu_mask & highpt_mask & leptons[f"isTightMuon_{muWP_highpt}"],
            "minpt": 15.0001,
            "minp": 50.0001,
            "maxeta": 2.3999
        },
        "iso": {
            "column": ("Lepton", "IsoSF"),
            "wrap_lowpt": correctionlib_wrapper(ceval_lowpt[isosf_key_lowpt]),
            "wrap_highpt": correctionlib_wrapper(ceval_highpt[isosf_key_highpt]),
            "mask_lowpt": mu_mask & lowpt_mask & leptons["isTightMuon_RelIso"],
            "mask_highpt": mu_mask & highpt_mask & leptons["isTightMuon_TkRelIso"],
            "minpt": 15.0001,
            "minp": 50.0001,
            "maxeta": 2.3999
        }
    }
    
    muon_sf = {}
    
    for sf in sfs_dict:
        if not sf in components:
            components[sf] = ["stat", "syst"]

        maxeta = sfs_dict[sf]["maxeta"]
        eta = ak.mask(leptons.eta, mu_mask)
        eta = ak.where(eta < -maxeta, -maxeta, eta)
        eta = ak.where(eta > maxeta, maxeta, eta)
        
        minpt = sfs_dict[sf]["minpt"]
        pt = ak.mask(leptons.pt, mu_mask)
        pt = ak.where(pt < minpt, minpt, pt)

        minp = sfs_dict[sf]["minp"]
        p = ak.mask(leptons.p, mu_mask)
        p = ak.where(p < minp, minp, p)

        # load SF
        mask_lowpt = sfs_dict[sf]["mask_lowpt"]
        mask_highpt = sfs_dict[sf]["mask_highpt"]
        clib_wrap_lowpt = sfs_dict[sf]["wrap_lowpt"]
        clib_wrap_highpt = sfs_dict[sf]["wrap_highpt"]
        
        lowpt_nominal = ak.where(mask_lowpt, clib_wrap_lowpt(eta, pt, "nominal"), 1.)
        highpt_nominal = ak.where(mask_highpt, clib_wrap_highpt(eta, p, "nominal"), 1.)
        
        muon_sf[sf] = {
            "lowpt_nominal": lowpt_nominal,
            "highpt_nominal": highpt_nominal,
        }
        
        for category in components[sf]:
            lowpt_unc = ak.where(mask_lowpt, clib_wrap_lowpt(eta, pt, category), 0.)
            highpt_unc = ak.where(mask_highpt, clib_wrap_highpt(eta, p, category), 0.)
            muon_sf[sf][f"lowpt_{category}_up"] = lowpt_nominal + lowpt_unc
            muon_sf[sf][f"lowpt_{category}_down"] = lowpt_nominal - lowpt_unc
            muon_sf[sf][f"highpt_{category}_up"] = highpt_nominal + highpt_unc
            muon_sf[sf][f"highpt_{category}_down"] = highpt_nominal - highpt_unc

        # save nominal SF per lepton
        column = sfs_dict[sf]["column"]
        events[column] = ak.where(
            leptons.isHighPtMuon, muon_sf[sf]["highpt_nominal"], muon_sf[sf]["lowpt_nominal"]
        )
        
        # save before variation
        var_name_lowpt = f"mu_{sf}_lowpt_before"
        varied_col_lowpt = format_varied_column(column, var_name_lowpt)
        events[varied_col_lowpt] = ak.where(
            leptons.isHighPtMuon, muon_sf[sf]["highpt_nominal"], ak.ones_like(muon_sf[sf]["lowpt_nominal"])
        )
        variations.register_variation([column], var_name_lowpt)

        var_name_highpt = f"mu_{sf}_highpt_before"
        varied_col_highpt = format_varied_column(column, var_name_highpt)
        events[varied_col_highpt] = ak.where(
            leptons.isHighPtMuon, ak.ones_like(muon_sf[sf]["highpt_nominal"]), muon_sf[sf]["lowpt_nominal"]
        )
        variations.register_variation([column], var_name_highpt)

        var_name_combined = f"mu_{sf}_combined_before"
        varied_col_combined = format_varied_column(column, var_name_combined)
        events[varied_col_combined] = ak.ones_like(muon_sf[sf]["highpt_nominal"])
        variations.register_variation([column], var_name_combined)

        # save up and down variations
        for variation in [k for k in muon_sf[sf] if not k in ["lowpt_nominal", "highpt_nominal"]]:
            var_name = f"mu_{sf}_{variation}"
            varied_col = format_varied_column(column, var_name)
            if variation.startswith("lowpt_"):
                events[varied_col] = ak.where(
                    leptons.isHighPtMuon, muon_sf[sf]["highpt_nominal"], muon_sf[sf][variation]
                )
            elif variation.startswith("highpt_"):
                events[varied_col] = ak.where(
                    leptons.isHighPtMuon, muon_sf[sf][variation], muon_sf[sf]["lowpt_nominal"]
                )
            variations.register_variation([column], var_name)

    return events, variations
