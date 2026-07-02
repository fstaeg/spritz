import awkward as ak
import numpy as np
import spritz.framework.variation as variation_module
from spritz.framework.framework import correctionlib_wrapper

format_varied_column = variation_module.Variation.format_varied_column

def lepton_sf(events, variations, ceval_lep, cfg):
    muWP = cfg["leptonsWP"]["muWP"]

    recosf_key = "Muon_RecoSF_highPtId" if muWP=="cut_highPtId" else "Muon_RecoSF"
    idsf_key = f"Muon_IdSF_{muWP.split("_")[-1]}"
    isosf_key = f"Muon_IsoSF_{muWP.split("_")[-1]}"

    leptons = ak.copy(events.Lepton)
    mu_mask = abs(leptons.pdgId) == 13

    sfs_dict = {
        "reco": {
            "wrap": correctionlib_wrapper(ceval_lep[recosf_key]),
            "mask": mu_mask,
            "column": ("Lepton", "RecoSF"),
            "minpt": 50.0001 if muWP=="cut_highPtId" else 15.0001,
            "maxeta": 2.3999
        },
        "id": {
            "wrap": correctionlib_wrapper(ceval_lep[idsf_key]),
            "mask": mu_mask & leptons[f"isTightMuon_{muWP}"],
            "column": ("Lepton", "IdSF"),
            "minpt": 50.0001 if muWP=="cut_highPtId" else 15.0001,
            "maxeta": 2.3999
        },
        "iso": {
            "wrap": correctionlib_wrapper(ceval_lep[isosf_key]),
            "mask": mu_mask & leptons["isTightMuon_RelIso"],
            "column": ("Lepton", "IsoSF"),
            "minpt": 50.0001 if muWP=="cut_highPtId" else 15.0001,
            "maxeta": 2.3999
        }
    }
    
    muon_sf = {}

    for sf in ["reco", "id", "iso"]:
        mask = sfs_dict[sf]["mask"]
        eta = ak.mask(leptons.eta, mask)
        pt = ak.mask(leptons.p if (sf=="reco" and muWP=="cut_highPtId") else leptons.pt, mask)
        
        maxeta = sfs_dict[sf]["maxeta"]
        eta = ak.where(eta < -maxeta, -maxeta, eta)
        eta = ak.where(eta > maxeta, maxeta, eta)
        
        minpt = sfs_dict[sf]["minpt"]
        pt = ak.where(pt < minpt, minpt, pt)

        # load SF
        clib_wrap = sfs_dict[sf]["wrap"]
        muon_sf[sf] = {
            "nominal": ak.where(mask, clib_wrap(eta, pt, "nominal"), 1.),
            "stat": ak.where(mask, clib_wrap(eta, pt, "stat"), 0.),
            "syst": ak.where(mask, clib_wrap(eta, pt, "syst"), 0.)
        }

        # get up and down variations
        err = np.sqrt( muon_sf[sf]["stat"]**2 + muon_sf[sf]["syst"]**2 )
        muon_sf[sf]["up"] = muon_sf[sf]["nominal"] + err
        muon_sf[sf]["down"] = muon_sf[sf]["nominal"] - err

        # save nominal SF per lepton
        column = sfs_dict[sf]["column"]
        events[column] = muon_sf[sf]["nominal"]
        
        # save before variation
        var_name = f"mu_{sf}_before"
        varied_col = format_varied_column(column, var_name)
        events[varied_col] = ak.ones_like(muon_sf[sf]["nominal"])
        variations.register_variation([column], var_name)

        # save up and down variations
        for variation in ["up", "down"]:
            var_name = f"mu_{sf}_{variation}"
            varied_col = format_varied_column(column, var_name)
            events[varied_col] = muon_sf[sf][variation]
            variations.register_variation([column], var_name)

    return events, variations
