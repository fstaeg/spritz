import awkward as ak
import numpy as np

def HO_reweight(events, variations, weights, weights_err, edges, name="EWWeight",
                observable=None):

    if observable not in (None, "mll"):
        raise ValueError(f"lhe_ho_reweight only supports observable 'mll', got '{observable}'")

    outgoing_mask = (events.LHEPart.status == 1)
    lepton_mask = ((abs(events.LHEPart.pdgId) == 11) | (abs(events.LHEPart.pdgId) == 13) | (abs(events.LHEPart.pdgId) == 15)) # electrons or muons or taus --> available at LHE level not at gendressed level
    lhe_leptons = events.LHEPart[outgoing_mask & lepton_mask]
        
    assert ak.all(ak.num(lhe_leptons) == 2)
    lhe_mll = (lhe_leptons[:, 0] + lhe_leptons[:, 1]).mass

    mll_bins = np.digitize(lhe_mll, edges[0]) - 1
    mll_bins = np.clip(mll_bins, 0, weights.shape[0]-1)
    
    nloweights = weights[mll_bins]
    nloweights_err = weights_err[mll_bins]

    events[name] = nloweights
    events[f"{name}_nlo_up"] = nloweights + nloweights_err
    events[f"{name}_nlo_down"] = nloweights - nloweights_err

    variations.register_variation([name], "nlo_up")
    variations.register_variation([name], "nlo_down")

    events[f"{name}_nlo_before"] = ak.ones_like(nloweights)
    variations.register_variation([name], "nlo_before")

    return events, variations