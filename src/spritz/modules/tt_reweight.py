import awkward as ak
import numpy as np

def tt_reweight(events, variations):
    
    isLastCopy = ak.values_astype((events.GenPart.statusFlags >> 13) & 1, bool)
    top_mask = (events.GenPart.pdgId == 6) & isLastCopy
    antitop_mask = (events.GenPart.pdgId == -6) & isLastCopy

    tops = ak.pad_none(events.GenPart[top_mask], 1, clip=True)
    antitops = ak.pad_none(events.GenPart[antitop_mask], 1, clip=True)
    
    top_pt = ak.fill_none(tops[:,0].pt, 0)
    antitop_pt = ak.fill_none(antitops[:,0].pt, 0)
    
    top_weight = 0.103*np.exp(-0.0118*top_pt) - 0.000134*top_pt + 0.973
    antitop_weight = 0.103*np.exp(-0.0118*antitop_pt) - 0.000134*antitop_pt + 0.973

    events["topPtWeight"] = ak.where(
        top_pt * antitop_pt > 0, 
        np.sqrt(top_weight * antitop_weight), 
        ak.ones_like(events.weight)
    )

    topPtWeight_err = np.abs(events.topPtWeight - ak.ones_like(events.weight))
    events["topPtWeight_tt_ptrw_up"] = events.topPtWeight + topPtWeight_err
    events["topPtWeight_tt_ptrw_down"] = events.topPtWeight - topPtWeight_err
    events["topPtWeight_tt_ptrw_before"] = ak.ones_like(events.weight)

    variations.register_variation(["topPtWeight"], "tt_ptrw_up")
    variations.register_variation(["topPtWeight"], "tt_ptrw_down")
    variations.register_variation(["topPtWeight"], "tt_ptrw_before")
    
    return events, variations
