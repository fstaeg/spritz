import awkward as ak
import numpy as np

def tt_reweight(events, variations):
    
    top_mask = (
        (events.GenPart.pdgId == 6) 
        & ak.values_astype((events.GenPart.statusFlags >> 13) & 1, bool)
    )
    top_pt = ak.fill_none(
        ak.mask(events, ak.num(events.GenPart[top_mask])>=1).GenPart[top_mask][:,-1].pt,
        0.0,
    )
    top_weight = 0.103*np.exp(-0.0118*top_pt) - 0.000134*top_pt + 0.973

    antitop_mask = (
        (events.GenPart.pdgId == -6) 
        & ak.values_astype((events.GenPart.statusFlags >> 13) & 1, bool)
    )
    antitop_pt = ak.fill_none(
        ak.mask(events, ak.num(events.GenPart[antitop_mask])>=1).GenPart[antitop_mask][:,-1].pt,
        0.0,
    )
    antitop_weight = 0.103*np.exp(-0.0118*antitop_pt) - 0.000134*antitop_pt + 0.973
    
    events['topPtWeight'] = ak.where(
        top_pt * antitop_pt > 0.0,
        np.sqrt(top_weight * antitop_weight),
        ak.ones_like(events.weight)
    )

    topPtWeight_err = np.abs(events.topPtWeight - ak.ones_like(events.weight))
    events['topPtWeight_tt_ptrw_up'] = events['topPtWeight'] + topPtWeight_err
    events['topPtWeight_tt_ptrw_down'] = events['topPtWeight'] - topPtWeight_err

    variations.register_variation(['topPtWeight'], 'tt_ptrw_up')
    variations.register_variation(['topPtWeight'], 'tt_ptrw_down')

    events['topPtWeight_before'] = ak.ones_like(events.weight)
    variations.register_variation(['topPtWeight'], 'before')
    
    return events, variations
