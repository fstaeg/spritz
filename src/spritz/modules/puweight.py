from spritz.framework.variation import Variation
import awkward as ak

def puweight_sf(events, variations: Variation, ceval_puWeight, cfg):
    events["puWeight"] = ceval_puWeight[cfg["puWeightsKey"]].evaluate(
        events.Pileup.nTrueInt, "nominal"
    )
    events["puWeight_PU_up"] = ceval_puWeight[cfg["puWeightsKey"]].evaluate(
        events.Pileup.nTrueInt, "up"
    )
    events["puWeight_PU_down"] = ceval_puWeight[cfg["puWeightsKey"]].evaluate(
        events.Pileup.nTrueInt, "down"
    )
    variations.register_variation(['puWeight'], 'PU_up')
    variations.register_variation(['puWeight'], 'PU_down')

    events["puWeight_before"] = ak.ones_like(events.weight)
    variations.register_variation(['puWeight'], 'before')

    return events, variations
