from typing import NewType

import awkward as ak
import spritz.framework.variation as variation
from spritz.framework.framework import correctionlib_wrapper

correctionlib_evaluator = NewType("correctionlib_evaluator", any)

def format_rule(column, variation_name):
    tag = variation_name.split("_")[-1]
    if isinstance(column, str):
        return f"{column}_{tag}"
    elif isinstance(column, tuple):
        _list = list(column[:-1])
        _list.append(f"{column[-1]}_{tag}")
        return tuple(_list)
    else:
        print("Cannot format varied column", column, "for variation", variation_name)
        raise Exception


@variation.vary(reads_columns=[("Jet", "pt"), ("Jet", "puId"), ("Jet", "genJetIdx")])
def func(
    events: ak.Array,
    variations: variation.Variation,
    ceval_puid: correctionlib_evaluator,
    cfg,
    doVariations: bool = False,
):
    wrap_c = correctionlib_wrapper(ceval_puid["PUJetID_eff"])
    jets = ak.copy(events.Jet)

    btagged = (jets.btagDeepFlavB >= cfg["bTag"][f"btag{cfg["bVeto"]["wp"]}"])
    genmatched = (jets.genJetIdx >= 0) & (jets.genJetIdx < ak.num(events.GenJet))

    mask = btagged & ~jets.pass_highPt
    jets = ak.mask(jets, mask)
    
    eta = ak.copy(jets.eta)
    pt = ak.copy(jets.pt)
    
    minpt, maxpt = 12.5001, 57.4999
    pt = ak.where(pt < minpt, minpt, pt)
    pt = ak.where(pt > maxpt, maxpt, pt)

    if not doVariations:
        sf = ak.Array(wrap_c(eta, pt, "nom", "L"))
        eff_mc = ak.Array(wrap_c(eta, pt, "MCEff", "L"))
        eff_data = sf*eff_mc
        ones = ak.ones_like(pt)

        puidSF = ak.where(jets.pass_puId & genmatched, sf, ak.ones_like(sf))
        puidSF = ak.where(~jets.pass_puId, (ones-eff_data) / (ones-eff_mc), puidSF)

        events[("Jet", "puidSF")] = ak.fill_none(puidSF, 1.0)
    else:        
        sf_up = ak.Array(wrap_c(eta, pt, "up", "L"))
        sf_down = ak.Array(wrap_c(eta, pt, "down", "L"))
        
        eff_mc = ak.Array(wrap_c(eta, pt, "MCEff", "L"))
        eff_data_up = sf_up*eff_mc
        eff_data_down = sf_down*eff_mc
        ones = ak.ones_like(pt)

        puidSF_up = ak.where(jets.pass_puId & genmatched, sf_up, ak.ones_like(sf_up))
        puidSF_up = ak.where(~jets.pass_puId, (ones-eff_data_up) / (ones-eff_mc), puidSF_up)
        puidSF_down = ak.where(jets.pass_puId & genmatched, sf_down, ak.ones_like(sf_down))
        puidSF_down = ak.where(~jets.pass_puId, (ones-eff_data_down) / (ones-eff_mc), puidSF_down)

        events[("Jet", "puidSF_up")] = ak.fill_none(puidSF_up, 1.0)
        events[("Jet", "puidSF_down")] = ak.fill_none(puidSF_down, 1.0)
        events[("Jet", "puidSF_before")] = ak.fill_none(ones, 1.0)

        variations.register_variation([("Jet", "puidSF")], "puidSF_up", format_rule=format_rule)
        variations.register_variation([("Jet", "puidSF")], "puidSF_down", format_rule=format_rule)
        variations.register_variation([("Jet", "puidSF")], "puidSF_before", format_rule=format_rule)

    return events, variations


def puid_sf(events, variations, ceval_puid, cfg):
    events, variations = func(events, variations, ceval_puid, cfg, doVariations=False)
    events, variations = func(events, variations, ceval_puid, cfg, doVariations=True)

    return events, variations
