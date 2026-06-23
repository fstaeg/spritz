import awkward as ak
import spritz.framework.variation as variation_module
import correctionlib
from spritz.framework.framework import correctionlib_wrapper


def scale_factor(tag, wp, jets, wrap_sf_l, wrap_sf_cb):
    abseta = ak.where(abs(jets.eta) >= 2.5, 2.49999, abs(jets.eta))
    pt = ak.copy(jets.pt)
    
    btag_sf = ak.ones_like(pt)
    btag_sf = ak.where(
        jets.hadronFlavour == 0,
        ak.Array(wrap_sf_l(tag, wp, 0, abseta, pt)),
        btag_sf
    )
    btag_sf = ak.where(
        jets.hadronFlavour == 4,
        ak.Array(wrap_sf_cb(tag, wp, 4, abseta, pt)),
        btag_sf
    )
    btag_sf = ak.where(
        jets.hadronFlavour == 5,
        ak.Array(wrap_sf_cb(tag, wp, 5, abseta, pt)),
        btag_sf
    )

    return btag_sf

@variation_module.vary(reads_columns=[("Jet", "pt"), ("Jet", "eta")])
def func(events, variations, ceval_btag, ceval_btageff, cfg, dataset, wp, doVariations: bool = False):
    wrap_sf_l = correctionlib_wrapper(ceval_btag["deepJet_incl"])
    wrap_sf_cb = correctionlib_wrapper(ceval_btag["deepJet_mujets"])

    if dataset.startswith("TT") or dataset.startswith("ST"):
        wrap_eff = correctionlib_wrapper(ceval_btageff["btag_efficiency_Top"])
    else:
        wrap_eff = correctionlib_wrapper(ceval_btageff["btag_efficiency_Electroweak"])

    abseta = ak.copy(abs(events.Jet.eta))
    pt = ak.copy(events.Jet.pt)
    ones = ak.ones_like(events.Jet.pt)

    if not doVariations:
        btag_sf = scale_factor("central", wp[0], events.Jet, wrap_sf_l, wrap_sf_cb)
        btag_eff = ak.Array(wrap_eff("central", wp[0], events.Jet.hadronFlavour, abseta, pt))

        p_data = ak.where(
            events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{wp}"],
            btag_sf*btag_eff,
            ones-btag_sf*btag_eff
        )
        p_mc = ak.where(
            events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{wp}"],
            btag_eff,
            ones-btag_eff
        )
        
        events[("Jet", "btagSF")] = p_data/p_mc
        varied_col = variation_module.Variation.format_varied_column(
            ("Jet", "btagSF"), "btagSF_before"
        )
        events[varied_col] = ones
        variations.register_variation([("Jet", "btagSF")], "btagSF_before")

    else:
        btag_sf = {"central": scale_factor("central", wp[0], events.Jet, wrap_sf_l, wrap_sf_cb)}
        btag_eff = {"central": ak.Array(wrap_eff("central", wp[0], events.Jet.hadronFlavour, abseta, pt))}
        btag_eff_stat = ak.Array(wrap_eff("stat", wp[0], events.Jet.hadronFlavour, abseta, pt))
        
        p_data,p_mc = {},{}

        for sign,tag in zip([+1,-1], ["up","down"]):
            btag_sf[tag] = scale_factor(tag, wp[0], events.Jet, wrap_sf_l, wrap_sf_cb)
            btag_eff[tag] = btag_eff["central"] + sign*btag_eff_stat

            p_data[tag] = {
                "sf": ak.where(
                    events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{wp}"],
                    btag_sf[tag]*btag_eff["central"],
                    ones-btag_sf[tag]*btag_eff["central"]
                ),
                "eff": ak.where(
                    events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{wp}"],
                    btag_sf["central"]*btag_eff[tag],
                    ones-btag_sf["central"]*btag_eff[tag]
                )
            }
            p_mc[tag] = {
                "sf": ak.where(
                    events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{wp}"],
                    btag_eff["central"],
                    ones-btag_eff["central"]
                ),
                "eff": ak.where(
                    events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{wp}"],
                    btag_eff[tag],
                    ones-btag_eff[tag]
                )
            }

            for syst in ["sf", "eff"]:
                varied_col = variation_module.Variation.format_varied_column(
                    ("Jet", "btagSF"), f"btagSF_{syst}_{tag}"
                )
                events[varied_col] = p_data[tag][syst]/p_mc[tag][syst]
                variations.register_variation([("Jet", "btagSF")], f"btagSF_{syst}_{tag}")
        
    return events, variations


def btag_sf(events, variations, ceval_btag, ceval_btageff, cfg, dataset, wp="Medium"):
    events,variations = func(events, variations, ceval_btag, ceval_btageff, cfg, dataset, wp, doVariations=False)
    events,variations = func(events, variations, ceval_btag, ceval_btageff, cfg, dataset, wp, doVariations=True)
    
    return events, variations
