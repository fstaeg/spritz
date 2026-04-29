import awkward as ak
import numpy as np
from spritz.framework.variation import Variation
from spritz.lookup_tools import rochester_lookup, txt_converters


def getRochester(cfg):
    rochester_file = cfg["rochester_file"]
    rochester_data = txt_converters.convert_rochester_file(
        rochester_file, loaduncs=True
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)
    return rochester


def varyRochester(events, variations, is_data, rochester):
    scaleFactors = {
        "set0": RochesterCorrections(events, is_data, rochester, s=0, m=0),
        "set1": [RochesterCorrections(events, is_data, rochester, s=1, m=i) for i in range(100)],
        "set2": RochesterCorrections(events, is_data, rochester, s=2, m=0),
        "set3": RochesterCorrections(events, is_data, rochester, s=3, m=0),
        "set4": RochesterCorrections(events, is_data, rochester, s=4, m=0),
        "set5": RochesterCorrections(events, is_data, rochester, s=5, m=0),
    }
    
    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    mu_mask = abs(events.Lepton.pdgId) == 13

    for member_i in range(100):
        muSF = scaleFactors["set1"][member_i] / scaleFactors["set0"] * scaleFactors["set5"]
        mu_pt = muSF * events.Muon.pt
        mu_pt = mu_pt[mu_idx]

        vcol = Variation.format_varied_column(("Lepton", "pt"), f"rochester_stat{member_i}")
        events[vcol] = ak.where(mu_mask, mu_pt, events.Lepton.pt)
        variations.register_variation(
            columns=[("Lepton","pt")], variation_name=f"rochester_stat{member_i}"
        )
    
    for set_i in ["set2","set3","set4"]:
        muSF = scaleFactors[set_i] / scaleFactors["set0"] * scaleFactors["set5"]
        mu_pt = muSF * events.Muon.pt
        mu_pt = mu_pt[mu_idx]

        vcol = Variation.format_varied_column(("Lepton", "pt"), f"rochester_{set_i}")
        events[vcol] = ak.where(mu_mask, mu_pt, events.Lepton.pt)
        variations.register_variation(
            columns=[("Lepton","pt")], variation_name=f"rochester_{set_i}"
        )

    return events, variations


def correctRochester(events, variations, is_data, rochester, s=5, m=0):
    muSF = RochesterCorrections(events, is_data, rochester, s, m)
    mu_pt = muSF * events.Muon.pt
    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    mu_pt = mu_pt[mu_idx]
    mu_mask = abs(events.Lepton.pdgId) == 13

    vcol = Variation.format_varied_column(("Lepton", "pt"), "rochester_before")
    events[vcol] = ak.copy(events.Lepton.pt)
    events[("Lepton", "pt")] = ak.where(mu_mask, mu_pt, events.Lepton.pt)
    
    variations.register_variation(
            columns=[("Lepton","pt")], variation_name=f"rochester_before"
        )
    return events, variations


def RochesterCorrections(events, is_data, rochester, s, m):
    print(f"Rochester corrections, s={s}, m={m}")
    muons = events.Muon
    muons["charge"] = muons.pdgId / (-abs(muons.pdgId))

    if is_data:
        muSF = rochester.kScaleDT(
            muons["charge"], muons["pt"], muons["eta"], muons["phi"], s, m
        )
    else:
        muons["right_genPartIdx"] = ak.mask(
            muons.genPartIdx,
            (muons.genPartIdx >= 0) & (muons.genPartIdx < ak.num(events.GenPart)),
        )
        # if reco pt has corresponding gen pt
        mcSF1 = rochester.kSpreadMC(
            muons["charge"],
            muons["pt"],
            muons["eta"],
            muons["phi"],
            events.GenPart[muons.right_genPartIdx].pt,
            s, 
            m
        )
        # if reco pt has no corresponding gen pt
        counts = ak.num(muons["pt"])
        #mc_rand = np.random.uniform(size=ak.sum(counts))
        rng = np.random.default_rng(seed=0)
        mc_rand = rng.uniform(size=ak.sum(counts))
        mc_rand = ak.unflatten(mc_rand, counts)
        mcSF2 = rochester.kSmearMC(
            muons["charge"],
            muons["pt"],
            muons["eta"],
            muons["phi"],
            muons["nTrackerLayers"],
            mc_rand,
            s,
            m
        )
        # Combine the two scale factors and scale the pt
        muSF = ak.where(ak.is_none(muons.right_genPartIdx, axis=1), mcSF2, mcSF1)
        # Remove masking from layout, none of the SF are masked here
        muSF = ak.fill_none(muSF, 1.0)
    
    return muSF
