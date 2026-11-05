import awkward as ak
import numpy as np
from spritz.framework.variation import Variation
from spritz.lookup_tools import rochester_lookup, txt_converters


def get_rochester(cfg):
    rochester_file = cfg["rochester_file"]
    rochester_data = txt_converters.convert_rochester_file(
        rochester_file, loaduncs=True
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)
    return rochester


def vary_rochester(events, variations, is_data, rochester, do_stat=False, scale_only=False, max_pt=None):
    corr = lambda s, m: rochester_corrections(events, is_data, rochester, s, m, scale_only, max_pt)  # noqa: E731
    scaleFactors = {f"set{s}": corr(s, 0) for s in [0, 2, 3, 4, 5]}
    if do_stat:
        scaleFactors["set1"] = [corr(1, i) for i in range(100)]
    
    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    mu_mask = abs(events.Lepton.pdgId) == 13

    if do_stat:
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


def correct_rochester(events, variations, is_data, rochester, s=5, m=0, scale_only=False, max_pt=None):
    muSF = rochester_corrections(events, is_data, rochester, s, m, scale_only, max_pt)
    mu_pt = muSF * events.Muon.pt
    mu_idx = ak.to_packed(events.Lepton.muonIdx)
    mu_pt = mu_pt[mu_idx]
    mu_mask = abs(events.Lepton.pdgId) == 13

    vcol = Variation.format_varied_column(("Lepton", "pt"), "rochester_before")
    events[vcol] = ak.copy(events.Lepton.pt)
    events[("Lepton", "pt")] = ak.where(mu_mask, mu_pt, events.Lepton.pt)
    
    variations.register_variation(
            columns=[("Lepton","pt")], variation_name="rochester_before"
        )
    return events, variations


def rochester_corrections(events, is_data, rochester, s, m, scale_only=False, max_pt=None):
    """Rochester factor on the muon pt. scale_only: in MC, only the scale
    part (kScaleMC), without the resolution correction (kSpreadMC/kSmearMC);
    data are always scale-only (kScaleDT). max_pt: 1 for muons with pt above
    it (e.g. the high-pT recipe: Rochester only below 200 GeV)"""
    muons = events.Muon
    muons["charge"] = muons.pdgId / (-abs(muons.pdgId))

    if is_data:
        muSF = rochester.kScaleDT(
            muons["charge"], muons["pt"], muons["eta"], muons["phi"], s, m
        )
    elif scale_only:
        muSF = rochester.kScaleMC(
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

    if max_pt is not None:
        muSF = ak.where(muons.pt < max_pt, muSF, 1.0)
    return muSF
