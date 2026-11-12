import awkward as ak
import numpy as np
from spritz.framework.variation import Variation
from spritz.lookup_tools import rochester_lookup, txt_converters


def get_rochester(cfg):
    rochester_file = cfg["rochester_file"]
    rochester_data = txt_converters.convert_rochester_file(rochester_file, loaduncs=True)
    return rochester_lookup.rochester_lookup(rochester_data)


def rochester_corrections(events, is_data, rochester, s, m, scale_only=False):
    """scale_only: in MC, only the scale part (kScaleMC), without the resolution 
    correction (kSpreadMC/kSmearMC)"""
    muons = events.Muon
    muons["charge"] = muons.pdgId / (-abs(muons.pdgId))

    if is_data:
        muSF = rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi, s, m)
    elif scale_only:
        muSF = rochester.kScaleMC(muons.charge, muons.pt, muons.eta, muons.phi, s, m)
    else:
        muons["right_genPartIdx"] = ak.mask(
            muons.genPartIdx, (muons.genPartIdx >= 0) & (muons.genPartIdx < ak.num(events.GenPart)),
        )
        # if reco pt has corresponding gen pt
        mcSF1 = rochester.kSpreadMC(
            muons.charge, muons.pt, muons.eta, muons.phi, 
            events.GenPart[muons.right_genPartIdx].pt, s, m
        )
        # if reco pt has no corresponding gen pt
        counts = ak.num(muons["pt"])
        rng = np.random.default_rng(seed=0)
        mc_rand = ak.unflatten(rng.uniform(size=ak.sum(counts)), counts)
        mcSF2 = rochester.kSmearMC(
            muons.charge, muons.pt, muons.eta, muons.phi,
            muons.nTrackerLayers, mc_rand, s, m
        )
        # Combine the two scale factors and scale the pt
        muSF = ak.where(ak.is_none(muons.right_genPartIdx, axis=1), mcSF2, mcSF1)
        # Remove masking from layout, none of the SF are masked here
        muSF = ak.fill_none(muSF, 1.0)

    return muSF

