import sys

import awkward as ak
import numpy as np
from spritz.framework.variation import Variation
from spritz.modules.rochester import rochester_corrections

NOMINAL_SET = 5  # as correct_rochester


def resolution_ratio(rochester, abseta, s=NOMINAL_SET, m=0):
    """r = kRes_data / kRes_MC of Rochester set s, member m"""
    kres = rochester._kRes[s][m]
    return kres[1](abseta) / kres[0](abseta)  # type 1 is data, type 0 is MC


def vary_muon_resolution(events, variations, is_data, rochester, name="mu_res", s=NOMINAL_SET, m=0):
    """Register, in MC, the variation {name} of Lepton.pt: the muons get the
    full Rochester correction of set s, member m (scale and resolution, as
    correct_rochester without scale_only), the other leptons keep their pt.
    Call it after correct_rochester(..., scale_only=True), whose nominal is the
    same correction without the resolution part, so that the variation is the
    resolution correction alone."""
    if is_data:
        return events, variations
    muSF = rochester_corrections(events, is_data, rochester, s, m)
    mu_pt = (muSF * events.Muon.pt)[ak.to_packed(events.Lepton.muonIdx)]
    mu_mask = abs(events.Lepton.pdgId) == 13
    events[Variation.format_varied_column(("Lepton", "pt"), name)] = ak.where(mu_mask, mu_pt, events.Lepton.pt)
    variations.register_variation(columns=[("Lepton", "pt")], variation_name=name)
    return events, variations


if __name__ == "__main__":
    from spritz.modules.rochester import get_rochester

    rochester = get_rochester({"rochester_file": sys.argv[1]})
    edges = next([float(x) for x in line.split()[2:]] for line in open(sys.argv[1]) if line.startswith("RETA"))
    centers = np.array([0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
    r = resolution_ratio(rochester, centers)
    print(f"{'|eta|':>11} {'r (set ' + str(NOMINAL_SET) + ')':>10} {'r - 1':>8}")
    for i in range(len(centers)):
        print(f"{edges[i]:4.2f}-{edges[i + 1]:4.2f} {r[i]:10.4f} {r[i] - 1:8.2%}")
