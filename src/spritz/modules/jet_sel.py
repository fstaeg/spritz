import awkward as ak
import numba
import numpy as np


def jet_sel(events, cfg):
    if "2016" not in cfg["era"]:
        puId_shift = 1 << 2
        maxeta = 2.5
    else:
        puId_shift = 1 << 0
        maxeta = 2.4

    jet = ak.copy(events.Jet)

    select = (abs(events.Jet.eta) <= maxeta) & (events.Jet.jetId >= 2)
    events["Jet"] = events.Jet[select]
    
    pass_puId = ak.values_astype(events.Jet.puId & puId_shift, bool)

    events[("Jet", "pass_puId")] = pass_puId & (events.Jet.pt < 50.)
    events[("Jet", "pass_highPt")] = (events.Jet.pt >= 50.)

    return events


@numba.njit
def goodJet_kernel(jet, lepton, builder):
    for ievent in range(len(jet)):
        builder.begin_list()
        for ijet in range(len(jet[ievent])):
            dRs = np.ones(len(lepton[ievent])) * 10
            for ipart in range(len(lepton[ievent])):
                single_jet = jet[ievent][ijet]
                single_lepton = lepton[ievent][ipart]
                dRs[ipart] = single_jet.deltaR(single_lepton)
            builder.boolean(~np.any(dRs < 0.3))
        builder.end_list()
    return builder


def goodJet_func(jets, leptons):
    if ak.backend(jets) == "typetracer":
        # here we fake the output of find_4lep_kernel since
        # operating on length-zero data returns the wrong layout!
        ak.typetracer.length_zero_if_typetracer(
            jets.pt
        )  # force touching of the necessary data
        return ak.Array(ak.Array([[True]]).layout.to_typetracer(forget_length=True))

    return goodJet_kernel(jets, leptons, ak.ArrayBuilder()).snapshot()


def clean_jet(events):
    mask = goodJet_func(events.Jet, events.Lepton[events.Lepton.pt >= 10])
    mask = ak.values_astype(mask, bool, including_unknown=True)

    events["Jet"] = events.Jet[mask]
    return events
