import awkward as ak

def prefireweight(events, variations):
    if "L1PreFiringWeight" in ak.fields(events):
        events["prefireWeightMuon"] = events.L1PreFiringWeight.Muon_Nom
        events["prefireWeightMuon_stat_up"] = events.L1PreFiringWeight.Muon_StatUp
        events["prefireWeightMuon_stat_down"] = events.L1PreFiringWeight.Muon_StatDn
        events["prefireWeightMuon_syst_up"] = events.L1PreFiringWeight.Muon_SystUp
        events["prefireWeightMuon_syst_down"] = events.L1PreFiringWeight.Muon_SystDn
        events["prefireWeightMuon_before"] = ak.ones_like(events.weight)

        for tag in ["stat_up", "stat_down", "syst_up", "syst_down", "before"]:
            variations.register_variation(
                columns=["prefireWeightMuon"],
                variation_name=f"prefireWeightMuon_{tag}",
                format_rule=lambda _, var_name: var_name,
            )

        events["prefireWeightECAL"] = events.L1PreFiringWeight.ECAL_Nom
        events["prefireWeightECAL_up"] = events.L1PreFiringWeight.ECAL_Up
        events["prefireWeightECAL_down"] = events.L1PreFiringWeight.ECAL_Dn
        events["prefireWeightECAL_before"] = ak.ones_like(events.weight)

        for tag in ["up", "down", "before"]:
            variations.register_variation(
                columns=["prefireWeightECAL"],
                variation_name=f"prefireWeightECAL_{tag}",
                format_rule=lambda _, var_name: var_name,
            )

        # events["prefireWeight"] = events.L1PreFiringWeight.Nom
        # events["prefireWeight_up"] = events.L1PreFiringWeight.Up
        # events["prefireWeight_down"] = events.L1PreFiringWeight.Dn
        
        # events["prefireWeight_before"] = ak.ones_like(events.L1PreFiringWeight.Nom)

        # for tag in ["up", "down", "before"]:
        #     variations.register_variation(
        #         columns=["prefireWeight"],
        #         variation_name=f"prefireWeight_{tag}",
        #         format_rule=lambda _, var_name: var_name,
        #     )
    else:
        events["prefireWeightMuon"] = ak.ones_like(events.weight)
        events["prefireWeightECAL"] = ak.ones_like(events.weight)
        # events["prefireWeight"] = ak.ones_like(events.weight)


    return events, variations
