import awkward as ak

def prefireweight(events, variations):
    if "L1PreFiringWeight" in ak.fields(events):
        events["prefireWeight"] = events.L1PreFiringWeight.Nom
        events["prefireWeight_up"] = events.L1PreFiringWeight.Up
        events["prefireWeight_down"] = events.L1PreFiringWeight.Dn
        events["prefireWeight_before"] = ak.ones_like(events.L1PreFiringWeight.Nom)

        for tag in ["up", "down", "before"]:
            variations.register_variation(
                columns=["prefireWeight"],
                variation_name=f"prefireWeight_{tag}",
                format_rule=lambda _, var_name: var_name,
            )
    else:
        events["prefireWeight"] = ak.ones_like(events.weight)

    return events, variations
