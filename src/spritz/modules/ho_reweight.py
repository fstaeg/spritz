import awkward as ak
import numpy as np

def HO_reweight(events, variations, correction):
    weights, weights_err = correction["weight"], correction["weight_err"]
    edges, name = correction["edges"], correction["name"]

    # LHELeptons
    lhe_ele_mask = (abs(events.LHEPart.pdgId) == 11)
    lhe_mu_mask = (abs(events.LHEPart.pdgId) == 13)
    lhe_tau_mask = (abs(events.LHEPart.pdgId) == 15)
    lhe_lep_mask = (lhe_ele_mask | lhe_mu_mask | lhe_tau_mask)

    lhe_leptons = events.LHEPart[lhe_lep_mask]
    lhe_Z = (lhe_leptons[:, 0] + lhe_leptons[:, 1])

    # GenDressedLeptons
    gen_ele_mask = (abs(events.GenDressedLepton.pdgId) == 11)
    gen_mu_mask = (abs(events.GenDressedLepton.pdgId) == 13)
    gen_tau_mask = (abs(events.GenDressedLepton.pdgId) == 15)
    gen_lep_mask = (gen_ele_mask | gen_mu_mask | gen_tau_mask)
    
    gen_leptons = events.GenDressedLepton[gen_lep_mask]
    gen_leptons = ak.pad_none(gen_leptons, 2)
    gen_Z = (gen_leptons[:, 0] + gen_leptons[:, 1])

    # Is there a pair of GenDressedLeptons with same flavour as the LHELeptons?
    elepair_mask = (
        (ak.num(events.GenDressedLepton[gen_ele_mask])==2) &
        (ak.num(events.LHEPart[lhe_ele_mask])==2)
    )
    mupair_mask = (
        (ak.num(events.GenDressedLepton[gen_mu_mask])==2) &
        (ak.num(events.LHEPart[lhe_mu_mask])==2)
    )
    taupair_mask = (
        (ak.num(events.GenDressedLepton[gen_tau_mask])==2) &
        (ak.num(events.LHEPart[lhe_tau_mask])==2)
    )
    genpair_mask = (elepair_mask | mupair_mask | taupair_mask)
    
    Z = ak.where(genpair_mask, gen_Z, lhe_Z)

    # Reweight
    ndim = weights.ndim

    if ndim == 1:
        # axes: (mll)
        mll = np.asarray(Z.mass)
        i0 = np.clip(np.digitize(mll, edges[0]) - 1, 0, weights.shape[0] - 1)
        ho_weights = weights[i0]
        ho_weights_err = weights_err[i0]

    elif ndim == 2:
        # axes: (pt, mll)
        pt = np.asarray(Z.pt)
        mll = np.asarray(Z.mass)
        i0 = np.clip(np.digitize(pt, edges[0]) - 1, 0, weights.shape[0] - 1)
        i1 = np.clip(np.digitize(mll, edges[1]) - 1, 0, weights.shape[1] - 1)
        ho_weights = weights[i0, i1]
        ho_weights_err = weights_err[i0, i1]

    elif ndim == 3:
        # axes: (pt, rapidity, mll)
        pt = np.asarray(Z.pt)
        rap = np.asarray(Z.rapidity)
        mll = np.asarray(Z.mass)
        i0 = np.clip(np.digitize(pt, edges[0]) - 1, 0, weights.shape[0] - 1)
        i1 = np.clip(np.digitize(rap, edges[1]) - 1, 0, weights.shape[1] - 1)
        i2 = np.clip(np.digitize(mll, edges[2]) - 1, 0, weights.shape[2] - 1)
        ho_weights = weights[i0, i1, i2]
        ho_weights_err = weights_err[i0, i1, i2]

    else:
        raise ValueError(f"Unsupported histogram dimensionality: {ndim}")

    events[name] = ho_weights
    events[f"{name}_{name}_up"] = ho_weights + ho_weights_err
    events[f"{name}_{name}_down"] = ho_weights - ho_weights_err
    events[f"{name}_{name}_before"] = ak.ones_like(events.weight)

    variations.register_variation([name], f"{name}_up")
    variations.register_variation([name], f"{name}_down")
    variations.register_variation([name], f"{name}_before")

    return events, variations
