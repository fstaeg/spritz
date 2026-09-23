import awkward as ak
import numpy as np


def HO_reweight(events, variations, weights, weights_err, edges, name="EWWeight",
                observable=None):

    # axis conventions are fixed per dimensionality here; only validate
    if observable is not None:
        expected = {1: "mll", 2: "pt:mll", 3: "pt:rapidity:mll"}.get(weights.ndim)
        if observable != expected:
            raise ValueError(
                f"gendressed_ho_reweight has fixed axes '{expected}' for "
                f"{weights.ndim}D maps, got observable '{observable}'"
            )

    muon_mask = (abs(events.GenDressedLepton.pdgId) == 13)
    gen_leptons = events.GenDressedLepton[muon_mask]
    gen_leptons = gen_leptons[ak.argsort(gen_leptons.pt, ascending=False, axis=-1)]
    gen_leptons = ak.pad_none(gen_leptons, 2)
    gen_Z = gen_leptons[:, 0] + gen_leptons[:, 1]
    gen_mll = gen_Z.mass
    is_none = ak.is_none(gen_mll)

    ndim = weights.ndim
    
    if ndim == 1:
        mll_arr = np.asarray(ak.fill_none(gen_mll, -1.0))
        i0 = np.clip(np.digitize(mll_arr, edges[0]) - 1, 0, weights.shape[0] - 1)
        nloweights = ak.where(is_none, ak.ones_like(i0), weights[i0])
        nloweights_err = ak.where(is_none, ak.zeros_like(i0), weights_err[i0])

    elif ndim == 2:
        # axes: (pt, mll)
        pt_arr = np.asarray(ak.fill_none(gen_Z.pt, -1.0))
        mll_arr = np.asarray(ak.fill_none(gen_mll, -1.0))
        i0 = np.clip(np.digitize(pt_arr, edges[0]) - 1, 0, weights.shape[0] - 1)
        i1 = np.clip(np.digitize(mll_arr, edges[1]) - 1, 0, weights.shape[1] - 1)
        nloweights = ak.where(is_none, ak.ones_like(i0), weights[i0, i1])
        nloweights_err = ak.where(is_none, ak.zeros_like(i0), weights_err[i0, i1])

    elif ndim == 3:
        # axes: (pt, rapidity, mll)
        pt_arr = np.asarray(ak.fill_none(gen_Z.pt, -1.0))
        rap_arr = np.asarray(ak.fill_none(gen_Z.rapidity, -99.0))
        mll_arr = np.asarray(ak.fill_none(gen_mll, -1.0))
        i0 = np.clip(np.digitize(pt_arr, edges[0]) - 1, 0, weights.shape[0] - 1)
        i1 = np.clip(np.digitize(rap_arr, edges[1]) - 1, 0, weights.shape[1] - 1)
        i2 = np.clip(np.digitize(mll_arr, edges[2]) - 1, 0, weights.shape[2] - 1)
        # print("PTll: ", pt_arr)
        # print("rapll: ", rap_arr)
        # print("mll: ", mll_arr)
        nloweights = ak.where(is_none, ak.ones_like(i0), weights[i0, i1, i2])
        # print("Weights: ", nloweights)
        nloweights_err = ak.where(is_none, ak.zeros_like(i0), weights_err[i0, i1, i2])

    else:
        raise ValueError(f"Unsupported histogram dimensionality: {ndim}")

    events[name] = nloweights
    events[f"{name}_nlo_up"] = nloweights + nloweights_err
    events[f"{name}_nlo_down"] = nloweights - nloweights_err

    variations.register_variation([name], "nlo_up")
    variations.register_variation([name], "nlo_down")

    events[f"{name}_nlo_before"] = ak.ones_like(nloweights)
    variations.register_variation([name], "nlo_before")

    return events, variations
