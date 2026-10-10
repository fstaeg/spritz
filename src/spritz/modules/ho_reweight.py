import awkward as ak
import numpy as np


def gen_boson(events):
    """Last copy of the generator-level Z (option type: None if absent)."""
    gen = events.GenPart
    is_last_Z = (gen.pdgId == 23) & ((gen.statusFlags & 8192) > 0)
    return ak.firsts(gen[is_last_Z]), ak.firsts(ak.local_index(gen)[is_last_Z])


def gen_leptons(events, z_idx):
    """Pre-FSR (l-, l+) from the decay of the gen Z. With PHOTOS the Z
    daughters are the post-FSR leptons (status 1, isHardProcess), the FSR
    photons and, only if something was radiated, copies of the leptons
    before FSR (status 746): use those when present."""
    gen = events.GenPart
    abs_id = abs(gen.pdgId)
    daughters = gen[
        (gen.genPartIdxMother == ak.fill_none(z_idx, -2))
        & ((abs_id == 11) | (abs_id == 13) | (abs_id == 15))
    ]
    has_pre_fsr = ak.any(daughters.status == 746, axis=1)
    is_born = ak.where(has_pre_fsr, daughters.status == 746, (daughters.statusFlags & 128) > 0)
    leptons = daughters[is_born]
    leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
    return ak.firsts(leptons[leptons.pdgId > 0]), ak.firsts(leptons[leptons.pdgId < 0])


def cos_theta_cs(l_minus, l_plus):
    """Collins-Soper cos(theta*) of the negative lepton, z axis along sign(pz_ll)."""
    ll = l_minus + l_plus
    sign = np.where(ll.pz >= 0, 1.0, -1.0)
    return (
        sign * 2 * (l_minus.pz * l_plus.E - l_minus.E * l_plus.pz)
        / (ll.mass * np.sqrt(ll.mass**2 + ll.pt**2))
    )


# Gen observables that can be used as map axes; costhetastar needs the leptons
OBSERVABLES = {
    "pt": lambda Z, lep: Z.pt,
    "rapidity": lambda Z, lep: Z.rapidity,
    "absrapidity": lambda Z, lep: abs(Z.rapidity),
    "mll": lambda Z, lep: Z.mass,
    "costhetastar": lambda Z, lep: cos_theta_cs(*lep),
}

# Axes assumed when the correction does not specify "observable"
DEFAULT_OBSERVABLE = {1: "mll", 2: "pt:mll", 3: "pt:rapidity:mll"}


def get_axes(correction, ndim):
    """Map axes from correction["observable"], colon separated in the order
    of the histogram axes (e.g. "pt:rapidity:mll")"""
    observable = correction.get("observable")
    if observable is None:
        if ndim not in DEFAULT_OBSERVABLE:
            raise ValueError(f"Unsupported histogram dimensionality: {ndim}")
        observable = DEFAULT_OBSERVABLE[ndim]

    axes = observable.split(":")
    if len(axes) != ndim:
        raise ValueError(f"observable '{observable}' has {len(axes)} axes but the histogram is {ndim}D")
    unknown = [ax for ax in axes if ax not in OBSERVABLES]
    if unknown:
        raise ValueError(f"Unknown observable(s) {unknown}; supported: {list(OBSERVABLES)}")
    return axes


def HO_reweight(events, variations, correction):
    weights, weights_err = correction["weight"], correction["weight_err"]
    edges, name = correction["edges"], correction["name"]
    axes = get_axes(correction, weights.ndim)

    gen_Z, z_idx = gen_boson(events)
    valid = ~ak.is_none(gen_Z)
    lep = None
    if "costhetastar" in axes:
        lep = gen_leptons(events, z_idx)
        valid = valid & ~ak.is_none(lep[0]) & ~ak.is_none(lep[1])
    valid = np.asarray(ak.fill_none(valid, False))
    if not valid.all():
        print(f"HO_reweight {name}: {(~valid).sum()}/{len(valid)} events without a gen Z (or its leptons) get weight 1")

    # Events outside the map are clamped to the nearest edge bin
    indices = []
    for iax, ax in enumerate(axes):
        values = np.asarray(ak.fill_none(OBSERVABLES[ax](gen_Z, lep), 0.0), dtype=np.float64)
        idx = np.digitize(values, edges[iax]) - 1
        indices.append(np.clip(idx, 0, weights.shape[iax] - 1))

    ho_weights = np.where(valid, weights[tuple(indices)], 1.0)
    ho_weights_err = np.where(valid, weights_err[tuple(indices)], 0.0)

    events[name] = ho_weights
    events[f"{name}_{name}_up"] = ho_weights + ho_weights_err
    events[f"{name}_{name}_down"] = ho_weights - ho_weights_err
    events[f"{name}_{name}_before"] = ak.ones_like(events.weight)

    variations.register_variation([name], f"{name}_up")
    variations.register_variation([name], f"{name}_down")
    variations.register_variation([name], f"{name}_before")

    return events, variations
