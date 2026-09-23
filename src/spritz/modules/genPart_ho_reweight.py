import awkward as ak
import numpy as np

# Observables that can be used as map axes: name -> (accessor, fill_none sentinel)
_OBSERVABLES = {
    "pt": (lambda gen_Z: gen_Z.pt, -1.0),
    "rapidity": (lambda gen_Z: gen_Z.rapidity, -99.0),
    "mll": (lambda gen_Z: gen_Z.mass, -1.0),
}

# Axis conventions assumed when no observable is specified
_DEFAULT_OBSERVABLE = {1: "mll", 2: "pt:mll", 3: "pt:rapidity:mll"}


def _digitize(arr, ax_edges, nbins):
    """Bin index along one axis, clamped to the nearest edge bin if out of range."""
    raw = np.digitize(arr, ax_edges) - 1
    return np.clip(raw, 0, nbins - 1)


def HO_reweight(events, variations, weights, weights_err, edges, name="EWWeight",
                observable=None):
    """observable: colon-separated axis observables, e.g. "pt:rapidity:mll" for a
    3D map or "pt" for a 1D one. Defaults to the conventions in _DEFAULT_OBSERVABLE."""

    z_mask = (
        (events.GenPart.pdgId == 23) &
        ((events.GenPart.statusFlags & 8192) > 0)  # isLastCopy
    )
    gen_Z_cands = events.GenPart[z_mask]
    gen_Z_cands = ak.pad_none(gen_Z_cands, 1)
    gen_Z = gen_Z_cands[:, 0]
    is_none = np.asarray(ak.is_none(gen_Z.mass))

    ndim = weights.ndim
    if observable is None:
        if ndim not in _DEFAULT_OBSERVABLE:
            raise ValueError(f"Unsupported histogram dimensionality: {ndim}")
        observable = _DEFAULT_OBSERVABLE[ndim]

    axes = observable.split(":")
    if len(axes) != ndim:
        raise ValueError(
            f"observable '{observable}' has {len(axes)} axes "
            f"but the histogram is {ndim}D"
        )
    unknown = [ax for ax in axes if ax not in _OBSERVABLES]
    if unknown:
        raise ValueError(
            f"Unknown observable(s) {unknown}; supported: {list(_OBSERVABLES)}"
        )

    indices = []
    for iax, ax in enumerate(axes):
        accessor, sentinel = _OBSERVABLES[ax]
        arr = np.asarray(ak.fill_none(accessor(gen_Z), sentinel))
        idx = _digitize(arr, edges[iax], weights.shape[iax])
        indices.append(idx)

    # Events outside the map boundaries are clamped to the nearest edge bin
    # (see _digitize) rather than defaulting to 1.0; only events with no gen Z
    # candidate at all get weight 1 +- 0.
    nloweights = ak.Array(np.where(is_none, 1.0, weights[tuple(indices)]))
    nloweights_err = ak.Array(np.where(is_none, 0.0, weights_err[tuple(indices)]))

    events[name] = nloweights
    events[f"{name}_nlo_up"] = nloweights + nloweights_err
    events[f"{name}_nlo_down"] = nloweights - nloweights_err

    variations.register_variation([name], "nlo_up")
    variations.register_variation([name], "nlo_down")

    events[f"{name}_nlo_before"] = ak.ones_like(nloweights)
    variations.register_variation([name], "nlo_before")

    return events, variations
