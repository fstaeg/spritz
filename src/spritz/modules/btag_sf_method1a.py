import awkward as ak
import hist
import numpy as np
from spritz.framework.framework import correctionlib_wrapper

PT_BINS = [30, 50, 70, 100, 140, 200, 300, 600, 1000]
ETA_BINS = [0, 0.6, 1.2, 2.5]
FLAVOURS = [0, 4, 5]


# ── Efficiency map I/O ──────────────────────────────────────────────────────

def make_eff_histos():
    axes = [
        hist.axis.Variable(PT_BINS, name="pt"),
        hist.axis.Variable(ETA_BINS, name="abseta"),
        hist.axis.IntCategory(FLAVOURS, name="flav"),
    ]
    return {
        "h_total":  hist.Hist(*axes, hist.storage.Weight()),
        "h_tagged": hist.Hist(*axes, hist.storage.Weight()),
    }


def fill_eff_histos(eff_histos, jets_acc, is_tagged, gen_weight):
    """Fill genWeight-weighted jet counts in acceptance. Call for nominal, MC only.

    genWeight is used (not PU/lepton SFs) to stay consistent with the BTV SF
    definition: SF = eps_data / eps_MC where eps_MC is raw simulation weighted
    only by the generator weight (needed for NLO negative-weight events).
    """
    pts   = ak.to_numpy(ak.flatten(jets_acc.pt))
    etas  = ak.to_numpy(ak.flatten(abs(jets_acc.eta)))
    flavs = ak.to_numpy(ak.flatten(jets_acc.hadronFlavour))
    tag   = ak.to_numpy(ak.flatten(is_tagged))

    # broadcast per-event genWeight to per-jet
    w = ak.to_numpy(ak.flatten(ak.broadcast_arrays(gen_weight, jets_acc.pt)[0]))

    eff_histos["h_total"].fill(pt=pts, abseta=etas, flav=flavs, weight=w)
    eff_histos["h_tagged"].fill(pt=pts[tag], abseta=etas[tag], flav=flavs[tag], weight=w[tag])
    return eff_histos


def save_eff_maps(eff_histos, filename):
    total  = eff_histos["h_total"].values()
    tagged = eff_histos["h_tagged"].values()
    W2_tot = eff_histos["h_total"].variances()
    W2_tag = eff_histos["h_tagged"].variances()
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(total > 0, tagged / total, 0.0)
        # error propagation for ε = W_tag / W_tot (independent-source approximation)
        sigma2 = np.where(total > 0, (W2_tag + eff**2 * W2_tot) / total**2, 0.0)
    eff_err = np.sqrt(np.maximum(sigma2, 0.0))
    # Read bin edges from the histogram axes so the saved arrays are always
    # consistent with the eff shape, regardless of what PT_BINS/ETA_BINS say.
    pt_bins_out  = eff_histos["h_total"].axes["pt"].edges.tolist()
    eta_bins_out = eff_histos["h_total"].axes["abseta"].edges.tolist()
    flavours_out = list(eff_histos["h_total"].axes["flav"])
    np.savez(filename, eff=eff, eff_err=eff_err,
             pt_bins=pt_bins_out, eta_bins=eta_bins_out, flavours=flavours_out)
    print(f"Saved efficiency maps → {filename}")


def load_eff_maps(filename):
    data = np.load(filename)
    return {
        "eff":      data["eff"],
        "eff_err":  data["eff_err"],
        "pt_bins":  data["pt_bins"].tolist(),
        "eta_bins": data["eta_bins"].tolist(),
        "flavours": data["flavours"].tolist(),
    }


# ── Per-jet SF from correctionlib ────────────────────────────────────────────

def compute_fixed_wp_sf(events, ceval_btag, wp, sf_name="mujets"):
    """
    Compute per-jet fixed-WP SFs (central, up, down) for all jets in events.Jet.
    Stores them as new Jet fields and registers the up/down variations.
    Returns events, variations (variations are registered outside to avoid import issues).
    """
    # wrap_comb = correctionlib_wrapper(ceval_btag["deepJet_comb"])
    wrap_comb = correctionlib_wrapper(ceval_btag[f"deepJet_{sf_name}"])
    wrap_incl = correctionlib_wrapper(ceval_btag["deepJet_incl"])

    flav   = events.Jet.hadronFlavour
    abseta = abs(events.Jet.eta)
    pt     = events.Jet.pt

    # correctionlib is only calibrated within the tracker acceptance
    acc_mask   = abseta < 2.5
    bc_mask    = ((flav == 4) | (flav == 5)) & acc_mask
    light_mask = (flav == 0) & acc_mask

    def _sf(syst):
        sf = ak.ones_like(pt, dtype=float)
        # b and c jets
        jets_bc = ak.mask(events.Jet, bc_mask)
        sf = ak.where(
            bc_mask,
            ak.fill_none(
                wrap_comb(syst, wp, jets_bc.hadronFlavour, abs(jets_bc.eta), jets_bc.pt),
                1.0,
            ),
            sf,
        )
        # light jets
        jets_l = ak.mask(events.Jet, light_mask)
        sf = ak.where(
            light_mask,
            ak.fill_none(
                wrap_incl(syst, wp, jets_l.hadronFlavour, abs(jets_l.eta), jets_l.pt),
                1.0,
            ),
            sf,
        )
        return sf

    events[("Jet", "fixedWPSF")]              = _sf("central")
    events[("Jet", "fixedWPSF_btag_wp_up")]   = _sf("up")
    events[("Jet", "fixedWPSF_btag_wp_down")] = _sf("down")
    return events


# ── Method 1a weight ─────────────────────────────────────────────────────────

def _lookup_eff(jets_acc, eff_maps):
    """Return per-jet efficiency as an awkward array matching jets_acc."""
    eff      = eff_maps["eff"]
    pt_bins  = eff_maps["pt_bins"]
    eta_bins = eff_maps["eta_bins"]
    flavours = eff_maps["flavours"]

    pts_f  = ak.to_numpy(ak.flatten(jets_acc.pt))
    etas_f = ak.to_numpy(ak.flatten(abs(jets_acc.eta)))
    flavs_f = ak.to_numpy(ak.flatten(jets_acc.hadronFlavour))

    pt_idx   = np.digitize(np.clip(pts_f,  pt_bins[0],  pt_bins[-1]  - 1e-3), pt_bins)  - 1
    eta_idx  = np.digitize(np.clip(etas_f, eta_bins[0], eta_bins[-1] - 1e-3), eta_bins) - 1
    flav_idx = np.array([flavours.index(int(f)) if int(f) in flavours else 0 for f in flavs_f], dtype=np.intp)

    eff_vals = eff[pt_idx, eta_idx, flav_idx]
    return ak.unflatten(eff_vals, ak.num(jets_acc.pt, axis=1))


def compute_method1a_weight(jets_acc, is_tagged, sf, eff_maps):
    """
    BTV Method 1a event weight.

    w = ∏_{tagged} SF_i  ×  ∏_{not tagged} (1 - SF_j·ε_j) / (1 - ε_j)

    Unphysical negative numerators are set to 1 per BTV recommendation.
    """
    print("HELLO")
    eff = _lookup_eff(jets_acc, eff_maps)
    print("compute_method1a_weight")
    print(eff)

    w_tagged = ak.prod(ak.where(is_tagged, sf, 1.0), axis=1)

    denom = 1.0 - eff
    numer = 1.0 - sf * eff
    ratio = ak.where((denom > 1e-6) & (numer >= 0.0), numer / denom, 1.0)
    w_not_tagged = ak.prod(ak.where(~is_tagged, ratio, 1.0), axis=1)

    return w_tagged * w_not_tagged
