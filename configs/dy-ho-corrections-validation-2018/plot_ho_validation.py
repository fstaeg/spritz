"""
Compare, for every region/variable and for DYmm, DYtt and their sum:
    bare MiNNLO, MiNNLO + NLO EW, MiNNLO + NNLO+N3LL QCD, MiNNLO + both
from histos.pkl (spritz-postproc). All four come from the same events (the
"<name>Before" corrections of the nominal), see config.py. The band on the
fully corrected prediction is the NLO EW (+) NNLO+N3LL QCD uncertainty.

Run from the config directory:
    python plot_ho_validation.py
"""

import os
import pickle
from functools import reduce

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np

from spritz.framework.framework import get_analysis_dict

mh.style.use("CMS")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "plots_ho_validation")

# (label, histogram suffix, color, linestyle)
PREDICTIONS = [
    ("MiNNLO", "_hoBefore", "black", "-"),
    ("MiNNLO + NLO EW", "_NNLO_N3LL_QCDBefore", "tab:blue", "--"),
    ("MiNNLO + NNLO+N3LL QCD", "_NLO_EWBefore", "tab:orange", "-."),
    ("MiNNLO + NLO EW + NNLO+N3LL QCD", "", "tab:red", "-"),
]
NUISANCES = ["NLO_EW", "NNLO_N3LL_QCD"]


def get(histos, prefix, sample, suffix):
    if sample == "DY":
        return sum(get(histos, prefix, s, suffix) for s in ("DYmm", "DYtt"))
    return histos[f"{prefix}_{sample}{suffix}"].values()


def get_variance(histos, prefix, sample, suffix):
    """Sum of squared weights (MC statistical variance)."""
    if sample == "DY":
        return sum(get_variance(histos, prefix, s, suffix) for s in ("DYmm", "DYtt"))
    return histos[f"{prefix}_{sample}{suffix}"].variances()


def bin_widths(var_cfg):
    """Bin widths (bin volumes for N-dim variables, in the unrolling order of spritz-postproc)."""
    axes = var_cfg["axis"] if isinstance(var_cfg["axis"], list) else [var_cfg["axis"]]
    return reduce(np.multiply.outer, [a.widths for a in axes]).T.flatten()


def plot(histos, region, variable, var_cfg, sample, lumi):
    prefix = f"{region}/{variable}/histo"
    h_nom = histos[f"{prefix}_DYmm"]
    edges = h_nom.axes[0].edges.copy()
    unrolled = isinstance(var_cfg.get("axis"), list)
    xlog = var_cfg.get("xlog", False) and not unrolled
    if xlog and edges[0] <= 0:
        edges[0] = edges[1] / 2  # keep the first bin visible on a log axis

    preds = {label: get(histos, prefix, sample, suffix) for label, suffix, _, _ in PREDICTIONS}
    bare = preds["MiNNLO"]
    nom = preds[PREDICTIONS[-1][0]]

    # HO uncertainty on the fully corrected prediction, EW and QCD in quadrature
    up2, down2 = np.zeros_like(nom), np.zeros_like(nom)
    for nuis in NUISANCES:
        shifts = np.array([get(histos, prefix, sample, f"_{nuis}{tag}") - nom for tag in ("Up", "Down")])
        up2 += np.max([shifts.max(axis=0), np.zeros_like(nom)], axis=0) ** 2
        down2 += np.min([shifts.min(axis=0), np.zeros_like(nom)], axis=0) ** 2
    up, down = nom + np.sqrt(up2), nom - np.sqrt(down2)

    # MiNNLO MC statistical uncertainty, from the uncorrected prediction
    bare_stat = np.sqrt(get_variance(histos, prefix, sample, PREDICTIONS[0][1]))

    ratio = lambda a: np.divide(a, bare, out=np.full_like(a, np.nan), where=bare != 0)  # noqa: E731
    units = [u for u in np.atleast_1d(var_cfg.get("unit", "")) if u]
    widths = bin_widths(var_cfg)
    figsize = (28, 11) if unrolled else (12, 11)

    # one plot with events per bin, one divided by the bin width (volume for unrolled variables)
    for per_width in (False, True):
        scale = 1 / widths if per_width else 1
        fig, (ax, rax) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})

        for label, _, color, ls in PREDICTIONS:
            mh.histplot(preds[label] * scale, edges, ax=ax, label=label, color=color, linestyle=ls, linewidth=2)
            mh.histplot(ratio(preds[label]), edges, ax=rax, color=color, linestyle=ls, linewidth=2)
        ax.stairs(up * scale, edges, baseline=down * scale, fill=True, color="tab:red", alpha=0.25, label="HO corr. unc.")
        rax.stairs(ratio(up), edges, baseline=ratio(down), fill=True, color="tab:red", alpha=0.25)
        stat_style = dict(fill=False, hatch="////", color="gray", linewidth=0)
        ax.stairs((bare + bare_stat) * scale, edges, baseline=(bare - bare_stat) * scale, label="MiNNLO MC stat. unc.", **stat_style)
        rax.stairs(ratio(bare + bare_stat), edges, baseline=ratio(bare - bare_stat), **stat_style)

        if per_width:
            ax.set_ylabel("Events / " + (" ".join(units) if units else "bin width"))
        else:
            ax.set_ylabel("Events / bin")
        ax.set_yscale("log")
        ax.legend(fontsize=16, title=f"{sample}, {region}", title_fontsize=16)
        mh.cms.label("Preliminary", data=False, lumi=round(lumi, 1), ax=ax, fontsize=20)
        rax.axhline(1, color="gray", lw=1)
        rax.set_ylabel("Ratio to MiNNLO")
        all_ratios = np.concatenate([ratio(down), ratio(up), *[ratio(v) for v in preds.values()]])
        rax.set_ylim(np.nanmin(all_ratios) * 0.99, np.nanmax(all_ratios) * 1.01)

        if unrolled:
            rax.set_xlabel(f"{variable} bin (unrolled)")
        else:
            unit = var_cfg.get("unit")
            rax.set_xlabel(var_cfg.get("label", variable) + (f" [{unit}]" if unit else ""))
            if xlog:
                ax.set_xscale("log")

        os.makedirs(f"{OUTDIR}/{region}", exist_ok=True)
        name = f"{OUTDIR}/{region}/{variable}_{sample}" + ("_perBinWidth" if per_width else "")
        fig.savefig(f"{name}.png", bbox_inches="tight")
        fig.savefig(f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    total = {label: v.sum() for label, v in preds.items()}
    return total


def main():
    cfg = get_analysis_dict()
    with open(os.path.join(HERE, "histos.pkl"), "rb") as f:
        histos = pickle.load(f)

    for region in cfg["regions"]:
        for variable, var_cfg in cfg["variables"].items():
            if "axis" not in var_cfg:
                continue
            for sample in ("DYmm", "DYtt", "DY"):
                total = plot(histos, region, variable, var_cfg, sample, cfg["lumi"])
            if variable == "mll":
                print(f"{region} yields (DY), relative to bare MiNNLO:")
                for label, y in total.items():
                    print(f"  {label:34s} {y:12.1f}  {y / total['MiNNLO'] - 1:+.2%}")
    print(f"plots in {OUTDIR}")


if __name__ == "__main__":
    main()
