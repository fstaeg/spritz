"""
Compare MiNNLO GenPart distributions with DYTurbo predictions at multiple orders.

MiNNLO : configs/dy-eft-genDressed/histos.root          [pb, lumi=1 pb^-1]
DYTurbo: N3LO+N3LL and NNLO+NNLL (ewscheme3)            [pb, same units]

Run from the config directory:
    python plot_vs_dyturbo.py
"""

import os

import hist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

import mplhep as mh

mh.style.use("CMS")


# ── paths ────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
MINNLO_FILE = os.path.join(HERE, "histos.root")
OUTDIR      = os.path.join(HERE, "plots_vs_dyturbo")
os.makedirs(OUTDIR, exist_ok=True)

# List of DYTurbo predictions to overlay.  Each entry: (file_path, label, color, linestyle)
DYTURBO_SAMPLES = [
    (
        "root://eosuser.cern.ch//eos/user/g/gboldrin/www/DYTurbo/N3LO_N3LL_ewscheme3_mll_yll_qt.root",
        "DYTurbo N3LO+N3LL",
        "firebrick",
        "--",
    ),
    (
        "root://eosuser.cern.ch//eos/user/g/gboldrin/www/DYTurbo/NNLO_NNLL_ewscheme3_mll_yll_qt.root",
        "DYTurbo NNLO+NNLL",
        "darkorange",
        "-.",
    ),
]

DYTURBO_SAMPLES = [
 ("/eos/user/g/gboldrin/www/DYTurbo/run_3d_prediction/combined_result.root", "DYTurbo NNLO+N3LO", "firebrick", "--")
]

DYTURBO_SAMPLES = [
 ("/eos/user/g/gboldrin/www/DYTurbo/NNPDF_corrected_NNLO_N3LL/example.root", "DYTurbo NNLO+N3LO", "firebrick", "--")
]

DYTURBO_SAMPLES = [
 # ("/eos/user/g/gboldrin/www/DYTurbo/run_3d_prediction_NNLO_N3LL_mll_30_3500_Cuts/combined_result.root", "DYTurbo NNLO+N3LO Cuts", "firebrick", "--"),
 
 ("/eos/user/g/gboldrin/www/DYTurbo/run_3d_prediction_NNLO_N3LL_mll_30_3500_noCuts/combined_result.root", "DYTurbo NNLO+N3LO noCuts", "firebrick", "--"),
 
]




# ── binning ───────────────────────────────────────────────────────────────────

# EDGES_QT  = np.array(  [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 65, 70, 80, 90, 100], dtype=float)
# EDGES_YLL  = np.array( [ -5.0, -4.0, -3.5, -3.25, -3.0, -2.75, -2.5, -2.0, -1.8, -1.5, -1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0 ] , dtype=float)
# EDGES_MLL  = np.array( [ 60, 120 ] , dtype=float)

EDGES_QT  = np.array(  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50,75,100,150,200,250,300,400,500,600] , dtype=float)
EDGES_YLL  = np.array( [ -3.0, -2.0, -1.0 , 0.0, 1.0, 2.0, 3.0 ] , dtype=float)
EDGES_MLL  = np.array( [30, 50, 120, 200, 600, 1000, 2000, 3500 ] , dtype=float)

N_QT  = len(EDGES_QT)  - 1   # 14
N_YLL = len(EDGES_YLL) - 1   # 6
N_MLL = len(EDGES_MLL) - 1   # 12

LABEL_QT  = r"$q_T^{\rm gen}\ [\rm GeV]$"
LABEL_YLL = r"$y_{\ell\ell}^{\rm gen}$"
LABEL_MLL = r"$m_{\ell\ell}^{\rm gen}\ [\rm GeV]$"

# 1D bin widths
WIDTH_QT  = np.diff(EDGES_QT)
WIDTH_YLL = np.diff(EDGES_YLL)
WIDTH_MLL = np.diff(EDGES_MLL)

# 3D bin-width array: (qT, yll, mll)
bin_width_3d = (
    WIDTH_QT[:, None, None]
    * WIDTH_YLL[None, :, None]
    * WIDTH_MLL[None, None, :]
).transpose(2, 1, 0)

# ── helpers ───────────────────────────────────────────────────────────────────

def ratio_error_split(num, num_var, den, den_var):
    """Ratio r = num/den and its error split by source (both are std devs, not variances):
    err_num = contribution from the numerator (DYTurbo) statistics only,
    err_den = contribution from the denominator (MiNNLO) statistics only.
    Total error = sqrt(err_num**2 + err_den**2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
        err_num = np.where(den > 0, np.sqrt(num_var) / den, np.nan)
        err_den = np.where(den > 0, np.abs(r) * np.sqrt(den_var) / den, np.nan)
    return r, err_num, err_den


def rel_error(vals, var):
    """Relative statistical error sqrt(var)/vals, NaN where vals <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vals > 0, np.sqrt(var) / vals, np.nan)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_1d(edges, minnlo_vals, minnlo_var, dyturbo_list, xlabel, varname, divide=True):
    """dyturbo_list: list of (vals, var, label, color, linestyle)."""

    if divide:
        minnlo_vals = minnlo_vals/np.diff(edges)
        minnlo_var = minnlo_var /(np.diff(edges)**2)
        
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_top.stairs(minnlo_vals, edges, label="MiNNLO (GenPart)", color="steelblue", linewidth=1.5)
    ax_top.fill_between(
        np.repeat(edges, 2)[1:-1],
        np.repeat(minnlo_vals - np.sqrt(minnlo_var), 2),
        np.repeat(minnlo_vals + np.sqrt(minnlo_var), 2),
        color="steelblue", alpha=0.2,
    )

    # MiNNLO stat uncertainty: gray band around 1 in the ratio panel
    m_rel = rel_error(minnlo_vals, minnlo_var)
    ax_bot.fill_between(
        np.repeat(edges, 2)[1:-1],
        np.repeat(1 - m_rel, 2),
        np.repeat(1 + m_rel, 2),
        color="gray", alpha=0.35, linewidth=0, label="MiNNLO stat.",
    )

    for dt_vals, dt_var, dt_label, dt_color, dt_ls in dyturbo_list:
        if divide: 
            dt_vals = dt_vals / np.diff(edges)
            dt_var = dt_var / (np.diff(edges)**2)
                        
        ax_top.stairs(dt_vals, edges, label=dt_label, color=dt_color, linewidth=1.5, linestyle=dt_ls)
        # r, r_err_dt, _ = ratio_error_split(dt_vals, dt_var, minnlo_vals, minnlo_var)
        r_err_dt = rel_error(minnlo_vals, dt_var)
        r = np.where(minnlo_vals > 0, dt_vals / minnlo_vals, np.nan)
        ax_bot.stairs(r, edges, color=dt_color, linewidth=1.2, linestyle=dt_ls, label=dt_label)
        # DYTurbo-only stat uncertainty: colored band around the ratio
        ax_bot.fill_between(
            np.repeat(edges, 2)[1:-1],
            np.repeat(r + r_err_dt, 2),
            np.repeat(r - r_err_dt, 2),
            color=dt_color, alpha=0.2, linewidth=0, label=f"{dt_label} stat.",
        )
        # print((r - r_err_dt*100)/r)

    ax_top.set_ylabel(r"$d\sigma/d\mathrm{var}$")
    ax_top.legend(fontsize=15)
    ax_top.set_yscale("log")
    ax_top.set_xscale("log")
    mh.cms.label("Preliminary", data=False, com=13, ax = ax_top)
    #ax_top.set_title(varname)

    ax_bot.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    if varname == "gen_qT":
        ax_bot.set_ylim(0.8, 1.3)
    else: 
        ax_bot.set_ylim(0.92, 1.08)
    ax_bot.set_ylabel("K-factor (DYTurbo / MiNNLO)", fontsize=12)
    ax_bot.set_xlabel(xlabel)
    ax_bot.legend(fontsize=15, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUTDIR, f"{varname}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_3d_slices(minnlo_3d, minnlo_3d_var, dyturbo_list_3d, divide=True):
    """K-factor vs qT for each (mll, yll) slice. Shape: (N_QT, N_YLL, N_MLL).
    dyturbo_list_3d: list of (vals_3d, var_3d, label, color, linestyle)."""
    qt_centers = 0.5 * (EDGES_QT[:-1] + EDGES_QT[1:])
    mll_lo = EDGES_MLL[:-1]
    mll_hi = EDGES_MLL[1:]
    cmap   = plt.cm.viridis
    colors = [cmap(i / (N_YLL - 1)) for i in range(N_YLL)]

    dt_markers = ["s", "^", "D", "v"]

    for i_mll in range(N_MLL):
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        dt_labels = " vs ".join(e[2] for e in dyturbo_list_3d)
        ax_top.set_title(
            f"MiNNLO vs {dt_labels}  |  "
            f"${mll_lo[i_mll]:.0f} < m_{{\\ell\\ell}} < {mll_hi[i_mll]:.0f}$ GeV",
            fontsize=10,
        )

        for i_yll in range(N_YLL):
            m  = minnlo_3d[:, i_yll, i_mll]
            mv = minnlo_3d_var[:, i_yll, i_mll]
            if np.nansum(m) <= 0:
                continue
            label = f"$y = [{EDGES_YLL[i_yll]:.1f}, {EDGES_YLL[i_yll+1]:.1f}]$"
            ax_top.errorbar(qt_centers, m, yerr=np.sqrt(mv),
                            fmt="o", markersize=3, color=colors[i_yll],
                            label=f"MiNNLO {label}")

            # MiNNLO stat uncertainty of this slice: faint band around 1
            m_rel = rel_error(m, mv)
            ax_bot.fill_between(
                np.repeat(EDGES_QT, 2)[1:-1],
                np.repeat(1 - m_rel, 2),
                np.repeat(1 + m_rel, 2),
                color=colors[i_yll], alpha=0.12, linewidth=0,
            )

            for i_dt, (d_3d, dv_3d, dt_label, dt_color, dt_ls) in enumerate(dyturbo_list_3d):
                d  = d_3d[:, i_yll, i_mll]
                dv = dv_3d[:, i_yll, i_mll]
                if np.nansum(d) <= 0:
                    continue
                r, r_err_dt, _ = ratio_error_split(d, dv, m, mv)
                mk = dt_markers[i_dt % len(dt_markers)]
                ax_top.errorbar(qt_centers, d, yerr=np.sqrt(dv),
                                fmt=mk, markersize=3, color=colors[i_yll],
                                linestyle=dt_ls, fillstyle="none")
                ax_bot.errorbar(qt_centers, r, yerr=r_err_dt,
                                fmt=mk, markersize=3, color=colors[i_yll],
                                linestyle=dt_ls,
                                label=f"{dt_label} {label}" if len(dyturbo_list_3d) > 1 else label)

        ax_top.set_yscale("log")
        ax_top.set_ylabel(r"$d\sigma/dq_T$ [pb]")
        ax_top.legend(fontsize=7, ncol=2, loc="upper right")
        
        mh.cms.label("Preliminary", data=False, com=13, ax = ax_top)

        ax_bot.axhline(1, color="gray", linewidth=0.8, linestyle=":")
        ax_bot.set_ylim(0.7, 1.3)
        ax_bot.set_ylabel("K-factor (DYTurbo / MiNNLO)")
        ax_bot.set_xlabel(LABEL_QT)
        ax_bot.legend(fontsize=6, ncol=2, loc="upper right")
        ax_bot.text(
            0.02, 0.95, "error bars: DYTurbo stat.  |  bands around 1: MiNNLO stat.",
            transform=ax_bot.transAxes, fontsize=6, va="top",
        )

        plt.tight_layout()
        out = os.path.join(OUTDIR, f"gen_3D_mll{i_mll}_{mll_lo[i_mll]:.0f}to{mll_hi[i_mll]:.0f}.pdf")
        # Label CMS posizionata sull'asse superiore (ax[0])
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


def plot_unrolled_3d(minnlo_3d, minnlo_3d_var, dyturbo_list_3d, divide=True):
    """
    Single canvas, unrolling order: outermost=mll, middle=yll, innermost=qT.
    Top: absolute dσ/dbin [pb]. Bottom: K-factor = DYTurbo / MiNNLO.
    dyturbo_list_3d: list of (vals_3d, var_3d, label, color, linestyle).
    """
    n_per_yll = N_QT
    n_per_mll = N_YLL * N_QT

    m_flat = minnlo_3d.transpose(2, 1, 0).flatten()
    m_var  = minnlo_3d_var.transpose(2, 1, 0).flatten()
    
    if divide:
        m_flat = m_flat / bin_width_3d.flatten()
        m_var = m_var / (bin_width_3d.flatten()**2)

    n_bins = len(m_flat)
    x = np.arange(n_bins)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(25, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    ax_top.step(x, m_flat, where="mid", color="steelblue", lw=0.8, label="MiNNLO (GenPart)")
    ax_top.fill_between(
        x, m_flat - np.sqrt(m_var), m_flat + np.sqrt(m_var),
        step="mid", alpha=0.25, color="steelblue",
    )

    # MiNNLO stat uncertainty: gray band around 1 in the ratio panel
    m_rel = rel_error(m_flat, m_var)
    ax_bot.fill_between(
        x, 1 - m_rel, 1 + m_rel,
        step="mid", alpha=0.35, color="gray", linewidth=0, label="MiNNLO stat.",
    )

    for d_3d, dv_3d, dt_label, dt_color, dt_ls in dyturbo_list_3d:
        d_flat = d_3d.transpose(2, 1, 0).flatten()
        d_var  = dv_3d.transpose(2, 1, 0).flatten()
        
        if divide:
            d_flat = d_flat / bin_width_3d.flatten()
            d_var = d_var / (bin_width_3d.flatten()**2)
            
        r, r_err_dt, _ = ratio_error_split(d_flat, d_var, m_flat, m_var)
        ax_top.step(x, d_flat, where="mid", color=dt_color, lw=0.8, ls=dt_ls, label=dt_label)
        ax_bot.step(x, r, where="mid", color=dt_color, lw=0.8, ls=dt_ls, label=dt_label)
        # DYTurbo-only stat uncertainty: colored band around the ratio
        ax_bot.fill_between(
            x, r - r_err_dt, r + r_err_dt,
            step="mid", alpha=0.15, color=dt_color, label=f"{dt_label} stat.",
        )

    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$d\sigma/d\mathrm{bin}$ [pb]", fontsize=11)
    ax_top.legend(fontsize=10)
    mh.cms.label("Preliminary", data=False, com=13, ax = ax_top)

    ax_bot.axhline(1, color="gray", lw=0.8, ls=":")
    ax_bot.set_ylim(0.5, 1.5)
    ax_bot.set_ylabel("K-factor (DYTurbo / MiNNLO)", fontsize=11)
    ax_bot.set_xlabel(
        r"Bin index  [$m_{\ell\ell}$ $\times$ $y_{\ell\ell}$ $\times$ $q_T$]",
        fontsize=11,
    )
    ax_bot.set_xlim(-0.5, n_bins - 0.5)
    ax_bot.legend(fontsize=8, ncol=2)

    mll_lo = EDGES_MLL[:-1]
    mll_hi = EDGES_MLL[1:]
    xtick_pos, xtick_labels = [], []

    for i_mll in range(N_MLL):
        x_mll = i_mll * n_per_mll
        if i_mll > 0:
            for ax in (ax_top, ax_bot):
                ax.axvline(x_mll - 0.5, color="black", lw=1.0, alpha=0.6)
        ax_top.text(
            x_mll + n_per_mll / 2, 1.01,
            f"${mll_lo[i_mll]:.0f}$–${mll_hi[i_mll]:.0f}$ GeV",
            transform=ax_top.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.5, color="black",
        )
        for i_yll in range(N_YLL):
            x_yll = x_mll + i_yll * n_per_yll
            if i_yll > 0:
                for ax in (ax_top, ax_bot):
                    ax.axvline(x_yll - 0.5, color="gray", lw=0.4, ls="--", alpha=0.5)
            xtick_pos.append(x_yll + n_per_yll / 2)
            xtick_labels.append(f"$[{EDGES_YLL[i_yll]:.1f},{EDGES_YLL[i_yll+1]:.1f}]$")

    ax_bot.set_xticks(xtick_pos)
    ax_bot.set_xticklabels(xtick_labels, rotation=90, fontsize=5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, "unrolled_3D.pdf")
    # Label CMS posizionata sull'asse superiore (ax[0])
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def _kfactor_hist(dyturbo, dyturbo_var, minnlo, minnlo_var, axes):
    """K = DYTurbo / MiNNLO as a hist with the given axes; K=1 +- 0 where undefined."""
    k, k_err_dt, k_err_m = ratio_error_split(dyturbo, dyturbo_var, minnlo, minnlo_var)
    k_var = k_err_dt**2 + k_err_m**2
    k     = np.where(np.isfinite(k),     k,     1.0)
    k_var = np.where(np.isfinite(k_var), k_var, 0.0)

    h = hist.Hist(*axes, hist.storage.Weight())
    h.view().value[:]    = k
    h.view().variance[:] = k_var
    return h


def save_kfactor(minnlo, dyturbo):
    """
    K = DYTurbo / MiNNLO (both in pb, same units) in every projection:
    3D (qT, yll, mll), 2D (qT, yll) from the 3D summed over mll, and 1D
    qT / yll / mll from the 1D distributions.
    Bins with no MiNNLO content get K=1. Saved for use by parse_ho_corrections.

    minnlo, dyturbo: dicts with keys 'qt', 'yll', 'mll', '3d' -> (values, variances).
    """
    ax_qt  = hist.axis.Variable(EDGES_QT,  name="gen_qT")
    ax_yll = hist.axis.Variable(EDGES_YLL, name="gen_yll")
    ax_mll = hist.axis.Variable(EDGES_MLL, name="gen_mll")

    m_3d,  m_3d_var  = minnlo["3d"]
    dt_3d, dt_3d_var = dyturbo["3d"]

    outpath = os.path.join(HERE, "kfactor_N3LO_GenPart.root")
    with uproot.recreate(outpath) as f:
        f["kfactor_N3LO_GenPart"] = _kfactor_hist(
            dt_3d, dt_3d_var, m_3d, m_3d_var, [ax_qt, ax_yll, ax_mll]
        )
        f["kfactor_N3LO_GenPart_qT_yll"] = _kfactor_hist(
            dt_3d.sum(axis=2), dt_3d_var.sum(axis=2),
            m_3d.sum(axis=2), m_3d_var.sum(axis=2), [ax_qt, ax_yll]
        )
        for key, ax in [("qt", ax_qt), ("yll", ax_yll), ("mll", ax_mll)]:
            f[f"kfactor_N3LO_GenPart_{ax.name.removeprefix('gen_')}"] = _kfactor_hist(
                *dyturbo[key], *minnlo[key], [ax]
            )
    print(f"Saved K-factor histogram to {outpath}")


# ── load MiNNLO [pb] ──────────────────────────────────────────────────────────
print("Loading MiNNLO histos.root ...")
with uproot.open(MINNLO_FILE) as f:
    def load(key):
        h = f[key].to_boost()
        return h.values(), h.variances()

    m_qt_v,  m_qt_e  = load("gen_inclusive/gen_qT/histo_DYll")
    m_yll_v, m_yll_e = load("gen_inclusive/gen_yll/histo_DYll")
    m_mll_v, m_mll_e = load("gen_inclusive/gen_mll/histo_DYll")

    # hist_unroll stores as .T.flatten() → flat order is (mll, yll, qT); reshape and transpose back
    h3_flat, h3_flat_var = load("gen_inclusive/gen_3D/histo_DYll")
    m_3d     = h3_flat.reshape(N_MLL, N_YLL, N_QT).transpose(2, 1, 0)
    m_3d_var = h3_flat_var.reshape(N_MLL, N_YLL, N_QT).transpose(2, 1, 0)

# ── load DYTurbo [pb] ─────────────────────────────────────────────────────────
dyturbo_datasets = []  # list of dicts keyed by var + style fields

for dt_path, dt_label, dt_color, dt_ls in DYTURBO_SAMPLES:
    print(f"Loading {dt_label} ...")
    with uproot.open(dt_path) as f:
        def load_dt(key):
            h = f[key].to_boost()
            v = h.values()
            e = h.variances() if h.variances() is not None else np.zeros_like(v)
            return v, e

        qt_v,  qt_e  = load_dt("s_qt")
        yll_v, yll_e = load_dt("s_y")
        mll_v, mll_e = load_dt("s_m")
        d3,    d3v   = load_dt("s_qt_vs_y_vs_m")

    dyturbo_datasets.append({
        "qt_v": qt_v, "qt_e": qt_e,
        "yll_v": yll_v, "yll_e": yll_e,
        "mll_v": mll_v, "mll_e": mll_e,
        "3d": d3, "3d_var": d3v,
        "label": dt_label, "color": dt_color, "ls": dt_ls,
    })

def _dt_list_1d(key_v, key_e):
    return [(ds[key_v], ds[key_e], ds["label"], ds["color"], ds["ls"]) for ds in dyturbo_datasets]

def _dt_list_3d():
    return [(ds["3d"], ds["3d_var"], ds["label"], ds["color"], ds["ls"]) for ds in dyturbo_datasets]

# ── 1D plots ──────────────────────────────────────────────────────────────────
print("\nMaking 1D plots ...")
plot_1d(EDGES_QT,  m_qt_v,  m_qt_e,  _dt_list_1d("qt_v", "qt_e"),   LABEL_QT,  "gen_qT")
plot_1d(EDGES_YLL, m_yll_v, m_yll_e, _dt_list_1d("yll_v", "yll_e"), LABEL_YLL, "gen_yll")
plot_1d(EDGES_MLL, m_mll_v, m_mll_e, _dt_list_1d("mll_v", "mll_e"), LABEL_MLL, "gen_mll")

# ── 3D sliced plots ───────────────────────────────────────────────────────────
print("\nMaking 3D slice plots ...")
plot_3d_slices(m_3d, m_3d_var, _dt_list_3d())

# ── unrolled 3D canvas ────────────────────────────────────────────────────────
print("\nMaking unrolled 3D canvas ...")
plot_unrolled_3d(m_3d, m_3d_var, _dt_list_3d())

# ── save K-factor ROOT file (use first DYTurbo entry = N3LO+N3LL) ─────────────
print("\nSaving K-factor histograms ...")
first = dyturbo_datasets[0]
save_kfactor(
    {
        "qt":  (m_qt_v,  m_qt_e),
        "yll": (m_yll_v, m_yll_e),
        "mll": (m_mll_v, m_mll_e),
        "3d":  (m_3d,    m_3d_var),
    },
    {
        "qt":  (first["qt_v"],  first["qt_e"]),
        "yll": (first["yll_v"], first["yll_e"]),
        "mll": (first["mll_v"], first["mll_e"]),
        "3d":  (first["3d"],    first["3d_var"]),
    },
)

print(f"\nAll plots saved to {OUTDIR}/")
