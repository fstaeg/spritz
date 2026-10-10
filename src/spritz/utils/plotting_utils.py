import hist
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import mplhep as hep
from spritz.utils.histogram import Histogram, StackedHistogram

plt = mpl.pyplot

def add_to_samples(samples, sample):
    if isinstance(samples, list):
        samples = samples + [sample]
    elif isinstance(samples, dict):
        samples = samples | {sample: "1.00"}
    return samples

def plot_panel(ax, histo_dict, histos, denominator=None, slice_bins=None, short_label=False, absolute=False):
    if denominator is not None:
        histogram = histos[denominator].get("histogram")
        variation = histos[denominator].get("variation")
        denominator_nom = histo_dict[histogram].nominal
        
        if variation is not None:
            denominator_nom += histo_dict[histogram][variation]
        
        divide = np.where(denominator_nom >= 1e-6, denominator_nom, 1e-6)
    elif absolute:
        divide = np.ones_like(histo_dict[list(histo_dict.keys())[0]].nominal)
    else:
        divide = None

    for key,args in histos.items():
        histo = histo_dict[args["histogram"]]
        if key==denominator and args.get("linestyle") is None:
            args["linestyle"] = "solid" if histo.is_data else "dashed"

        if isinstance(histo, StackedHistogram):
            histo.plot_stack(ax, divide=divide, slice_bins=slice_bins, short_label=short_label, **args)
        elif histo.is_data and not key==denominator:
            histo.plot_data(ax, divide=divide, slice_bins=slice_bins, short_label=short_label, **args)
        else:
            histo.plot_mc_unc(ax, divide=divide, slice_bins=slice_bins, short_label=short_label, **args)
            histo.plot_mc(ax, divide=divide, slice_bins=slice_bins, short_label=short_label, **args)


def make_plots(axes, histo_dict, panels, variable_dict, slice_bins=None, short_label=False, hide_xlabel=False, hide_ylabel=False, hide_legend=False, legend_ncols=5, legend_fontsize=8):
    xlog = variable_dict.get("xlog", False)
    ylog = variable_dict.get("ylog", True)
    xlabel = variable_dict.get("label", "x")
    if isinstance(xlabel, list):
        xlabel = xlabel[0]
    unit = variable_dict.get("unit")
    if isinstance(unit, list):
        unit = unit[0]
    if unit is not None:
        xlabel += f" ({unit})"

    textbox = {"fontsize": 8, "verticalalignment": "top", "backgroundcolor": ("white", 0.75)}
        #"bbox": {"boxstyle": "square", "alpha": 1.0, "fc": "white", "ec": "white"}}

    if slice_bins is None:
        h0 = histo_dict[list(panels[0].get("histos").values())[0]["histogram"]]
        variable_width = isinstance(h0.axis, hist.axis.Variable)
        edges = h0.edges
    else:
        axis = slice_bins[0]
        variable_width = isinstance(axis, hist.axis.Variable)
        edges = axis.edges

    for i,panel in enumerate(panels):
        histos = panel.get("histos")
        denominator = panel.get("denominator")
        absolute = panel.get("absolute", False)

        plot_panel(
            ax=axes[i], histo_dict=histo_dict, histos=histos, denominator=denominator, 
            slice_bins=slice_bins, short_label=short_label, absolute=absolute
        )

        left_label, right_label = panel.get("left_label"), panel.get("right_label")
        if left_label is not None:
            axes[i].text(0.04, 0.95, left_label, **textbox,
                transform=axes[i].transAxes, horizontalalignment="left")
        if right_label is not None:
            axes[i].text(0.96, 0.95, right_label, **textbox,
                transform=axes[i].transAxes, horizontalalignment="right")

        is_ratio = denominator is not None
        ylabel = panel.get("ylabel")

        if ylabel is None:
            if is_ratio:
                ylabel = f"Ratio to {denominator}"
            else:
                ylabel = "Events"
                if variable_width:
                    ylabel += f" / {(unit if unit is not None else xlabel)}"

        handles, labels = axes[i].get_legend_handles_labels()
        if not hide_legend and len(handles) > 0:
            axes[i].legend(
                loc="upper center", ncols=legend_ncols, frameon=True, framealpha=0.9,
                linewidth=0, fontsize=legend_fontsize
            )

        axes[i].tick_params(labelbottom=False)
        axes[i].set_xlabel("")
        
        if not hide_ylabel:
            axes[i].set_ylabel(ylabel)
        if ylog and not is_ratio:
            axes[i].set_yscale("log")

        yrange = panel.get("yrange")
        if yrange is None:
            yrange = get_yrange(
                axes[i], ylog, is_ratio, variable_width, panel.get("more_space", False)
            )

        axes[i].set_ylim(*yrange)

    # x axis
    axes[-1].set_xlim(edges[0], edges[-1])
    if xlog: 
        if edges[0] == 0:
            axes[-1].set_xlim(edges[1]/4, edges[-1])
        axes[-1].set_xscale("log")
    
    if not hide_xlabel:
        axes[-1].tick_params(labelbottom=True)
        axes[-1].set_xlabel(xlabel)
    else:
        axes[-1].tick_params(labelbottom=False)


def setup_fig(analysis_dict, npanels=2, height_ratios=[3,1]):
    plot_label = analysis_dict.get("plot_label", "Run 2")
    lumi = analysis_dict.get("lumi")

    if npanels==1:
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax = np.array([ax])
    else:
        fig, ax = plt.subplots(npanels, 1, dpi=200, 
            sharex=True, gridspec_kw={"height_ratios": height_ratios})

    fig.tight_layout(pad=-0.4)
    hep.cms.label("Preliminary", data=True, lumi=round(lumi, 2), ax=ax[0], year=plot_label)
    
    return fig, ax


def setup_multifig(analysis_dict, variable_dict, npanels=2, height_ratios=[3,1]):
    plot_label = analysis_dict.get("plot_label", "Run 2")
    lumi = analysis_dict.get("lumi")
    axis = variable_dict.get("axis")
    variable_label = variable_dict.get("label")
    
    if len(axis)==3:
        x_label, y_label, z_label = variable_label
        x_axis, y_axis, z_axis = axis
        nrows = len(z_axis.centers)
        ncols = len(y_axis.centers)
        nbins = len(x_axis.centers)
    elif len(axis)==2:
        x_label, y_label = variable_label
        x_axis, y_axis = axis
        nrows = math.floor(math.sqrt(len(y_axis.centers)))
        ncols = math.ceil(len(y_axis.centers)/nrows)

    fig = plt.figure(dpi=200)
    ax = np.empty((npanels*nrows, ncols), dtype=plt.Axes)
    
    if len(height_ratios) > npanels:
        height_ratios = height_ratios[:npanels]
    while len(height_ratios) < npanels:
        height_ratios += height_ratios[-1:]

    gs = mpl.gridspec.GridSpec(
        (npanels+1)*nrows-1, 2*ncols-1, figure=fig,
        height_ratios=((height_ratios+[0])*nrows)[:-1],
        width_ratios=([1,0.02]*ncols)[:-1]
    )
    
    l_labels = np.empty((nrows, ncols), dtype=object)
    r_labels = np.empty((nrows, ncols), dtype=object)
    bins = np.empty((nrows, ncols), dtype=slice)

    for i in range(nrows):
        for j in range(ncols):
            ax[npanels*i,j] = fig.add_subplot(gs[(npanels+1)*i,2*j], sharex=ax[0,0], sharey=ax[0,0])
            for k in range(1,npanels):
                ax[npanels*i+k,j] = fig.add_subplot(gs[(npanels+1)*i+k,2*j], sharex=ax[0,0], sharey=ax[k,0])

            bins[i,j] = slice((ncols*nbins)*i+nbins*j, (ncols*nbins)*i+nbins*(j+1))

            if len(axis)==3:
                r_labels[i,j] = f"${y_axis.edges[j]} <${y_label}$< {y_axis.edges[j+1]}$"
                l_labels[i,j] = f"${z_axis.edges[i]} <${z_label}$< {z_axis.edges[i+1]}$"

            elif len(axis)==2:
                l_labels[i,j] = f"${y_axis.edges[nrows*i+j]} <${y_label}$< {y_axis.edges[nrows*i+j+1]}$"

    for axij in ax.flat:
        axij.label_outer()

    fig.tight_layout(pad=-0.5, rect=[0,0,1,0.96])
    hep.cms.label("Preliminary", rlabel="", data=True, ax=ax[0,0], fontsize=12)
    hep.label.exp_label(data=True, lumi=round(lumi, 2), year=plot_label, ax=ax[0,-1], fontsize=12)

    return fig, ax, {"bins": bins, "l_labels": l_labels, "r_labels": r_labels}


def oom(number):
    if number==0: return 0
    else: return int(math.floor(math.log(number, 10)))


def get_yrange(ax, ylog, is_ratio, variable_binwidth, more_space=False):
    ymin, ymax = [ax.dataLim.y0, ax.dataLim.y1]
    if is_ratio:
        fact = 1.25 if not more_space else 1.4
        ylim = fact * max(abs(ymin-1), abs(ymax-1))
        ylim = min(ylim, 1)
        ymin, ymax = 1-ylim, 1+ylim
    elif ylog:
        if variable_binwidth:
            ymin = max(1e-1, 0.5*ymin)
        else:
            ymin = max(1, 0.5*ymin)
        ymax = ymax * 10**((oom(ymax)-oom(ymin))/4)
        if more_space:
            ymin, ymax = ymin/10, 50*ymax
    else:
        ymin = 0
        ymax = ymax + ymax/5
        if more_space:
            ymax = ymax + ymax/5
    
    return [ymin, ymax]

