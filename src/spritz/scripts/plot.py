import concurrent.futures
import json
import subprocess
import sys
from copy import deepcopy

import matplotlib as mpl
import mplhep as hep
import numpy as np
import math
import uproot
import hist
from spritz.framework.framework import (
    get_analysis_dict, 
    get_fw_path
)
from spritz.utils.plotting_utils import (
    HistVariation, 
    Histogram, 
    StackedHistogram, 
    darker_color, 
    union, 
    unc_colors, 
    get_yrange
)

mpl.use("Agg")
from matplotlib import pyplot as plt

d = deepcopy(hep.style.CMS)

d["font.size"] = 8
d["figure.figsize"] = (5, 5)

d_multidim = deepcopy(d)
d_multidim["font.size"] = 10
d_multidim["figure.figsize"] = (10, 10)

plt.style.use(d)


def print_unc(ax, histo, highlight):
    unc_up = round(np.sum(histo.up()) / np.sum(histo.nominal) * 100, 2)
    unc_down = round(np.sum(histo.down()) / np.sum(histo.nominal) * 100, 2)

    text = ax.text(
        0.7, 0.75, fontsize=6, transform=ax.transAxes,
        s=f"Syst [-{unc_down}, +{unc_up}]%"
    )
    
    i_highlight = 0

    for syst in histo.variation_names:
        highlighted = syst in highlight
        up = round(np.sum(histo.up([syst])) / np.sum(histo.nominal) * 100, 2)
        down = round(np.sum(histo.down([syst])) / np.sum(histo.nominal) * 100, 2)
        text = ax.annotate(
            text=f"{syst} [-{down}, +{up}]%", xycoords=text, xy=(0,0), 
            verticalalignment="top", fontsize=6, color=darker_color(unc_colors[i_highlight%12]) if highlighted else "black"
        )
        i_highlight += int(highlighted)
        

def ratio_print(numerator, denominator):
    denominator_nom = np.where(denominator.nominal >= 1e-6, denominator.nominal, 1e-6)
    edges = numerator.edges
    ratio = numerator.nominal / denominator_nom
    up = ratio * np.sqrt( np.square(numerator.rel_up()) + np.square(denominator.rel_down()) )
    down = ratio * np.sqrt( np.square(numerator.rel_down()) + np.square(denominator.rel_up()) )

    print(json.dumps({
        "edges": list(edges), "nominal": list(ratio), "up": list(up), "down": list(down)
    }))


def plot_panel(ax, histos, denominator=None, labels=[], labels_unc=[], highlight_unc=[], plot_unc=None, short_label=False, mc_alpha=1., print_unc=False, print_ratio=False):

    if denominator is not None:
        denominator_nom = np.where(histos[denominator].nominal >= 1e-6, histos[denominator].nominal, 1e-6)

        # plot denominator
        histos[denominator].plot_mc_unc(ax, divide=denominator_nom, label=denominator in labels_unc, highlight=highlight_unc, uncertainties=plot_unc)
        histos[denominator].plot_mc(ax, divide=denominator_nom, label=denominator in labels, alpha=mc_alpha, linestyle="dashed")

    else:
        denominator_nom = None

    for key,histo in histos.items():
        if key==denominator: continue

        if isinstance(histo, StackedHistogram):
            histo.plot_stack(ax, divide=denominator_nom, label=key in labels, short_label=short_label)
        elif histo.is_data:
            histo.plot_data(ax, divide=denominator_nom, label=key in labels, short_label=short_label)
        else:
            histo.plot_mc_unc(ax, divide=denominator_nom, label=key in labels_unc, highlight=highlight_unc, uncertainties=plot_unc)
            histo.plot_mc(ax, divide=denominator_nom, label=key in labels, short_label=short_label, alpha=mc_alpha)

        if print_unc:
            print_unc(ax, histo, highlight_unc)

        if print_ratio and denominator is not None:
            ratio_print(histo, histos[denominator])


def make_plots(axes, histo_dict, panels=[], xaxis={}, ylog=True, short_label=False, mc_alpha=1., print_ratio=False, print_unc=False, hide_xlabel=False, hide_ylabel=False):
    
    h0 = list(histo_dict.values())[0]
    variable_binwidth = h0.variable_width

    xlog = xaxis.get("xlog", False)
    xlabel = xaxis.get("xlabel", "x") 
    unit = xaxis.get("unit")
    if unit is not None:
        xlabel = f"{xlabel} ({unit})"

    for i,panel in enumerate(panels):
        denominator = panel.get("denominator")
        do_ratio = denominator is not None
        histos = panel.get("histos", list())
        labels = panel.get("labels", list())
        labels_unc = panel.get("labels_unc", list())
        highlight_unc = panel.get("highlight_unc", list())
        plot_unc = panel.get("plot_unc")
        ylabel = panel.get("ylabel")
        
        histo_dict_panel = {  } 
        for h in histos:
            if h in histo_dict: 
                histo_dict_panel[h] = histo_dict[h]
            else:
                for h2 in histo_dict:
                    if isinstance(histo_dict[h2], StackedHistogram) and histo_dict[h2].contains(h):
                        histo_dict_panel[h] = histo_dict[h2][h]

        yrange = panel.get("yrange", get_yrange(histo_dict_panel, denominator, ylog, variations=plot_unc))

        if ylabel is None:
            if do_ratio:
                ylabel = f"Ratio to {denominator}"
            else:
                ylabel = "Events" + (f" / {(unit if unit is not None else xlabel)}" if variable_binwidth else "")

        plot_panel(
            ax=axes[i], 
            histos=histo_dict_panel,
            denominator=denominator,
            labels=labels, 
            labels_unc=labels_unc,
            highlight_unc=highlight_unc,
            plot_unc=plot_unc,
            short_label=short_label,
            mc_alpha=mc_alpha,
            print_unc=print_unc,
            print_ratio=print_ratio,
        )

        if len(labels+labels_unc) > 0:
            axes[i].legend(
                loc="upper center",
                frameon=True,
                ncols=4,
                framealpha=0.8,
                fontsize=6,
            )

        axes[i].tick_params(labelbottom=False)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("" if hide_ylabel else ylabel)
        axes[i].set_ylim(yrange[0], yrange[1])

        if ylog and not do_ratio:
            axes[i].set_yscale("log")

    # x axis
    axes[-1].tick_params(labelbottom=not hide_xlabel)
    axes[-1].set_xlabel("" if hide_xlabel else xlabel)
    axes[-1].set_xlim(h0.edges[0], h0.edges[-1])

    if xlog: 
        axes[-1].set_xscale("log")
        if axes[-1].get_xlim()[0] == 0:
            xmin = h0.edges[1]/2
            axes[-1].set_xlim(xmin, h0.edges[-1])


def make_plots_multidim(axes, histo_dict, h_axis, panels=[], xaxis={}, ylog=True, short_label=False, mc_alpha=1.):
    
    xlabel = xaxis.get("xlabel") 
    if xlabel is None:
        xlabel = [h_axis[i].name for i in range(len(h_axis))]

    xaxis["xlabel"] = xlabel[0]

    unit = xaxis.get("unit", [None for i in range(len(h_axis))])
    xaxis["unit"] = unit[0]

    nbins = len(h_axis[0].centers)
    if len(h_axis)==3:
        ncols = len(h_axis[1].centers)
        nrows = len(h_axis[2].centers)
    elif len(h_axis)==2:
        nrows = math.floor(math.sqrt(len(h_axis[1].centers)))
        ncols = math.ceil(len(h_axis[1].centers)/nrows)

    ncells = int(len(list(histo_dict.values())[0].centers)/nbins)
    widths = np.tile(h_axis[0].widths, ncells)
    panels[0]["yrange"] = get_yrange(histo_dict, ylog=ylog, divide=widths)
    npanels = len(panels)

    bbox = {"boxstyle":"square", "alpha":1.0, "fc":"white", "ec":"black"} 
    finished = False
    for irow in range(nrows):
        for icol in range(ncols):
            histos_sliced = {}
            for key,histo in histo_dict.items():
                if (ncols*nbins)*irow+nbins*(icol+1) > len(histo.widths):
                    finished = True
                    break
                histos_sliced[key] = histo[(ncols*nbins)*irow+nbins*icol:(ncols*nbins)*irow+nbins*(icol+1)]
                histos_sliced[key].set_axis(h_axis[0])
            if finished:
                break

            make_plots(
                axes=(axes[npanels*irow,icol],*[axes[npanels*irow+k,icol] for k in range(1,npanels)]),
                histo_dict=histos_sliced,
                panels=panels,
                xaxis=xaxis,
                ylog=ylog,
                short_label=short_label,
                mc_alpha=mc_alpha,
                hide_xlabel=irow!=nrows-1,
                hide_ylabel=icol!=0,
            )

            if len(h_axis)==3:
                axes[npanels*irow,icol].text(0.96, 0.95,
                    f"${h_axis[1].edges[icol]} <${xlabel[1]}<$ {h_axis[1].edges[icol+1]}$",
                    fontsize=7,
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=axes[npanels*irow,icol].transAxes,
                    bbox=bbox
                )

                axes[npanels*irow,icol].text(0.04, 0.95,
                    f"${h_axis[2].edges[irow]} <${xlabel[2]}$< {h_axis[2].edges[irow+1]}$",
                    fontsize=7,
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=axes[npanels*irow,icol].transAxes,
                    bbox=bbox
                )

            elif len(h_axis)==2:
                axes[npanels*irow,icol].text(0.96, 0.95,
                    f"${h_axis[1].edges[nrows*irow+icol]} <${xlabel[1]}<$ {h_axis[1].edges[nrows*irow+icol+1]}$",
                    fontsize=7,
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=axes[npanels*irow,icol].transAxes,
                    bbox=bbox
                )


def setup_multifig(ncols, nrows, npanels=2):
    fig = plt.figure(dpi=200) 
    ax = np.empty((npanels*nrows,ncols), dtype=plt.Axes)
    gs = mpl.gridspec.GridSpec(
        (npanels+1)*nrows-1, 2*ncols-1,
        figure=fig,
        height_ratios=(([2]+[1]*(npanels-1)+[0.05])*nrows)[:-1],
        width_ratios=([1,0]*ncols)[:-1]
    )
    
    for i in range(nrows):
        for j in range(ncols): 
            ax[npanels*i,j] = fig.add_subplot(gs[(npanels+1)*i,2*j], sharex=ax[0,0], sharey=ax[0,0])
            for k in range(1,npanels):
                ax[npanels*i+k,j] = fig.add_subplot(gs[(npanels+1)*i+k,2*j], sharex=ax[0,0], sharey=ax[k,0])

    for axij in ax.flat:
        axij.label_outer()

    return fig,ax


def plot(
    region,
    variable,
    analysis_dict,
    variable_dict,
    addFakes=False,
    plotVariations=False,
    plotCorrections=False,
    threePanels=False,
    noRatio=False
):
    
    print("Doing ", region, variable)

    input_file = uproot.open("histos.root")

    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    corrections = analysis_dict.get("corrections", dict())
    colors = analysis_dict["colors"]
    year_label = analysis_dict.get("year_label", "Run-II")
    lumi = analysis_dict["lumi"]

    variable_label = variable_dict.get("label", variable)
    unit = variable_dict.get("unit")
    xlog = variable_dict.get("xlog", False)
    ylog = variable_dict.get("ylog", True)
    axis = variable_dict.get("axis")

    nuisances["stat"] = {
        "name": "stat",
        "type": "stat",
        "samples": dict((skey, "1.00") for skey in samples),
    }

    directory = input_file[f"{region}/{variable}"]
    mc_samples = [x for x in samples if not samples[x].get("is_data", False)]# and not x in ["W+Jets","QCD","TTToSemiLeptonic"]]
    mcfakes_samples = [x for x in samples if x in ["W+Jets","QCD","TTToSemiLeptonic"]]
    
    # get the histograms
    histos = {
        sample: Histogram.make_hist(directory, nuisances, corrections, sample, is_data=samples[sample].get("is_data", False), color=colors.get(sample,"black"))
        for sample in samples
    }

    # prepare total MC histogram
    stack_mc = StackedHistogram([histos[sample] for sample in mc_samples])
    histo_mc = stack_mc.sum("Tot MC", color="black")

    if addFakes and not "_ss" in region:
        directory_ss = input_file[f"{region}_ss/{variable}"]
        histos_ss = {
            sample: Histogram.make_hist(directory_ss, nuisances, corrections, sample, is_data=samples[sample].get("is_data", False), color=colors.get(sample,"black"))
            for sample in samples
        }
        stack_mc_ss = StackedHistogram([histos_ss[sample] for sample in mc_samples])
        histo_mc_ss = stack_mc_ss.sum("Tot MC SS", color="black")

        if "Data" in histos_ss:
            histo_data_ss = histos_ss["Data"]
        else:
            histo_data_ss = Histogram.empty_like(histo_mc_ss, name="Data", is_data=True, color="black")

        # subtract MC from data to get fakes
        variations_fakes = {}
        variations_fakes["stat"] = HistVariation({
            "up": np.sqrt(np.square(histo_data_ss.up(["stat"]))+np.square(histo_mc_ss.down(["stat"]))),
            "down": np.sqrt(np.square(histo_data_ss.down(["stat"]))+np.square(histo_mc_ss.up(["stat"]))) 
        })

        if "fakerw_param" in histo_data_ss.variation_names:
            variations_fakes["fakerw_param"] = HistVariation({   
                "up": histo_data_ss.varied["fakerw_param"].up()-histo_mc_ss.varied["fakerw_param"].up(),
                "down": histo_data_ss.varied["fakerw_param"].down()-histo_mc_ss.varied["fakerw_param"].down() 
            }, "weight")
        
        if "fakerw_model" in histo_data_ss.variation_names:
            variations_fakes["fakerw_model"] = HistVariation({
                "fakerw_model_exp": histo_data_ss.varied["fakerw_model"]["fakerw_model_exp"]-histo_mc_ss.varied["fakerw_model"]["fakerw_model_exp"],
                "fakerw_model_erf": histo_data_ss.varied["fakerw_model"]["fakerw_model_erf"]-histo_mc_ss.varied["fakerw_model"]["fakerw_model_erf"] 
            }, "envelope")

        correction_names = union([histo_data_ss.correction_names, histo_mc_ss.correction_names])
        corrections_fakes = {}

        for corr in correction_names:
            corr_data = histo_data_ss.corrected[corr] if corr in histo_data_ss.correction_names else histo_data_ss.nominal
            corr_mc = histo_mc_ss.corrected[corr] if corr in histo_mc_ss.correction_names else histo_mc_ss.nominal
            corrections_fakes[corr] = corr_data-corr_mc

        histo_fakes = Histogram(
            "Fakes", 
            histo_data_ss.nominal-histo_mc_ss.nominal, 
            varied=variations_fakes,
            corrected=corrections_fakes, 
            is_data=True, 
            color=colors["Fakes"],
            axis=histo_data_ss.axis
        )

        stack_mc.add(histo_fakes, position=0)
        histo_mc = stack_mc.sum("Tot MC", color="black")

    # prepare data histogram
    if "Data" in histos:
        histo_data = histos["Data"]
    else:
        histo_data = Histogram.empty_like(histo_mc, name="Data", is_data=True, color="black")

    # make plots
    if isinstance(axis, list):
        plt.style.use(d_multidim)
        if len(axis)==3:
            ncols = len(axis[1].centers)
            nrows = len(axis[2].centers)
        elif len(axis)==2:
            nrows = math.floor(math.sqrt(len(axis[1].centers)))
            ncols = math.ceil(len(axis[1].centers)/nrows)

        fig, ax = setup_multifig(ncols, nrows, npanels=2-int(noRatio))
        fig.tight_layout(pad=-0.4)
        hep.cms.label('Preliminary', rlabel="", data=True, ax=ax[0,0])
        hep.label.exp_label(data=True, lumi=round(lumi, 2), year=year_label, ax=ax[0,-1])

        xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog}
        panels = [{"histos": ["MC Stack","MC","Data"]}]
        if not noRatio:
            panels.append({"histos": ["Data","MC"], "denominator": "MC", "yrange":(0.9,1.1)})

        make_plots_multidim(
            axes=ax, 
            histo_dict={"MC Stack": stack_mc, "MC": histo_mc, "Data": histo_data},
            h_axis=axis,
            panels=panels,
            xaxis=xaxis_dict,
            ylog=ylog
        )
    else:
        plt.style.use(d)
        panels = [
            {"histos": ["MC Stack","MC","Data"], "labels": ["MC Stack","MC","Data"], "labels_unc": ["MC"]},
        ]
        if noRatio:
            fig, ax = plt.subplots(1, 1, dpi=200)
            ax = np.array([ax])
        else:
            fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,1]}, dpi=200)
            panels.append({"histos": ["Data","MC"], "denominator": "MC", "yrange":(0.9,1.1)})
        
        hep.cms.label('Preliminary', data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)
        fig.tight_layout(pad=-0.5)

        xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }
        
        make_plots(
            axes=ax, 
            histo_dict={"MC Stack": stack_mc, "MC": histo_mc, "Data": histo_data},
            panels=panels,
            xaxis=xaxis_dict,
            ylog=ylog
        )
    
    fig.savefig(
        f"plots/{region}_{variable}.pdf",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )

    plt.close()
    
    if plotVariations:

        for nuis in nuisances:
            name = nuisances[nuis]["name"]
            type = nuisances[nuis]["type"]
            kind = nuisances[nuis].get("kind")
            samples = nuisances[nuis]["samples"]
            
            histo_mc_varied = {}
            var_colors = ["red","blue","green","purple","cyan","magenta","grey","brown","orange"]
            if kind in ["envelope","square","stdev"]:
                for i,key in enumerate(histo_mc.varied[nuis].keys()):
                    histo_mc_varied[key] = Histogram.empty_like(histo_mc, name=key)
                    histo_mc_varied[key].nominal = histo_mc.nominal + histo_mc.varied[nuis][key]
                    histo_mc_varied[key].color = var_colors[i % 9]
            else:
                histo_mc_varied["up"] = Histogram.empty_like(histo_mc, name=f"{nuis} $+1\\sigma$")
                histo_mc_varied["down"] = Histogram.empty_like(histo_mc, name=f"{nuis} $-1\\sigma$")
                histo_mc_varied["up"].nominal = histo_mc.nominal + histo_mc.varied[nuis].up()
                histo_mc_varied["down"].nominal = histo_mc.nominal - histo_mc.varied[nuis].down()
                histo_mc_varied["up"].color = var_colors[0]
                histo_mc_varied["down"].color = var_colors[1]

            if isinstance(axis, list):
                plt.style.use(d_multidim)
                if len(axis)==3:
                    ncols = len(axis[1].centers)
                    nrows = len(axis[2].centers)
                elif len(axis)==2:
                    nrows = math.floor(math.sqrt(len(axis[1].centers)))
                    ncols = math.ceil(len(axis[1].centers)/nrows)

                fig, ax = setup_multifig(ncols, nrows)
                fig.tight_layout(pad=-0.4)
                hep.cms.label('Preliminary', rlabel="", data=True, ax=ax[0,0])
                hep.label.exp_label(data=True, lumi=round(lumi, 2), year=year_label, ax=ax[0,-1])

                xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog}

                make_plots_multidim(
                    axes=ax, 
                    histo_dict={"MC": histo_mc} | {k: histo_mc_varied[k] for k in histo_mc_varied.keys()} | {"Data": histo_data},
                    h_axis=axis,
                    panels=[
                        {"histos": ["MC"]+[k for k in histo_mc_varied.keys()]+["Data"]},
                        {"histos": ["MC"]+[k for k in histo_mc_varied.keys()]+["Data"], "denominator": "MC"},
                    ],
                    xaxis=xaxis_dict,
                    ylog=ylog,
                    mc_alpha=0.3 if len(histo_mc_varied.keys())>10 else 1.
                )

            else:
                plt.style.use(d)
                fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1,1]}, dpi=200)
                hep.cms.label('Preliminary', data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)
                fig.tight_layout(pad=-0.5)

                xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }

                make_plots(
                    axes=ax, 
                    histo_dict={"MC": histo_mc} | {k: histo_mc_varied[k] for k in histo_mc_varied.keys()} | {"Data": histo_data, "MC": histo_mc},
                    panels=[{
                        "histos": ["MC"]+[k for k in histo_mc_varied.keys()], 
                        "denominator": "MC", 
                        "labels": [k for i,k in enumerate(histo_mc_varied.keys()) if i<8],
                        "plot_unc": [nuis], "highlight_unc": [nuis]
                    }, {
                        "histos": ["MC","Data"], "denominator": "MC", 
                        "labels": ["Data"], "labels_unc": ["MC"], 
                        "highlight_unc": [nuis]
                    }],
                    xaxis=xaxis_dict,
                    ylog=ylog,
                    short_label=True,
                    mc_alpha=0.3 if len(histo_mc_varied.keys())>10 else 1.
                )

            fig.savefig(
                f"plots/variations/{region}_{variable}_{name}.pdf",
                facecolor="white",
                pad_inches=0.1,
                bbox_inches="tight",
            )

            plt.close()


    if plotCorrections:
        for corr in corrections:
            name = corrections[corr].get("name", corr)
            related_nuisances = corrections[corr].get("related_nuisances", [corr])
            histo_mc_before = Histogram.empty_like(histo_mc, name=f"MC (before {corr})", color="blue")
            histo_mc_before.nominal = histo_mc.corrected[corr]
            histo_mc_before.linestyle = "dashed"
            
            histo_mc_after = Histogram.empty_like(histo_mc, name=f"MC (after {corr})", color="red")
            histo_mc_after.nominal = histo_mc.nominal
            
            for nuis in related_nuisances:
                if nuis in histo_mc.variation_names:
                    histo_mc_after.varied[nuis] = histo_mc.varied[nuis]

            if corr in histo_data.corrected:
                histo_data_before = Histogram.empty_like(histo_data, name=f"Data (before {corr})", color="blue")
                histo_data_before.nominal = histo_data.corrected[corr]

                histo_data_after = Histogram.empty_like(histo_data, name=f"Data (after {corr})", color="red")
                histo_data_after.nominal = histo_data.nominal
                
                for nuis in related_nuisances:
                    if nuis in histo_data.variation_names:
                        histo_data_after.varied[nuis] = histo_data.varied[nuis]

                histo_dict = {"MC": histo_mc_before, "MC after": histo_mc_after, "Data": histo_data_before, "Data after": histo_data_after}
                labels_list = ["MC", "MC after", "Data", "Data after"]
            else:
                histo_dict = {"MC": histo_mc_before, "MC after": histo_mc_after, "Data": histo_data}
                labels_list = ["MC", "MC after", "Data"]

            if not isinstance(axis, list):
                plt.style.use(d)
                if threePanels:
                    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1,1,1]}, dpi=200)
                    panels = [
                        {"histos": labels_list, "labels": labels_list},
                        {"histos": labels_list, "denominator": "MC"},
                        {"histos": labels_list, "denominator": "Data"}
                    ]
                else:
                    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1,1]}, dpi=200)
                    panels = [
                        {"histos": labels_list, "denominator": "MC", "labels": labels_list},
                        {"histos": labels_list, "denominator": "Data"}
                    ]
                hep.cms.label('Preliminary', data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)
                fig.tight_layout(pad=-0.5)

                xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }
                
                make_plots(
                    axes=ax, 
                    histo_dict=histo_dict,
                    panels=panels,
                    xaxis=xaxis_dict,
                    ylog=ylog,
                    short_label=True,
                )

                fig.savefig(
                    f"plots/corrections/{region}_{variable}_{name}.pdf",
                    facecolor="white",
                    pad_inches=0.1,
                    bbox_inches="tight",
                )

                plt.close()


def main():
    analysis_dict = get_analysis_dict()

    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]

    keep_keys = ["samples", "nuisances", "corrections", "colors", "year_label", "lumi"]
    analysis_dict = { k:v for k,v in analysis_dict.items() if k in keep_keys }

    addFakes = "--fakes" in sys.argv
    plotVariations = "--variations" in sys.argv
    plotCorrections = "--corrections" in sys.argv
    threePanels = "--3panels" in sys.argv
    noRatio = "--noratio" in sys.argv


    cmd_mkdir = f"mkdir -p plots && cp {get_fw_path()}/data/common/index.php plots/"
    if plotVariations:
        cmd_mkdir += f" && mkdir -p plots/variations && cp {get_fw_path()}/data/common/index.php plots/variations/"
    if plotCorrections:
        cmd_mkdir += f" && mkdir -p plots/corrections && cp {get_fw_path()}/data/common/index.php plots/corrections/"
    
    proc = subprocess.Popen(cmd_mkdir, shell=True)
    proc.wait()

    cpus = 15

    print("Doing plots")

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = []

        for region in regions:
            for variable in variables:
                keep_keys = ["label", "unit", "xlog", "ylog", "axis"]
                variable_dict = { k:v for k,v in variables[variable].items() if k in keep_keys }
                if "axis" not in variable_dict:
                    continue
                tasks.append(
                    executor.submit(
                        plot,
                        region,
                        variable,
                        analysis_dict,
                        variable_dict,
                        addFakes,
                        plotVariations,
                        plotCorrections,
                        threePanels,
                        noRatio
                    )
                )
        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()

if __name__ == "__main__":
    main()
