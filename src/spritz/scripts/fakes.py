import concurrent.futures
import os, sys, subprocess
from copy import deepcopy

import matplotlib as mpl, mplhep as hep
import numpy as np, scipy as sc
import iminuit, math
import json
import pickle

from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.utils.histogram import Histogram, StackedHistogram
from spritz.utils.plotting_utils import plot_panel, make_plots, setup_fig, setup_multifig, add_to_samples

veto = (50, 120)

fit_functions = {
    "logistic": { 
        "initial": (2,0.5,200,20), "string": "%.2f + %.2f/(1+exp((m-%.2f)/%.2f)" 
    },
    "erf": { 
        "initial": (2,0.5,200,20), "string": "%.2f - %.2f * erf((m-%.2f)/%.2f)" 
    },
    # "exp": { 
    #     "initial": (1,1,1000), "string": "%.2f + %.2f * exp(-m / %.2f)" 
    # },
}

def f_logistic(x,a,b,c,d):
    return a+b/(1+np.exp((x-c)/d))

def f_erf(x,a,b,c,d):
    return a-b*sc.special.erf((x-c)/d)

def f_exp(x,a,b,c):
    return a+b*np.exp(-x/c)

def get_chi2(f, x, y, yerr, *params):
    model = eval(f"f_{f}")(x, *params)
    return np.sum(((y-model)/yerr)**2)


mpl.use("Agg")
d = deepcopy(hep.style.CMS)
d["font.size"] = 8
d["figure.figsize"] = (5, 5)
d_multidim = deepcopy(d)
d_multidim["font.size"] = 10
d_multidim["figure.figsize"] = (10, 10)

plt = mpl.pyplot
plt.style.use(d)


def plot_channel(
    histo_dict,
    analysis_dict,
    variable_dict,
    filename
):
    axis = variable_dict.get("axis")
    panels = [
        { 
            "histos": {
                "MC Stack": {"histogram": "stack_mc", "show_label": True}, 
                "Fakes": {"histogram": "histo_fakes", "show_label": True}
            }
        }, {
            "denominator": "Fakes", 
            "histos": {
                "MC Stack": {"histogram": "stack_mc"}, 
                "Fakes": {"histogram": "histo_fakes"}
            }, 
            "yrange": (0.,1.5)
        }
    ]

    npanels = len(panels)
    
    if isinstance(axis, list):
        if plt.rcParams["figure.figsize"] != [10, 10]:
            plt.style.use(d_multidim)

        fig, ax, multifig_cfg = setup_multifig(analysis_dict, variable_dict, npanels=npanels)
        nrows, ncols = multifig_cfg["bins"].shape

        for i in range(nrows):
            for j in range(ncols):
                _panels = [p for p in panels]
                _panels[0] |= {
                    "left_label": multifig_cfg["l_labels"][i,j], 
                    "right_label": multifig_cfg["r_labels"][i,j], 
                    "more_space": True
                }

                make_plots(
                    axes=[ax[npanels*i+k,j] for k in range(npanels)],
                    histo_dict=histo_dict,
                    panels=_panels,
                    variable_dict=variable_dict,
                    slice_bins=(axis[0], multifig_cfg["bins"][i,j]),
                    hide_xlabel=i!=nrows-1,
                    hide_ylabel=j!=0,
                    hide_legend=True
                )

        handles, labels = ax[npanels*i,j].get_legend_handles_labels()
        if len(handles) > 0:
            if len(handles) > 2 * 5:
                legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncols=5, fontsize=7)
            else:
                legend = fig.legend(handles, labels, loc="upper center", ncols=5, fontsize=7)
            legend.get_frame().set_linewidth(0)

    else:
        if plt.rcParams["figure.figsize"] != [5, 5]:
            plt.style.use(d)

        fig, ax = setup_fig(analysis_dict, npanels=npanels)

        make_plots(
            axes=ax, 
            histo_dict=histo_dict,
            panels=panels,
            variable_dict=variable_dict,
            legend_ncols=5,
            legend_fontsize=5
        )

    if not os.path.isdir(os.path.split(filename)[0]):
        subprocess.Popen(f"mkdir -p {os.path.split(filename)[0]}", shell=True).wait()

    fig.savefig(filename, facecolor="white", bbox_inches="tight")
    print(f">> {filename}")
    plt.close()


def plot_combined(
    # histo_os,
    # histo_ss,
    # plot_cfg,
    histo_dict,
    analysis_dict,
    variable_dict,
    filename,
):
    axis = variable_dict.get("axis")
    panels = [
        { 
            "histos": {
                "OS": {"histogram": "histo_fakes_os", "show_label": True}, 
                "SS": {"histogram": "histo_fakes_ss", "show_label": True}
            }
        }, {
            "denominator": "SS", 
            "histos": {
                "OS": {"histogram": "histo_fakes_os"}, 
                "SS": {"histogram": "histo_fakes_ss"}
            }, 
            "yrange": (0.,3.5)
        }
    ]

    npanels = len(panels)

    if isinstance(axis, list):
        if plt.rcParams["figure.figsize"] != [10, 10]:
            plt.style.use(d_multidim)

        fig, ax, multifig_cfg = setup_multifig(analysis_dict, variable_dict, npanels=npanels)
        nrows, ncols = multifig_cfg["bins"].shape

        for i in range(nrows):
            for j in range(ncols):
                _panels = [p for p in panels]
                _panels[0] |= {
                    "left_label": multifig_cfg["l_labels"][i,j], 
                    "right_label": multifig_cfg["r_labels"][i,j], 
                    "more_space": True
                }

                make_plots(
                    axes=[ax[npanels*i+k,j] for k in range(npanels)],
                    histo_dict=histo_dict,
                    panels=_panels,
                    variable_dict=variable_dict,
                    slice_bins=(axis[0], multifig_cfg["bins"][i,j]),
                    hide_xlabel=i!=nrows-1,
                    hide_ylabel=j!=0,
                    hide_legend=True
                )

        handles, labels = ax[npanels*i,j].get_legend_handles_labels()
        if len(handles) > 0:
            if len(handles) > 2 * 5:
                legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncols=5, fontsize=7)
            else:
                legend = fig.legend(handles, labels, loc="upper center", ncols=5, fontsize=7)
            legend.get_frame().set_linewidth(0)
    else:
        if plt.rcParams["figure.figsize"] != [5, 5]:
            plt.style.use(d)

        fig, ax = setup_fig(analysis_dict, npanels=npanels)

        make_plots(
            axes=ax, 
            histo_dict=histo_dict,
            panels=panels,
            variable_dict=variable_dict,
            legend_ncols=5,
            legend_fontsize=5
        )
    
    fig.savefig(filename, facecolor="white", bbox_inches="tight")
    print(f">> {filename}")
    plt.close()


def fit_ratio(
    histo_dict,
    analysis_dict,
    variable_dict
):
    fitresult = {}
    h_fakes_os = histo_dict["histo_fakes_os"]
    h_fakes_ss = histo_dict["histo_fakes_ss"]

    # take OS / SS ratio
    numerator = np.where(h_fakes_os.nominal >= 1e-6, h_fakes_os.nominal, 1e-6)
    denominator = np.where(h_fakes_ss.nominal >= 1e-6, h_fakes_ss.nominal, 1e-6)
    y = numerator / denominator
    yup = y * np.sqrt( 
        np.square(h_fakes_os.up()/h_fakes_os.nominal) + np.square(h_fakes_ss.down()/h_fakes_ss.nominal) 
    )
    ydown = y * np.sqrt( 
        np.square(h_fakes_os.down()/h_fakes_os.nominal) + np.square(h_fakes_ss.up()/h_fakes_ss.nominal) 
    )
    yerr = (ydown + yup) / 2.
    x = h_fakes_os.centers

    # mask data points around the Z peak
    edges = h_fakes_os.edges
    mask = np.array([edges[i+1]<=veto[0] or edges[i]>=veto[1] for i in range(len(edges)-1)])

    # setup figure
    npanels = len(fit_functions)
    fig, ax = plt.subplots(npanels, 1, sharex=True, gridspec_kw={"height_ratios": [1]*npanels}, dpi=300)
    fig.tight_layout(pad=-0.5)
    hep.cms.label(f"Preliminary", data=True, lumi=round(analysis_dict.get("lumi"), 2), ax=ax[0], year=analysis_dict.get("plot_label", "Run 2"))

    for i,f in enumerate(fit_functions):
        
        # do fit
        chi2 = lambda *initial: get_chi2(f, x[mask], y[mask], yerr[mask], *initial)
        
        model = iminuit.Minuit(chi2, *fit_functions[f]["initial"])
        model.migrad()

        ndof = len(x[mask]) - model.nfit

        # evaluate function at best fit value
        fit_x = np.linspace(edges[0], edges[-1], 400)
        fit_y = eval(f"f_{f}")(fit_x, *model.values)

        # evaluate uncertainty on best fit
        rng = np.random.default_rng(seed=0)
        params = rng.multivariate_normal(model.values, model.covariance, size=100)
        y_random = [eval(f"f_{f}")(fit_x, *p) for p in params]
        fit_yerr = np.std(y_random, axis=0)

        # print fit result
        print(f"\n-------------\n {f}\n-------------")
        print(f"chi2 / ndof = {round(model.fval, 1)} / {ndof} = {round(model.fval/ndof, 3)}\n")
        print(f"p = {np.array2string(np.array(model.values), separator=", ")}")
        print(f"err = {np.array2string(np.array(model.errors), separator=", ")}")
        print(f"\ncov =\n{np.array2string(model.covariance, separator=", ")}")

        # save fit result
        fitresult[f] = {
            "parameters": list(model.values),
            "covariance": model.covariance.tolist()
        }
    
        ax[i].hlines(1, edges[0], edges[-1], linestyles="dashed", colors="black")
        
        ax[i].fill_between(
            fit_x, fit_y-fit_yerr, fit_y+fit_yerr, 
            facecolor="red", alpha=0.25
        )
        ax[i].plot(
            fit_x, fit_y, color="red", 
            label=fit_functions[f]["string"] % tuple(model.values))

        ax[i].errorbar(
            x, y, yerr=(ydown, yup), 
            fmt="o", markersize=4, color="cornflowerblue",
            label="Data points not used in fit"
        )
        ax[i].errorbar(
            x[mask], y[mask], yerr=(ydown[mask], yup[mask]),
            fmt="o", markersize=4, color="black",
            label="Data points used in fit"
        )

        ax[i].legend(loc="upper center", frameon=True, framealpha=0.8, fontsize=6, ncols=3)
        ax[i].set_ylabel("OS / SS")
        ax[i].set_ylim(0.8, 4.0)
        ax[i].set_xlabel(variable_dict.get("label", "x"))


    fig.savefig(
        "transferfactor.pdf",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )

    print()

    with open("fakes_rw.json", "w") as f:
        json.dump(fitresult, f, indent=2)


def fakes(
    region,
    variable,
    analysis_dict,
    variable_dict,
    do_plots,
    do_fit
):
    print(region, variable)
    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    corrections = analysis_dict.get("corrections", dict())
    colors = analysis_dict["colors"]

    fakes_dict = analysis_dict.get("fakes_dict", {})
    fakes_regions = [region.get("target") for region in fakes_dict.get("regions",[])]

    mc_samples = [
        k for k,v in samples.items() if not (
            k in fakes_dict.get("subtract_mc", []) or v.get("is_data"))
    ]

    with open("histos.pkl", "rb") as fin:
        din = pickle.load(fin)
    
    directories = {
        "os": {k:v for k,v in din.items() if k.startswith(f"{region}/{variable}/")},
        "ss": {k:v for k,v in din.items() if k.startswith(f"{region}_ss/{variable}/")}
    }
    directories["os"] = {k.replace(f"{region}/{variable}/", ""):v for k,v in directories["os"].items()}
    directories["ss"] = {k.replace(f"{region}_ss/{variable}/", ""):v for k,v in directories["ss"].items()}
    """
    plot_cfg = {
        "plot_label": analysis_dict.get("plot_label", "Run-II"),
        "lumi": analysis_dict.get("lumi"),
        "variable_label": variable_dict.get("label", variable), 
        "unit": variable_dict.get("unit"), 
        "xlog": variable_dict.get("xlog", False),
        "ylog": variable_dict.get("ylog", True),
        "axis": variable_dict.get("axis")
    }
    """
    nuisances["stat"] = {
        "name": "stat",
        "type": "stat",
        "samples": dict((skey, "1.00") for skey in list(samples.keys())+["Fakes"]),
    }

    histos_mc, stack_mc, histo_fakes = {}, {}, {}
    
    labels = { "os": "opposite-sign (fakes)", "ss": "same-sign (fakes)" }
    colors = colors | { "os": "blue", "ss": "red" }

    for channel in ["os", "ss"]:
        directory = directories[channel]
    
        # MC histograms (e.g. W+Jets or QCD)
        histos_mc[channel] = {
            sample: Histogram.make_hist(
                directory, sample, nuisances, corrections, color=colors.get(sample,"black"),
                label=labels.get(sample), is_data=False
            )
            for sample in mc_samples
        }
        
        # fakes histogram
        for nuis in fakes_dict.get("nuisances", []) + ["stat"]:
            nuisances[nuis]["samples"] = add_to_samples(nuisances[nuis]["samples"].copy(), "Fakes")
        for corr in corrections:
            corrections[corr]["samples"] = add_to_samples(corrections[corr]["samples"].copy(), "Fakes")

        histo_fakes[channel] = Histogram.make_hist(
            directory, "Fakes", nuisances, corrections, color=colors[channel], is_data=True
        )
        histo_fakes[channel].name = labels[channel]

        # make plots per channel
        if do_plots and len(mc_samples) > 0:
            stack_mc[channel] = StackedHistogram([histos_mc[channel][sample] for sample in mc_samples])
            histo_dict = {
                "stack_mc": stack_mc[channel], 
                "histo_mc": histo_mc[channel], 
                "histo_fakes": histo_fakes[channel]
            }

            plot_channel(
                histo_dict,
                analysis_dict,
                variable_dict,
                f"plots_fakes/{channel}/{region}_{channel}_{variable}.pdf"
            )

    histo_dict = {
        "histo_fakes_os": histo_fakes["os"], 
        "histo_fakes_ss": histo_fakes["ss"]
    }

    # make plots of ratio OS / SS
    if do_plots:
        plot_combined(
            histo_dict,
            analysis_dict,
            variable_dict,
            f"plots_fakes/{region}_{variable}.pdf"
        )

    # fit transferfactor
    if do_fit and region=="bveto_mm" and variable=="mll_medium":
        fit_ratio(
            histo_dict,
            analysis_dict,
            variable_dict,
        )
    

def main():
    analysis_dict = get_analysis_dict()

    regions = analysis_dict["regions"]
    regions = [region for region in regions if f"{region}_ss" in regions]
    variables = analysis_dict["variables"]

    keep_keys = ["samples", "nuisances", "corrections", "colors", "plot_label", "lumi", "fakes_dict"]
    analysis_dict = { k:v for k,v in analysis_dict.items() if k in keep_keys }

    do_fit = "--fit" in sys.argv
    do_plots = "--plot" in sys.argv or not do_fit

    if do_plots:
        cmd_mkdir = f"mkdir -p plots_fakes && cp {get_fw_path()}/data/common/index.php plots_fakes/"
        proc = subprocess.Popen(cmd_mkdir, shell=True)
        proc.wait()
        cpus = 10
        print("Doing plots")
    else:
        regions = ["bveto_mm"]
        variables = { k:v for k,v in variables.items() if k=="mll_medium" }
        cpus = 1

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
                        fakes,
                        region,
                        variable,
                        analysis_dict,
                        variable_dict,
                        do_plots,
                        do_fit
                    )
                )
        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()

if __name__ == "__main__":
    main()
