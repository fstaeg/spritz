import concurrent.futures
import os, sys, subprocess
from copy import deepcopy

import matplotlib as mpl, mplhep as hep
import numpy as np, scipy as sc
import iminuit, math
import uproot
import json

from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.utils.plotting_utils import Histogram, StackedHistogram, get_fakes
from spritz.scripts.plot import make_plots, make_plots_multidim, setup_multifig

mc_fakes = ["W+Jets","QCD"]
veto = (50, 110)

fit_functions = {
    "logistic": { 
        "initial": (2,1,200,10), "string": "%.2f + %.2f/(1+exp((m-%.2f)/%.2f)" 
    },
    "erf": { 
        "initial": (2,1,200,10), "string": "%.2f - %.2f * erf((m-%.2f)/%.2f)" 
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
    histo_fakes,
    stack_mc,
    plot_cfg,
    filename
):
    plot_label = plot_cfg["plot_label"]
    variable_label = plot_cfg["variable_label"]
    lumi = plot_cfg["lumi"]
    unit = plot_cfg["unit"]
    xlog = plot_cfg["xlog"]
    ylog = plot_cfg["ylog"]
    axis = plot_cfg["axis"]

    xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }
    histo_dict = { "Fakes": histo_fakes, "MC Stack": stack_mc }
    panels = [
        { "histos": ["Fakes", "MC Stack"], "labels": ["Fakes", "MC Stack"] },
        { "histos": ["Fakes"]+[s for s in stack_mc], "denominator": "Fakes", "yrange": (0.,1.5) }
    ]
    
    if isinstance(axis, list):
        if len(axis)==3:
            ncols, nrows = len(axis[1].centers), len(axis[2].centers)
        elif len(axis)==2:
            nrows = math.floor(math.sqrt(len(axis[1].centers)))
            ncols = math.ceil(len(axis[1].centers)/nrows)

        plt.style.use(d_multidim)
        fig, ax = setup_multifig(ncols, nrows)
        fig.tight_layout(pad=-0.4)
        hep.cms.label("Preliminary", rlabel="", data=True, ax=ax[0,0])
        hep.label.exp_label(data=True, lumi=round(lumi, 2), year=plot_label, ax=ax[0,-1])

        make_plots_multidim(
            axes=ax, histo_dict=histo_dict, h_axis=axis,
            panels=panels, xaxis=xaxis_dict, ylog=ylog
        )

    else:
        plt.style.use(d)
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,1]}, dpi=200)
        hep.cms.label("Preliminary", data=True, lumi=round(lumi, 2), ax=ax[0], year=plot_label)
        fig.tight_layout(pad=-0.5)
        
        make_plots(
            axes=ax, histo_dict=histo_dict, panels=panels,
            xaxis=xaxis_dict, ylog=ylog
        )

    if not os.path.isdir(os.path.split(filename)[0]):
        subprocess.Popen(f"mkdir -p {os.path.split(filename)[0]}", shell=True).wait()

    fig.savefig(filename, facecolor="white", pad_inches=0.1, bbox_inches="tight")
    plt.close()


def plot_combined(
    histo_os,
    histo_ss,
    plot_cfg,
    filename,
):
    plot_label = plot_cfg["plot_label"]
    variable_label = plot_cfg["variable_label"]
    lumi = plot_cfg["lumi"]
    unit = plot_cfg["unit"]
    xlog = plot_cfg["xlog"]
    ylog = plot_cfg["ylog"]
    axis = plot_cfg["axis"]

    xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }
    histo_dict = { "OS": histo_os, "SS": histo_ss }
    panels = [
        { "histos": ["OS","SS"], "labels": ["OS","SS"] },
        { "histos": ["OS","SS"], "denominator": "SS", "yrange": (0.,3.5) }
    ]

    if isinstance(axis, list):
        if len(axis)==3:
            ncols, nrows = len(axis[1].centers), len(axis[2].centers)
        elif len(axis)==2:
            nrows = math.floor(math.sqrt(len(axis[1].centers)))
            ncols = math.ceil(len(axis[1].centers)/nrows)

        plt.style.use(d_multidim)
        fig, ax = setup_multifig(ncols, nrows)
        fig.tight_layout(pad=-0.4)
        hep.cms.label("Preliminary", rlabel="", data=True, ax=ax[0,0])
        hep.label.exp_label(data=True, lumi=round(lumi, 2), year=plot_label, ax=ax[0,-1])

        make_plots_multidim(
            axes=ax, histo_dict=histo_dict, h_axis=axis,
            panels=panels, xaxis=xaxis_dict, ylog=ylog
        )
    else:
        plt.style.use(d)
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,1]}, dpi=200)
        hep.cms.label("Preliminary", data=True, lumi=round(lumi, 2), ax=ax[0], year=plot_label)
        fig.tight_layout(pad=-0.5)
        
        make_plots(
            axes=ax, histo_dict=histo_dict, panels=panels, xaxis=xaxis_dict, ylog=ylog
        )
    
    fig.savefig(filename, facecolor="white", pad_inches=0.1, bbox_inches="tight")
    plt.close()


def fit_ratio(
    h_fakes_os,
    h_fakes_ss,
    plot_cfg,
):
    
    variable_label = plot_cfg["variable_label"]
    plot_label = plot_cfg["plot_label"]
    lumi = plot_cfg["lumi"]
    fitresult = {}

    # take OS / SS ratio
    numerator = np.where(h_fakes_os.nominal >= 1e-6, h_fakes_os.nominal, 1e-6)
    denominator = np.where(h_fakes_ss.nominal >= 1e-6, h_fakes_ss.nominal, 1e-6)
    y = numerator / denominator
    yup = y * np.sqrt( np.square(h_fakes_os.rel_up()) + np.square(h_fakes_ss.rel_down()) )
    ydown = y * np.sqrt( np.square(h_fakes_os.rel_down()) + np.square(h_fakes_ss.rel_up()) )
    yerr = (ydown + yup) / 2.
    x = h_fakes_os.centers

    # mask data points around the Z peak
    edges = h_fakes_os.edges
    mask = np.array([edges[i+1]<=veto[0] or edges[i]>=veto[1] for i in range(len(edges)-1)])

    # setup figure
    npanels = len(fit_functions)
    fig, ax = plt.subplots(npanels, 1, sharex=True, gridspec_kw={"height_ratios": [1]*npanels}, dpi=300)
    fig.tight_layout(pad=-0.5)
    hep.cms.label(f"Preliminary", data=True, lumi=round(lumi, 2), ax=ax[0], year=plot_label)

    for i,f in enumerate(fit_functions):
        
        # do fit
        chi2 = lambda *initial: get_chi2(f, x[mask], y[mask], yerr[mask], *initial)
        
        model = iminuit.Minuit(chi2, *fit_functions[f]["initial"])
        model.errordef = 1
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
        ax[i].set_xlabel(variable_label)


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
    
    print("Doing ", region, variable)

    input_file = uproot.open("histos.root")

    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    corrections = analysis_dict.get("corrections", dict())
    colors = analysis_dict["colors"]

    plot_cfg = {
        "plot_label": analysis_dict.get("plot_label", "Run-II"),
        "lumi": analysis_dict.get("lumi"),
        "variable_label": variable_dict.get("label", variable), 
        "unit": variable_dict.get("unit"), 
        "xlog": variable_dict.get("xlog", False),
        "ylog": variable_dict.get("ylog", True),
        "axis": variable_dict.get("axis")
    }

    nuisances["stat"] = {
        "name": "stat",
        "type": "stat",
        "samples": dict((skey, "1.00") for skey in samples),
    }

    mc_samples = [k for k,v in samples.items() if not (v.get("is_data", False) or k in mc_fakes)]
    mcfakes_samples = [k for k in samples if k in mc_fakes]

    histos, stack_mcfakes, histo_fakes = {}, {}, {}
    
    filenames = { "os": f"{region}_os_{variable}", "ss": f"{region}_ss_{variable}" }
    directories = { "os": region, "ss": f"{region}_ss" }
    labels = { "os": "opposite-sign (fakes)", "ss": "same-sign (fakes)" }
    colors = colors | { "os": "blue", "ss": "red" }

    for channel in ["os", "ss"]:
        directory = input_file[f"{directories[channel]}/{variable}"]
    
        # get the histograms
        histos[channel] = {
            sample: Histogram.make_hist(
                directory, nuisances, corrections, sample, 
                is_data=samples[sample].get("is_data", False), color=colors.get(sample,"black")
            )
            for sample in samples
        }

        # "fakes" MC histograms
        stack_mcfakes[channel] = StackedHistogram([histos[channel][sample] for sample in mcfakes_samples])
        
        # fakes histogram (= data-mc)
        histo_fakes[channel] = get_fakes(
            histos[channel]["Data"], 
            StackedHistogram([histos[channel][sample] for sample in mc_samples]).sum()
        )
        histo_fakes[channel].is_data = True
        histo_fakes[channel].name = labels[channel]
        histo_fakes[channel].color = colors[channel]

        # make plots per channel
        if do_plots and len(mcfakes_samples) > 0:
            plot_channel(
                histo_fakes[channel], 
                stack_mcfakes[channel], 
                plot_cfg, 
                f"plots_fakes/{channel}/{filenames[channel]}.pdf"
            )

    # make plots of ratio OS / SS
    if do_plots:
        plot_combined(
            histo_fakes["os"],
            histo_fakes["ss"],
            plot_cfg,
            f"plots_fakes/{region}_{variable}.pdf"
        )

    # fit transferfactor
    if do_fit and region=="bveto_mm" and variable=="mll_medium":
        fit_ratio(
            histo_fakes["os"], 
            histo_fakes["ss"], 
            plot_cfg
        )
    

def main():
    analysis_dict = get_analysis_dict()

    regions = analysis_dict["regions"]
    regions = [region for region in regions if f"{region}_ss" in regions]
    variables = analysis_dict["variables"]

    keep_keys = ["samples", "nuisances", "corrections", "colors", "plot_label", "lumi"]
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
