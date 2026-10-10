import concurrent.futures
import json
import subprocess
import sys
import time
import matplotlib as mpl
import mplhep as hep
import pickle
from argparse import ArgumentParser
from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.utils.histogram import Histogram, StackedHistogram
from spritz.utils.plotting_utils import (
    get_yrange, 
    plot_panel,
    make_plots,
    setup_fig,
    setup_multifig,
    add_to_samples
)

mpl.use("Agg")
plt = mpl.pyplot

d = hep.style.CMS.copy()
d["font.size"] = 8
d["figure.figsize"] = (5, 5)

d_vertical = hep.style.CMS.copy()
d_vertical["font.size"] = 8
d_vertical["figure.figsize"] = (5, 6)

d_multidim = hep.style.CMS.copy()
d_multidim["font.size"] = 9
d_multidim["figure.figsize"] = (10, 10)

plt.style.use(d)

# pretty labels but very slow
# d["text.usetex"] = True


def prepare_plot(
    region,
    variable,
    input_file,
    analysis_dict,
    variable_dict,
):
    print(region, variable)
    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    corrections = analysis_dict.get("corrections", dict())
    colors = analysis_dict["colors"]
    labels = analysis_dict.get("labels", {})

    fakes_dict = analysis_dict.get("fakes_dict", {})
    fakes_regions = [r.get("target") for r in fakes_dict.get("regions",[])]

    mc_samples = [x for x in samples if not samples[x].get("is_data") 
        or samples[x].get("is_smeft") or samples[x].get("is_variance")]    

    directory = {k:v for k,v in input_file.items() if k.startswith(f"{region}/{variable}/")}
    directory = {k.replace(f"{region}/{variable}/", ""):v for k,v in directory.items()}

    # get the histograms
    histos = {
        sample: Histogram.make_hist(
            directory, sample, nuisances, corrections, color=colors.get(sample,"black"),
            label=labels.get(sample), is_data=samples[sample].get("is_data", False)
        ) for sample in samples
    }

    # prepare total MC histogram
    stack_mc = StackedHistogram([histos[sample] for sample in mc_samples])

    if region in fakes_regions:
        for nuis in fakes_dict.get("nuisances", []) + ["stat"]:
            nuisances[nuis]["samples"] = add_to_samples(nuisances[nuis]["samples"].copy(), "Fakes")
        for corr in corrections:
            corrections[corr]["samples"] = add_to_samples(corrections[corr]["samples"].copy(), "Fakes")

        histo_fakes = Histogram.make_hist(
            directory, "Fakes", nuisances, corrections, label="Fakes", color=colors["Fakes"]
        )

        stack_mc.add(histo_fakes, position=0)
    
    histo_mc = stack_mc.sum(name="Tot MC")

    # prepare data histogram
    if "Data" in histos:
        histo_data = histos["Data"]
    else:
        histo_data = Histogram.empty_like(histo_mc, name="Data", is_data=True, color="black")

    histo_dict = {
        "stack_mc": stack_mc, 
        "histo_mc": histo_mc, 
        "histo_data": histo_data
    }

    return histo_dict


def plot_main(
    region,
    variable,
    histo_dict,
    analysis_dict,
    variable_dict,
):
    axis = variable_dict.get("axis")
    short_label = analysis_dict.get("short_label", False)

    panels = [
        { 
            "histos": {
                "MC Stack": {"histogram": "stack_mc", "show_label": True}, 
                "MC": {"histogram": "histo_mc", "show_label": not short_label, "show_label_unc": True},
                "Data": {"histogram": "histo_data", "show_label": True} 
            }
        }, {
            "denominator": "MC", 
            "histos": {
                "MC": {"histogram": "histo_mc"}, 
                "Data": {"histogram": "histo_data"} 
            }, 
            "yrange": (0.9,1.1)
        }
    ]
    
    if analysis_dict.get("no_ratio", False): 
        panels = panels[:1]
    
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
                    short_label=short_label,
                    hide_xlabel=i!=nrows-1,
                    hide_ylabel=j!=0,
                    hide_legend=True
                )

        handles, labels = ax[npanels*i,j].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper center", ncols=5, linewidth=0, fontsize=7)

    else:
        if plt.rcParams["figure.figsize"] != [5, 5]:
            plt.style.use(d)

        fig, ax = setup_fig(analysis_dict, npanels=npanels)

        make_plots(
            axes=ax, 
            histo_dict=histo_dict,
            panels=panels,
            variable_dict=variable_dict,
            short_label=short_label,
            legend_fontsize=8 if short_label else 5
        )

    figname = f"plots/{region}_{variable}.pdf"
    fig.savefig(figname, facecolor="white", bbox_inches="tight")
    print(f">> {figname}")
    plt.close()

    return histo_dict


def plot_variations(
    region,
    variable,
    nuis,
    histo_dict,
    analysis_dict,
    variable_dict,
):
    short_label = analysis_dict.get("short_label", False)

    nuisances = analysis_dict["nuisances"]
    name = nuisances[nuis].get("name", nuis)
    type = nuisances[nuis].get("type")
    kind = nuisances[nuis].get("kind")
    
    highlight_nuis = nuisances[nuis].get("nuisances") if type=="group" else [nuis]
    
    var_colors = ["red","blue","green","purple","cyan","magenta","grey","brown","orange"]

    panels = [
        {
            "denominator": "MC",
            "histos": {
                "MC": {
                    "histogram": "histo_mc", "nuisances": [], "highlight_nuis": highlight_nuis}
            }, 
        }, {
            "denominator": "MC",
            "histos": {
                "MC": {
                    "histogram": "histo_mc", "highlight_nuis": highlight_nuis, "show_label_unc": True}
            }, 
        }
    ]
    
    if kind in ["envelope","square","stdev"]:
        variations = nuisances[nuis]["variations"]
        for i,var in enumerate(variations):
            label = f"{name}_{i}"
            panels[0]["histos"][f"MC {label}"] = {
                "histogram": "histo_mc", "variation": label, "label": label,
                "color": var_colors[i % 8], "nuisances": [],
                "show_label": i<8, "alpha": 0.3 if len(variations)>10 else 1.
            }
    else:
        for i,(var,label) in enumerate(zip(["Up","Down"], ["$+1\\sigma$","$-1\\sigma$"])):
            panels[0]["histos"][f"MC {nuis} {var}"] = {
                "histogram": "histo_mc", "variation": f"{nuis} {var}", "label": f"{nuis} {label}",
                "color": var_colors[i], "nuisances": [], "show_label": True
            }
            if type=="group":
                panels[1]["histos"]["MC"]["label_highlight"] = nuis
    
    panels[1]["histos"]["Data"] = {"histogram": "histo_data", "show_label": True}
    npanels = len(panels)
    
    if plt.rcParams["figure.figsize"] != [5, 5]:
        plt.style.use(d)

    fig, ax = setup_fig(analysis_dict, npanels=npanels, height_ratios=[1]*npanels)

    make_plots(
        axes=ax, 
        histo_dict=histo_dict,
        panels=panels,
        variable_dict=variable_dict,
        short_label=short_label,
        legend_ncols=4,
        legend_fontsize=7 if short_label else 5
    )

    figname = f"plots/variations/{region}_{variable}_{name}.pdf"
    fig.savefig(figname, facecolor="white", bbox_inches="tight")
    print(f">> {figname}")
    plt.close()


def plot_corrections(
    region,
    variable,
    corr,
    histo_dict,
    analysis_dict,
    variable_dict,
):
    short_label = analysis_dict.get("short_label", False)
    three_panels = variable=="nPVs" and corr=="Pile-up corr."

    corrections = analysis_dict["corrections"]
    name = corrections[corr].get("name")
    corr_samples = corrections[corr].get("samples")
    nuisances = corrections[corr].get("related_nuisances")
    if nuisances is None:
        nuisances = [corr] if corr in analysis_dict["nuisances"] else []

    if corr in histo_dict["histo_data"].corrections:
        panels = [
            {
                "denominator": "MC before corr.",
                "histos": {
                    "MC before corr.": { 
                        "histogram": "histo_mc", "label": f"MC (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "show_label": True },
                    "MC after corr.": { 
                        "histogram": "histo_mc", "label": f"MC (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True },
                },
            }, {
                "denominator": "Data before corr.",
                "histos": {
                    "Data before corr.": { 
                        "histogram": "histo_data", "label": f"Data (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "show_label": True },
                    "Data after corr.": { 
                        "histogram": "histo_data", "label": f"Data (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True },
                }
            }, {
                "denominator": "MC before corr.",
                "histos": {
                    "MC before corr.": { 
                        "histogram": "histo_mc", "label": f"MC (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "show_label": True },
                    "Data before corr.": { 
                        "histogram": "histo_data", "label": f"Data (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "show_label": True },
                },
            }, {
                "denominator": "MC after corr.",
                "histos": {
                    "MC after corr.": { 
                        "histogram": "histo_mc", "label": f"MC (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True },
                    "Data after corr.": { 
                        "histogram": "histo_data", "label": f"Data (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True },
                }
            }
        ]
    else:
        if three_panels:
            panels = [{
                "histos": {
                    "MC before corr.": { 
                        "histogram": "histo_mc", "label": f"MC (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "linestyle": "dashed", "show_label": True },
                    "MC": {
                        "histogram": "histo_mc", "label": f"MC (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True},
                    "Data": {
                        "histogram": "histo_data", "show_label": True} 
                }
            }]
        else:
            panels = []
        
        panels += [
            {
                "denominator": "MC before corr.",
                "histos": {
                    "MC before corr.": { 
                        "histogram": "histo_mc", "label": f"MC (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "linestyle": "dashed", "show_label": True },
                    "MC after corr.": { 
                        "histogram": "histo_mc", "label": f"MC (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True },
                    "Data": { 
                        "histogram": "histo_data", "show_label": True }
                },
            }, 
            {
                "denominator": "Data",
                "histos": {
                    "MC before corr.": { 
                        "histogram": "histo_mc", "label": f"MC (before {corr})", 
                        "variation": f"{corr} Before", "nuisances": [], "color": "blue", 
                        "linestyle": "dashed", "show_label": True },
                    "MC after corr.": { 
                        "histogram": "histo_mc", "label": f"MC (after {corr})", 
                        "nuisances": nuisances, "color": "red", "show_label": True },
                    "Data": { 
                        "histogram": "histo_data", "show_label": True }
                }, 
            }
        ]
    
    npanels = len(panels)

    figsize = plt.rcParams["figure.figsize"]
    
    if npanels==2 and figsize != [5,5]:
        plt.style.use(d)
    elif npanels==4 and figsize != [5,6]:
        plt.style.use(d_vertical)

    fig, ax = setup_fig(analysis_dict, npanels=npanels, height_ratios=[1]*npanels)

    make_plots(
        axes=ax, 
        histo_dict=histo_dict,
        panels=panels,
        variable_dict=variable_dict,
        short_label=short_label,
        legend_ncols=4,
        legend_fontsize=7 if short_label else 5
    )

    figname = f"plots/corrections/{region}_{variable}_{name}.pdf"
    fig.savefig(figname, facecolor="white", bbox_inches="tight")
    print(f">> {figname}")
    plt.close()


def main():
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument("-f", "--fakes", action="store_true")
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--variations", action="store_true")
    parser.add_argument("--corrections", action="store_true")
    parser.add_argument("--no-ratio", action="store_true")
    parser.add_argument("--short-label", action="store_true")
    args = parser.parse_args()

    if not args.main and not args.variations and not args.corrections:
        args.main = True
    
    analysis_dict = get_analysis_dict()
    print()

    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]

    if args.variables is not None:
        variables = {k:v for k,v in variables.items() if k in args.variables}

    keep_keys = ["samples", "nuisances", "corrections", "colors", "labels", "lumi", "plot_label","fakes_dict"]
    analysis_dict = { k:v for k,v in analysis_dict.items() if k in keep_keys }

    analysis_dict |= {
        "add_fakes": args.fakes,
        "no_ratio": args.no_ratio,
        "short_label": args.short_label
    }

    cmd_mkdir = f"mkdir -p plots && cp {get_fw_path()}/data/common/index.php plots/"
    if args.variations:
        cmd_mkdir += f" && mkdir -p plots/variations && cp {get_fw_path()}/data/common/index.php plots/variations/"
    if args.corrections:
        cmd_mkdir += f" && mkdir -p plots/corrections && cp {get_fw_path()}/data/common/index.php plots/corrections/"
    
    proc = subprocess.Popen(cmd_mkdir, shell=True)
    proc.wait()

    with open("histos.pkl", "rb") as f:
        input_file = pickle.load(f)

    input_dict = {}

    cpus = 10

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = {}

        histo_dict = {}

        for region in regions:
            histo_dict[region], input_dict[region] = {}, {}
            
            for variable in variables:
                keep_keys = ["label", "unit", "xlog", "ylog", "axis"]
                variable_dict = { k:v for k,v in variables[variable].items() if k in keep_keys }
                if "axis" not in variable_dict:
                    continue

                # input_dict[region][variable] = {}
                # for k in list(input_file.keys()):
                #     if k.startswith(f"{region}/{variable}"):
                #         h = k.replace(f"{region}/{variable}/", "")
                #         input_dict[region][variable][h] = input_file.pop(k)
                
                task = executor.submit(
                    prepare_plot,
                    region,
                    variable,
                    input_file,
                    #input_dict[region][variable],
                    analysis_dict,
                    variable_dict,
                )
                tasks[task] = (region, variable)

        for task in concurrent.futures.as_completed(tasks):
            #result_dict_ = task.result()
            region, variable = tasks[task]
            histo_dict[region][variable] = task.result()

        tasks = []

        for region in regions:
            for variable in variables:
                keep_keys = ["label", "unit", "xlog", "ylog", "axis"]
                variable_dict = { k:v for k,v in variables[variable].items() if k in keep_keys }
                if "axis" not in variable_dict:
                    continue

                # main plots
                if args.main:
                    tasks.append(
                        executor.submit(
                            plot_main,
                            region,
                            variable,
                            histo_dict[region][variable],
                            analysis_dict,
                            variable_dict
                        )
                    )

                if isinstance(variable_dict.get("axis"), list):
                    continue

                # corrections
                if args.corrections:
                    for corr in analysis_dict["corrections"]:
                        tasks.append(
                            executor.submit(
                                plot_corrections,
                                region,
                                variable,
                                corr,
                                histo_dict[region][variable],
                                analysis_dict,
                                variable_dict,
                            )
                        )
                
                # systematics
                if args.variations:
                    for nuis in analysis_dict["nuisances"]:
                        tasks.append(
                            executor.submit(
                                plot_variations,
                                region,
                                variable,
                                nuis,
                                histo_dict[region][variable],
                                analysis_dict,
                                variable_dict
                            )
                        )

        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()

    end = time.time()
    print(f"\n>> done in {end-start:.0f}s\n")

if __name__ == "__main__":
    main()

