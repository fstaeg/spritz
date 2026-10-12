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
from spritz.scripts.plot import (
    # ratio_print,
    # plot_panel,
    make_plots,
    make_plots_multidim,
    setup_multifig
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


def plot_fakes(
    region,
    variable,
    analysis_dict,
    variable_dict,
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

    mc_samples = [x for x in samples if not samples[x].get("is_data", False) and not x in ["W+Jets","QCD"]]
    mcfakes_samples = [x for x in samples if x in ["W+Jets","QCD"]]

    histos, stack_mc, histo_mc, stack_mcfakes, histo_mcfakes, histo_data, histo_fakes = {}, {}, {}, {}, {}, {}, {}
    filenames = { region: f'{region}_os_{variable}', f'{region}_ss': f'{region}_ss_{variable}' }

    for region_ in [region, f'{region}_ss']:
        directory = input_file[f"{region_}/{variable}"]
    
        # get the histograms
        histos[region_] = {
            sample: Histogram.make_hist(directory, nuisances, corrections, sample, is_data=samples[sample].get("is_data", False), color=colors.get(sample,"black"))
            for sample in samples
        }

        # prepare total MC histogram
        stack_mc[region_] = StackedHistogram([histos[region_][sample] for sample in mc_samples])
        histo_mc[region_] = stack_mc[region_].sum("Tot MC", color="black")

        # prepare fake MC histograms
        stack_mcfakes[region_] = StackedHistogram([histos[region_][sample] for sample in mcfakes_samples])
        histo_mcfakes[region_] = stack_mcfakes[region_].sum("MC Fakes", color="black")

        # prepare data histogram
        if "Data" in histos[region_]:
            histo_data[region_] = histos[region_]["Data"]
        else:
            histo_data[region_] = Histogram.empty_like(histo_mc[region_], name="Data", is_data=True, color="black")

        # subtract MC from data to get fakes
        variations_fakes = {}
        variations_fakes["stat"] = HistVariation({
            "up": np.sqrt(np.square(histo_data[region_].up(["stat"]))+np.square(histo_mc[region_].up(["stat"]))),
            "down": np.sqrt(np.square(histo_data[region_].down(["stat"]))+np.square(histo_mc[region_].down(["stat"]))) 
        })

        if "fakerw_param" in histo_data[region_].variation_names:
            variations_fakes["fakerw_param"] = HistVariation({   
                "up": histo_data[region_].varied["fakerw_param"].up()-histo_mc[region_].varied["fakerw_param"].up(),
                "down": histo_data[region_].varied["fakerw_param"].down()-histo_mc[region_].varied["fakerw_param"].down() 
            }, kind="weight")
        
        if "fakerw_model" in histo_data[region_].variation_names:
            variations_fakes["fakerw_model"] = HistVariation({
                "fakerw_model_exp": histo_data[region_].varied["fakerw_model"]["fakerw_model_exp"]-histo_mc[region_].varied["fakerw_model"]["fakerw_model_exp"],
                "fakerw_model_erf": histo_data[region_].varied["fakerw_model"]["fakerw_model_erf"]-histo_mc[region_].varied["fakerw_model"]["fakerw_model_erf"] 
            }, kind="envelope")

        correction_names = union([histo_data[region_].correction_names, histo_mc[region_].correction_names])
        corrections_fakes = {}

        for corr in correction_names:
            corr_data = histo_data[region_].corrected[corr] if corr in histo_data[region_].correction_names else histo_data[region_].nominal
            corr_mc = histo_mc[region_].corrected[corr] if corr in histo_mc[region_].correction_names else histo_mc[region_].nominal
            corrections_fakes[corr] = corr_data-corr_mc

        # prepare fakes histogram (= data-mc)
        histo_fakes[region_] = Histogram(
            name="Fakes", 
            nominal=histo_data[region_].nominal-histo_mc[region_].nominal, 
            varied=variations_fakes,
            corrected=corrections_fakes, 
            is_data=True, 
            color="black",
            axis=histo_data[region_].axis
        )

        # make plots
        xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }
        histo_dict = { "Fakes": histo_fakes[region_], "MC Stack": stack_mcfakes[region_],
                       **{s: histos[region_][s] for s in mcfakes_samples} }
        panels = [{ "histos": ["Fakes","MC Stack"], "labels": ["Fakes","MC Stack"] }]
        if len(mcfakes_samples) > 0:
            panels.append({ "histos": ["Fakes"]+[s for s in mcfakes_samples], "denominator": "Fakes", "yrange": (0.,1.5) })

        if isinstance(axis, list):
            if len(axis)==3:
                ncols, nrows = len(axis[1].centers), len(axis[2].centers)
            elif len(axis)==2:
                nrows = math.floor(math.sqrt(len(axis[1].centers)))
                ncols = math.ceil(len(axis[1].centers)/nrows)

            plt.style.use(d_multidim)
            fig, ax = setup_multifig(ncols, nrows)
            fig.tight_layout(pad=-0.4)
            hep.cms.label('Preliminary', rlabel="", data=True, ax=ax[0,0])
            hep.label.exp_label(data=True, lumi=round(lumi, 2), year=year_label, ax=ax[0,-1])

            make_plots_multidim(
                axes=ax, histo_dict=histo_dict, h_axis=axis,
                panels=panels, xaxis=xaxis_dict, ylog=ylog
            )

        else:
            plt.style.use(d)
            if len(mcfakes_samples) > 0:
                fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,1]}, dpi=200)
            else:
                fig, ax = plt.subplots(1, 1, dpi=200)
                ax = np.array([ax])
            hep.cms.label('Preliminary', data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)
            fig.tight_layout(pad=-0.5)
            
            make_plots(
                axes=ax, histo_dict=histo_dict, panels=panels,
                xaxis=xaxis_dict, ylog=ylog
            )
    
        fig.savefig(
            f"plots_fakes/{filenames[region_]}.pdf",
            facecolor="white",
            pad_inches=0.1,
            bbox_inches="tight",
        )

        plt.close()

    # make plots
    histo_fakes[region].name = 'opposite-sign (fakes)'
    histo_fakes[region].color = 'blue'
    histo_fakes[f'{region}_ss'].name = 'same-sign (fakes)'
    histo_fakes[f'{region}_ss'].color = 'red'

    xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog}
    histo_dict = { "OS": histo_fakes[region], "SS": histo_fakes[f'{region}_ss'] }
    panels = [
        { "histos": ["OS","SS"] },
        { "histos": ["OS","SS"], "denominator": "SS", "yrange": (0.,2.5) }
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
        hep.cms.label('Preliminary', rlabel="", data=True, ax=ax[0,0])
        hep.label.exp_label(data=True, lumi=round(lumi, 2), year=year_label, ax=ax[0,-1])

        make_plots_multidim(
            axes=ax, histo_dict=histo_dict, h_axis=axis,
            panels=panels, xaxis=xaxis_dict, ylog=ylog
        )
    else:
        plt.style.use(d)
        fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3,1]}, dpi=200)
        hep.cms.label('Preliminary', data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label)
        fig.tight_layout(pad=-0.5)
        
        make_plots(
            axes=ax, histo_dict=histo_dict, panels=panels,
            xaxis=xaxis_dict, ylog=ylog, print_ratio=(region=='inc_mm')&(variable=='mll_medium')
        )
    
    fig.savefig(
        f"plots_fakes/{region}_{variable}.pdf",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )

    plt.close()
    

def main():
    analysis_dict = get_analysis_dict()

    regions = analysis_dict["regions"]
    regions = [region for region in regions if f'{region}_ss' in regions]
    variables = analysis_dict["variables"]

    keep_keys = ["samples", "nuisances", "corrections", "colors", "year_label", "lumi"]
    analysis_dict = { k:v for k,v in analysis_dict.items() if k in keep_keys }

    cmd_mkdir = f"mkdir -p plots_fakes && cp {get_fw_path()}/data/common/index.php plots_fakes/"
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
                        plot_fakes,
                        region,
                        variable,
                        analysis_dict,
                        variable_dict,
                    )
                )
        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()

if __name__ == "__main__":
    main()
