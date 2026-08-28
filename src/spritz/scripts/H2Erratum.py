import concurrent.futures
import json
import subprocess
import sys
from copy import deepcopy

import matplotlib as mpl
import mplhep as hep
import numpy as np
import uproot
from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.scripts.plot import make_plots
from spritz.utils.plotting_utils import Histogram, StackedHistogram

mpl.use("Agg")

d = deepcopy(hep.style.CMS)
d["font.size"] = 8
d["figure.figsize"] = (5, 5)

plt = mpl.pyplot
plt.style.use(d)


def H2Erratum(
    region,
    variable,
    analysis_dict,
    variable_dict,
    do_plots,
    do_json
):
    
    print("Doing ", region, variable)

    input_file = uproot.open("histos.root")

    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    corrections = analysis_dict.get("corrections", dict())
    colors = analysis_dict["colors"]
    plot_label = analysis_dict.get("plot_label", "Run-II")
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

    # get the histograms
    histos = {
        sample: Histogram.make_hist(
            directory, nuisances, corrections, sample, 
            is_data=samples[sample].get("is_data", False), color=colors.get(sample,"black")
        )
        for sample in samples
    }
    
    DYmm_H2ErratumFix = histos["DYll"]
    DYmm_noH2ErratumFix = histos["DYll_noH2ErratumFix"]
    DYtt_H2ErratumFix = histos["DYtt"]
    DYtt_noH2ErratumFix = histos["DYtt_noH2ErratumFix"]

    # make plots
    if do_plots:
        fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [2,1,1]}, dpi=200)
        hep.cms.label("Preliminary", data=True, lumi=round(lumi, 2), ax=ax[0], year=plot_label)
        fig.tight_layout(pad=-0.5)

        xaxis_dict = { 
            "xlabel": variable_label, "unit": unit, "xlog": xlog 
        }
        histo_dict = {
            "DYmm": DYmm_noH2ErratumFix,
            "DYmm H2ErratumFix": DYmm_H2ErratumFix,
            "DYtt": DYtt_noH2ErratumFix, 
            "DYtt H2ErratumFix": DYtt_H2ErratumFix,
        }
        panels = [{
            "histos": ["DYmm","DYmm H2ErratumFix","DYtt","DYtt H2ErratumFix"],
            "labels": ["DYmm","DYmm H2ErratumFix","DYtt","DYtt H2ErratumFix"]
        }, {
            "histos": ["DYmm","DYmm H2ErratumFix"], "denominator": "DYmm", 
            "yrange": (0.95,1.05) if "ptll" in variable else (0.99,1.01)
        }, {
            "histos": ["DYtt","DYtt H2ErratumFix"], "denominator": "DYtt", 
            "yrange": (0.95,1.05) if "ptll" in variable else (0.99,1.01)
        }]

        make_plots(
            axes=ax, histo_dict=histo_dict, panels=panels, xaxis=xaxis_dict, ylog=ylog
        )
        
        fig.savefig(f"plots/{region}_{variable}.pdf", facecolor="white", pad_inches=0.1, bbox_inches="tight")
        plt.close()

    # make json
    if do_json and variable == "ptll":
        edges = DYmm_H2ErratumFix.edges
        
        mm_H2, mm_err_H2 = DYmm_H2ErratumFix.nominal, (DYmm_H2ErratumFix.up()+DYmm_H2ErratumFix.down()) / 2
        mm_noH2, mm_err_noH2 = DYmm_noH2ErratumFix.nominal, (DYmm_noH2ErratumFix.up()+DYmm_noH2ErratumFix.down()) / 2
        mm_ratio = mm_H2 / mm_noH2
        mm_err = mm_ratio * np.sqrt((mm_err_H2/mm_H2)**2 + (mm_err_noH2/mm_noH2)**2)

        tt_H2, tt_err_H2 = DYtt_H2ErratumFix.nominal, (DYtt_H2ErratumFix.up()+DYtt_H2ErratumFix.down()) / 2
        tt_noH2, tt_err_noH2 = DYtt_noH2ErratumFix.nominal, (DYtt_noH2ErratumFix.up()+DYtt_noH2ErratumFix.down()) / 2
        tt_ratio = tt_H2 / tt_noH2
        tt_err = tt_ratio * np.sqrt((tt_err_H2/tt_H2)**2 + (tt_err_noH2/tt_noH2)**2)

        ratio_dict = {
            "DYmm": {
                "edges": edges.tolist(),
                "nominal": mm_ratio.tolist(),
                "err": mm_err.tolist()
            },
            "DYtt": {
                "edges": edges.tolist(),
                "nominal": tt_ratio.tolist(),
                "err": tt_err.tolist()
            }
        }

        with open("H2Erratum_weights.json", "w") as f:
            json.dump(ratio_dict, f, indent=2)


def main():
    analysis_dict = get_analysis_dict()

    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]

    keep_keys = ["samples", "nuisances", "corrections", "colors", "lumi", "plot_label"]
    analysis_dict = { k:v for k,v in analysis_dict.items() if k in keep_keys }

    do_plots = "--plot" in sys.argv
    do_json = "--json" in sys.argv

    if do_plots:
        cmd_mkdir = f"mkdir -p plots && cp {get_fw_path()}/data/common/index.php plots/"
        proc = subprocess.Popen(cmd_mkdir, shell=True)
        proc.wait()
        cpus = 10
        print("Doing plots")
    else:
        variables = { k:v for k,v in variables.items() if k=="ptll" }
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
                        H2Erratum,
                        region,
                        variable,
                        analysis_dict,
                        variable_dict,
                        do_plots,
                        do_json
                    )
                )
        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()

if __name__ == "__main__":
    main()
