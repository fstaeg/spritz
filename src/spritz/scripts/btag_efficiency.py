import concurrent.futures
import json
import subprocess
import sys
import correctionlib
import correctionlib.schemav2 as cs
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
)
from spritz.scripts.plot import make_plots

mpl.use("Agg")
d = deepcopy(hep.style.CMS)
d["font.size"] = 10
d["figure.figsize"] = (10, 5)
plt = mpl.pyplot
plt.style.use(d)

def btageff(
    region,
    sample,
    variable,
    analysis_dict,
    variable_dict,
    do_plots,
    do_json
):
    
    print("Doing", sample, variable)

    variable_label = variable_dict.get("label", variable)
    unit = variable_dict.get("unit")
    xlog = variable_dict.get("xlog", False)
    axis = variable_dict.get("axis")

    input_file = uproot.open("histos.root")

    nuisances = analysis_dict["nuisances"]
    corrections = analysis_dict.get("corrections", dict())
    plot_label = analysis_dict.get("plot_label", "Run-II")
    lumi = analysis_dict["lumi"]

    nuisances["stat"] = {
        "name": "stat",
        "type": "stat",
        "samples": {sample: "1.00"},
    }

    variable_tagged, variable_all = variable, "_".join(variable.split("_")[:-1])
    histos = {}

    # get the histograms
    for variable_ in [variable_tagged, variable_all]:
        directory = input_file[f"{region}/{variable_}"]

        histos[variable_] = Histogram.make_hist(directory, nuisances, corrections, sample)

    if do_plots:
        cmd_mkdir = f"mkdir -p plots_efficiency/{sample.replace(" ", "_")}"
        proc = subprocess.Popen(cmd_mkdir, shell=True)
        proc.wait()

    # take ratio
    all_denom = np.where(histos[variable_all].nominal > 0, histos[variable_all].nominal, 1e-6)
    tag_denom = np.where(histos[variable_tagged].nominal > 0, histos[variable_tagged].nominal, 1e-6)
    
    nominal = histos[variable_tagged].nominal / all_denom
    num_stat = histos[variable_tagged].up(["stat"]) / tag_denom
    denom_stat = histos[variable_all].up(["stat"]) / all_denom
    variations_efficiency = {
        "stat": HistVariation({
            "up": np.sqrt(np.square(num_stat)+np.square(denom_stat))*nominal,
            "down": np.sqrt(np.square(num_stat)+np.square(denom_stat))*nominal 
    })}
    histo_efficiency = Histogram(
        name="Efficiency", 
        nominal=nominal, 
        varied=variations_efficiency,
        corrected={}, 
        color="black",
        axis=histos[variable_tagged].axis
    )

    matrix = histo_efficiency.nominal
    matrix_unc = histo_efficiency.up()

    # make plots
    if do_plots:
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax = np.array([ax])
        fig.tight_layout(pad=-0.5)
        hep.cms.label("Preliminary", data=True, lumi=round(lumi, 2), ax=ax[0], year=plot_label)

        xaxis_dict = { "xlabel": variable_label, "unit": unit, "xlog": xlog }
        
        make_plots(
            axes=ax, 
            histo_dict={"Efficiency": histo_efficiency},
            panels=[{
                "histos": ["Efficiency"], "absolute": True, "ylabel": "Efficiency", "yrange": (0, 1)
            }],
            xaxis=xaxis_dict,
            ylog=False
        )

        fig.savefig(
            f"plots_efficiency/{sample.replace(" ", "_")}/{variable}.pdf",
            facecolor="white",
            pad_inches=0.1,
            bbox_inches="tight",
        )

        plt.close()

    return matrix.tolist(), matrix_unc.tolist()


def main():
    region = "inc_mm"
    variable = "pt"

    samples = ["Inclusive", "DYll", "DYtt", "Single Top", "TT", "VV"]
    jet_flavours = {0: "lj", 4: "cj", 5: "bj"}
    working_points = {"L": "btagloose", "M": "btagmedium", "T": "btagtight"}

    _dict = get_analysis_dict()
    analysis_dict = { 
        k:v for k,v in _dict.items() if k in [
            "nuisances", "corrections", "plot_label", "lumi"
        ]
    }

    cpus = 15

    do_plots = "--plot" in sys.argv
    do_json = "--json" in sys.argv

    if do_plots:
        cmd_mkdir = f"mkdir -p plots_efficiency && cp {get_fw_path()}/data/common/index.php plots_efficiency/"
        proc = subprocess.Popen(cmd_mkdir, shell=True)
        proc.wait()

    if do_json:
        values = {
            sample: {
                jf: {
                    wp: {} for wp in working_points
                } for jf in jet_flavours
            } for sample in samples
        }

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = {}

        for sample in samples:
            for jf in jet_flavours:
                variable_dict = _dict["variables"][f"{variable}{jet_flavours[jf]}"]
                variable_dict = {
                    k:v for k,v in variable_dict.items() if k in [
                        "label", "unit", "xlog", "axis"
                    ]
                }
                for wp in working_points:
                    var = f"{variable}{jet_flavours[jf]}_{working_points[wp]}"
                    task = executor.submit(
                        btageff,
                        region,
                        sample,
                        var,
                        analysis_dict,
                        variable_dict,
                        do_plots,
                        do_json
                    )
                    tasks[task] = (sample, jf, wp)
        
        for task in concurrent.futures.as_completed(tasks):
            (central, stat) = task.result()
            if do_json:
                sample, jf, wp = tasks[task]
                values[sample][jf][wp]["central"] = central
                values[sample][jf][wp]["stat"] = stat
            
    if do_json:
        axis_dict = {
            jf: _dict["variables"][f"{variable}{jet_flavours[jf]}"]["axis"] for jf in jet_flavours
        }
        cset = cs.CorrectionSet(
            schema_version=2,
            corrections=[
                cs.Correction(
                    name=sample, version=1,
                    inputs=[
                        cs.Variable(name="systematic", type="string"),
                        cs.Variable(name="working_point", type="string", description="L/M/T"),
                        cs.Variable(name="flavor", type="int", description="hadron flavor definition: 5=b, 4=c, 0=udsg"),
                        cs.Variable(name="pt", type="real"),
                    ],
                    output=cs.Variable(name="efficiency", type="real"),
                    data=cs.Category(
                        input="systematic", nodetype="category",
                        content=[
                            cs.CategoryItem(
                                key=syst,
                                value=cs.Category(
                                    input="working_point", nodetype="category",
                                    content=[
                                        cs.CategoryItem(
                                            key=wp,
                                            value=cs.Category(
                                                input="flavor", nodetype="category",
                                                content=[
                                                    cs.CategoryItem(
                                                        key=flavor,
                                                        value=cs.Binning(
                                                            input="pt", nodetype="binning",
                                                            edges=axis_dict[flavor].edges.tolist(), flow="clamp",
                                                            content=values[sample][flavor][wp][syst]
                                                        )
                                                    ) for flavor in jet_flavours
                                                ]
                                            )
                                        ) for wp in working_points
                                    ]
                                )
                            ) for syst in ["central", "stat"]
                        ]
                    )
                ) for sample in samples
            ]
        )

        with open("btagging_eff.json", "w") as f:
            f.write(cset.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
