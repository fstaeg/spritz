"""
spritz-plot-eft: the main spritz-plot plots (stack, data, uncertainties) with
the total prediction (MC + EFT) for some operators drawn on top as lines, and
its ratio to MC in the ratio panel.

    spritz-plot-eft --ops cHDD=1.1 cHDD=-1.1 cHl1=0.5 [--model DYmm_SMEFTsim0j] [--variables mll]

For an operator op with Wilson coefficient c (default 1), from the templates
sm, w1_op (c = 1) and wm1_op (c = -1) of histos.pkl:
    lin  = 0.5 * (w1_op - wm1_op)
    quad = 0.5 * (w1_op + wm1_op) - sm
the line is the expected number of events (bkg + SM + EFT) for that value of
the Wilson coefficient: Tot MC + c * lin + c^2 * quad.
"""
import concurrent.futures
import pickle
import subprocess
import sys
import time
from argparse import ArgumentParser

import numpy as np
from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.scripts.plot import histo_owner, plot_main, plotted_samples, prepare_plot
from spritz.utils.histogram import Histogram

op_colors = [
    "crimson", "royalblue", "forestgreen", "darkviolet",
    "darkorange", "deepskyblue", "saddlebrown", "olive",
]


def parse_ops(values):
    """["cHDD=1.1", "cHDD=-1.1", "cHl1"] -> [("cHDD", 1.1), ("cHDD", -1.1), ("cHl1", 1.0)]:
    a list, the same operator can be drawn with several values"""
    ops = []
    for value in values:
        op, _, c = value.partition("=")
        ops.append((op, float(c) if c else 1.0))
    return ops


def eft_models(samples):
    """{model: {point: sample}} of the SMEFT templates: an "is_smeft" sample
    with datacard name "point" is called "<model>_<point>" (DYmm_sm,
    DYmm_SMEFTsim0j_w1_cHDD, ...)"""
    models = {}
    for sample, cfg in samples.items():
        point = cfg.get("name")
        if not cfg.get("is_smeft") or not point or not sample.endswith(f"_{point}"):
            continue
        models.setdefault(sample[: -len(point) - 1], {})[point] = sample
    return models


def choose_model(samples, ops, model=None):
    """{point: sample} of the sm, w1_op and wm1_op templates of the EFT model
    (default: the only one, or the one whose sm is in the stack)"""
    models = eft_models(samples)
    if not models:
        sys.exit("No SMEFT templates (is_smeft samples) in the config")
    if model is None:
        candidates = list(models)
        if len(candidates) > 1:
            candidates = [m for m in models if models[m].get("sm") in plotted_samples(samples)[0]]
        if len(candidates) != 1:
            sys.exit(f"Several EFT models {list(models)}: choose one with --model")
        model = candidates[0]
    elif model not in models:
        sys.exit(f"Unknown EFT model {model}, the config has {list(models)}")

    points = models[model]
    available = [p[len("w1_"):] for p in points if p.startswith("w1_")]
    operators = list(dict.fromkeys(op for op, _ in ops))
    missing = [op for op in operators if f"w1_{op}" not in points or f"wm1_{op}" not in points]
    if "sm" not in points or missing:
        sys.exit(f"Model {model}: no sm or w1/wm1 templates for {missing}, available operators: {available}")

    points = ["sm"] + [f"{w}_{op}" for op in operators for w in ["w1", "wm1"]]
    return model, {p: models[model][p] for p in points}


def eft_histogram(like, values, name, color):
    histo = Histogram.empty_like(like, name=name, color=color)
    histo.data.loc["nominal"] = values
    return histo


def plot_eft(region, variable, input_dict, analysis_dict, variable_dict, eft_samples, ops, figdir):
    histo_dict = prepare_plot(region, variable, input_dict, analysis_dict, variable_dict)

    def values(point):
        return input_dict[f"{region}/{variable}/histo_{eft_samples[point]}"].values()

    histo_mc = histo_dict["histo_mc"]
    mc = histo_mc.nominal
    sm = values("sm")

    main, ratio = {}, {}
    deviation = 0.1  # spritz-plot's ratio range is 0.9-1.1, widened to show the EFT
    for i, (op, c) in enumerate(ops):
        w1, wm1 = values(f"w1_{op}"), values(f"wm1_{op}")
        eft = c * 0.5 * (w1 - wm1) + c**2 * (0.5 * (w1 + wm1) - sm)
        color = op_colors[i % len(op_colors)]
        tag = f"{op}={c:g}"

        key = f"MC + EFT {i}"
        histo_dict[key] = eft_histogram(histo_mc, mc + eft, f"Tot MC + {tag}", color)
        main[key] = {
            "histogram": key, "nuisances": [], "show_label": True, "color": color,
            "linewidth": 1.5, "zorder": 3,
        }
        ratio[key] = {"histogram": key, "nuisances": [], "color": color, "linewidth": 1.5, "zorder": 3}
        filled = mc > 1e-6
        if np.any(filled):
            deviation = max(deviation, 1.25 * np.max(np.abs(eft[filled] / mc[filled])))

    deviation = min(deviation, 1)
    # room for the longer legend: more space above the histograms and, in
    # the 1D plots, fewer columns so that it stays as wide as the axes
    extra = [
        {"histos": main, "more_space": True},
        {"histos": ratio, "yrange": (1 - deviation, 1 + deviation)},
    ]
    legend_ncols = 5 if isinstance(variable_dict["axis"], list) else 4
    plot_main(
        region, variable, histo_dict, analysis_dict, variable_dict,
        extra=extra, figdir=figdir, legend_ncols=legend_ncols,
    )


def main():
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument(
        "--ops", nargs="+", required=True,
        help="operators to draw, with their Wilson coefficient (default 1): cHDD=1.1 cHl1=0.5 ...",
    )
    parser.add_argument("--model", default=None, help="EFT model, e.g. DYmm_SMEFTsim0j (needed only with several)")
    parser.add_argument("--variables", nargs="+", default=None)
    parser.add_argument("--outdir", default="plots/eft")
    args = parser.parse_args()

    ops = parse_ops(args.ops)

    analysis_dict = get_analysis_dict()
    print()

    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]
    if args.variables is not None:
        variables = {k: v for k, v in variables.items() if k in args.variables}

    model, eft_samples = choose_model(analysis_dict["samples"], ops, args.model)
    print(f"EFT model {model}: " + ", ".join(f"{op}={c:g}" for op, c in ops))

    keep_keys = ["samples", "nuisances", "corrections", "colors", "labels", "lumi", "plot_label", "fakes_dict", "split_uncertainties"]
    analysis_dict = {k: v for k, v in analysis_dict.items() if k in keep_keys}

    proc = subprocess.Popen(
        f"mkdir -p {args.outdir} && cp {get_fw_path()}/data/common/index.php {args.outdir}/",
        shell=True,
    )
    proc.wait()

    with open("histos.pkl", "rb") as f:
        input_file = pickle.load(f)

    # as in spritz-plot, every task gets only the histograms it plots: those of
    # the stacked samples and data, and the nominal EFT templates
    samples = analysis_dict["samples"]
    mc_samples, data_samples = plotted_samples(samples)
    needed = set(mc_samples + data_samples + ["Fakes"])
    sample_names = set(samples) | {"Fakes"}
    eft_names = {f"histo_{sample}" for sample in eft_samples.values()}
    wanted = {(r, v) for r in regions for v in variables if "axis" in variables[v]}
    input_dict = {}
    for key, h in input_file.items():
        region, variable, name = key.split("/", 2)
        if (region, variable) in wanted and (name in eft_names or histo_owner(name, sample_names) in needed):
            input_dict.setdefault((region, variable), {})[key] = h
    del input_file

    cpus = 10

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = []
        for region in regions:
            for variable in variables:
                keep_keys = ["label", "unit", "xlog", "ylog", "axis"]
                variable_dict = {k: v for k, v in variables[variable].items() if k in keep_keys}
                if "axis" not in variable_dict:
                    continue
                tasks.append(
                    executor.submit(
                        plot_eft,
                        region,
                        variable,
                        input_dict.pop((region, variable), {}),
                        analysis_dict,
                        variable_dict,
                        eft_samples,
                        ops,
                        args.outdir,
                    )
                )

        concurrent.futures.wait(tasks)
        for task in tasks:
            task.result()

    end = time.time()
    print(f"\n>> done in {end-start:.0f}s\n")


if __name__ == "__main__":
    main()
