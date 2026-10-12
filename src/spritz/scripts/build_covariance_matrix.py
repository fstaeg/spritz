"""
spritz-cov-matrix: build the full per-bin MC-stat covariance matrix between a
set of templates that were built from the same underlying MC events (e.g. EFT
reweight points), using auxiliary "covariance" entries in an analysis
config's `samples` dict.

This tool is generic: it never looks at anything in `config.py` beyond the
standard `samples` dict, and only acts on entries flagged `is_variance: True`
with a `covariance_of: (name_i, name_j)` pair -- everything else in `samples`
(other backgrounds, unrelated processes, etc.) is ignored. A `samples` entry
that opts into this tool needs:

    samples["cov_A_B"] = {
        "samples": [...],           # same as any other sample: raw dataset(s)
                                     # whose per-bin content is Sum(events.weight**2
                                     # * weight_A_expr * weight_B_expr), i.e. the
                                     # MC-stat covariance between templates "A"
                                     # and "B" (same events, different weights).
        "is_variance": True,        # post_process.py: normalize by scale**2, not scale.
        "exclude_from_datacard": True,  # make_cards.py: never a process row.
        "covariance_of": ("A", "B"),    # this tool: which two templates.
    }

The "diagonal" term covariance_of=(A, A) is redundant with template A's own
histogram variance (hist.storage.Weight() already accumulates
Sum(weight_A**2)), but including it lets --check cross-validate the two
against each other.

Usage (from anywhere -- config is an explicit argument, not implied by cwd):
    spritz-cov-matrix path/to/config_dir histos.root (or histos.pkl) \
        -o covariance.root --region inc_mm --variable mll --check
"""
import argparse
import os
import numpy as np
import uproot
from hist import Hist

from spritz.framework.framework import get_analysis_dict
from spritz.utils.histo_io import open_histos


def get_args():
    parser = argparse.ArgumentParser(
        description="Build the per-bin MC-stat covariance matrix between templates sharing MC events"
    )
    parser.add_argument("config", help="Path to the analysis config directory (or its config.py)")
    parser.add_argument(
        "histos",
        help="histos.pkl from spritz-postproc or histos.root from spritz-pkl2root (format from the extension)",
    )
    parser.add_argument("-o", "--output", default="covariance.root")
    parser.add_argument(
        "--region", nargs="+", default=None,
        help="Regions to process (default: config's covariance_regions, else cards_regions, else all regions)",
    )
    parser.add_argument(
        "--variable", nargs="+", default=None,
        help="Variables to process (default: config's covariance_variables, else cards_variables, else all axis variables)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Print max|cov(A,A) - histo_A.variances()| per template as a consistency check",
    )
    return parser.parse_args()


def resolve_config_dir(config_arg):
    path = os.path.abspath(config_arg)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return path


def find_covariance_pairs(samples):
    """Return {covariance_sample_name: (name_i, name_j)} for every `samples`
    entry flagged is_variance with a covariance_of pair. Anything else in
    `samples` (real processes with no covariance info, data, unrelated
    backgrounds, ...) is silently ignored."""
    pairs = {}
    for name, info in samples.items():
        if not info.get("is_variance", False):
            continue
        cov_of = info.get("covariance_of")
        if not cov_of or len(cov_of) != 2:
            raise Exception(
                f"samples['{name}'] has is_variance=True but no valid "
                "'covariance_of': (name_i, name_j) pair -- spritz-cov-matrix "
                "can't tell which two templates this covariance term is for."
            )
        pairs[name] = tuple(cov_of)
    return pairs


def ordered_templates(samples, pairs):
    """Templates referenced by at least one covariance pair, in `samples`'s
    own declaration order (not sorted) -- so unrelated `samples` entries with
    no covariance info are excluded, and axis ordering stays predictable."""
    referenced = {n for pair in pairs.values() for n in pair}
    return [
        name for name, info in samples.items()
        if name in referenced and not (info.get("is_variance") or info.get("is_data"))
    ]


def default_regions_variables(analysis_cfg, args):
    regions = (
        args.region or analysis_cfg.get("covariance_regions") or analysis_cfg.get("cards_regions")
        or list(analysis_cfg["regions"].keys())
    )
    variables = args.variable or analysis_cfg.get("covariance_variables") or analysis_cfg.get("cards_variables") or [
        v for v in analysis_cfg["variables"] if "axis" in analysis_cfg["variables"][v]
    ]
    return regions, variables


def build_matrix_for(fin, region, variable, templates, pairs, check, labels=None):
    # labels: template -> datacard process name. combine matches the matrix
    # axis labels to the process names, so they must follow samples' "name"
    labels = labels or {name: name for name in templates}
    nominal = {name: fin[f"{region}/{variable}/histo_{name}"] for name in templates}

    cov = {}
    for cov_name, (name_i, name_j) in pairs.items():
        cov[(name_i, name_j)] = fin[f"{region}/{variable}/histo_{cov_name}"]

    if check:
        print(f"[{region}/{variable}] consistency check: cov(A,A) vs histo_A.variances()")
        for name in templates:
            if (name, name) not in cov:
                continue
            c = cov[(name, name)].values()
            v = nominal[name].variances()
            maxdiff = np.max(np.abs(c - v)/c) if len(c) else 0.0
            print(f"  {name}: max rel(|diff|) = {maxdiff:.6g}")

    x_axis = nominal[templates[0]].axes[0]
    x_edges = x_axis.edges

    h3 = (
        Hist.new.Reg(len(x_edges) - 1, x_edges[0], x_edges[-1], name=x_axis.name)
        .StrCategory([labels[name] for name in templates], name="template_i")
        .StrCategory([labels[name] for name in templates], name="template_j")
        .Double()
    )

    # (x bin, template i, template j), symmetric in i <-> j; combine reads
    # the matrix by bin index, so x bin k is the template's bin k
    index = {name: i for i, name in enumerate(templates)}
    matrix = np.zeros((len(x_edges) - 1, len(templates), len(templates)))
    for (name_i, name_j), h in cov.items():
        i, j = index[name_i], index[name_j]
        matrix[:, i, j] = h.values()
        matrix[:, j, i] = h.values()
    h3.view()[...] = matrix

    return h3


def main():
    args = get_args()
    config_dir = resolve_config_dir(args.config)
    analysis_cfg = get_analysis_dict(config_dir)
    samples = analysis_cfg["samples"]

    pairs = find_covariance_pairs(samples)
    if not pairs:
        raise Exception(
            f"No samples in {config_dir}/config.py are flagged is_variance "
            "with a covariance_of pair -- nothing to build a matrix from."
        )
    templates = ordered_templates(samples, pairs)
    labels = {name: samples[name].get("name", name) for name in templates}
    print(f"Found {len(pairs)} covariance terms across {len(templates)} templates: {templates}")

    regions, variables = default_regions_variables(analysis_cfg, args)

    fin = open_histos(args.histos)
    with uproot.recreate(args.output) as fout:
        for region in regions:
            for variable in variables:
                key = f"{region}/{variable}/covariance_matrix"
                try:
                    fout[key] = build_matrix_for(fin, region, variable, templates, pairs, args.check, labels)
                except KeyError as e:
                    print(f"Skipping {region}/{variable}: {e}")
                    continue
                print(f"Wrote {key}")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
