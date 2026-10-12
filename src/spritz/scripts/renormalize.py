"""
spritz-renormalize: rescale a set of processes in histos.pkl bin by bin to a
reference prediction, e.g. the SMEFT templates to the higher-order DY one.

For every region/variable, k = numerator / denominator is computed bin by bin
from the nominal histograms, and every histogram (nominal and all of its
variations) of the processes selected by --apply is multiplied by k. Histogram
variances are scaled by k**2, and so are the values of "is_variance" samples
(covariance terms, quadratic in the per-event weight).
"""

import argparse
import fnmatch
import re

import numpy as np

from spritz.framework.framework import get_analysis_dict
from spritz.utils.histo_io import load_histos, save_histos


def get_args():
    parser = argparse.ArgumentParser(
        description="Rescale processes bin by bin by k = numerator / denominator"
    )
    parser.add_argument("--numerator", required=True, help="Process giving the target prediction")
    parser.add_argument("--denominator", required=True, help="Process to be matched to the numerator")
    parser.add_argument(
        "--apply", nargs="+", required=True,
        help="Processes to rescale: names, glob patterns or regular expressions",
    )
    parser.add_argument("-i", "--input", default="histos.pkl", help=".pkl or .root")
    parser.add_argument("-o", "--output", default="histos_renormalized.pkl", help=".pkl or .root")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print the selected processes and the k factors, write nothing",
    )
    return parser.parse_args()


def select_processes(patterns, processes):
    selected = []
    for pattern in patterns:
        if pattern in processes:
            matched = [pattern]
        else:
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = None
            matched = [
                p for p in processes
                if fnmatch.fnmatchcase(p, pattern) or (regex and regex.fullmatch(p))
            ]
        if not matched:
            raise Exception(f"--apply '{pattern}' matches no process in the config")
        selected += [p for p in matched if p not in selected]
    return selected


def owner_process(name, processes):
    """process a histogram belongs to: histo_{process} or
    histo_{process}_{variation}. The longest matching prefix wins, so that
    e.g. DYmm_w1_cHDD is not mistaken for a variation of a shorter name"""
    parts = name.split("_")
    for n in range(len(parts), 0, -1):
        candidate = "_".join(parts[:n])
        if candidate in processes:
            return candidate
    return None


def scale(h, k, square=False):
    h = h.copy()
    view = h.view()
    factor = k**2 if square else k
    if view.dtype.names and "value" in view.dtype.names:
        view.value = view.value * factor
        if "variance" in view.dtype.names:
            view.variance = view.variance * k**2
    else:
        view[...] = view * factor
    return h


def main():
    args = get_args()
    samples = get_analysis_dict()["samples"]

    for process in (args.numerator, args.denominator):
        if process not in samples:
            raise Exception(f"'{process}' is not a process in the config's samples")

    selected = select_processes(args.apply, list(samples))
    if args.numerator in selected:
        print(f"Not rescaling the numerator {args.numerator} itself")
        selected.remove(args.numerator)
    print(f"Rescaling {len(selected)} processes: {selected}")

    histos = load_histos(args.input)

    processes = set(samples)
    groups = {}  # region/variable -> {process: [histogram keys]}
    for key in histos:
        directory, _, name = key.rpartition("/")
        if not name.startswith("histo_"):
            continue
        process = owner_process(name[len("histo_"):], processes)
        if process in selected:
            groups.setdefault(directory, {}).setdefault(process, []).append(key)

    n_scaled = 0
    for directory in sorted(groups):
        num_key = f"{directory}/histo_{args.numerator}"
        den_key = f"{directory}/histo_{args.denominator}"
        if num_key not in histos or den_key not in histos:
            print(f"[{directory}] no {args.numerator} or {args.denominator}, skipped")
            continue

        num = histos[num_key].values()
        den = histos[den_key].values()
        k = np.divide(num, den, out=np.ones_like(num, dtype=float), where=den != 0)
        print(
            f"[{directory}] k: min {k.min():.4f}, max {k.max():.4f}, "
            f"integral ratio {num.sum() / den.sum():.4f}, "
            f"{np.count_nonzero(den == 0)} bins with empty denominator left as they are"
        )

        for process, keys in groups[directory].items():
            square = samples[process].get("is_variance", False)
            for key in keys:
                if not args.dry_run:
                    histos[key] = scale(histos[key], k, square)
                n_scaled += 1

    if args.dry_run:
        print(f"Dry run: would rescale {n_scaled} histograms, nothing written")
        return

    print(f"Rescaled {n_scaled} histograms, saving to {args.output}")
    save_histos(histos, args.output)


if __name__ == "__main__":
    main()
