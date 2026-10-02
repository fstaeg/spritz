import concurrent.futures
import fnmatch
import json
import sys

import hist
import numpy as np
import uproot
from spritz.framework.framework import (
    add_dict_iterable,
    expand_eft_combined,
    get_analysis_dict,
    get_fw_path,
    get_batch_cfg,
    read_chunks,
)

path_fw = get_fw_path()


def renorm(h, xs, sumw, lumi, square=False):
    scale = xs * 1000 * lumi / sumw
    if square:
        # For samples whose bin content is itself a product of two per-event
        # weights (e.g. Sum(weight_i * weight_j), used to propagate MC-stat
        # covariance between two differently-reweighted views of the same
        # events), the correct normalization is scale**2, not scale -- each
        # of the two weights individually carries one factor of the
        # normalization. This must happen before summing across datasets
        # with different xs/sumw, since the two operations don't commute.
        scale = scale ** 2
    # print(scale)
    _h = h.copy()
    a = _h.view(True)
    a.value = a.value * scale
    a.variance = a.variance * scale * scale
    return _h


def hist_move_content(h, ifrom, ito):
    """
    Moves content of a histogram from `ifrom` bin to `ito` bin.
    Content and sumw2 of bin `ito` will be the sum of the original `ibin`
    and `ito`.
    Content and sumw2 of bin `ifrom` will be 0.
    Modifies in place the histogram.

    Parameters
    ----------
    h : hist
        Histogram
    ifrom : int
        the index of the bin where content will be reset
    ito : int
        the index of the bin where content will be the sum
    """
    dimension = len(h.axes)
    # numpy view is a numpy array containing two keys, value
    # and variances for each bin
    numpy_view = h.view(True)
    content = numpy_view.value
    sumw2 = numpy_view.variance

    if dimension == 1:
        content[ito] += content[ifrom]
        content[ifrom] = 0.0

        sumw2[ito] += sumw2[ifrom]
        sumw2[ifrom] = 0.0

    elif dimension == 2:
        content[ito, :] += content[ifrom, :]
        content[ifrom, :] = 0.0
        content[:, ito] += content[:, ifrom]
        content[:, ifrom] = 0.0

        sumw2[ito, :] += sumw2[ifrom, :]
        sumw2[ifrom, :] = 0.0
        sumw2[:, ito] += sumw2[:, ifrom]
        sumw2[:, ifrom] = 0.0

    elif dimension == 3:
        content[ito, :, :] += content[ifrom, :, :]
        content[ifrom, :, :] = 0.0
        content[:, ito, :] += content[:, ifrom, :]
        content[:, ifrom, :] = 0.0
        content[:, :, ito] += content[:, :, ifrom]
        content[:, :, ifrom] = 0.0

        sumw2[ito, :, :] += sumw2[ifrom, :, :]
        sumw2[ifrom, :, :] = 0.0
        sumw2[:, ito, :] += sumw2[:, ifrom, :]
        sumw2[:, ifrom, :] = 0.0
        sumw2[:, :, ito] += sumw2[:, :, ifrom]
        sumw2[:, :, ifrom] = 0.0


def hist_fold(h, fold_method):
    """
    Fold a histogram (hist object)

    Parameters
    ----------
    h : hist
        Histogram to fold, will be modified in place (aka no copy)
    fold_method : int
        choices 0: no fold
        choices 1: fold underflow
        choices 2: fold overflow
        choices 3: fold both underflow and overflow
    """
    if fold_method == 1 or fold_method == 3:
        hist_move_content(h, 0, 1)
    if fold_method == 2 or fold_method == 3:
        hist_move_content(h, -1, -2)


def hist_unroll(h):
    """
    Unrolls n-dimensional histogram

    Parameters
    ----------
    h : hist
        Histogram to unroll

    Returns
    -------
    hist
        Unrolled 1-dimensional histogram
    """
    dimension = len(h.axes)
    
    if dimension == 1:
        return h

    if dimension == 2:
        numpy_view = h.view()  # no under/overflow!
        nx = numpy_view.shape[0]
        ny = numpy_view.shape[1]
        h_unroll = hist.Hist(hist.axis.Regular(nx * ny, 0, nx * ny), hist.storage.Weight())

        numpy_view_unroll = h_unroll.view()
        numpy_view_unroll.value = numpy_view.value.T.flatten()
        numpy_view_unroll.variance = numpy_view.variance.T.flatten()

        return h_unroll

    if dimension == 3:
        numpy_view = h.view()  # no under/overflow!
        nx = numpy_view.shape[0]
        ny = numpy_view.shape[1]
        nz = numpy_view.shape[2]

        h_unroll = hist.Hist(hist.axis.Regular(nx * ny * nz, 0, nx * ny * nz), hist.storage.Weight())

        numpy_view_unroll = h_unroll.view()
        numpy_view_unroll.value = numpy_view.value.T.flatten()
        numpy_view_unroll.variance = numpy_view.variance.T.flatten()

        return h_unroll



def get_variations(h):
    axis = h.axes[-1]
    variation_names = [axis.value(i) for i in range(len(axis.centers))]
    return variation_names


def blind(region, variable, edges):
    if "sr" in region and "dnn" in variable:
        return np.arange(0, len(edges)) > len(edges) / 2


def single_post_process(results, region, variable, samples, xss, nuisances, corrections, lumi, do_renorm=True):
    dout = {}
    for histoName in samples:
        for sample in samples[histoName]["samples"]:
            try:
                results[sample]["histos"][variable]
            except KeyError:
                print(f"Could not find key {sample} in {variable}")
                continue
            h = results[sample]["histos"][variable].copy()
            real_axis = list([slice(None) for _ in range(len(h.axes) - 2)])
            h = h[tuple(real_axis + [hist.loc(region), slice(None)])].copy()
            is_data = samples[histoName].get("is_data", False)
            is_variance = samples[histoName].get("is_variance", False)
            # renorm mcs
            if do_renorm and not is_data:
                h = renorm(h, xss[sample], results[sample]["sumw"], lumi, square=is_variance)
            tmp_histo = h[tuple(real_axis + [hist.loc("nom")])].copy()
            if len(real_axis) > 1:
                tmp_histo = hist_unroll(tmp_histo)
            key = f"{region}/{variable}/histo_{histoName}"
            if key not in dout:
                dout[key] = tmp_histo.copy()
            else:
                dout[key] += tmp_histo.copy()
            nom_histo = tmp_histo.copy()

            for nuis in nuisances:
                if nuisances[nuis]["type"] != "shape":
                    continue
                if histoName not in nuisances[nuis]["samples"]:
                    continue
                nuis_kind = nuisances[nuis]["kind"]
                nuis_name = nuisances[nuis]["name"]
                if nuis_kind in ["suffix", "weight"]:
                    for tag in ["up", "down"]:
                        h_axis = tuple(real_axis + [hist.loc(f"{nuis_name}_{tag}")])
                        try:
                            tmp_histo = h[h_axis].copy()
                        except:
                            tmp_histo = nom_histo.copy()
                        if len(real_axis) > 1:
                            tmp_histo = hist_unroll(tmp_histo)
                        key = f"{region}/{variable}/histo_{histoName}_{nuis_name}{tag.capitalize()}"
                        if key not in dout:
                            dout[key] = tmp_histo.copy()
                        else:
                            dout[key] += tmp_histo.copy()
                if nuis_kind in ["envelope", "square", "stdev"]:
                    variations = []
                    for i,variation in enumerate(nuisances[nuis]["variations"]):
                        skip_sample = False
                        if isinstance(nuisances[nuis]["variations"][i]["tag"], dict):
                            if histoName in nuisances[nuis]["variations"][i]["tag"]:
                                nuis_sample_key = histoName
                            else:
                                nuis_sample_key = sample
                            if nuis_sample_key in nuisances[nuis]["variations"][i]["tag"]:
                                nuis_tag = nuisances[nuis]["variations"][i]["tag"][nuis_sample_key]
                            else:
                                skip_sample = True 
                        else:
                            nuis_tag = nuisances[nuis]["variations"][i]["tag"]

                        if skip_sample:
                            tmp_histo = nom_histo.copy()
                        else:
                            tmp_histo = h[
                                tuple(real_axis + [hist.loc(nuis_tag)])
                            ].copy()

                        if len(real_axis) > 1:
                            tmp_histo = hist_unroll(tmp_histo)
                        key = f"{region}/{variable}/histo_{histoName}_{nuis_name}_{i}"
                        if key not in dout:
                            dout[key] = tmp_histo.copy()
                        else:
                            dout[key] += tmp_histo.copy()
                        variations.append(tmp_histo.values())

                    variations = np.array(variations)
                    arrup = 0
                    arrdo = 0

                    if nuis_kind.endswith("envelope"):
                        arrup = np.max(variations, axis=0)
                        arrdo = np.min(variations, axis=0)
                    elif nuis_kind.endswith("square"):
                        arrnom = np.tile(nom_histo.values(), (variations.shape[0], 1))
                        arrv = np.sqrt(np.sum(np.square(variations - arrnom), axis=0))
                        arrup = nom_histo.values() + arrv
                        arrdo = nom_histo.values() - arrv
                    elif nuis_kind.endswith("stdev"):
                        arrv = np.std(variations, axis=0)
                        arrup = nom_histo.values() + arrv
                        arrdo = nom_histo.values() - arrv

                    hists = {}
                    hists["Up"] = nom_histo.copy()
                    a = hists["Up"].view()
                    a.value = arrup

                    hists["Down"] = nom_histo.copy()
                    a = hists["Down"].view()
                    a.value = arrdo

                    for tag in ["Up", "Down"]:
                        key = f"{region}/{variable}/histo_{histoName}_{nuis_name}{tag.capitalize()}"
                        tmp_histo = hists[tag]
                        if key not in dout:
                            dout[key] = tmp_histo.copy()
                        else:
                            dout[key] += tmp_histo.copy()

            for corr in corrections:
                corr_sample_key = histoName if histoName in corrections[corr]["samples"] else sample
                if corr_sample_key not in corrections[corr]["samples"]:
                    continue
                corr_name = corrections[corr].get("name", corr)
                h_axis = tuple(real_axis + [hist.loc(f"{corr_name}_before")])
                try:
                    tmp_histo = h[h_axis].copy()
                except:
                    tmp_histo = nom_histo.copy()
                if len(real_axis) > 1:
                    tmp_histo = hist_unroll(tmp_histo)
                key = f"{region}/{variable}/histo_{histoName}_{corr_name}Before"
                if key not in dout:
                    dout[key] = tmp_histo.copy()
                else:
                    dout[key] += tmp_histo.copy()


    return dout


def post_process(results, regions, variables, samples, xss, nuisances, corrections, lumi, do_renorm=True):
    print("Start converting histograms")

    cpus = 10

    region_variable_pairs = [
        (region, variable)
        for region in regions
        for variable in variables
        if "axis" in variables[variable]
    ]

    if len(region_variable_pairs) <= 1:
        # ProcessPoolExecutor.submit() pickles `results` fresh for every
        # task, regardless of how many workers exist -- for an
        # eft_reweighting-heavy config, `results` (post expand_eft_combined)
        # can be hundreds of thousands of hist.Hist objects, tens of GB
        # pickled. Confirmed (single_post_process called directly on real
        # data vs. through the executor, same inputs) that this multiprocessing
        # IPC path silently drops most entries at that scale -- no exception,
        # no error, dout just ends up far smaller than expected. With only
        # one (region, variable) task there's no parallelism to gain anyway,
        # so skip the executor and call directly in-process, which is both
        # correct (validated) and avoids the pickling cost entirely.
        print("only one region/variable task, running in-process (no executor)")
        dout_list = [
            single_post_process(results, region, variable, samples, xss, nuisances, corrections, lumi, do_renorm)
            for region, variable in region_variable_pairs
        ]
        dout = add_dict_iterable(dout_list) if dout_list else {}
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            tasks = []
            print("start post-proc in parallel")
            for region, variable in region_variable_pairs:
                tasks.append(
                    executor.submit(
                        single_post_process,
                        results,
                        region,
                        variable,
                        samples,
                        xss,
                        nuisances,
                        corrections,
                        lumi,
                        do_renorm,
                ))
            concurrent.futures.wait(tasks)
            print("done post-proc in parallel")
            task_results = []
            for task in tasks:
                task_results.append(task.result())
            dout = add_dict_iterable(task_results)

    print("start saving in root file")
    with uproot.recreate("histos.root") as fout:
        fout.update(dout)


def main():
    analysis_dict = get_analysis_dict()
    year = analysis_dict["year"]
    lumi = analysis_dict["lumi"]
    datasets = analysis_dict["datasets"]
    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]
    corrections = analysis_dict.get("corrections", dict())
    do_renorm = not '--no-renorm' in sys.argv

    if not do_renorm:
        print("\ncross sections are not normalized\n")

    with open(f"{path_fw}/data/{year}/samples/samples.json") as file:
        samples_xs = json.load(file)

    results = read_chunks(f"{get_batch_cfg()["BATCH_SYSTEM"]}/results_merged_new.pkl")
    # Transparently unpacks any "megahisto" combined eft_reweighting entries
    # back into one-hist-per-name entries; a no-op for any dataset that
    # doesn't use that layout, so this is safe unconditionally.
    results = expand_eft_combined(results)

    # Every flat_dataset derived from a given raw dataset (a `subsamples`
    # split, an `eft_reweighting` point/covariance term, or the bare dataset
    # itself) shares that dataset's cross section -- they're all just
    # different reweightings/selections of the exact same underlying MC
    # events. `eft_reweighting`'s point/covariance names aren't listed
    # anywhere in config.py itself (only in `results`, via `expand_eft_
    # combined`'s f"{dataset}_{name}" keys -- see its docstring), so those
    # names are recovered from `results` directly instead of re-deriving
    # them (which would need the runner's covariance_name() convention
    # duplicated here too).
    xss = {}
    for dataset in datasets:
        if datasets[dataset].get("is_data", False):
            continue
        key = datasets[dataset]["files"]
        print(key)
        dataset_xs = eval(samples_xs["samples"][key]["xsec"])

        if "subsamples" in datasets[dataset]:
            for sub in datasets[dataset]["subsamples"]:
                flat_dataset = f"{dataset}_{sub}"
                xss[flat_dataset] = dataset_xs
                print(flat_dataset, xss[flat_dataset])
        elif "eft_reweighting" in datasets[dataset]:
            prefix = f"{dataset}_"
            for flat_dataset in results:
                if flat_dataset.startswith(prefix):
                    xss[flat_dataset] = dataset_xs
        else:
            xss[dataset] = dataset_xs

    print(results.keys())
    post_process(results, regions, variables, samples, xss, nuisances, corrections, lumi, do_renorm)


if __name__ == "__main__":
    main()
