import concurrent.futures
import fnmatch
import json
import pickle
import sys
import time

import hist
import numpy as np
from spritz.framework.framework import (
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
        scale = scale ** 2
    a = h.view(True)
    a.value = a.value * scale
    a.variance = a.variance * scale * scale
    return h


def hist_unroll(h):
    """
    Unrolls n-dimensional histogram
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

def single_post_process(results_sample, histoName, sample, regions, variables, samples, nuisances, corrections, xss, lumi, do_renorm_xs=True):
    is_variance = samples[histoName].get("is_variance", False)
    is_data = samples[histoName].get("is_data", False)

    dout = {}
    
    for variable in variables:
        try:
            h0 = results_sample["histos"][variable]
            # h0 = results_sample["histos"].pop(variable)
        except KeyError:
            print(f"Could not find key {sample} in {variable}")
            continue
        
        for region in regions:
            h_prefix = f"{region}/{variable}/histo"

            real_axis = list([slice(None) for _ in range(len(h0.axes) - 2)])
            h = h0[tuple(real_axis + [hist.loc(region), slice(None)])]

            # renorm mcs
            if do_renorm_xs and not is_data:
                h = renorm(h, xss[sample], results_sample["sumw"], lumi, square=is_variance)

            nom_histo = h[tuple(real_axis + [hist.loc("nom")])].copy()
            
            # unroll N-dim histogram
            if len(real_axis) > 1:
                nom_histo = hist_unroll(nom_histo)

            # save nominal
            key = f"{h_prefix}_{histoName}"
            if key not in dout:
                dout[key] = nom_histo
            else:
                dout[key] += nom_histo

            # skip variations for covariance terms
            if is_variance:
                continue

            # varied histograms
            variation_names = get_variations(h)

            for nuis in samples[histoName]["nuisances"]:
                nuis_kind, nuis_name = nuisances[nuis]["kind"], nuisances[nuis]["name"]
                nuis_variations = nuisances[nuis].get("variations")
                
                if nuis_kind in ["suffix", "weight"]:
                    for variation in ["up", "down"]:
                        if isinstance(nuis_variations, dict) and variation in nuis_variations:
                            tag = nuis_variations[variation]["tag"]
                            if isinstance(tag, dict):
                                tag = tag.get(histoName, tag.get(sample))
                        else:
                            tag = f"{nuis_name}_{variation}"
                        
                        if tag in variation_names:
                            tmp_histo = h[tuple(real_axis + [hist.loc(tag)])].copy()
                            if len(real_axis) > 1:
                                tmp_histo = hist_unroll(tmp_histo)
                        else:
                            tmp_histo = nom_histo.copy()

                        key = f"{h_prefix}_{histoName}_{nuis_name}{variation.capitalize()}"
                        if key not in dout:
                            dout[key] = tmp_histo
                        else:
                            dout[key] += tmp_histo

                if nuis_kind in ["envelope", "square", "stdev"]:
                    if nuis_kind == "envelope":
                        max_values, min_values = None, None
                    elif nuis_kind == "square":
                        sum_sq = np.zeros_like(nom_histo.values())
                    elif nuis_kind == "stdev":
                        varied_histos = list()

                    for i,variation in enumerate(nuis_variations):
                        tag = nuis_variations[i]["tag"]
                        if isinstance(tag, dict):
                            tag = tag.get(histoName, tag.get(sample))

                        if tag in variation_names:
                            tmp_histo = h[tuple(real_axis + [hist.loc(tag)])].copy()
                            if len(real_axis) > 1:
                                tmp_histo = hist_unroll(tmp_histo)
                        else:
                            tmp_histo = nom_histo.copy()

                        # Single top samples saved PDF variations as variation - nominals
                        # add back nominal to make it consistent with all other samples
                        if "ST_tW" in sample and tag is not None and "PDFWeight" in tag:
                            a = tmp_histo.view()
                            a.value = tmp_histo.values() + nom_histo.values()

                        key = f"{h_prefix}_{histoName}_{nuis_name}_{i}"
                        if key not in dout:
                            dout[key] = tmp_histo
                        else:
                            dout[key] += tmp_histo

                        if nuis_kind == "envelope":
                            if max_values is None:
                                max_values = tmp_histo.values()
                                min_values = tmp_histo.values()
                            else:
                                max_values = np.max(max_values, tmp_histo.values(), axis=0)
                                min_values = np.min(min_values, tmp_histo.values(), axis=0)
                        elif nuis_kind == "square":
                            sum_sq += (tmp_histo.values()-nom_histo.values())**2
                        elif nuis_kind == "stdev":
                            varied_histos.append(tmp_histo.values())


                    # construct up and down variations
                    arr = {}
                    if nuis_kind == "envelope":
                        arr["Up"] = max_values
                        arr["Down"] = min_values
                    elif nuis_kind == "square":
                        #arrnom = np.tile(nom_histo.values(), (varied_histos.shape[0], 1))
                        #arrv = np.sqrt(np.sum(np.square(varied_histos - arrnom), axis=0))
                        arrv = np.sqrt(sum_sq)
                        arr["Up"] = nom_histo.values() + arrv
                        arr["Down"] = nom_histo.values() - arrv
                    elif nuis_kind == "stdev":
                        arrv = np.std(np.array(varied_histos), axis=0)
                        arr["Up"] = nom_histo.values() + arrv
                        arr["Down"] = nom_histo.values() - arrv

                    for tag in ["Up", "Down"]:
                        tmp_histo = nom_histo.copy()
                        a = tmp_histo.view()
                        a.value = arr[tag]

                        key = f"{h_prefix}_{histoName}_{nuis_name}{tag.capitalize()}"
                        if key not in dout:
                            dout[key] = tmp_histo
                        else:
                            dout[key] += tmp_histo

            for corr in samples[histoName]["corrections"]:
                if not histoName in corrections[corr]["samples"]:
                    continue

                corr_name = corrections[corr].get("name", corr)
                if f"{corr_name}_before" in variation_names:
                    tmp_histo = h[tuple(real_axis + [hist.loc(f"{corr_name}_before")])].copy()
                    if len(real_axis) > 1:
                        tmp_histo = hist_unroll(tmp_histo)
                else:
                    tmp_histo = nom_histo.copy()

                key = f"{h_prefix}_{histoName}_{corr_name}Before"
                if key not in dout:
                    dout[key] = tmp_histo
                else:
                    dout[key] += tmp_histo

            #dout = renormalize_hists(dout, region, variable, samples, renorm_samples)

    return dout


def renormalize_hists(dout, regions, variables, samples, target, reference, samples_rw):
    for variable in variables:
        for region in regions:
            h_prefix = f"{region}/{variable}/histo"

            # renormalize SMEFT histograms to higher-order
            h_reference = dout[f"{h_prefix}_{reference}"].values()
            h_target = dout[f"{h_prefix}_{target}"].values()
            h_rw = {}

            k = np.divide(h_target, h_reference, where=h_reference!=0, out=np.zeros_like(h_reference))

            # renormalize nominal
            for sample in samples_rw:
                nom_histo = dout[f"{h_prefix}_{sample}"]
                a = nom_histo.view()
                if samples[sample].get("is_variance"):
                    a.value = np.where(k!=0, a.value*k**2, a.value)
                else:
                    a.value = np.where(k!=0, a.value*k, h_target)
                    a.variance = np.where(k!=0, a.variance*k**2, a.variance)

                h_rw[sample] = nom_histo.values()
                dout[f"{h_prefix}_{sample}"] = nom_histo.copy()

            # renormalize variations
            variations = [v for v in dout if v.startswith(f"{h_prefix}_{target}_")]
            variations = [v.replace(f"{h_prefix}_{target}_", "") for v in variations]

            for variation in variations:
                vkey = f"{h_prefix}_%s_{variation}"

                v_target = dout[vkey % target].values()
                if vkey % reference in dout:
                    v_reference = dout[vkey % reference].values()
                    k_var = np.divide(v_target, v_reference, where=v_reference!=0, out=np.zeros_like(v_reference))

                for sample in samples_rw:
                    if samples[sample].get("is_variance", False):
                        continue

                    if vkey % sample in dout:
                        tmp_histo = dout[vkey % sample].copy()
                        v_nom = tmp_histo.values()
                        v_rw = np.where(k_var!=0, k_var*v_nom, h_rw[sample]+v_target-h_target)
                        a = tmp_histo.view()
                        a.value = v_rw
                    else:
                        tmp_histo = dout[f"{h_prefix}_{sample}"].copy()
                        a = tmp_histo.view()
                        a.value = a.value+v_target-h_target

                    dout[vkey % sample] = tmp_histo

            #return dout


def get_fakes(dout, variable, samples, nuisances, corrections, fakes_dict):
    nuisances_ = {k:v for k,v in nuisances.items() if k in fakes_dict["nuisances"]}
    samples_ = {k:v for k,v in samples.items() if k in fakes_dict["subtract_mc"]}
    
    for region in fakes_dict["regions"]:
        target, source = region["target"], region["source"]

        h_prefix_target = f"{target}/{variable}/histo"
        h_prefix_source = f"{source}/{variable}/histo"

        if not f"{h_prefix_source}_Data" in dout:
            print(f"{h_prefix_source}_Data not found, skipping fakes for {target}_{variable}")
            continue

        h_data = dout[f"{h_prefix_source}_Data"]
        h_mc = { sample: dout[f"{h_prefix_source}_{sample}"] for sample in samples_ }

        h_fakes = h_data.copy()
        a = h_fakes.view()

        for sample in h_mc:
            a.value = a.value - h_mc[sample].values()
            a.variance = a.variance + h_mc[sample].variances()

        dout[f"{h_prefix_target}_Fakes"] = h_fakes.copy()

        for nuis in nuisances_:
            nuis_name = nuisances_[nuis].get("name")
            nuis_variations = nuisances_[nuis].get("variations")
            nuis_kind = nuisances_[nuis].get("kind")
            varied_histos = []

            if nuis_kind in ["envelope", "square", "stdev"]:
                variations = [f"{nuis_name}_{i}" for i in range(len(nuis_variations))]
                if nuis_kind == "envelope":
                    max_values, min_values = None, None
                elif nuis_kind == "square":
                    sum_sq = np.zeros_like(nom_histo.values())
                elif nuis_kind == "stdev":
                    varied_histos = list()
            else:
                variations = [f"{nuis_name}Up", f"{nuis_name}Down"]

            for vari_tag in variations:
                if f"{h_prefix_source}_Data_{vari_tag}" in dout:
                    v_fakes = dout[f"{h_prefix_source}_Data_{vari_tag}"].copy()
                else:
                    v_fakes = h_data.copy()

                a = v_fakes.view()

                for sample in samples_:
                    if f"{h_prefix_source}_{sample}_{vari_tag}" in dout:
                        v_sample = dout[f"{h_prefix_source}_{sample}_{vari_tag}"].values()
                    else:
                        v_sample = h_mc[sample].values()

                    a.value = a.value - v_sample

                dout[f"{h_prefix_target}_Fakes_{vari_tag}"] = v_fakes
                
                if nuis_kind == "envelope":
                    if max_values is None:
                        max_values = v_fakes.values()
                        min_values = v_fakes.values()
                    else:
                        max_values = np.max(max_values, v_fakes.values(), axis=0)
                        min_values = np.min(min_values, v_fakes.values(), axis=0)
                elif nuis_kind == "square":
                    sum_sq += (v_fakes.values()-h_fakes.values())**2
                elif nuis_kind == "stdev":
                    varied_histos.append(v_fakes.values())

            # construct up and down variations
            if nuis_kind in ["envelope", "square", "stdev"]:
                arr = {}
                if nuis_kind == "envelope":
                    arr["Up"] = max_values
                    arr["Down"] = min_values
                elif nuis_kind == "square":
                    arrv = np.sqrt(sum_sq)
                    arr["Up"] = h_fakes.values() + arrv
                    arr["Down"] = h_fakes.values() - arrv
                elif nuis_kind == "stdev":
                    arrv = np.std(np.array(varied_histos), axis=0)
                    arr["Up"] = h_fakes.values() + arrv
                    arr["Down"] = h_fakes.values() - arrv

                for tag in ["Up", "Down"]:
                    v_fakes = h_fakes.copy()
                    a = v_fakes.view()
                    a.value = arr[tag]

                    dout[f"{h_prefix_target}_Fakes_{nuis_name}{tag}"] = v_fakes

        for corr in corrections:
            corr_tag = f"{corrections[corr].get("name", corr)}Before"

            if f"{h_prefix_source}_Data_{corr_tag}" in dout:
                c_fakes = dout[f"{h_prefix_source}_Data_{corr_tag}"].copy()
            else:
                c_fakes = h_data.copy()

            a = c_fakes.view()

            for sample in samples_:
                if f"{h_prefix_source}_{sample}_{corr_tag}" in dout:
                    c_sample = dout[f"{h_prefix_source}_{sample}_{corr_tag}"].values()
                else:
                    c_sample = h_mc[sample].values()
                
                a.value = a.value - c_sample

            dout[f"{h_prefix_target}_Fakes_{corr_tag}"] = c_fakes

    #return dout


def post_process(results, regions, variables, samples, xss, nuisances, corrections, lumi, fakes_dict, renorm_samples, do_renorm_xs=True):
    print("Start converting histograms")
    cpus = 5
    dout = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = []
        print("start post-proc in parallel")
        
        for histoName in samples:
            samples[histoName]["nuisances"] = [
                nuis for nuis in nuisances if histoName in nuisances[nuis]["samples"] 
                and nuisances[nuis]["type"] == "shape" ]
            samples[histoName]["corrections"] = [
                corr for corr in corrections if histoName in corrections[corr]["samples"] ]
            
            for sample in samples[histoName]["samples"]:
                if not sample in results.keys():
                    print(f"Could not find key {sample} in results")
                    continue

                results_sample = results.pop(sample)

                tasks.append(
                    executor.submit(
                        single_post_process,
                        results_sample,
                        histoName,
                        sample,
                        regions,
                        variables,
                        samples,
                        nuisances,
                        corrections,
                        xss,
                        lumi,
                        do_renorm_xs,
                    )
                )

                del results_sample

        concurrent.futures.wait(tasks)
            
        for task in tasks:
            task_dout = task.result()
            for k,v in task_dout.items():
                if k in dout.keys():
                    dout[k] += v
                else:
                    dout[k] = v
            del task_dout
        print("done post-proc in parallel")

    if renorm_samples is not None:   
        target = renorm_samples.get("target")
        reference = renorm_samples.get("reference")
        samples_rw = [s for s in renorm_samples.get("samples", []) if s in samples]

        if target in samples and reference in samples and len(samples_rw) > 0:  
            renormalize_hists(dout, regions, variables, samples, target, reference, samples_rw)

    t0 = time.time()
    if fakes_dict is not None:
        for variable in variables:
           get_fakes(dout, variable, samples, nuisances, corrections, fakes_dict)
    print(f"get_fakes took {time.time()-t0:.1f} s")

    t0 = time.time()
    print("start saving in pickle file")
    with open("histos.pkl", "wb") as fout:
        pickle.dump(dout, fout)
    print(f">> done in {time.time()-t0:.1f} s")


def main():
    t0 = time.time()
    analysis_dict = get_analysis_dict()
    year = analysis_dict["year"]
    lumi = analysis_dict["lumi"]
    datasets = analysis_dict["datasets"]
    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    regions = list(analysis_dict["regions"].keys())
    variables = list(analysis_dict["variables"].keys())
    corrections = analysis_dict.get("corrections", dict())
    fakes_dict = analysis_dict.get("fakes_dict")
    renorm_samples = analysis_dict.get("renorm_samples")
    do_renorm_xs = not '--no-renorm' in sys.argv

    if not do_renorm_xs:
        print("\ncross sections are not normalized\n")

    with open(f"{path_fw}/data/{year}/samples/samples.json") as file:
        samples_xs = json.load(file)

    t1 = time.time()
    results = read_chunks(f"{get_batch_cfg()["BATCH_SYSTEM"]}/results_merged_new.pkl")
    print(f"read_chunks took {time.time()-t0:.1f} s")
    t1 = time.time()
    results = expand_eft_combined(results)
    print(f"expand_eft_combined took {time.time()-t0:.1f} s")

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

    post_process(results, regions, variables, samples, xss, nuisances, corrections, lumi, fakes_dict, renorm_samples, do_renorm_xs)
    print(f"done in {time.time()-t0} s")

if __name__ == "__main__":
    main()
