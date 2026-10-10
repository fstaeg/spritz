import gc
import sys
import traceback as tb

import awkward as ak
import hist
import numpy as np
import spritz.framework.variation as variation_module
import uproot
import vector
from copy import deepcopy
from spritz.framework.framework import (
    big_process,
    get_analysis_dict,
    read_chunks,
    write_chunks,
)
from spritz.modules.basic_selections import pass_weightfilter

vector.register_awkward()

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)


def process(events, **kwargs):
    dataset = kwargs["dataset"]
    subsamples = kwargs.get("subsamples", {})

    events["weight"] = events.genWeight
    events = pass_weightfilter(events, kwargs.get("max_weight", None))
    events = events[events.pass_weightfilter]

    # genWeight is float32 in NanoAOD: sum in float64 (see runner_3DY_eft.py)
    sumw = ak.sum(ak.values_astype(events.weight, "float64"))
    nevents = ak.num(events.weight, axis=0)

    # Select gen dressed muons, require at least 2
    muon_mask = abs(events.GenDressedLepton.pdgId) == 13
    gen_muons = events.GenDressedLepton[muon_mask]
    gen_muons = gen_muons[ak.argsort(gen_muons.pt, ascending=False, axis=-1)]
    has_2mu = ak.num(gen_muons) >= 2
    events = events[has_2mu]
    gen_muons = gen_muons[has_2mu]

    # Reconstruct gen Z boson observables
    gen_Z = gen_muons[:, 0] + gen_muons[:, 1]
    events["gen_qT"]  = gen_Z.pt
    events["gen_yll"] = gen_Z.rapidity
    events["gen_mll"] = gen_Z.mass

    # Avoid double-counting: M-50to100 is the inclusive M>50 sample;
    # exclude events already covered by the dedicated higher-mass slices
    if dataset == "DYmm_M-50to100":
        events = events[events.gen_mll < 100]

    regions = deepcopy(analysis_cfg["regions"])
    variables = deepcopy(analysis_cfg["variables"])

    default_axis = [
        hist.axis.StrCategory([region for region in regions], name="category"),
        hist.axis.StrCategory(["nom"], name="syst"),
    ]

    results = {dataset: {"sumw": sumw, "nevents": nevents, "events": {}, "histos": {}}}
    if subsamples:
        results = {
            f"{dataset}_{sub}": {"sumw": sumw, "nevents": nevents, "events": {}, "histos": {}}
            for sub in subsamples
        }

    for dataset_name in results:
        histos = {}
        for variable in variables:
            ax = variables[variable]["axis"]
            if isinstance(ax, list):
                histos[variable] = hist.Hist(*ax, *default_axis, hist.storage.Weight())
            else:
                histos[variable] = hist.Hist(ax, *default_axis, hist.storage.Weight())
        results[dataset_name]["histos"] = histos

    events[dataset] = ak.ones_like(events.run) == 1
    if subsamples:
        for sub in subsamples:
            events[f"{dataset}_{sub}"] = eval(subsamples[sub])

    for region in regions:
        regions[region]["mask"] = regions[region]["func"](events)

    for dataset_name in results:
        for region in regions:
            mask = regions[region]["mask"] & events[dataset_name]
            if ak.sum(mask) == 0:
                continue
            for variable in results[dataset_name]["histos"]:
                ax = variables[variable]["axis"]
                if isinstance(ax, list):
                    var_names = [k.name for k in ax]
                    vals = {vn: events[vn][mask] for vn in var_names}
                    results[dataset_name]["histos"][variable].fill(
                        **vals,
                        category=region,
                        syst="nom",
                        weight=events["weight"][mask],
                    )
                else:
                    var_name = ax.name
                    results[dataset_name]["histos"][variable].fill(
                        events[var_name][mask],
                        category=region,
                        syst="nom",
                        weight=events["weight"][mask],
                    )

    gc.collect()
    return results


if __name__ == "__main__":
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks))

    for i in range(len(new_chunks)):
        new_chunk = new_chunks[i]

        if new_chunk["result"] != {}:
            print(
                "Skip chunk",
                {k: v for k, v in new_chunk["data"].items() if k != "read_form"},
                "already processed",
            )
            continue

        print(new_chunk["data"]["dataset"])

        try:
            new_chunks[i]["result"] = big_process(process=process, **new_chunk["data"])
            new_chunks[i]["error"] = ""
        except Exception as e:
            print("\n\nError for chunk", new_chunk, file=sys.stderr)
            nice_exception = "".join(tb.format_exception(None, e, e.__traceback__))
            print(nice_exception, file=sys.stderr)
            new_chunks[i]["result"] = {}
            new_chunks[i]["error"] = nice_exception

        print(f"Done {i+1}/{len(new_chunks)}")

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
