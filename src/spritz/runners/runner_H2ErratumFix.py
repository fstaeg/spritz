import gc
import json
import sys
import traceback as tb
import awkward as ak
import correctionlib
import hist
import vector
from copy import deepcopy
from spritz.framework.framework import (
    big_process,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
)
import spritz.framework.variation as variation_module
from spritz.modules.basic_selections import pass_weightfilter

vector.register_awkward()

print("awkward version", ak.__version__)

path_fw = get_fw_path()
with open("cfg.json") as file:
    txt = file.read()
    txt = txt.replace("RPLME_PATH_FW", path_fw)
    cfg = json.loads(txt)

ceval_assign_run = correctionlib.CorrectionSet.from_file(cfg["run_to_era"])

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)

special_analysis_cfg = analysis_cfg["special_analysis_cfg"]
do_variations = special_analysis_cfg.get("do_variations", True)


def process(events, **kwargs):
    dataset = kwargs["dataset"]
    isData = kwargs.get("is_data", False)
    subsamples = kwargs.get("subsamples", {})
    max_weight = kwargs.get("max_weight", None)

    print(f"\nmax_weight = {max_weight}")

    variations = variation_module.Variation()
    variations.register_variation([], "nom")

    events["weight"] = events.genWeight

    events = pass_weightfilter(events, max_weight)
    events = events[events.pass_weightfilter]

    sumw = ak.sum(events.weight)
    nevents = ak.num(events.weight, axis=0)

    ele_mask = (abs(events.LHEPart.pdgId) == 11)
    mu_mask = (abs(events.LHEPart.pdgId) == 13)
    tau_mask = (abs(events.LHEPart.pdgId) == 15)
    events["LHELepton"] = events.LHEPart[ele_mask | mu_mask | tau_mask]

    if "DYmm" in dataset:
        mll = (events.LHELepton[:, 0] + events.LHELepton[:, 1]).mass
        events = events[(50 < mll) & (mll < 100)]

    lep_sort = ak.argsort(events.LHELepton.pt, ascending=False, axis=-1)
    events["LHELepton"] = events.LHELepton[lep_sort]

    # Apply a skim!
    events = events[ak.num(events.LHELepton) >= 2]

    highest_weight_index = ak.argmax(abs(events.weight))
    highest_weight = events.weight[highest_weight_index]
    highest_ptll = (events.LHELepton[highest_weight_index, 0] + events.LHELepton[highest_weight_index, 1]).pt
    print(f"\nhighest weight = {highest_weight}")
    print(f"ptll = {highest_ptll} GeV\n")

    # Regions definitions
    regions = deepcopy(analysis_cfg["regions"])
    variables = deepcopy(analysis_cfg["variables"])

    if not do_variations:
        variations.variations_dict = {
            k: v for k, v in variations.variations_dict.items() if k == "nom"
        }

    default_axis = [
        hist.axis.StrCategory(
            [region for region in regions],
            name="category",
        ),
        hist.axis.StrCategory(
            sorted(list(variations.get_variations_all())), 
            name="syst"
        ),
    ]

    results = {dataset: {"sumw": sumw, "nevents": nevents, "events": 0, "histos": 0}}
    if subsamples != {}:
        results = {}
        for subsample in subsamples:
            results[f"{dataset}_{subsample}"] = {
                "sumw": sumw,
                "nevents": nevents,
                "events": 0,
                "histos": 0,
            }

    for dataset_name in results:
        _events = {}
        histos = {}
        for variable in variables:
            _events[variable] = ak.Array([])

            if "axis" in variables[variable]:
                if isinstance(variables[variable]["axis"], list):
                    histos[variable] = hist.Hist(
                        *variables[variable]["axis"],
                        *default_axis,
                        hist.storage.Weight(),
                    )
                else:
                    histos[variable] = hist.Hist(
                        variables[variable]["axis"],
                        *default_axis,
                        hist.storage.Weight(),
                    )

        results[dataset_name]["histos"] = histos
        results[dataset_name]["events"] = _events

    originalEvents = ak.copy(events)

    print("Doing variations")
    for variation in sorted(variations.get_variations_all()):
        events = ak.copy(originalEvents)

        print(variation)
        for switch in variations.get_variation_subs(variation):
            if len(switch) == 2:
                variation_dest, variation_source = switch
                events[variation_dest] = events[variation_source]

        ##################################################
        
        # Variable definitions
        for variable in variables:
            if "func" in variables[variable]:
                events[variable] = variables[variable]["func"](events)

        events[dataset] = ak.ones_like(events.run) == 1.0

        if subsamples != {}:
            for subsample in subsamples:
                events[f"{dataset}_{subsample}"] = eval(subsamples[subsample])

        for region in regions:
            regions[region]["mask"] = regions[region]["func"](events)

        # Fill histograms
        for dataset_name in results:
            for region in regions:
                # Apply mask for specific region, category and dataset_name
                mask = regions[region]["mask"] & events[dataset_name]

                if len(events[mask]) == 0:
                    continue

                for variable in results[dataset_name]["histos"]:
                    if isinstance(variables[variable]["axis"], list):
                        var_names = [k.name for k in variables[variable]["axis"]]
                        vals = {
                            var_name: events[var_name][mask] for var_name in var_names
                        }
                        results[dataset_name]["histos"][variable].fill(
                            **vals,
                            category=region,
                            syst=variation,
                            weight=events["weight"][mask],
                        )
                    else:
                        var_name = variables[variable]["axis"].name
                        results[dataset_name]["histos"][variable].fill(
                            events[var_name][mask],
                            category=region,
                            syst=variation,
                            weight=events["weight"][mask],
                        )

    gc.collect()
    return results


if __name__ == "__main__":
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks))

    results = {}
    errors = []
    processed = []

    for i in range(len(new_chunks)):
        new_chunk = new_chunks[i]

        if new_chunk["result"] != {}:
            print(
                "Skip chunk",
                {k: v for k, v in new_chunk["data"].items() if k != "read_form"},
                "was already processed",
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

    datasets = list(filter(lambda k: "root:/" not in k, results.keys()))

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
