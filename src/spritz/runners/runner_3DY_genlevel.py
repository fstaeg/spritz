import gc
import json
import sys
import traceback as tb
import awkward as ak
import correctionlib
import hist
import numpy as np
import onnxruntime as ort
import spritz.framework.variation as variation_module
import uproot
import vector
from copy import deepcopy
from spritz.framework.framework import (
    big_process,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
)
from spritz.modules.basic_selections import pass_weightfilter
from spritz.modules.theory_unc import theory_unc

vector.register_awkward()

print("uproot version", uproot.__version__)
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
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1


def process(events, **kwargs):
    dataset = kwargs["dataset"]
    isData = kwargs.get("is_data", False)
    subsamples = kwargs.get("subsamples", {})
    special_weight = eval(kwargs.get("weight", "1.0"))

    variations = variation_module.Variation()
    variations.register_variation([], "nom")

    events["weight"] = events.genWeight

    events = pass_weightfilter(events, kwargs.get("max_weight", None))
    events = events[events.pass_weightfilter]

    sumw = ak.sum(events.weight)
    nevents = ak.num(events.weight, axis=0)

    # Add special weight for each dataset (not subsamples)
    if special_weight != 1.0:
        print(f"Using special weight for {dataset}: {special_weight}")

    events["weight"] = events.weight * special_weight

    GenDressedLepton = events.GenDressedLepton[abs(events.GenDressedLepton.pdgId) == 13]
    GenDressedLepton = GenDressedLepton[GenDressedLepton.pt >= 15]
    GenDressedLepton = GenDressedLepton[abs(GenDressedLepton.eta) <= 2.4]
    GenDressedLepton = GenDressedLepton[ak.argsort(GenDressedLepton.pt, ascending=False, axis=-1)]
    events["GenDressedLepton"] = GenDressedLepton

    GenLepton = events.GenPart[(abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1)]
    GenLepton = GenLepton[GenLepton.pt >= 15]
    GenLepton = GenLepton[abs(GenLepton.eta) <= 2.4]
    GenLepton = GenLepton[ak.argsort(GenLepton.pt, ascending=False, axis=-1)]
    events["GenLepton"] = GenLepton

    LHELepton = events.LHEPart[(abs(events.LHEPart.pdgId) == 13) & (events.LHEPart.status == 1)]
    events["EmptyLepton"] = ak.zeros_like(LHELepton)

    if dataset in ["DYmm_M-50to100", "DYmm_M-50"]: # for mll > 100 GeV we have separate DY samples
        assert ak.all(ak.num(LHELepton) == 2)
        lhe_mll = (LHELepton[:, 0] + LHELepton[:, 1]).mass
        events = events[(50 < lhe_mll) & (lhe_mll < 100)]
        LHELepton = LHELepton[(50 < lhe_mll) & (lhe_mll < 100)]

    LHELepton = LHELepton[LHELepton.pt >= 15]
    LHELepton = LHELepton[abs(LHELepton.eta) <= 2.4]
    LHELepton = LHELepton[ak.argsort(LHELepton.pt, ascending=False, axis=-1)]
    events["LHELepton"] = LHELepton

    # Apply a skim!
    events = events[
        (ak.num(events.GenDressedLepton) >= 2) 
        | (ak.num(events.GenLepton) >= 2)
        | (ak.num(events.LHELepton) >= 2)
    ]

    # select events with two LHELeptons passing the cuts
    events["LHELepton"] = ak.where(
        ak.num(events.LHELepton) >= 2,
        events.LHELepton,
        events.EmptyLepton
    )

    events["lhe"] = (
        ((events.LHELepton[:, 0].pdgId * events.LHELepton[:, 1].pdgId) == -13*13)
        & (events.LHELepton[:, 0].pt >= 29)
        & (events.LHELepton[:, 1].pt >= 15)
    )

    # select events with two GenDressedLeptons passing the cuts
    events["GenDressedLepton"] = ak.where(
        ak.num(events.GenDressedLepton) >= 2,
        events.GenDressedLepton,
        events.EmptyLepton
    )

    events["dressed"] = (
        ((events.GenDressedLepton[:, 0].pdgId * events.GenDressedLepton[:, 1].pdgId) == -13*13)
        & (events.GenDressedLepton[:, 0].pt >= 29)
        & (events.GenDressedLepton[:, 1].pt >= 15)
    )

    # select events with two GenLeptons passing the cuts
    events["GenLepton"] = ak.where(
        ak.num(events.GenLepton) >= 2,
        events.GenLepton,
        events.EmptyLepton
    )

    events["gen"] = (
        ((events.GenLepton[:, 0].pdgId * events.GenLepton[:, 1].pdgId) == -13*13)
        & (events.GenLepton[:, 0].pt >= 29)
        & (events.GenLepton[:, 1].pt >= 15)
    )

    # event selection
    events = events[events.dressed | events.lhe | events.gen]

    # Theory unc.
    if special_analysis_cfg.get("do_theory_variations", True):
        events, variations = theory_unc(events, variations)

    # Regions definitions
    regions = deepcopy(analysis_cfg["regions"])
    variables = deepcopy(analysis_cfg["variables"])

    if not special_analysis_cfg.get("do_variations", False):
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

        if len(events) == 0:
            continue
        
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
