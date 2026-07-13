import gc
import json
import sys
import traceback as tb
import awkward as ak
import correctionlib
import hist
import spritz.framework.variation as variation_module
import vector
from copy import deepcopy
from spritz.framework.framework import (
    big_process,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
)
from spritz.modules.basic_selections import (
    LumiMask,
    lumi_mask,
    pass_flags,
    pass_trigger,
    pass_weightfilter,
)
from spritz.modules.jet_sel import cleanJet, jetSel
from spritz.modules.jme import (
    correct_jets_mc,
    jet_veto,
    remove_jets_HEM_issue,
)
from spritz.modules.lepton_sel import createLepton, leptonSel
from spritz.modules.prompt_gen import prompt_gen_match_leptons
from spritz.modules.rochester import (
    correctRochester, 
    getRochester,
)
from spritz.modules.run_assign import assign_run_period

vector.register_awkward()

print("awkward version", ak.__version__)

path_fw = get_fw_path()
with open("cfg.json") as file:
    txt = file.read()
    txt = txt.replace("RPLME_PATH_FW", path_fw)
    cfg = json.loads(txt)

ceval_assign_run = correctionlib.CorrectionSet.from_file(cfg["run_to_era"])

rochester = getRochester(cfg)

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)
regions = deepcopy(analysis_cfg["regions"])
variables = deepcopy(analysis_cfg["variables"])

def process(events, **kwargs):
    dataset = kwargs["dataset"]
    era = kwargs.get("era", None)
    subsamples = kwargs.get("subsamples", {})
    max_weight = kwargs.get("max_weight", None)
    genmatching_nlep = kwargs.get("genmatching_nlep", 2)

    variations = variation_module.Variation()
    variations.register_variation([], "nom")

    events["weight"] = events.genWeight
    events = pass_weightfilter(events, max_weight)
    events = events[events.pass_weightfilter]

    events["weight"] = ak.ones_like(events.run)

    sumw = ak.sum(events.weight)
    nevents = ak.num(events.weight, axis=0)

    # LHE level selections
    if dataset in ["DYmm_M-50to100", "DYmm_M-50"]: # for mll > 100 GeV we have separate DY samples
        outgoing_mask = (events.LHEPart.status == 1)
        lepton_mask = (abs(events.LHEPart.pdgId) == 13)
        lhe_leptons = events.LHEPart[outgoing_mask & lepton_mask]
        
        assert ak.all(ak.num(lhe_leptons) == 2)
        lhe_mll = (lhe_leptons[:, 0] + lhe_leptons[:, 1]).mass
        events = events[(50 < lhe_mll) & (lhe_mll < 100)]

    # pass trigger and flags
    events = assign_run_period(events, False, cfg, ceval_assign_run)
    events = pass_trigger(events, cfg["era"])
    events = pass_flags(events, cfg["flags"])
    events = events[events.pass_flags & events.pass_trigger]

    # Require at least one good PV
    events = events[events.PV.npvsGood > 0]

    # Lepton preselection
    events = createLepton(events)
    events = leptonSel(events, cfg)
    events["Lepton"] = events.Lepton[events.Lepton.isLoose]
    
    # Apply a skim!
    events = events[ak.num(events.Lepton) >= 2]
    events = events[events.Lepton[:, 0].pt >= 24]
    events = events[events.Lepton[:, 1].pt >= 10]

    # Gen matching
    events = prompt_gen_match_leptons(events)

    # Jet preselection
    events = jetSel(events, cfg)
    events = cleanJet(events)
    events = remove_jets_HEM_issue(events, cfg)
    events = jet_veto(events, cfg)

    # Muon Rochester corrections
    events, variations = correctRochester(events, variations, False, rochester)

    # JEC + JER + JES
    events, variations = correct_jets_mc(events, variations, cfg, run_variations=False)

    ##################################################
    if len(events) == 0: 
        print("0 events, skipping variations")
        return {}

    # Set up results
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
        )
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

    ##################################################
    # Loop over variations
    print("Doing variations")
    originalEvents = ak.copy(events)

    for variation in sorted(variations.get_variations_all()):
        print(variation)
        events = ak.copy(originalEvents)
        
        for switch in variations.get_variation_subs(variation):
            if len(switch) == 2:
                variation_dest, variation_source = switch
                events[variation_dest] = events[variation_source]

        # resort Leptons
        lepton_sort = ak.argsort(events[("Lepton", "pt")], ascending=False, axis=1)
        events["Lepton"] = events.Lepton[lepton_sort]

        # Define categories
        events["mm"] = (
            events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId
        ) == -13 * 13
        events["mm_ss"] = (
            events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId
        ) == 13 * 13
        events = events[events.mm | events.mm_ss]

        # Cut on pt of two leading leptons
        ptcut = (events.Lepton[:, 0].pt > 29) & (events.Lepton[:, 1].pt > 15)
        events = events[ptcut]

        # tight ID requirement
        muWP = cfg["leptonsWP"]["muWP"]
        lTight = events.Lepton[:, 0]["isTightMuon_" + muWP] & events.Lepton[:, 1]["isTightMuon_" + muWP]
        events = events[lTight]
        
        # isolation requirement
        l1Iso = events.Lepton[:, 0]["isTightMuon_RelIso"]
        l2Iso = events.Lepton[:, 1]["isTightMuon_RelIso"]
        lIso = l1Iso & l2Iso
        events = events[lIso]

        # third lepton veto
        events["Lepton"] = events.Lepton[events.Lepton.pt >= 10]
        l3Veto = ak.num(events.Lepton) < 3
        events = events[l3Veto]

        # prompt gen matching
        events["prompt_gen_match_1l"] = (
            events.Lepton[:, 0].promptgenmatched | events.Lepton[:, 1].promptgenmatched
        )
        events["prompt_gen_match_2l"] = (
            events.Lepton[:, 0].promptgenmatched & events.Lepton[:, 1].promptgenmatched
        )
        if genmatching_nlep == 1:
            events = events[events.prompt_gen_match_1l]
        elif genmatching_nlep > 1:
            events = events[events.prompt_gen_match_2l]

        if len(events) == 0:
            continue

        events["Ljet"] = events.Jet[events.Jet.hadronFlavour == 0]
        events["Cjet"] = events.Jet[events.Jet.hadronFlavour == 4]
        events["Bjet"] = events.Jet[events.Jet.hadronFlavour == 5]
        
        events["LjetBtagLoose"] = events.Ljet[events.Ljet.btagDeepFlavB > cfg["bTag"]["btagLoose"]]
        events["LjetBtagMedium"] = events.Ljet[events.Ljet.btagDeepFlavB > cfg["bTag"]["btagMedium"]]
        events["LjetBtagTight"] = events.Ljet[events.Ljet.btagDeepFlavB > cfg["bTag"]["btagTight"]]
        events["CjetBtagLoose"] = events.Cjet[events.Cjet.btagDeepFlavB > cfg["bTag"]["btagLoose"]]
        events["CjetBtagMedium"] = events.Cjet[events.Cjet.btagDeepFlavB > cfg["bTag"]["btagMedium"]]
        events["CjetBtagTight"] = events.Cjet[events.Cjet.btagDeepFlavB > cfg["bTag"]["btagTight"]]
        events["BjetBtagLoose"] = events.Bjet[events.Bjet.btagDeepFlavB > cfg["bTag"]["btagLoose"]]
        events["BjetBtagMedium"] = events.Bjet[events.Bjet.btagDeepFlavB > cfg["bTag"]["btagMedium"]]
        events["BjetBtagTight"] = events.Bjet[events.Bjet.btagDeepFlavB > cfg["bTag"]["btagTight"]]

        events = events[ak.num(events.Jet) > 0]

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
                        try:
                            results[dataset_name]["histos"][variable].fill(
                                **vals,
                                category=region,
                                syst=variation,
                                weight=events["weight"][mask],
                            )
                        except:
                            weights = ak.flatten(
                                ak.broadcast_arrays(next(iter(vals.values())), events["weight"][mask])[1]
                            )
                            vals = {k: ak.flatten(v) for k,v in vals.items()}
                            results[dataset_name]["histos"][variable].fill(
                                **vals,
                                category=region,
                                syst=variation,
                                weight=weights,
                            )
                    else:
                        var_name = variables[variable]["axis"].name
                        try:
                            results[dataset_name]["histos"][variable].fill(
                                events[var_name][mask],
                                category=region,
                                syst=variation,
                                weight=events["weight"][mask],
                            )
                        except:
                            results[dataset_name]["histos"][variable].fill(
                                ak.flatten(events[var_name][mask]),
                                category=region,
                                syst=variation,
                                weight=ak.flatten(
                                    ak.broadcast_arrays(events[var_name][mask], events["weight"][mask])[1]
                                ),
                            )

    gc.collect()
    return results


if __name__ == "__main__":
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks))

    results = {}

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

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
