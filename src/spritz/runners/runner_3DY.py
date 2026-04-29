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
from spritz.modules.basic_selections import (
    LumiMask,
    lumi_mask,
    pass_flags,
    pass_trigger,
    pass_weightfilter,
)
from spritz.modules.fake_leptons import reweightFakeLep
from spritz.modules.jme import correct_met
from spritz.modules.lepton_sel import createLepton, leptonSel
from spritz.modules.lepton_sf import lepton_sf
from spritz.modules.prompt_gen import prompt_gen_match_leptons
from spritz.modules.puweight import puweight_sf
from spritz.modules.rochester import (
    correctRochester, 
    getRochester,
    varyRochester, 
)
from spritz.modules.run_assign import assign_run_period
from spritz.modules.theory_unc import theory_unc
from spritz.modules.trigger_sf import (
    match_trigger_object,
    trigger_sf,
)
from spritz.modules.tt_reweight import tt_reweight

vector.register_awkward()

print("uproot version", uproot.__version__)
print("awkward version", ak.__version__)

path_fw = get_fw_path()
with open("cfg.json") as file:
    txt = file.read()
    txt = txt.replace("RPLME_PATH_FW", path_fw)
    cfg = json.loads(txt)

ceval_puWeight = correctionlib.CorrectionSet.from_file(cfg["puWeights"])
ceval_lepton_sf = correctionlib.CorrectionSet.from_file(cfg["leptonSF"])
ceval_assign_run = correctionlib.CorrectionSet.from_file(cfg["run_to_era"])
ceval_met = correctionlib.CorrectionSet.from_file(cfg["met"])

rochester = getRochester(cfg)

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)
special_analysis_cfg = analysis_cfg["special_analysis_cfg"]
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1


def process(events, **kwargs):
    dataset = kwargs["dataset"]
    trigger_sel = kwargs.get("trigger_sel", "")
    isData = kwargs.get("is_data", False)
    era = kwargs.get("era", None)
    subsamples = kwargs.get("subsamples", {})
    special_weight = eval(kwargs.get("weight", "1.0"))

    variations = variation_module.Variation()
    variations.register_variation([], "nom")

    if isData:
        events["weight"] = ak.ones_like(events.run)
    else:
        events["weight"] = events.genWeight

    if isData:
        lumimask = LumiMask(cfg["lumiMask"])
        events = lumi_mask(events, lumimask)
    else:
        events = pass_weightfilter(events, kwargs.get("max_weight", None))
        events = events[events.pass_weightfilter]

    sumw = ak.sum(events.weight)
    nevents = ak.num(events.weight, axis=0)

    # Add special weight for each dataset (not subsamples)
    if special_weight != 1.0:
        print(f"Using special weight for {dataset}: {special_weight}")

    events["weight"] = events.weight * special_weight

    # pass trigger and flags
    events = assign_run_period(events, isData, cfg, ceval_assign_run)
    events = pass_trigger(events, cfg["era"])
    events = pass_flags(events, cfg["flags"])

    events = events[events.pass_flags & events.pass_trigger]

    if isData: # each data DataSet has its own trigger_sel
        events = events[eval(trigger_sel)]

    events = createLepton(events)
    events = leptonSel(events, cfg)
    events["Lepton"] = events.Lepton[events.Lepton.isLoose]
    
    # Apply a skim!
    events = events[ak.num(events.Lepton) >= 2]
    events = events[events.Lepton[:, 0].pt >= 24]
    events = events[events.Lepton[:, 1].pt >= 10]

    if dataset in ["DYmm_M-50to100", "DYmm_M-50"]: # for mll > 100 GeV we have separate DY samples
        outgoing_mask = (events.LHEPart.status == 1)
        lepton_mask = (abs(events.LHEPart.pdgId) == 13)
        lhe_leptons = events.LHEPart[outgoing_mask & lepton_mask]
        
        assert ak.all(ak.num(lhe_leptons) == 2)
        lhe_mll = (lhe_leptons[:, 0] + lhe_leptons[:, 1]).mass
        events = events[(50 < lhe_mll) & (lhe_mll < 100)]

    if not isData:
        events = prompt_gen_match_leptons(events)

    # Require at least one good PV
    events = events[events.PV.npvsGood > 0]

    # Top pT reweighting
    if kwargs.get("top_pt_rwgt", False):
        events, variations = tt_reweight(events, variations)
    else:
        events['topPtWeight'] = ak.ones_like(events.weight)

    # Correct Muons with rochester
    if special_analysis_cfg.get("do_rochester_variations", False):
        events, variations = varyRochester(events, variations, isData, rochester)
    events, variations = correctRochester(events, variations, isData, rochester)
    
    # Trigger matching
    events = match_trigger_object(events, cfg)

    # MET corrections
    events = correct_met(events, ceval_met, isData)

    # Fake lepton reweighting
    if special_analysis_cfg.get("reweight_fakes", False):
        events, variations = reweightFakeLep(events, variations)
    else:
        events["fakeLepWeight"] = ak.ones_like(events.weight)

    if not isData:
        # puWeight
        events, variations = puweight_sf(events, variations, ceval_puWeight, cfg)

        # add trigger SF
        events, variations = trigger_sf(events, variations, ceval_lepton_sf, cfg)

        # add LeptonSF
        events, variations = lepton_sf(events, variations, ceval_lepton_sf, cfg)

        # prefire
        if "L1PreFiringWeight" in ak.fields(events):
            events["prefireWeight"] = events.L1PreFiringWeight.Nom
            events["prefireWeight_up"] = events.L1PreFiringWeight.Up
            events["prefireWeight_down"] = events.L1PreFiringWeight.Dn
            events["prefireWeight_before"] = ak.ones_like(events.L1PreFiringWeight.Nom)

            for tag in ["up", "down", "before"]:
                variations.register_variation(
                    columns=["prefireWeight"],
                    variation_name=f"prefireWeight_{tag}",
                    format_rule=lambda _, var_name: var_name,
                )
        else:
            events["prefireWeight"] = ak.ones_like(events.weight)

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

    originalEvents = ak.copy(events)

    print("Doing variations")
    for variation in sorted(variations.get_variations_all()):
        events = ak.copy(originalEvents)

        print(variation)
        for switch in variations.get_variation_subs(variation):
            if len(switch) == 2:
                variation_dest, variation_source = switch
                events[variation_dest] = events[variation_source]

        # resort Leptons
        lepton_sort = ak.argsort(events[("Lepton", "pt")], ascending=False, axis=1)
        events["Lepton"] = events.Lepton[lepton_sort]

        # l2tight
        events = events[(ak.num(events.Lepton, axis=1) >= 2)]
        muWP = cfg["leptonsWP"]["muWP"]

        comb = (
            events.Lepton[:, 0]["isTightMuon_" + muWP] & events.Lepton[:, 1]["isTightMuon_" + muWP]
        )
        if special_analysis_cfg.get("invert_one_isolation", False):
            comb = comb & (
                (events.Lepton[:, 0]["isTightMuon_RelIso"] & ~events.Lepton[:, 1]["isTightMuon_RelIso"])
                | (~events.Lepton[:, 0]["isTightMuon_RelIso"] & events.Lepton[:, 1]["isTightMuon_RelIso"])
            )
        elif special_analysis_cfg.get("invert_one_isolation_loose", False):
            comb = comb & (
                (events.Lepton[:, 0]["isTightMuon_RelIso"] & ~events.Lepton[:, 1]["isTightMuon_RelIso_loose"])
                | (~events.Lepton[:, 0]["isTightMuon_RelIso_loose"] & events.Lepton[:, 1]["isTightMuon_RelIso"])
            )
        elif special_analysis_cfg.get("invert_one_isolation_control", False):
            comb = comb & (
                (events.Lepton[:, 0]["isTightMuon_RelIso"] & ~events.Lepton[:, 1]["isTightMuon_RelIso"] & events.Lepton[:, 1]["isTightMuon_RelIso_loose"])
                | (~events.Lepton[:, 0]["isTightMuon_RelIso"] & events.Lepton[:, 0]["isTightMuon_RelIso_loose"] & events.Lepton[:, 1]["isTightMuon_RelIso"])
            )
        elif special_analysis_cfg.get("invert_both_isolation", False):
            comb = comb & (
                ~events.Lepton[:, 0]["isTightMuon_RelIso"] & ~events.Lepton[:, 1]["isTightMuon_RelIso"]
            )
        else:
            comb = comb & (
                events.Lepton[:, 0]["isTightMuon_RelIso"] & events.Lepton[:, 1]["isTightMuon_RelIso"]
            )
        events["l2Tight"] = ak.copy(comb)
        events = events[events.l2Tight]

        if len(events) == 0:
            continue


        # Define categories
        events["mm"] = (
            events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId
        ) == -13 * 13
        events["mm_ss"] = (
            events.Lepton[:, 0].pdgId * events.Lepton[:, 1].pdgId
        ) == 13 * 13

        if not isData and not special_analysis_cfg.get("skip_genmatching", False):
            events["prompt_gen_match_2l"] = (
                events.Lepton[:, 0].promptgenmatched & events.Lepton[:, 1].promptgenmatched
            )
            events = events[events.prompt_gen_match_2l]

        # Analysis level cuts
        leptoncut = (events.mm | events.mm_ss)

        # third lepton veto
        leptoncut = leptoncut & (
            ak.fill_none(
                ak.mask(
                    ak.all(events.Lepton[:, 2:].pt < 10, axis=1),
                    ak.num(events.Lepton) >= 3,
                ),
                True,
                axis=0,
            )
        )

        # Cut on pt of two leading leptons
        leptoncut = leptoncut & (events.Lepton[:, 0].pt > 29) & (events.Lepton[:, 1].pt > 15)
        events = events[leptoncut]

        ##################################################

        if len(events) == 0:
            continue

        if not isData:
            # Load all SFs
            events["RecoSF"] = events.Lepton[:, 0].RecoSF * events.Lepton[:, 1].RecoSF
            events["TightSF"] = events.Lepton[:, 0].TightSF * events.Lepton[:, 1].TightSF

            events["weight"] = (
                events.weight
                * events.puWeight
                * events.topPtWeight
                * events.RecoSF
                * events.TightSF
                * events.prefireWeight
                * events.TriggerSFweight_2l
            )

        # Fake lepton reweighting is only applied in the same-sign region
        events["fakeLepWeight"] = ak.where(
            events.mm_ss, events.fakeLepWeight, ak.ones_like(events.weight)
        )

        events["weight"] = events.weight * events.fakeLepWeight
        
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
