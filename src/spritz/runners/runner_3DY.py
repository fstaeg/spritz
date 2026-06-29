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
from spritz.modules.btag_sf import btag_sf
from spritz.modules.fake_leptons import reweightFakeLep
from spritz.modules.jet_sel import cleanJet, jetSel
from spritz.modules.jme import (
    correct_jets_data,
    correct_jets_mc,
    jet_veto,
    remove_jets_HEM_issue,
)
from spritz.modules.lepton_sel import createLepton, leptonSel
from spritz.modules.lepton_sf import lepton_sf
from spritz.modules.prefireweight import prefireweight
from spritz.modules.prompt_gen import prompt_gen_match_leptons
from spritz.modules.puid_sf import puid_sf
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

print("awkward version", ak.__version__)

path_fw = get_fw_path()
with open("cfg.json") as file:
    txt = file.read()
    txt = txt.replace("RPLME_PATH_FW", path_fw)
    cfg = json.loads(txt)

ceval_puid = correctionlib.CorrectionSet.from_file(cfg["puidSF"])
ceval_btag = correctionlib.CorrectionSet.from_file(cfg["btagSF"])
ceval_btageff = correctionlib.CorrectionSet.from_file(cfg["btagEfficiency"])
ceval_puWeight = correctionlib.CorrectionSet.from_file(cfg["puWeights"])
ceval_lepton_sf = correctionlib.CorrectionSet.from_file(cfg["leptonSF"])
ceval_assign_run = correctionlib.CorrectionSet.from_file(cfg["run_to_era"])

rochester = getRochester(cfg)

analysis_path = sys.argv[1]
analysis_cfg = get_analysis_dict(analysis_path)
regions = deepcopy(analysis_cfg["regions"])
variables = deepcopy(analysis_cfg["variables"])

special_analysis_cfg = analysis_cfg["special_analysis_cfg"]
reweight_fakes = special_analysis_cfg.get("reweight_fakes", False)
bveto_wp = special_analysis_cfg.get("bveto_wp", "Medium")
do_variations = special_analysis_cfg.get("do_variations", True)
do_rochester_variations = special_analysis_cfg.get("do_rochester_variations", False)
do_jet_variations = special_analysis_cfg.get("do_jet_variations", False)
do_theory_variations = special_analysis_cfg.get("do_theory_variations", False)
invert_one_isolation = special_analysis_cfg.get("invert_one_isolation", False)
invert_one_isolation_loose = special_analysis_cfg.get("invert_one_isolation_loose", False)
invert_one_isolation_control = special_analysis_cfg.get("invert_one_isolation_control", False)
invert_both_isolation = special_analysis_cfg.get("invert_both_isolation", False)

def process(events, **kwargs):
    dataset = kwargs["dataset"]
    trigger_sel = kwargs.get("trigger_sel", "")
    isData = kwargs.get("is_data", False)
    era = kwargs.get("era", None)
    subsamples = kwargs.get("subsamples", {})
    max_weight = kwargs.get("max_weight", None)
    top_pt_rwgt = kwargs.get("top_pt_rwgt", False)
    genmatching_nlep = kwargs.get("genmatching_nlep", 2)

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
        events = pass_weightfilter(events, max_weight)
        events = events[events.pass_weightfilter]

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
    events = assign_run_period(events, isData, cfg, ceval_assign_run)
    events = pass_trigger(events, cfg["era"])
    events = pass_flags(events, cfg["flags"])
    events = events[events.pass_flags & events.pass_trigger]

    if isData: # each data DataSet has its own trigger_sel
        events = events[eval(trigger_sel)]

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
    if not isData:
        events = prompt_gen_match_leptons(events)

    # Jet preselection
    events = jetSel(events, cfg)
    events = cleanJet(events)
    events = remove_jets_HEM_issue(events, cfg)
    events = jet_veto(events, cfg)

    # Muon Rochester corrections
    if do_rochester_variations:
        events, variations = varyRochester(events, variations, isData, rochester)
    events, variations = correctRochester(events, variations, isData, rochester)
    
    # Trigger matching
    events = match_trigger_object(events, cfg)

    if not isData:
        # puWeight SF
        events, variations = puweight_sf(events, variations, ceval_puWeight, cfg)

        # trigger SF
        events, variations = trigger_sf(events, variations, ceval_lepton_sf, cfg)

        # lepton SF
        events, variations = lepton_sf(events, variations, ceval_lepton_sf, cfg)

        # JEC + JER + JES
        events, variations = correct_jets_mc(events, variations, cfg, run_variations=do_jet_variations)

        # puId SF
        events, variations = puid_sf(events, variations, ceval_puid, cfg)

        # btag SF
        events, variations = btag_sf(events, variations, ceval_btag, ceval_btageff, cfg, dataset, wp=bveto_wp)

        # prefire weight
        events, variations = prefireweight(events, variations)

        # Top pT reweighting
        if top_pt_rwgt:
            events, variations = tt_reweight(events, variations)
        else:
            events["topPtWeight"] = ak.ones_like(events.weight)

        # Theory unc.
        if do_theory_variations:
            events, variations = theory_unc(events, variations)

    else:
        events = correct_jets_data(events, cfg, era)

    # Fake lepton reweighting
    if reweight_fakes:
        events, variations = reweightFakeLep(events, variations)

    ##################################################
    if len(events) == 0: 
        print("0 events, skipping variations")
        return {}

    # Set up results
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
        l1IsoLoose = events.Lepton[:, 0]["isTightMuon_RelIso_loose"]
        l2Iso = events.Lepton[:, 1]["isTightMuon_RelIso"]
        l2IsoLoose = events.Lepton[:, 1]["isTightMuon_RelIso_loose"]

        if invert_one_isolation:
            lIso = (l1Iso & ~l2Iso) | (~l1Iso & l2Iso)
        elif invert_one_isolation_loose:
            lIso = (l1Iso & ~l2IsoLoose) | (~l1IsoLoose & l2Iso)
        elif invert_one_isolation_control:
            lIso = (l1Iso & l2IsoLoose & ~l2Iso) | (l1IsoLoose & ~l1Iso & l2Iso)
        elif invert_both_isolation:
            lIso = ~l1Iso & ~l2Iso
        else:
            lIso = l1Iso & l2Iso
        events = events[lIso]

        # third lepton veto
        events["Lepton"] = events.Lepton[events.Lepton.pt >= 10]
        l3Veto = ak.num(events.Lepton) < 3
        events = events[l3Veto]

        # prompt gen matching
        if not isData:
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

        # btag and bveto regions
        btagged = (events.Jet.btagDeepFlavB >= cfg["bTag"][f"btag{bveto_wp}"])
        events["bveto"] = ak.num(events.Jet[btagged]) == 0
        events["btag"] = ak.num(events.Jet[btagged]) >= 1

        # max btag score
        events["btagDeepFlavB_max"] = ak.fill_none(
            ak.max(events.Jet.btagDeepFlavB, axis=-1), 0.
        )

        ##################################################
        # Fake lepton reweighting (only in the same-sign region)
        if reweight_fakes:
            events["fakeLepWeight"] = ak.where(
                events.mm_ss, events.fakeLepWeight, ak.ones_like(events.weight)
            )
            events["weight"] = events.weight * events.fakeLepWeight

        # Load all SFs
        if not isData:
            events["btagSF"] = ak.prod(events.Jet.btagSF, axis=-1)
            events["PUID_SF"] = ak.prod(events.Jet.PUID_SF, axis=-1)
            events["RecoSF"] = events.Lepton[:, 0].RecoSF * events.Lepton[:, 1].RecoSF
            events["TightSF"] = events.Lepton[:, 0].TightSF * events.Lepton[:, 1].TightSF

            events["weight"] = (
                events.weight
                * events.puWeight
                * events.topPtWeight
                * events.btagSF
                * events.PUID_SF
                * events.RecoSF
                * events.TightSF
                * events.prefireWeight
                * events.TriggerSFweight_2l
            )
        
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
