import argparse
import datetime
import glob
import os
import shutil
import types
import numpy as np
from spritz.framework.framework import get_batch_cfg
from spritz.scripts.merge import MERGE_SPLIT_DIR
import importlib.util
from pathlib import Path

def same_code(c1, c2):
    # bytecode, names and constants, but not line numbers: a nested function
    # (e.g. a lambda inside a helper) is a code object in co_consts, and two
    # code objects only compare equal at the same line of their files
    if c1.co_code != c2.co_code or c1.co_names != c2.co_names or len(c1.co_consts) != len(c2.co_consts):
        return False
    for k1, k2 in zip(c1.co_consts, c2.co_consts):
        if isinstance(k1, types.CodeType) and isinstance(k2, types.CodeType):
            if not same_code(k1, k2):
                return False
        elif type(k1) != type(k2) or k1 != k2:
            return False
    return True


def code_names(code):
    # global (and attribute) names used by a code object and the ones nested in it
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= code_names(const)
    return names


def same_value(v1, v2, _seen):
    # what a lambda refers to: functions are compared like lambdas, modules,
    # classes and other callables must be the same object, the rest by value
    if isinstance(v1, types.FunctionType) and isinstance(v2, types.FunctionType):
        return compare_lambdas(v1, v2, _seen)
    if isinstance(v1, types.ModuleType) or callable(v1) or callable(v2):
        return v1 is v2
    try:
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        return bool(v1 == v2)
    except Exception:
        return repr(v1) == repr(v2)


def compare_lambdas(l1, l2, _seen=None):
    # function that tells if two lambda function
    # implement the same logical action
    _seen = set() if _seen is None else _seen
    if (id(l1), id(l2)) in _seen:  # recursive functions
        return True
    _seen.add((id(l1), id(l2)))

    if not same_code(l1.__code__, l2.__code__):
        return False

    # the same code can still do different things: the names it uses point
    # to each config's own objects, e.g. the preselections a region calls
    for name in code_names(l1.__code__):
        in1, in2 = name in l1.__globals__, name in l2.__globals__
        if in1 != in2:
            return False
        # attribute names (events.mll, ...) are in neither
        if in1 and not same_value(l1.__globals__[name], l2.__globals__[name], _seen):
            return False

    cells1, cells2 = l1.__closure__ or (), l2.__closure__ or ()
    if len(cells1) != len(cells2):
        return False
    for c1, c2 in zip(cells1, cells2):
        if not same_value(c1.cell_contents, c2.cell_contents, _seen):
            return False

    return True

def same_regions(r1, r2):
    # check if two configs define the same regions
    # the problem is that lambda functions can be different
    # even if they have the same definition
    if r1.keys() != r2.keys():
        return False

    for name in r1:
        if r1[name].get("mask") != r2[name].get("mask"):
            return False

        f1, f2 = r1[name]["func"], r2[name]["func"]
        if not compare_lambdas(f1, f2):
            return False

    return True

def same_axis(a, b):
    return (
        type(a) == type(b)
        and a.name == b.name
        and np.array_equal(a.edges, b.edges)
    )

def same_axes(a, b):
    # Support either a single axis or a list of axes
    a = a if isinstance(a, (list, tuple)) else [a]
    b = b if isinstance(b, (list, tuple)) else [b]

    return len(a) == len(b) and all(
        same_axis(x, y) for x, y in zip(a, b)
    )

def same_variables(r1, r2):

    # We only care about name, axis and func
    # All other options can be changed and
    # Do not require processing

    if r1.keys() != r2.keys():
        return False

    for name in r1:

        # a variable histogrammed (axis) or computed (func) in only one config
        for key in ["func", "axis"]:
            if (key in r1[name]) != (key in r2[name]):
                return False

        if "func" in r1[name] and "func" in r2[name]:

            f1, f2 = r1[name]["func"], r2[name]["func"]
            if not compare_lambdas(f1, f2):
                return False

        if "axis" in r1[name] and "axis" in r2[name]:

            a1, a2 = r1[name]["axis"], r2[name]["axis"]
            if not same_axes(a1, a2):
                return False

    return True


def check_consistency(cfg_dict):

    configs = list(cfg_dict.values())

    # basic things such as year, lumi etc

    equal_basic = all(
        (cfg.year, cfg.lumi, cfg.lumi_unc) ==
        (configs[0].year, configs[0].lumi, configs[0].lumi_unc)
        for cfg in configs
    ) if configs else True

    # Need to define the same regions

    equal_regions = all(
        same_regions(configs[0].regions, cfg.regions)
        for cfg in configs[1:]
    ) if configs else True

    # Need to define the same variables

    equal_variables = all(
        same_variables(configs[0].variables, cfg.variables)
        for cfg in configs[1:]
    ) if configs else True

    return equal_basic & equal_regions &  equal_variables


def load_config(path):
    # import a config.py as a module, its names are its attributes
    spec = importlib.util.spec_from_file_location("config", Path(path))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# The merged config is a short python file: it loads the input configs and
# builds every merged name with one of these functions, so it can be edited
# like any config (rename, drop or add entries, take a name from another
# config, ...). A literal dump would be huge: the covariance terms alone are
# ~80k samples per SMEFT model.

# names taken from one config, the last that defines them
SINGLE_KEYS = [
    "year", "year_label", "lumi", "lumi_unc", "plot_label",
    "runner", "njobs", "eft_chunks_per_job", "special_analysis_cfg",
    "regions", "variables",
    "cards_regions", "cards_variables", "covariance_regions", "covariance_variables",
    "covariance_file", "split_uncertainties",
]

# names merged over all configs: {name: merge function}
MERGED_KEYS = {
    "datasets": "update_dicts",
    "samples": "update_dicts",
    "colors": "update_dicts",
    "labels": "update_dicts",
    "nuisances": "merge_entries",
    "corrections": "merge_entries",
    "fakes_dict": "merge_fakes",
    "renorm_samples": "merge_renorm",
}


def union(a, b):
    # join two membership containers: lists keep their order, dicts
    # (e.g. lnN samples {sample: value}) are updated
    if isinstance(a, dict) and isinstance(b, dict):
        return a | b
    seen = set(a)
    return list(a) + [x for x in b if x not in seen]


def update_dicts(*dicts):
    # later dicts update (replace) the entries of earlier ones
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def merge_variations(old, new):
    # the same variation with per-sample tags ({sample: tag}, e.g. the QCD
    # scale weight indices, which differ between samples) in several configs
    # keeps the tags of all of them
    if len(old) != len(new):
        return new
    merged = []
    for a, b in zip(old, new):
        if isinstance(a, dict) and isinstance(b, dict) and isinstance(a.get("tag"), dict) and isinstance(b.get("tag"), dict):
            b = b | {"tag": a["tag"] | b["tag"]}
        merged.append(b)
    return merged


def merge_entries(*dicts, union_keys=("samples", "use_nominal_for_datasets")):
    # update_dicts for nuisances / corrections, but an entry defined in several
    # dicts keeps the samples of all of them
    merged = {}
    for d in dicts:
        for name, entry in d.items():
            if name in merged and isinstance(entry, dict):
                new = merged[name] | entry
                for key in union_keys:
                    if key in merged[name] and key in entry:
                        new[key] = union(merged[name][key], entry[key])
                if "variations" in merged[name] and "variations" in entry:
                    new["variations"] = merge_variations(merged[name]["variations"], entry["variations"])
                merged[name] = new
            else:
                merged[name] = entry
    return merged


def short(value, length=80):
    text = repr(value)
    return text if len(text) <= length else text[:length - 3] + "..."


def variation_conflicts(name, old, new, prev, label):
    # what merge_variations cannot combine: only per-sample tag dicts with
    # no sample in common (or the same tag for it) are merged
    if len(old) != len(new):
        return [f"[{name!r}] {len(old)} variations in {prev}, {len(new)} in {label}: {label}'s are used for all samples"]
    found = []
    for i, (a, b) in enumerate(zip(old, new)):
        if repr(a) == repr(b):
            continue
        if not (isinstance(a, dict) and isinstance(b, dict)):
            found.append(f"[{name!r}] variation {i}: {short(a)} ({prev}) replaced by {short(b)} ({label})")
            continue
        for field in sorted(a.keys() & b.keys()):
            va, vb = a[field], b[field]
            if repr(va) == repr(vb):
                continue
            if field == "tag" and isinstance(va, dict) and isinstance(vb, dict):
                clash = sorted(s for s in va.keys() & vb.keys() if va[s] != vb[s])
                if clash:
                    found.append(f"[{name!r}] variation {i}: tag of {short(clash, 60)} differs, {label}'s is used")
            else:
                found.append(f"[{name!r}] variation {i} {field}: {short(va)} ({prev}) replaced by {short(vb)} ({label})")
    if len(found) > 3:
        found = found[:2] + [f"[{name!r}] ... and {len(found) - 2} more variations differ"]
    return found


def entry_conflicts(labels, dicts, union_keys=("samples", "use_nominal_for_datasets")):
    # what merge_entries cannot combine: the fields of an entry (nuisance,
    # correction) defined in several configs with different values, where
    # the later config's value is used for the samples of all of them
    found = []
    merged, owner = {}, {}
    for label, d in zip(labels, dicts):
        for name, entry in d.items():
            if name in merged and isinstance(entry, dict) and isinstance(merged[name], dict):
                old = merged[name]
                for field in sorted(old.keys() & entry.keys()):
                    if field in union_keys or repr(old[field]) == repr(entry[field]):
                        continue
                    if field == "variations":
                        found += variation_conflicts(name, old[field], entry[field], owner[name], label)
                    else:
                        found.append(f"[{name!r}] {field}: {short(old[field])} ({owner[name]}) replaced by {short(entry[field])} ({label})")
                merged[name] = merge_entries({name: old}, {name: entry}, union_keys=union_keys)[name]
            else:
                merged[name] = entry
            owner[name] = label
    return found


def merge_fakes(*fakes_dicts):
    # fakes regions and the MC subtracted / nuisances of all configs
    merged = {}
    for fd in fakes_dicts:
        new = merged | fd
        new["regions"] = list(merged.get("regions", []))
        new["regions"] += [r for r in fd.get("regions", []) if r not in new["regions"]]
        for key in ["subtract_mc", "nuisances"]:
            new[key] = union(merged.get(key, []), fd.get(key, []))
        merged = new
    return merged


def merge_renorm(*renorms):
    # one renormalization per (target, reference): the same pair in several
    # configs renormalizes the samples of all of them. A single pair stays a
    # dict like in a config; several (e.g. one per SMEFT model, each with its
    # own SM reference) become a list, which spritz-postproc applies in turn
    entries = {}
    for renorm in renorms:
        for entry in (renorm if isinstance(renorm, (list, tuple)) else [renorm]):
            pair = (entry.get("target"), entry.get("reference"))
            if pair in entries:
                old = entries[pair]
                entries[pair] = old | entry | {"samples": union(old.get("samples", []), entry.get("samples", []))}
            else:
                entries[pair] = dict(entry)
    values = list(entries.values())
    return values[0] if len(values) == 1 else values


def other_keys(cfg_dict):
    # every other top-level name of the configs (plotting options, helpers,
    # ...), in the order the configs define them: carried over from the last
    # config defining it, the user drops or changes what is not needed
    keys = []
    for cfg in cfg_dict.values():
        for key, value in vars(cfg).items():
            if key.startswith("_") or isinstance(value, types.ModuleType):
                continue
            if key in SINGLE_KEYS or key in MERGED_KEYS or key in keys:
                continue
            keys.append(key)
    return keys


def merge_configs(cfg_dict):
    # {label: config module} -> ({name: merged value}, notes about what a
    # later config replaced), merged the way the dumped config.py does it
    labels = list(cfg_dict)
    merged, notes = {}, []

    differing = []
    for key in other_keys(cfg_dict):
        defined = [label for label in labels if hasattr(cfg_dict[label], key)]
        merged[key] = getattr(cfg_dict[defined[-1]], key)
        values = [getattr(cfg_dict[label], key) for label in defined]
        if not callable(values[0]) and len({repr(v) for v in values}) > 1:
            differing.append(key)
    if differing:
        notes.append(f"other names taken from the last config defining them, the configs differ: {differing}")

    for key in SINGLE_KEYS:
        defined = [label for label in labels if hasattr(cfg_dict[label], key)]
        if not defined:
            continue
        merged[key] = getattr(cfg_dict[defined[-1]], key)
        if key not in ("regions", "variables"):  # checked to be the same
            values = [repr(getattr(cfg_dict[label], key)) for label in defined]
            if len(set(values)) > 1:
                notes.append(f"{key}: taken from {defined[-1]}, the configs differ")

    for key, function in MERGED_KEYS.items():
        defined = [label for label in labels if hasattr(cfg_dict[label], key)]
        if not defined:
            continue
        values = [getattr(cfg_dict[label], key) for label in defined]
        merged[key] = globals()[function](*values)

        if function == "update_dicts":
            # entries a later config replaced
            owner = {}
            for label, value in zip(defined, values):
                for name, entry in value.items():
                    if name in owner and repr(entry) != repr(owner[name][1]):
                        why = ""
                        if key == "samples" and set(entry.get("samples", [])) != set(owner[name][1].get("samples", [])):
                            why = ", different datasets"
                        notes.append(f"{key}[{name!r}]: {label} replaces {owner[name][0]}{why}")
                    owner[name] = (label, entry)
        elif function == "merge_entries":
            # fields defined differently: one config's value wins for all samples
            for conflict in entry_conflicts(defined, values):
                notes.append(f"WARNING: {key}{conflict}")
        elif function == "merge_renorm" and isinstance(merged[key], list):
            pairs = [(e.get("target"), e.get("reference")) for e in merged[key]]
            notes.append(f"renorm_samples: {len(pairs)} renormalizations (target, reference): {pairs}")

    return merged, notes


def dump_config(cfg_dict, cfg_paths, output, notes):
    # copy the input configs to output/sources/ and write output/config.py
    os.makedirs(f"{output}/sources", exist_ok=True)
    labels = list(cfg_dict)
    sources = {}
    for label in labels:
        sources[label] = f"sources/{os.path.basename(label.rstrip('/'))}.py"
        while list(sources.values()).count(sources[label]) > 1:
            sources[label] = sources[label][:-3] + "_.py"
        shutil.copy2(cfg_paths[label], f"{output}/{sources[label]}")

    def ref(label, key):
        return f"_cfg[{label!r}].{key}"

    lines = [
        f"# Merged config, written by spritz-merge-configs on {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "# Input configs, in merge order (later ones update earlier ones):",
        *[f"#   {label}: {os.path.abspath(cfg_paths[label])} -> {sources[label]}" for label in labels],
        "# Every merged name is built explicitly below: edit freely.",
    ]
    if notes:
        lines += ["#", "# Merge notes (what a later config replaced):"]
        lines += [f"#   {note}" for note in notes]
    lines += [
        "import os",
        "from spritz.scripts.merge_configs import (",
        "    load_config, update_dicts, merge_entries, merge_fakes, merge_renorm,",
        ")",
        "",
        "_here = os.path.dirname(os.path.abspath(__file__))",
        "_cfg = {",
        *[f"    {label!r}: load_config(os.path.join(_here, {sources[label]!r}))," for label in labels],
        "}",
        "",
        "# taken from one config (the last that defines it)",
    ]
    for key in SINGLE_KEYS:
        defined = [label for label in labels if hasattr(cfg_dict[label], key)]
        if defined:
            lines.append(f"{key} = {ref(defined[-1], key)}")
    lines += ["", "# merged over all configs"]
    for key, function in MERGED_KEYS.items():
        defined = [label for label in labels if hasattr(cfg_dict[label], key)]
        if not defined:
            continue
        args = [ref(label, key) for label in defined]
        if len(", ".join(args)) + len(key) + len(function) < 80:
            lines.append(f"{key} = {function}({', '.join(args)})")
        else:
            lines += [f"{key} = {function}("] + [f"    {a}," for a in args] + [")"]

    lines += ["", "# every other name of the configs, from the last that defines it:",
              "# change or drop what is not needed"]
    for key in other_keys(cfg_dict):
        defined = [label for label in labels if hasattr(cfg_dict[label], key)]
        lines.append(f"{key} = {ref(defined[-1], key)}")
    lines += ["", "# your changes", ""]

    with open(f"{output}/config.py", "w") as f:
        f.write("\n".join(lines))


def dataset_files(dataset, info):
    # the results_merged_split/ files of a dataset: one per subsample if it
    # has subsamples (the runner never writes <dataset> itself then), else one
    subsamples = info.get("subsamples", {})
    if subsamples:
        return [f"{dataset}_{subsample}" for subsample in subsamples]
    return [dataset]


def resolve_samples(configs, basepath):
    # The merged datasets use the definition of the last config defining each
    # one: take its results from that config or, if it does not have them
    # (e.g. a SMEFT config that also defines the backgrounds but did not
    # process them), from an earlier config defining the dataset the same way.
    # A config defining the same name with other files processed other events:
    # its results are never used.
    definitions = {}
    for name in configs:
        for dataset, info in configs[name].datasets.items():
            definitions.setdefault(dataset, []).append((name, info))

    combined = {}
    for dataset, defs in definitions.items():
        last_name, last_info = defs[-1]
        stems = dataset_files(dataset, last_info)

        others = [name for name, info in defs if info.get("files") != last_info.get("files")]
        if others:
            print(f"WARNING: dataset {dataset} is {last_info.get('files')!r} in {last_name} "
                  f"but other files in {others}: the merged config and its results use {last_name}'s")

        source = None
        for name, info in reversed(defs):
            if info.get("files") != last_info.get("files"):
                continue
            split_dir = f"{name}/{basepath}/{MERGE_SPLIT_DIR}"
            if any(os.path.isfile(f"{split_dir}/{stem}.pkl") for stem in stems):
                source = name
                break
        if source is None:
            print(f"WARNING: no results for dataset {dataset} ({last_info.get('files')!r}) "
                  f"in {[name for name, info in defs if name not in others]}")
            continue
        combined[dataset] = (source, stems)

    # {dataset: (analysis where to take its .pkl files, [their names])}
    return combined


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge configs that processed different datasets with the same regions "
        "and variables (e.g. data + backgrounds, and SMEFT samples) into one config folder: "
        "their per-dataset results (spritz-merge --split-dataset) and an editable config.py"
    )
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="Configs folder paths to be merged, in merge order (later ones update earlier ones)",
    )
    parser.add_argument("-o", "--output", default="config_merged", help="output folder to store merged configs")
    return parser.parse_args()


def main():
    args = get_args()

    basepath = get_batch_cfg()["BATCH_SYSTEM"]

    cfg_dict = {}
    cfg_paths = {}

    for p__ in args.configs:

        module_name = f"{p__}/config.py"
        # folder_name = p__.split("/")[-1]

        if not os.path.isfile(module_name):
            print(f"Missing config file for config input {p__}, skipping ...")
            continue

        cfg_dict[p__] = load_config(module_name)
        cfg_paths[p__] = module_name

    # First check that the dicts can really be merged: are they implementing
    # the same variables regions etc?
    if not check_consistency(cfg_dict):
        raise ValueError("The input configs define different regions or variables, please check")

    # We need two ingredients:
    # 1) Which .pkl dataset files we need to copy from which folder --> Need to resolve duplicates
    # 2) The merged config as a dumpable python object
    #
    # Everything else will be handled under the hood
    # the output is cleaned below: it must not be one of the inputs
    if os.path.realpath(args.output) in [os.path.realpath(p__) for p__ in cfg_dict]:
        raise ValueError(f"The output {args.output} is one of the input configs")

    cs__ = resolve_samples(cfg_dict, basepath)

    split_dir = f"{args.output}/{basepath}/{MERGE_SPLIT_DIR}"
    os.makedirs(split_dir, exist_ok=True)
    # results of an earlier merge into this folder: spritz-postproc would
    # read all of them
    old = glob.glob(f"{split_dir}/*.pkl")
    if old:
        print(f"Removing {len(old)} files of an earlier merge from {split_dir}")
        for path in old:
            os.remove(path)

    # copy files into the new merged folder
    for dataset, (cfg, stems) in cs__.items():
        for stem in stems:
            pkl_path = f"{cfg}/{basepath}/{MERGE_SPLIT_DIR}/{stem}.pkl"
            if not os.path.isfile(pkl_path):
                print(f"Missing {pkl_path}, skipping ...")
                continue
            # a copy, so the merged folder does not depend on the inputs
            shutil.copy2(pkl_path, f"{split_dir}/{stem}.pkl")

    if os.path.isfile(f"{args.output}/{basepath}/results_merged_new.pkl"):
        print(f"WARNING: {args.output}/{basepath}/results_merged_new.pkl exists and spritz-postproc "
              "uses it instead of the merged per-dataset results: remove it")

    print(check_consistency(cfg_dict))

    # 2) the merged config, written as config.py
    merged, notes = merge_configs(cfg_dict)
    dump_config(cfg_dict, cfg_paths, args.output, notes)
    for note in notes:
        print(note if note.startswith("WARNING") else f"note: {note}")

    # the written config.py must give the same merge
    written = load_config(f"{args.output}/config.py")
    for key in merged:
        value, check = merged[key], getattr(written, key)
        if key in ("regions", "variables"):  # lambdas: compare the names only
            assert check.keys() == value.keys(), key
        elif callable(value):  # functions of the copies in sources/
            assert getattr(check, "__name__", None) == getattr(value, "__name__", None), key
        elif " at 0x" in repr(value):  # objects without a value repr
            assert type(check) == type(value), key
        else:
            assert repr(check) == repr(value), key
    print(f"Wrote {args.output}/config.py ({len(merged.get('datasets', {}))} datasets, "
          f"{len(merged.get('samples', {}))} samples, {len(merged.get('nuisances', {}))} nuisances)")



if __name__ == "__main__":
    main()
