import gc
import json
import os
import sys
import time
import traceback as tb
import zlib
from copy import deepcopy

import awkward as ak
import cloudpickle
import hist
import numpy as np
import uproot

from matplotlib.colors import LinearSegmentedColormap, to_hex


def get_fw_path():
    path_fw = os.getenv("SPRITZ_PATH")
    if path_fw is None:
        raise Exception("Could not find SPRITZ_PATH variable, remember to source!")
    return path_fw


get_fw_path()


def get_config_path():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    return path


def get_analysis_dict(path=None):
    if not path:
        path = get_config_path()
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    return analysis_cfg.__dict__  # type: ignore # noqa: F821


def get_batch_cfg():
    if os.path.isfile(f"{get_fw_path()}/batch_config.json"):
        with open(f"{get_fw_path()}/batch_config.json", "r") as file:
            batch_cfg = json.load(file)
    else:
        batch_cfg = dict()
    return {
        "X509_USER_PROXY": batch_cfg.get("X509_USER_PROXY", None),
        "SINGULARITY_IMAGE": batch_cfg.get("SINGULARITY_IMAGE", None),
        "BATCH_SYSTEM": batch_cfg.get("BATCH_SYSTEM", "condor"),
        "JOB_FLAVOUR": batch_cfg.get("JOB_FLAVOUR", None),
        "MACHINES": batch_cfg.get("MACHINES", []),
        "REQUEST_MEMORY": batch_cfg.get("REQUEST_MEMORY", 2048),
    }


def correctionlib_wrapper(ceval):
    return ceval.evaluate


def max_vec(vec, val):
    return ak.where(vec > val, vec, val)


def over_under(val, min, max):
    val = ak.where(val >= max, max, val)
    val = ak.where(val <= min, min, val)
    return val


def m_pi_pi(phi):
    return ak.where(
        phi > np.pi,
        phi - 2 * np.pi,
        ak.where(
            phi <= -np.pi,
            phi + 2 * np.pi,
            phi,
        ),
    )


def read_events(filename, start=0, stop=100, read_form={}):
    print("start reading", flush=True)
    _t0 = time.time()
    uproot_options = dict(
        timeout=30,
        handler=uproot.source.xrootd.XRootDSource,
        num_workers=1,
        use_threads=False,
    )
    f = uproot.open(filename, **uproot_options)
    print(f"  [timing] uproot.open() done +{time.time()-_t0:.2f}s", flush=True)
    tree = f["Events"]
    start = min(start, tree.num_entries)
    stop = min(stop, tree.num_entries)
    if start >= stop:
        return ak.Array([])

    branches = [k.name for k in tree.branches]
    print(f"  [timing] got branch list ({len(branches)}) +{time.time()-_t0:.2f}s", flush=True)

    events = {}
    form = deepcopy(read_form)

    all_branches = []
    for coll in form:
        coll_branches = form[coll]["branches"]
        if len(coll_branches) == 0:
            if coll in branches:
                all_branches.append(coll)
        else:
            for branch in coll_branches:
                branch_name = coll + "_" + branch
                if branch_name in branches:
                    all_branches.append(branch_name)

    print(f"  [timing] about to read {len(all_branches)} branches +{time.time()-_t0:.2f}s", flush=True)
    events_bad_form = tree.arrays(
        all_branches,
        entry_start=start,
        entry_stop=stop,
        decompression_executor=uproot.source.futures.TrivialExecutor(),
        interpretation_executor=uproot.source.futures.TrivialExecutor(),
    )
    print(f"  [timing] tree.arrays() done +{time.time()-_t0:.2f}s", flush=True)
    f.close()

    for coll in form:
        d = {}
        coll_branches = form[coll].pop("branches")

        if len(coll_branches) == 0:
            if coll in branches:
                events[coll] = events_bad_form[coll]
            continue

        for branch in coll_branches:
            branch_name = coll + "_" + branch
            if branch_name in branches:
                if branch_name.endswith("phi"):
                    vals = events_bad_form[branch_name]
                    vals = over_under(vals, -np.pi, np.pi)
                    d[branch] = vals
                else:
                    d[branch] = events_bad_form[branch_name]

        if len(d.keys()) == 0:
            print("did not find anything for", coll, filename, file=sys.stderr)
            continue

        events[coll] = ak.zip(d, **form[coll])
        del d

    print(f"created events (per-collection zip loop done +{time.time()-_t0:.2f}s)", flush=True)
    _events = ak.zip(events, depth_limit=1)
    print(f"  [timing] final ak.zip done +{time.time()-_t0:.2f}s", flush=True)
    del events
    gc.collect()
    print(f"  [timing] gc.collect done +{time.time()-_t0:.2f}s", flush=True)
    return _events


def add_dict(d1, d2):
    if isinstance(d1, dict):
        d = {}
        common_keys = d1.keys() & d2.keys()
        for key in common_keys:
            if key in ("eft_names", "eft_batch_size"):
                # Invariant per-dataset metadata (the eft_reweighting name
                # list / batch size), not an additive quantity -- identical
                # across every chunk of the same dataset by construction, so
                # merging just keeps one copy instead of falling through to
                # list-concatenation (or, for eft_batch_size, integer
                # addition) below.
                d[key] = d1[key]
            else:
                d[key] = add_dict(d1[key], d2[key])
        for key in d1.keys()-common_keys:
            d[key] = d1[key]
        for key in d2.keys()-common_keys:
            d[key] = d2[key]
        return d
    elif isinstance(d1, ak.Array):
        return ak.concatenate([d1, d2])
    elif isinstance(d1, np.ndarray) and d1.ndim != 0:
        print("Debug np", d1, d2)
        return np.concatenate([d1, d2])
    elif isinstance(d1, set):
        return d1 | d2
    elif isinstance(d1, list):
        # A list of per-batch hist.Hist objects (the megahisto
        # eft_reweighting layout) -- sum elementwise via recursion (falls
        # through to the hist.Hist "+" case below), not Python's list "+"
        # concatenation, which would double the list length at every
        # merge-tree level instead of summing histogram contents.
        return [add_dict(a, b) for a, b in zip(d1, d2)]
    else:
        try:
            return d1 + d2
        except:
            print()
            print('d1')
            print(d1)
            print()
            print('d2')
            print(d2)
            print()
            #raise


def add_dict_iterable(iterable):
    tmp = None
    for it in iterable:
        if tmp is None:
            tmp = it
        else:
            tmp = add_dict(tmp, it)
    return tmp


# Must match runner_3DY_eft_full_morphing_megahisto.py's EFT_COMBINED_KEY()
# (f"{dataset}__eft_combined") exactly.
EFT_COMBINED_SUFFIX = "__eft_combined"


def expand_eft_combined(results):
    """Back-compat shim for the "megahisto" eft_reweighting layout, where a
    dataset's 406-template + covariance-term histograms are stored as ONE
    f"{dataset}__eft_combined" entry -- a list of small batch hist.Hist
    objects (each with its own "subsample" IntCategory axis) plus an
    `eft_names` list mapping name -> (batch, position) -- instead of one
    f"{dataset}_{name}" entry per name (the older, one-hist-per-name
    layout). The batch layout only exists to keep the *runner*'s per-chunk
    histogram creation/fill/serialization cost from scaling with the number
    of EFT reweight points (tens of thousands for a full morphing fit); by
    the time results reach post_process.py, they've already been summed
    down to one copy per dataset, so there's no more reason not to go back
    to plain one-hist-per-name entries, and doing so means post_process.py
    and build_covariance_matrix.py (which key their ROOT output/input by
    individual "histo_{name}") need no changes at all to support either
    layout.

    Entries with no "__eft_combined" suffix -- any older-style result, or a
    dataset that never used eft_reweighting -- pass through untouched, so
    this is safe to call unconditionally on any results dict.
    """
    expanded = {}
    for key, entry in results.items():
        if not key.endswith(EFT_COMBINED_SUFFIX):
            expanded[key] = entry
            continue
        dataset = key[: -len(EFT_COMBINED_SUFFIX)]
        eft_names = entry["eft_names"]
        batch_size = entry["eft_batch_size"]
        for idx, name in enumerate(eft_names):
            batch_idx = idx // batch_size
            local_idx = idx % batch_size
            histos = {}
            for variable, batch_histos in entry["histos"].items():
                batch_histo = batch_histos[batch_idx]
                axis_names = [ax.name for ax in batch_histo.axes]
                sub_pos = axis_names.index("subsample")
                slicer = [slice(None)] * len(batch_histo.axes)
                slicer[sub_pos] = hist.loc(local_idx)
                histos[variable] = batch_histo[tuple(slicer)]
            expanded[f"{dataset}_{name}"] = {
                "sumw": entry["sumw"],
                "nevents": entry["nevents"],
                "events": entry.get("events", 0),
                "histos": histos,
            }
    return expanded


def big_process(process, filenames, start, stop, read_form, **kwargs):
    t_start = time.time()

    events = 0
    error = ""
    print(filenames)
    for filename in filenames:
        try:
            events = read_events(filename, start=start, stop=stop, read_form=read_form)
            break
        except Exception as e:
            error += "".join(tb.format_exception(None, e, e.__traceback__))
            # time.sleep(1)
            continue

    if isinstance(events, int):
        print(error, file=sys.stderr)
        raise Exception(
            "Error, could not read any of the filenames\n" + error, filenames
        )

    t_reading = time.time() - t_start
    print(f"  [timing] big_process: read_events total = {t_reading:.2f}s", flush=True)
    if len(events) == 0:
        return {}
    results = {"real_results": 0, "performance": {}}
    results["real_results"] = process(events, **kwargs)
    t_total = time.time() - t_start
    results["performance"][f"{filename}_{start}"] = {
        "total": t_total,
        "read": t_reading,
    }
    del events
    gc.collect()
    return results


def read_chunks(filename, readable=False):
    if not readable:
        with open(filename, "rb") as file:
            chunks = cloudpickle.loads(zlib.decompress(file.read()))
        return chunks
    else:
        with open(filename, "r") as file:
            chunks = json.load(file)
        return chunks


def write_chunks(d, filename, readable=False):
    if not readable:
        with open(filename, "wb") as file:
            file.write(zlib.compress(cloudpickle.dumps(d)))
    else:
        with open(filename, "w") as file:
            json.dump(d, file)


# plots
# cmap_petroff = [
#     "#5790fc",
#     "#f89c20",
#     "#e42536",
#     "#964a8b",
#     "#9c9ca1",
#     "#7a21dd",
# ]
# cmap_petroff = [
#     "#1845fb",
#     "#ff5e02",
#     "#c91f16",
#     "#c849a9",
#     "#adad7d",
#     "#86c8dd",
#     "#578dff",
#     "#656364",
# ]

def interpolate_colors(base_colors, n_colors):
    """
    Interpolate a list of hex colors.

    Parameters
    ----------
    base_colors : list[str]
        List of hex colors, e.g. ["#ff0000", "#00ff00"]
    n_colors : int
        Number of output colors requested

    Returns
    -------
    list[str]
        Interpolated hex colors
    """

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        base_colors,
        N=n_colors,
    )

    return [to_hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]

cmap_petroff = [ # https://github.com/mpetroff/accessible-color-cycles
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd"
]
cmap_pastel = [
    "#A1C9F4",
    "#FFB482",
    "#8DE5A1",
    "#FF9F9B",
    "#D0BBFF",
    "#DEBB9B",
    "#FAB0E4",
    "#CFCFCF",
    "#FFFEA3",
    "#B9F2F0",
]
