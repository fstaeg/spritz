#!/usr/bin/env python
"""Survey every raw input file of a config's datasets for the failure modes that
matter to an eft_reweighting analysis, and write one JSON record per file.

Per file (all retried up to 3 times before being called unreadable):
  - opens, has an `Events` tree                       -> else "unreadable"
  - entries == the nevents recorded in data/fileset.json -> else "nevents_mismatch"
  - every event has exactly the card width in nLHEReweightingWeight
    (`eft_reweighting["n_weights"]` of its dataset)   -> else "width_mismatch"
  - LHEReweightingWeight and genWeight fully decompress and are finite
                                                      -> else "nonfinite"
Usage (from the config directory, spritz env sourced):
    python scripts/survey_files.py -o data/file_survey.jsonl [--workers 32] [--limit N]
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import awkward as ak
import numpy as np
import uproot

from spritz.framework.framework import get_analysis_dict


def survey_one(task):
    dataset, index, path, expected_entries, expected_width = task
    rec = {"dataset": dataset, "index": index, "path": path, "expected_entries": expected_entries,
           "expected_width": expected_width}
    error = None
    t0 = time.time()
    for attempt in range(1, 4):
        try:
            with uproot.open(path, timeout=60, handler=uproot.source.xrootd.XRootDSource,
                             num_workers=1, use_threads=False) as f:
                tree = f["Events"]
                rec["entries"] = int(tree.num_entries)
                counts = tree["nLHEReweightingWeight"].array(library="np")
                rec["widths"] = [int(x) for x in np.unique(counts)]
                weights = ak.to_numpy(ak.flatten(tree["LHEReweightingWeight"].array(library="ak")))
                rec["weights_finite"] = bool(np.isfinite(weights).all())
                rec["genweight_finite"] = bool(np.isfinite(tree["genWeight"].array(library="np")).all())
            error = None
            break
        except Exception as e:  # noqa: BLE001 -- any read failure is what we are surveying
            error = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(3 * attempt)
    rec["attempts"] = attempt
    rec["seconds"] = round(time.time() - t0, 2)
    problems = []
    if error is not None:
        rec["error"] = error
        problems.append("unreadable")
    else:
        if rec["entries"] != expected_entries:
            problems.append("nevents_mismatch")
        if rec["widths"] != [expected_width]:
            problems.append("width_mismatch")
        if not (rec["weights_finite"] and rec["genweight_finite"]):
            problems.append("nonfinite")
    rec["problems"] = problems
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="data/file_survey.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N files of each dataset (testing)")
    args = ap.parse_args()

    cfg = get_analysis_dict(".")
    fileset = json.load(open("data/fileset.json"))
    tasks = []
    for name, d in cfg["datasets"].items():
        width = d["eft_reweighting"]["n_weights"]
        files = fileset[name]["files"]
        if args.limit:
            files = files[: args.limit]
        for i, f in enumerate(files):
            path = f["path"][0] if isinstance(f["path"], list) else f["path"]
            tasks.append((name, i, path, f["nevents"], width))
    print(f"surveying {len(tasks)} files with {args.workers} workers", flush=True)

    done, t0 = 0, time.time()
    bad = collections.Counter()
    with open(args.output, "w") as out, ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(survey_one, t) for t in tasks]
        for fut in as_completed(futures):
            rec = fut.result()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done += 1
            for p in rec["problems"]:
                bad[p] += 1
            if done % 500 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(tasks)}  {rate:.1f} files/s  eta {(len(tasks) - done) / rate / 60:.0f} min  problems so far: {dict(bad)}", flush=True)
    print("done", dict(bad), flush=True)


if __name__ == "__main__":
    main()
