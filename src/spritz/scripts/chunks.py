import json
import os
import random
from math import ceil
import uproot
import numpy as np

from spritz.framework.framework import get_analysis_dict, write_chunks

EXCLUDED_FILES_PATH = "excluded_files.json"


def load_excluded_paths(path=EXCLUDED_FILES_PATH):
    """Optional, per-config list of known-bad raw file paths to drop before
    ever building a chunk for them (e.g. files a survey found with a
    corrupted/incomplete reweight branch) -- a no-op for any config that
    doesn't have this file. Format: [{"path": "root://...", ...}, ...] (extra
    keys, e.g. why a path is excluded, are ignored here)."""
    if not os.path.isfile(path):
        return set()
    with open(path) as file:
        entries = json.load(file)
    return {e["path"] for e in entries}


def split_chunks(num_entries):
    chunksize = 100_000
    nIterations = ceil(num_entries / chunksize)
    file_results = []
    for i in range(nIterations):
        start = min(num_entries, chunksize * i)
        stop = min(num_entries, chunksize * (i + 1))
        if start >= stop:
            break
        file_results.append([start, stop])
    return file_results


def get_files(datasets):
    with open("data/fileset.json", "r") as file:
        files = json.load(file)

    excluded = load_excluded_paths()
    n_dropped = 0
    for dataset in datasets:
        dataset_files = files[datasets[dataset]["files"]]["files"]
        if excluded:
            kept = [f for f in dataset_files if not (excluded & set(f["path"]))]
            n_dropped += len(dataset_files) - len(kept)
            dataset_files = kept
        datasets[dataset]["files"] = dataset_files
    if excluded:
        print(f"excluded_files.json: dropped {n_dropped} of {len(excluded)} listed files "
              f"({len(excluded) - n_dropped} listed paths were not found in data/fileset.json)")
    return datasets


def parse_ho_corrections(argument):
    f = uproot.open(argument["file"])
    obj = f[argument["object"]]
    h = obj.to_boost()

    weights = h.values()
    weights_err = np.sqrt(h.variances())
    edges = [ax.edges for ax in h.axes]

    return weights, weights_err, edges
    
    
def create_chunks(datasets):
    chunks = []
    for dataset in datasets:
        is_data = datasets[dataset].get("is_data", False)
        max_chunks = datasets[dataset].get("max_chunks", None)
        files = datasets[dataset]["files"]
        ho_corrections = datasets[dataset].get("ho_corrections", False)
        if ho_corrections:
            for idx, h__ in enumerate(ho_corrections):
                w, w_e, e = parse_ho_corrections(h__)
                ho_corrections[idx]["weight"] = w 
                ho_corrections[idx]["weight_err"] = w_e
                ho_corrections[idx]["edges"] = e
                
        dataset_dict = {
            k: v
            for k, v in datasets[dataset].items()
            if k != "files" and k != "task_weight"
        }
        chunks_dataset = []
        for file in files:
            steps = split_chunks(file["nevents"])
            for start, stop in steps:
                replicas = file["path"]
                random.shuffle(replicas)
                d = {
                    "data": {
                        "dataset": dataset,
                        "filenames": replicas,
                        "start": start,
                        "stop": stop,
                        **dataset_dict,
                    },
                    "error": "",
                    "result": {},
                    "priority": 0,  # used for merging
                    "weight": datasets[dataset].get("task_weight", 1),
                }

                chunks_dataset.append(d)
        if not is_data and max_chunks:
            chunks_dataset = chunks_dataset[:max_chunks]
        chunks.extend(chunks_dataset)
    return chunks


def main():
    datasets = get_analysis_dict()["datasets"]
    datasets = get_files(datasets)
    chunks = create_chunks(datasets)
    print("Now got", len(chunks), "chunks")
    write_chunks(chunks, "data/chunks.pkl")


if __name__ == "__main__":
    main()
