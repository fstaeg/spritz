import argparse
import json
import os
import random
from math import ceil
import uproot
import numpy as np

from spritz.framework.framework import get_analysis_dict, get_fw_path, write_chunks

EXCLUDED_FILES_PATH = "excluded_files.json"
DEFAULT_CHUNK_SIZE = 100_000


def global_excluded_files_path(era):
    """Era-wide skip list, kept next to samples.json so every config of
    that era uses it without copying it around."""
    return f"{get_fw_path()}/data/{era}/samples/excluded_files.json"


def load_excluded_paths(paths=(EXCLUDED_FILES_PATH,)):
    """Raw file paths to drop before ever building a chunk for them: the union
    of the given lists, by default the era-wide data/<era>/samples/excluded_files.json
    (see get_files) and the optional per-config excluded_files.json. Typical
    entries are files a survey found with a corrupted/incomplete reweight
    branch, or files that repeat the events of another production of the same
    sample. A missing list is a no-op. Format: [{"path": "root://...", ...}, ...]
    (extra keys, e.g. why a path is excluded, are ignored here)."""
    excluded = set()
    for path in paths:
        if path and os.path.isfile(path):
            with open(path) as file:
                excluded |= {e["path"] for e in json.load(file)}
    return excluded


def split_chunks(num_entries, chunksize=DEFAULT_CHUNK_SIZE):
    nIterations = ceil(num_entries / chunksize)
    file_results = []
    for i in range(nIterations):
        start = min(num_entries, chunksize * i)
        stop = min(num_entries, chunksize * (i + 1))
        if start >= stop:
            break
        file_results.append([start, stop])
    return file_results


def get_files(datasets, era=None):
    with open("data/fileset.json", "r") as file:
        files = json.load(file)

    excluded = load_excluded_paths(
        [global_excluded_files_path(era) if era else None, EXCLUDED_FILES_PATH]
    )
    n_dropped = 0
    for dataset in datasets:
        dataset_files = files[datasets[dataset]["files"]]["files"]
        if excluded:
            kept = [f for f in dataset_files if not (excluded & set(f["path"]))]
            n_dropped += len(dataset_files) - len(kept)
            dataset_files = kept
        datasets[dataset]["files"] = dataset_files
    if excluded:
        print(f"excluded_files.json (era-wide + config): dropped {n_dropped} of {len(excluded)} listed files "
              f"({len(excluded) - n_dropped} listed paths were not found in data/fileset.json)")
    return datasets


def parse_ho_corrections(argument):
    f = uproot.open(argument["file"])
    obj = f[argument["object"]]
    h = obj.to_boost()

    weights = h.values()
    weights_err = np.sqrt(h.variances())
    edges = [ax.edges for ax in h.axes]
    
    if f"{argument["object"]}_muRmuF_down" in f:
        scale_keys = ["muRmuF_down", "muR_down", "muF_down", "muRes_down", "muRes_up", "muF_up", "muR_up", "muRmuF_up"]
        weights_scale = {}
        for key in scale_keys:
            obj = f[f"{argument["object"]}_{key}"]
            h_scale = obj.to_boost()
            weights_scale[key] = h_scale.values()
    else:
        weights_scale = None

    return weights, weights_err, edges, weights_scale
    
    
def create_chunks(datasets, chunk_size=DEFAULT_CHUNK_SIZE):
    chunks = []
    for dataset in datasets:
        is_data = datasets[dataset].get("is_data", False)
        max_chunks = datasets[dataset].get("max_chunks", None)
        files = datasets[dataset]["files"]
        ho_corrections = datasets[dataset].get("ho_corrections", False)
        if ho_corrections:
            for idx, h__ in enumerate(ho_corrections):
                w, w_e, e, s = parse_ho_corrections(h__)
                ho_corrections[idx]["weight"] = w 
                ho_corrections[idx]["weight_err"] = w_e
                ho_corrections[idx]["edges"] = e
                if s is not None:
                    ho_corrections[idx]["scale_weights"] = s
                
        dataset_dict = {
            k: v
            for k, v in datasets[dataset].items()
            if k != "files" and k != "task_weight"
        }
        chunks_dataset = []
        for file in files:
            steps = split_chunks(file["nevents"], chunk_size)
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


def get_args():
    parser = argparse.ArgumentParser(
        description="Split the files of data/fileset.json into chunks, written to data/chunks.pkl"
    )
    parser.add_argument(
        "--chunk-size", "--chunk_size",
        dest="chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum number of events per chunk (default {DEFAULT_CHUNK_SIZE}). A "
        "chunk never spans more than one file.",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    return args


def main():
    args = get_args()
    an_dict = get_analysis_dict()
    datasets = get_files(an_dict["datasets"], an_dict.get("year"))
    chunks = create_chunks(datasets, args.chunk_size)
    print("Now got", len(chunks), "chunks")
    write_chunks(chunks, "data/chunks.pkl")


if __name__ == "__main__":
    main()
