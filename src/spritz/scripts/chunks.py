import json
import os
import random
from math import ceil

from analysis.config import datasets
from framework import path_fw, write_chunks


def split_chunks(num_entries):
    # max_events = 200_000_000
    chunksize = 100_000
    # max_events = min(num_entries, max_events)
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
    with open(path_fw + "/data/files_all2.json", "r") as file:
        files = json.load(file)

    for dataset in datasets:
        datasets[dataset]["files"] = files[datasets[dataset]["files"]]["files"]
        # print(datasets[dataset]["files"])
    return datasets


def create_chunks(datasets, max_chunks=None):
    chunks = []
    for dataset in datasets:
        is_data = datasets[dataset].get("is_data", False)
        files = datasets[dataset]["files"]
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


if __name__ == "__main__":
    datasets = get_files(datasets)
    chunks = create_chunks(datasets, max_chunks=None)
    print("Now got", len(chunks), "chunks")
    write_chunks(chunks, "data/chunks.pkl")