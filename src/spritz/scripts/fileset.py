import glob
import json
import os
import sys
import multiprocessing as mp

import uproot
from tqdm import tqdm
from dbs.apis.dbsClient import DbsApi
from spritz.framework.framework import get_analysis_dict, get_fw_path
from spritz.utils import rucio_utils

path_fw = get_fw_path()


def list_directories(path):
    from XRootD import client

    host = path.split("//eos/")[0]
    path = path.split(host)[1]

    fs = client.FileSystem(host)

    dirs = []
    status, listing = fs.dirlist(path)

    if not status.ok:
        raise RuntimeError(f"Error: {status}")

    for entry in listing:
        name = entry.name
        if entry.statinfo is not None:
            # We got statinfo, so we can check flags
            if entry.statinfo.flags & client.flags.StatInfoFlags.IS_DIR:
                dirs.append(name)
        else:
            # No statinfo returned, fall back to stat() call
            fullpath = path.rstrip("/") + "/" + name
            stat_status, statinfo = fs.stat(fullpath)
            if stat_status.ok and statinfo.flags & client.flags.StatInfoFlags.IS_DIR:
                dirs.append(name)

    return dirs


def process_file(args):
    found_file, sample_name = args
    try:
        f = uproot.open(found_file)
        nevents = f["Events"].num_entries
        return {"sample_name": sample_name, "path": [found_file], "nevents": nevents}
    except Exception as e:
        return {"sample_name": sample_name, "path": [found_file], "nevents": 0, "error": str(e)}


def get_files(era, active_samples):
    Samples = {}

    with open(f"{path_fw}/data/{era}/samples/samples.json") as file:
        Samples = json.load(file)
        if active_samples == "ALL":
            Samples = {k: v for k, v in Samples["samples"].items()}
        else:
            Samples = {
                k: v for k, v in Samples["samples"].items() if k in active_samples
            }

    files = {}
    for sampleName in Samples:
        if "nanoAOD" in Samples[sampleName]:
            files[sampleName] = {"query": Samples[sampleName]["nanoAOD"], "files": []}
        elif "path" in Samples[sampleName]:
            files[sampleName] = {"files": []}

            if Samples[sampleName]["path"].startswith("root://"):
                import gfal2

                print("searching for directories in ", Samples[sampleName]["path"])
                dirs = list_directories(Samples[sampleName]["path"])
                ctx = gfal2.creat_context()
                found_files = []
                for d__ in dirs:
                    fp = os.path.join(Samples[sampleName]["path"], d__)
                    found_files += [os.path.join(fp, p__) for p__ in ctx.listdir(fp)]
                # sanity check: CRAB output dirs can contain non-.root files (logs, etc.)
                found_files = [f for f in found_files if f.endswith(".root")]
            else:
                found_files = glob.glob(Samples[sampleName]["path"])

            print(sampleName, f"({len(found_files)} files found)")
            # opening every file to read its event count is I/O bound; parallelize it
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap(process_file, [(f, sampleName) for f in found_files]),
                        total=len(found_files),
                    )
                )

            for result in results:
                if "error" in result:
                    print(f"Error processing {result['path'][0]}: {result['error']}")
                else:
                    files[result["sample_name"]]["files"].append(
                        {"path": result["path"], "nevents": result["nevents"]}
                    )

    return files


def main():
    an_dict = get_analysis_dict()
    era = an_dict["year"]
    datasets = [k["files"] for k in an_dict["datasets"].values()]
    files = get_files(era, datasets)
    print(files)
    rucio_client = rucio_utils.get_rucio_client()
    # DE|FR|IT|BE|CH|ES|UK
    good_sites = ["IT", "FR", "BE", "CH", "UK", "ES", "DE", "US"]
    for dname in files:
        if "query" not in files[dname]:
            continue
        dataset = files[dname]["query"]
        print("Checking", dname, "files with query", dataset)
        try:
            (
                outfiles,
                outsites,
                sites_counts,
            ) = rucio_utils.get_dataset_files_replicas(
                dataset,
                allowlist_sites=[],
                blocklist_sites=[],
                regex_sites=r"T[123]_(" + "|".join(good_sites) + ")_\w+",
                mode="full",  # full or first. "full"==all the available replicas
                client=rucio_client,
            )
        except Exception as e:
            print(f"\n[red bold] Exception: {e}[/]")
            sys.exit(1)

        url = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader"
        api = DbsApi(url=url)
        filelist = api.listFiles(dataset=dataset, detail=1)

        for replicas, _ in zip(outfiles, outsites):
            prefix = "/store/data"
            if prefix not in replicas[0]:
                prefix = "/store/mc"
            logical_name = prefix + replicas[0].split(prefix)[-1]

            right_file = list(
                filter(lambda k: k["logical_file_name"] == logical_name, filelist)
            )
            if len(right_file) == 0:
                raise Exception("File present in rucio but not dbs!", logical_name)
            if len(right_file) > 1:
                raise Exception(
                    "More files have the same logical_file_name, not support"
                )
            nevents = right_file[0]["event_count"]
            files[dname]["files"].append({"path": replicas, "nevents": nevents})

    os.makedirs("data", exist_ok=True)
    with open("data/fileset.json", "w") as file:
        json.dump(files, file, indent=2)


if __name__ == "__main__":
    main()
