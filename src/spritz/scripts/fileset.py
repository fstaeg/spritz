import argparse
import glob
import json
import os
import sys
import multiprocessing as mp

import uproot
from tqdm import tqdm
from dbs.apis.dbsClient import DbsApi
from spritz.framework.framework import get_analysis_dict, get_batch_cfg, get_fw_path
from spritz.scripts.batch import submit
from spritz.utils import rucio_utils

path_fw = get_fw_path()

# Separate directory from the real analysis batch_config["BATCH_SYSTEM"]
# (usually "condor"), so dispatching the file-listing step never collides
# with a real spritz-batch analysis submission in the same config directory.
FILESET_BATCH_SYSTEM = "condor_fileset"


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


def discover_files(era, active_samples):
    """Find which raw files exist for every requested sample -- fast, always
    local (just directory listing / globbing, never opens a file). For
    "path"-based samples, files[sampleName]["files"] is a list of raw file
    path strings (not yet annotated with event counts). For "nanoAOD"-based
    (DAS) samples, files[sampleName] is {"query": ..., "files": []} exactly
    as before -- those get resolved later via rucio/DBS, which is cheap
    metadata-only work that was never worth dispatching to condor."""
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
            files[sampleName] = {"files": found_files}

    return files


def count_events_local(files):
    """Open every raw file found by discover_files() to read its event
    count, in a local multiprocessing pool. This is the slow, I/O-bound step
    that --condor/--merge exist to dispatch instead."""
    for sampleName, entry in files.items():
        if "query" in entry:
            continue  # nanoAOD/DAS sample, resolved separately

        found_files = entry["files"]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(process_file, [(f, sampleName) for f in found_files]),
                    total=len(found_files),
                )
            )

        counted = []
        for result in results:
            if "error" in result:
                print(f"Error processing {result['path'][0]}: {result['error']}")
            else:
                counted.append({"path": result["path"], "nevents": result["nevents"]})
        files[sampleName]["files"] = counted

    return files


def submit_condor_counting(files, an_dict, njobs, dry_run=False):
    """Submit the per-file event-counting step to HTCondor via the same
    machinery spritz-batch uses (batch.submit()), one chunk per raw file.
    Run `spritz-fileset --merge` once the jobs finish to collect results."""
    chunks = []
    n_files = 0
    for sampleName, entry in files.items():
        if "query" in entry:
            continue  # nanoAOD/DAS sample, resolved separately, no dispatch needed
        for f in entry["files"]:
            chunks.append(
                {
                    "data": {"dataset": sampleName, "file_path": f},
                    "weight": 1,
                    "result": {},
                    "error": "",
                }
            )
        n_files += len(entry["files"])

    if n_files == 0:
        print("No path-based samples to dispatch to condor -- nothing to submit.")
        return

    batch_config = get_batch_cfg()

    print(f"Submitting {n_files} files across {min(njobs, n_files)} condor jobs to {FILESET_BATCH_SYSTEM}/")
    submit(
        chunks,
        path_an=os.path.abspath("."),
        an_dict=an_dict,
        njobs=njobs,
        dryRun=dry_run,
        script_name=f"{path_fw}/src/spritz/scripts/fileset_worker.py",
        batch_config=batch_config,
        job_dir=FILESET_BATCH_SYSTEM,
    )
    if dry_run:
        print(f"\nDry run: prepared {FILESET_BATCH_SYSTEM}/ but did not call condor_submit.")
    else:
        print(
            f"\nSubmitted. Once all jobs in {FILESET_BATCH_SYSTEM}/ have finished "
            "(check with condor_q / condor_history), run:\n"
            "  spritz-fileset --merge\n"
            "to collect the results into data/fileset.json."
        )


def merge_condor_counting():
    """Collect the results of a previous --condor dispatch into the same
    {sample: {"files": [{"path": [...], "nevents": n}, ...]}} shape
    count_events_local() would have produced."""
    from spritz.framework.framework import read_chunks

    job_dirs = sorted(glob.glob(f"{FILESET_BATCH_SYSTEM}/job_*"))
    if not job_dirs:
        raise Exception(
            f"No job directories found under {FILESET_BATCH_SYSTEM}/ -- "
            "did you run 'spritz-fileset --condor' first?"
        )

    files = {}
    n_missing = 0
    for job_dir in job_dirs:
        chunk_path = f"{job_dir}/chunks_job.pkl"
        if not os.path.isfile(chunk_path):
            print(f"Warning: {chunk_path} not found, skipping (job likely never ran)")
            continue
        chunks = read_chunks(chunk_path, readable=False)
        for chunk in chunks:
            sampleName = chunk["data"]["dataset"]
            file_path = chunk["data"]["file_path"]
            if chunk["result"] == {}:
                n_missing += 1
                print(f"Warning: no result yet for {sampleName}: {file_path} (job still running or failed)")
                continue
            files.setdefault(sampleName, {"files": []})
            files[sampleName]["files"].append(
                {"path": [file_path], "nevents": chunk["result"]["nevents"]}
            )

    if n_missing:
        print(
            f"\n{n_missing} file(s) have no result yet -- re-run once all "
            "condor jobs have actually finished for a complete fileset.json."
        )

    return files


def get_args():
    parser = argparse.ArgumentParser(description="Build data/fileset.json for the current analysis config")
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Dispatch the per-file event-counting step to HTCondor instead of running it locally, then exit "
        "(run again with --merge once the jobs finish)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Collect the results of a previous --condor dispatch and write data/fileset.json",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=50,
        help="Number of condor jobs to split the file list across (only used with --condor, default 50)",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="With --condor, prepare the job directory but don't actually call condor_submit "
        "(same convention as spritz-batch -dr)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    an_dict = get_analysis_dict()
    era = an_dict["year"]
    datasets = [k["files"] for k in an_dict["datasets"].values()]

    if args.merge:
        files = merge_condor_counting()
        # nanoAOD/DAS samples were never dispatched to condor -- discover
        # (cheap) and fold them back in now so they get resolved below.
        discovered = discover_files(era, datasets)
        for k, v in discovered.items():
            if "query" in v and k not in files:
                files[k] = v
    else:
        files = discover_files(era, datasets)
        if args.condor:
            submit_condor_counting(files, an_dict, njobs=args.njobs, dry_run=args.dry_run)
            return
        files = count_events_local(files)

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
