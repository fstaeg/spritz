import argparse
import concurrent.futures
import glob
import hashlib
import os
from math import ceil
from typing import NewType, Generator

from tqdm import tqdm

from spritz.framework.framework import (  # noqa: F401
    add_dict_iterable,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
    write_chunks,
    get_batch_cfg
)
from spritz.scripts.batch import submit

MERGE_RESULT_FNAME = "tmp_special_"

# Directory `spritz-merge --condor` stages its dispatch in, and
# `spritz-merge --merge` reads back from -- parallel to spritz-fileset's
# "condor_fileset" convention, kept separate from the real analysis batch's
# own job_dir (BATCH_SYSTEM, usually "condor") so the two never collide.
MERGE_CONDOR_DIR = "merge_condor"

"""
# Result is something like:
{
    "dataset1": {
        # result of single dataset
    }
}
"""
Result = NewType("Result", dict[str, dict])


def read_inputs(inputs: list[str]) -> Generator:
    for input in inputs:
        job_result = read_chunks(input)
        if isinstance(job_result, list):
            for job_result_single in job_result:
                if job_result_single["result"] != {}:
                    yield job_result_single["result"]["real_results"]
        else:
            yield job_result


def check_input(input: Result) -> bool:
    # Returns true if input is ok
    for chunk in input:
        if input["result"] == {} or input["error"] != "":
            return False
    return True


def postprocess_inputs(inputs):
    for input in inputs:
        if MERGE_RESULT_FNAME in input.split("/")[-1]:
            print("removing", input)
            os.remove(input)


def reduction(inputs, reduce_function, output):
    result = reduce_function(read_inputs(inputs))
    postprocess_inputs(inputs)
    write_chunks(result, output)


def split_inputs(inputs, elements_for_task):
    ntasks = ceil(len(inputs) / elements_for_task)
    for i in range(ntasks):
        start = min(i * elements_for_task, len(inputs) - 1)
        stop = min((i + 1) * elements_for_task, len(inputs))
        if start == stop:
            break
        yield slice(start, stop)


def count_tree_tasks(n_inputs, elements_for_task=10):
    """Total number of reduction() calls create_tree() will make across every
    level of the tree, for a progress bar total that covers the whole merge
    (not just one level) up front."""
    total = 0
    n = n_inputs
    while n > elements_for_task:
        n = ceil(n / elements_for_task)
        total += n
    return total + 1  # the final base-case reduction


def create_tree(inputs, reduce_function, output, executor, elements_for_task=10, pbar=None):
    if len(inputs) <= elements_for_task:
        reduction(inputs, reduce_function, output)
        if pbar is not None:
            pbar.update(1)

    else:
        output_dir = "/".join(output.split("/")[:-1])
        output_format = output.split(".")[-1]
        splits = split_inputs(inputs, elements_for_task)
        tasks = []
        new_inputs = []

        futures = {}
        for itask, split in enumerate(split_inputs(inputs, elements_for_task)):
            h = hashlib.new("sha256")
            h.update(str(itask).encode("utf-8"))
            for input in inputs[split]:
                h.update(input.encode("utf-8"))
            h = h.hexdigest()[:10]
            output_tmp = f"{output_dir}/{MERGE_RESULT_FNAME}_{h}.{output_format}"
            future = executor.submit(reduction, inputs[split], reduce_function, output_tmp)
            futures[future] = output_tmp
            new_inputs.append(output_tmp)

        for future in concurrent.futures.as_completed(futures):
            future.result()
            del futures[future]
            if pbar is not None:
                pbar.update(1)

        create_tree(new_inputs, reduce_function, output, executor, elements_for_task, pbar=pbar)


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge per-chunk analysis results into results_merged_new.pkl"
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Dispatch the merge to HTCondor instead of running it locally: splits the "
        "raw job_*/chunks_job.pkl into --njobs groups, one condor job per group, each "
        "producing one merged intermediate file under merge_condor/. Run again with "
        "--merge once those finish to produce the final results_merged_new.pkl.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Finish a previous --condor dispatch: locally reduce merge_condor/'s "
        "intermediate outputs into results_merged_new.pkl (same local reduction as "
        "the no-flag mode, just starting from far fewer, pre-merged inputs).",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=100,
        help="Target number of merged output files when using --condor (default 100) "
        "-- i.e. how many condor jobs to split the raw per-chunk outputs across.",
    )
    parser.add_argument(
        "-dr",
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="With --condor, prepare the job directory but don't actually call "
        "condor_submit (same convention as spritz-batch -dr)",
    )
    return parser.parse_args()


def submit_condor_merge(basepath, njobs, dry_run=False):
    """Dispatch the leaf-level merge work to condor: split the raw
    job_*/chunks_job.pkl files (there can be thousands) into `njobs` groups,
    one condor job per group, each just calling the same reduction()/
    add_dict_iterable() used locally on its own group. This is the
    expensive, embarrassingly-parallel part (per earlier profiling, nearly
    all of a local merge's time is the leaf level); the remaining upper
    tree levels are cheap enough to finish locally afterward with --merge.

    NOTE: workers read their assigned inputs by absolute /gwpool/... path
    directly rather than via HTCondor's transfer_input_files (the shared
    condor_submit() JDL template only supports one fixed file list, not an
    arbitrary per-job set) -- this assumes /gwpool is actually mounted on
    the execute nodes, which is the normal setup for a pool like this one
    but hasn't been verified here. Test with a small --njobs first.
    """
    inputs = [os.path.abspath(p) for p in glob.glob(f"{basepath}/job_*/chunks_job.pkl")]
    if not inputs:
        raise Exception(f"No {basepath}/job_*/chunks_job.pkl files found -- nothing to merge.")

    njobs = min(njobs, len(inputs))
    groups = [inputs[i::njobs] for i in range(njobs)]
    groups = [g for g in groups if len(g) > 0]

    chunks = [
        {
            "data": {"inputs": group, "dataset": f"merge_group_{i}"},
            "weight": 1,
            "result": {},
            "error": "",
        }
        for i, group in enumerate(groups)
    ]

    an_dict = get_analysis_dict()
    batch_config = get_batch_cfg()

    print(f"Merging {len(inputs)} inputs across {len(chunks)} condor jobs -> {len(chunks)} merged files in {MERGE_CONDOR_DIR}/")
    submit(
        chunks,
        path_an=os.path.abspath("."),
        an_dict=an_dict,
        njobs=len(chunks),
        dryRun=dry_run,
        script_name=f"{get_fw_path()}/src/spritz/scripts/merge_worker.py",
        batch_config=batch_config,
        job_dir=MERGE_CONDOR_DIR,
    )
    if dry_run:
        print(f"\nDry run: prepared {MERGE_CONDOR_DIR}/ but did not call condor_submit.")
    else:
        print(
            f"\nSubmitted. Once all jobs in {MERGE_CONDOR_DIR}/ have finished "
            "(check with condor_q / condor_history), run:\n"
            "  spritz-merge --merge\n"
            "to combine them into the final results_merged_new.pkl."
        )


def main():
    args = get_args()
    basepath = os.path.abspath(get_batch_cfg()["BATCH_SYSTEM"])
    output = f"{basepath}/results_merged_new.pkl"
    reduce_function = add_dict_iterable
    elements_for_task = 25
    cpus = 30

    if args.condor:
        submit_condor_merge(basepath, njobs=args.njobs, dry_run=args.dry_run)
        return

    if args.merge:
        inputs = glob.glob(f"{MERGE_CONDOR_DIR}/job_*/chunks_job.pkl")
        if not inputs:
            raise Exception(
                f"No {MERGE_CONDOR_DIR}/job_*/chunks_job.pkl files found -- "
                "did you run 'spritz-merge --condor' first?"
            )
    else:
        inputs = glob.glob(f"{basepath}/job_*/chunks_job.pkl")[:]

    total_tasks = count_tree_tasks(len(inputs), elements_for_task)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        with tqdm(total=total_tasks, desc="merging") as pbar:
            create_tree(
                inputs,
                reduce_function,
                output,
                executor,
                elements_for_task=elements_for_task,
                pbar=pbar,
            )

    results = read_chunks(output)
    print([(dataset, results[dataset]["sumw"]) for dataset in results])


if __name__ == "__main__":
    main()
