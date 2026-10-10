import json
import os
import subprocess
import sys  # noqa: F401

from spritz.framework.framework import (
    add_dict,
    get_analysis_dict,
    get_fw_path,
    get_batch_cfg,
    read_chunks,
    write_chunks,
)


def preprocess_chunks(year):
    with open(f"{get_fw_path()}/data/common/forms.json", "r") as file:
        forms_common = json.load(file)
    with open(f"{get_fw_path()}/data/{year}/forms.json", "r") as file:
        forms_era = json.load(file)
    forms = add_dict(forms_common, forms_era)
    new_chunks = read_chunks("data/chunks.pkl")

    for i, chunk in enumerate(new_chunks):
        new_chunks[i]["data"]["read_form"] = forms[chunk["data"]["read_form"]]
    return new_chunks


def split_chunks(chunks, n, eft_chunks_per_job=None):
    """
    Splits list l of chunks into n jobs with approximately equals sum of values
    see  http://stackoverflow.com/questions/6855394/splitting-list-in-chunks-of-balanced-weight

    Keep datasets with eft_reweighting together to reduce the 
    total size of results and prevent memory issues. With eft_chunks_per_job,
    each job gets exactly that many chunks of a single EFT dataset (the
    runner sums them into one result), instead of spreading them over all n jobs
    """
    jobs = [[] for i in range(n)]
    sums = {i: 0 for i in range(n)}
    
    chunks_eft, chunks_other = [], []
    for chunk in chunks:
        if chunk["data"].get("eft_reweighting") is None: 
            chunks_other.append(chunk)
        else: 
            chunks_eft.append(chunk)
    chunks = [chunk for chunk in chunks_other]
    
    if len(chunks_eft) > 0:
        if eft_chunks_per_job:
            by_dataset = {}
            for chunk in chunks_eft:
                by_dataset.setdefault(chunk["data"]["dataset"], []).append(chunk)
            eft_batches = [
                group[k: k+eft_chunks_per_job]
                for group in by_dataset.values()
                for k in range(0, len(group), eft_chunks_per_job)
            ]
            if len(eft_batches) > n:
                raise Exception(
                    f"{len(chunks_eft)} EFT chunks at eft_chunks_per_job={eft_chunks_per_job} "
                    f"need {len(eft_batches)} jobs, but njobs={n}"
                )
        else:
            n_eft = int( len(chunks_eft) / n + 1)
            eft_batches = [chunks_eft[k: k+n_eft] for k in range(0, len(chunks_eft), n_eft)]

        for i, batch in enumerate(eft_batches):
            jobs[i].extend(batch)
            sums[i] += sum(chunk["weight"] for chunk in batch)
    
    c = min(sums.values())
    
    for chunk in chunks:
        for i in sums:
            if c == sums[i]:
                jobs[i].append(chunk)
                break
        sums[i] += chunk["weight"]
        c = min(sums.values())
    return jobs

def slurm_script(image, runner, path_an, short_queue=False):
    short_queue_string = """#SBATCH --time=01:00:00
#SBATCH --partition=short
""" if short_queue else ""
    return f"""#!/bin/bash
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --mem=3000M
{short_queue_string}

export SLURM_ID=${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
export TMPDIR=/scratch/$USER/${{SLURM_ID}}

echo SLURM_ID: $SLURM_ID
echo HOSTNAME: $HOSTNAME

mkdir -p $TMPDIR
cp {image} $TMPDIR
cp {path_an}/config.py $TMPDIR
cp {path_an}/slurm/start.sh $TMPDIR
cp {path_an}/slurm/cfg.json $TMPDIR
cp {path_an}/slurm/{runner} $TMPDIR
cp {path_an}/slurm/data.tar.gz $TMPDIR
cp {path_an}/slurm/spritz.tar.gz $TMPDIR
cp {path_an}/slurm/job_${{SLURM_ARRAY_TASK_ID}}/chunks_job.pkl $TMPDIR

pushd $TMPDIR
singularity run {os.path.split(image)[-1]}
source start.sh
tar -xzf data.tar.gz
tar -xzf spritz.tar.gz

time python {runner} .

cp results.pkl {path_an}/slurm/job_${{SLURM_ARRAY_TASK_ID}}/chunks_job.pkl
popd

mv logs/${{SLURM_ID}}.out job_${{SLURM_ARRAY_TASK_ID}}/out.txt
mv logs/${{SLURM_ID}}.err job_${{SLURM_ARRAY_TASK_ID}}/err.txt
rm -rf $TMPDIR
"""


def condor_script(proxy, runner):
    return f"""#!/bin/bash
{f"export X509_USER_PROXY={proxy}" if proxy is not None else ""}

source start.sh
tar -xzf data.tar.gz
tar -xzf spritz.tar.gz

time python {runner} .
"""


def condor_submit(proxy, runner, image, machines, folders, path_an, has_special_start=False, job_flavour=None, job_dir="condor", request_memory=2048, folder_memory=None, delegate_proxy=True, stream_output=False):
    """folder_memory: {folder: request_memory} for the folders that need more
    than request_memory. Jobs held for exceeding their memory (HoldReasonCode
    34) are released with the largest request, at most 3 times."""
    extra_inputs = f", {path_an}/{job_dir}/special_start.sh" if has_special_start else ""
    folder_memory = folder_memory or {}
    max_memory = max([request_memory, *folder_memory.values()])
    queue_items = "\n".join(f"{folder} {folder_memory.get(folder, request_memory)}" for folder in folders)
    return f"""universe = vanilla
executable = run.sh
arguments = $(Folder)
use_x509userproxy = {"true" if proxy is not None and delegate_proxy else "false"}
should_transfer_files = YES
transfer_input_files = {path_an}/{job_dir}/$(Folder)/chunks_job.pkl, {path_an}/{job_dir}/{runner}, {path_an}/{job_dir}/cfg.json, {path_an}/config.py, {path_an}/{job_dir}/data.tar.gz, {path_an}/{job_dir}/spritz.tar.gz, {path_an}/{job_dir}/start.sh{extra_inputs}
{f'MY.SingularityImage = "{image}"' if image is not None else ""}
transfer_output_remaps = "results.pkl = $(Folder)/chunks_job.pkl"
output = $(Folder)/out.txt
error  = $(Folder)/err.txt
log    = $(Folder)/log.txt
{"stream_output = True" + chr(10) + "stream_error = True" if stream_output else ""}
request_memory = ifThenElse(LastHoldReasonCode =?= 34, {max_memory}, $(JobMemory))
periodic_release = (HoldReasonCode == 34) && (NumJobStarts < 4)
{("Requirements = " + " || ".join([f'(machine == "{machine}")' for machine in machines])) if len(machines)>0 else ""}
{f'+JobFlavour = "{job_flavour}"' if job_flavour is not None else ""}
queue Folder, JobMemory from (
{queue_items}
)
"""


def submit(
    new_chunks,
    path_an,
    an_dict,
    njobs=500,
    start=0,
    dryRun=False,
    script_name="script_worker.py",
    batch_config={},
    short_queue=False,
    job_dir=None,
):
    machines = batch_config.get("MACHINES", [])
    batch_system = batch_config["BATCH_SYSTEM"]
    if job_dir is None:
        job_dir = batch_system

    print(f"{len(new_chunks)} chunks")
    jobs = split_chunks(new_chunks, njobs, an_dict.get("eft_chunks_per_job"))

    print(f"{len(jobs)} jobs")
    print(sorted(list(set(list(map(lambda k: k["data"]["dataset"], new_chunks))))))
    print()

    if os.path.isdir(job_dir):
        if os.path.isdir(f"{job_dir}_backup"):
            proc = subprocess.Popen(f"rm -r {job_dir}_backup", shell=True)
            proc.wait()

        proc = subprocess.Popen(f"mv {job_dir} {job_dir}_backup", shell=True)
        proc.wait()

    folders = []

    for i, job in enumerate(jobs):
        folder = f"{job_dir}/job_{start+i}"
        proc = subprocess.Popen(f"mkdir -p {folder}", shell=True)
        proc.wait()
        write_chunks(job, f"{folder}/chunks_job.pkl")
        #write_chunks(job, f"{folder}/chunks_job_original.pkl")
        folders.append(folder.split("/")[-1])

    command = f"cp {script_name} {job_dir}/; "
    command += f"cp {get_fw_path()}/data/{an_dict["year"]}/cfg.json {job_dir}/; "
    command += f"cp {get_fw_path()}/start.sh {job_dir}/; "
    command += f"tar -zcf {job_dir}/data.tar.gz --directory={get_fw_path()} data/; "
    command += f"tar -zcf {job_dir}/spritz.tar.gz --directory={get_fw_path()}/src spritz/"

    special_start_src = f"{get_fw_path()}/special_start.sh"
    has_special_start = os.path.isfile(special_start_src)
    if has_special_start:
        command += f"; cp {special_start_src} {job_dir}/"

    proc = subprocess.Popen(command, shell=True)
    proc.wait()

    if batch_system == "condor":
        txtsh = condor_script(batch_config["X509_USER_PROXY"], os.path.split(script_name)[-1])
    elif batch_system == "slurm":
        txtsh = slurm_script(batch_config["SINGULARITY_IMAGE"], os.path.split(script_name)[-1], path_an, short_queue)

    with open(f"{job_dir}/run.sh", "w") as file:
        file.write(txtsh)

    if batch_system == "condor":
        folder_memory = {}
        if batch_config.get("REQUEST_MEMORY_EFT") is not None:
            folder_memory = {
                folder: batch_config["REQUEST_MEMORY_EFT"]
                for folder, job in zip(folders, jobs)
                if any(chunk["data"].get("eft_reweighting") is not None for chunk in job)
            }
        txtjdl = condor_submit(
            batch_config["X509_USER_PROXY"],
            os.path.split(script_name)[-1],
            batch_config["SINGULARITY_IMAGE"],
            machines,
            folders,
            path_an,
            has_special_start=has_special_start,
            job_flavour=batch_config.get("JOB_FLAVOUR"),
            job_dir=job_dir,
            request_memory=batch_config.get("REQUEST_MEMORY", 2048),
            folder_memory=folder_memory,
            delegate_proxy=batch_config.get("X509_DELEGATE", True),
            stream_output=batch_config.get("STREAM_OUTPUT", False),
        )
        with open(f"{job_dir}/submit.jdl", "w") as file:
            file.write(txtjdl)

    command = ""
    if not dryRun:
        if batch_system == "condor":
            command = f"cd {job_dir}/; chmod +x run.sh; condor_submit submit.jdl; cd -"
        elif batch_system == "slurm":
            command = f"cd {job_dir}/; sbatch --array=0-{len(jobs)-1} run.sh; cd -"
    elif batch_system == "condor":
        command = f"cd {job_dir}/; chmod +x run.sh; cd -"

    proc = subprocess.Popen(command, shell=True)
    proc.wait()


def main():
    start = 0
    path_an = os.path.abspath(".")
    an_dict = get_analysis_dict()
    chunks = preprocess_chunks(an_dict["year"])
    runner_default = f"{get_fw_path()}/src/spritz/runners/runner_default.py"
    runner = an_dict.get("runner", runner_default)
    dryRun = "-dr" in sys.argv
    short_queue = "-short" in sys.argv

    submit(
        chunks,
        path_an,
        an_dict,
        njobs=an_dict["njobs"],
        start=start,
        dryRun=dryRun,
        script_name=runner,
        batch_config=get_batch_cfg(),
        short_queue=short_queue
    )


if __name__ == "__main__":
    main()
