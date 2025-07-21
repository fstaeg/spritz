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


def split_chunks(chunks, n):
    """
    Splits list l of chunks into n jobs with approximately equals sum of values
    see  http://stackoverflow.com/questions/6855394/splitting-list-in-chunks-of-balanced-weight
    """
    jobs = [[] for i in range(n)]
    sums = {i: 0 for i in range(n)}
    c = 0
    for chunk in chunks:
        for i in sums:
            if c == sums[i]:
                jobs[i].append(chunk)
                break
        sums[i] += chunk["weight"]
        c = min(sums.values())
    return jobs


def slurm_script(image, runner, path_an):
    return f"""#!/bin/bash
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --mem=3000M

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


def condor_submit(proxy, runner, image, machines, folders):
    return f"""universe = vanilla
executable = run.sh
arguments = $(Folder)
use_x509userproxy = {"true" if proxy is not None else "false"}
should_transfer_files = YES
transfer_input_files = $(Folder)/chunks_job.pkl, {runner}, cfg.json, ../config.py, data.tar.gz, spritz.tar.gz, start.sh
{f'MY.SingularityImage = "{image}"' if image is not None else ""}
transfer_output_remaps = "results.pkl = $(Folder)/chunks_job.pkl"
output = $(Folder)/out.txt
error  = $(Folder)/err.txt
log    = $(Folder)/log.txt
request_cpus=1
request_memory=2000
request_disk=2500000
{("Requirements = " + " || ".join([f'(machine == "{machine}")' for machine in machines])) if len(machines)>0 else ""}
+JobFlavour = "longlunch"
queue 1 Folder in {", ".join(folders)}
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
):
    machines = []
    batch_system = batch_config["BATCH_SYSTEM"]

    print(f"{len(new_chunks)} chunks")
    jobs = split_chunks(new_chunks, njobs)

    print(f"{len(jobs)} jobs")
    print(sorted(list(set(list(map(lambda k: k["data"]["dataset"], new_chunks))))))
    print()

    if os.path.isdir(batch_system):
        if os.path.isdir(f"{batch_system}_backup"):
            proc = subprocess.Popen(f"rm -r {batch_system}_backup", shell=True)
            proc.wait()
        
        proc = subprocess.Popen(f"mv {batch_system} {batch_system}_backup", shell=True)
        proc.wait()

    folders = []

    for i, job in enumerate(jobs):
        folder = f"{batch_system}/job_{start+i}"
        proc = subprocess.Popen(f"mkdir -p {folder}", shell=True)
        proc.wait()
        write_chunks(job, f"{folder}/chunks_job.pkl")
        #write_chunks(job, f"{folder}/chunks_job_original.pkl")
        folders.append(folder.split("/")[-1])
    
    command = f"cp {script_name} {batch_system}/; "
    command += f"cp {get_fw_path()}/data/{an_dict["year"]}/cfg.json {batch_system}/; "
    command += f"cp {get_fw_path()}/start.sh {batch_system}/; "
    command += f"tar -zcf {batch_system}/data.tar.gz --directory={get_fw_path()} data/; "
    command += f"tar -zcf {batch_system}/spritz.tar.gz --directory={get_fw_path()}/src spritz/"
    proc = subprocess.Popen(command, shell=True)
    proc.wait()

    if batch_system == "condor":
        txtsh = condor_script(batch_config["X509_USER_PROXY"], os.path.split(script_name)[-1])
    elif batch_system == "slurm":
        txtsh = slurm_script(batch_config["SINGULARITY_IMAGE"], os.path.split(script_name)[-1], path_an)
    
    with open(f"{batch_system}/run.sh", "w") as file:
        file.write(txtsh)

    if batch_system == "condor":
        txtjdl = condor_submit(
            batch_config["X509_USER_PROXY"], 
            os.path.split(script_name)[-1], 
            batch_config["SINGULARITY_IMAGE"], 
            machines, 
            folders
        )
        with open(f"{batch_system}/submit.jdl", "w") as file:
            file.write(txtjdl)
    
    command = ""
    if not dryRun:
        if batch_system == "condor":
            command = "cd condor/; chmod +x run.sh; condor_submit submit.jdl; cd -"
        elif batch_system == "slurm":
            command = f"cd slurm/; sbatch --array=0-{len(jobs)-1} run.sh; cd -"
    elif batch_system == "condor":
        command = "cd condor/; chmod +x run.sh; cd -"
    
    proc = subprocess.Popen(command, shell=True)
    proc.wait()


def main():
    start = 0
    path_an = os.path.abspath(".")
    an_dict = get_analysis_dict()
    chunks = preprocess_chunks(an_dict["year"])
    runner_default = f"{get_fw_path()}/src/spritz/runners/runner_default.py"
    runner = an_dict.get("runner", runner_default)
    dryRun = False

    if len(sys.argv) > 1:
        dryRun = sys.argv[1] == "-dr"

    submit(
        chunks,
        path_an,
        an_dict,
        njobs=an_dict["njobs"],
        start=start,
        dryRun=dryRun,
        script_name=runner,
        batch_config=get_batch_cfg(),
    )


if __name__ == "__main__":
    main()
