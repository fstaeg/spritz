import sys  # noqa: F401
import subprocess
from spritz.framework.framework import get_batch_cfg

def resubmit(
    job_idx_list=[],
    dryRun=False,
    batch_system="condor"
):
    print(f"resubmitting {len(job_idx_list)} jobs")
    print(sorted(job_idx_list))

    folders = []
    for idx in job_idx_list:
        folder_path = f"{batch_system}/job_{idx}"
        folders.append(folder_path.split('/')[-1])
        proc = subprocess.Popen(f"rm {folder_path}/*.txt", shell=True)
        proc.wait()

    if batch_system == "condor":
        with open(f"{batch_system}/submit.jdl", "r") as file:
            txtjdl = file.readlines()
        for i,line in enumerate(txtjdl):
            if line.startswith("queue 1 Folder in"):
                txtjdl[i] = f'queue 1 Folder in {", ".join(folders)}\n'

        with open(f"{batch_system}/resubmit.jdl", "w") as file:
            file.writelines(txtjdl)

    if not dryRun:
        if batch_system == "condor":
            command = "cd condor/; chmod +x run.sh; condor_submit resubmit.jdl; cd -"
        elif batch_system == "slurm":
            command = f"cd slurm/; sbatch --array={','.join(job_idx_list)} run.sh; cd -"
    elif batch_system == "condor":
        command = "cd condor/; chmod +x run.sh; cd -"
    proc = subprocess.Popen(command, shell=True)
    proc.wait()


def main():
    dryRun = False 
    batch_system = get_batch_cfg()["BATCH_SYSTEM"]

    if len(sys.argv) > 1:
        dryRun = sys.argv[1] == "-dr"
        jobs = [i for i in sys.argv[1:] if not i == '-dr']
        
    resubmit(
        job_idx_list=jobs,
        dryRun=dryRun,
        batch_system=batch_system
    )


if __name__ == "__main__":
    main()
