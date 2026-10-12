import subprocess
import os
from spritz.framework.framework import get_analysis_dict, get_fw_path, get_batch_cfg


# Not an f-string: the generated script has plenty of {braces} of its own.
MULTICORE_TEMPLATE = """import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob


def run_script(i):
    script = "./run_local.sh"
    print(f"[INFO] Starting job {i}...")

    # Open subprocess to stream output
    process = subprocess.Popen(
        ["bash", script, str(i)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        env=os.environ,
    )

    # Stream the output line by line
    for line in process.stdout:
        print(f"[Job {i}] {line}", end="")  # Prefix each line with job ID

    returncode = process.wait()
    print(f"[INFO] Job {i} finished with code {returncode}")
    return i, returncode


def main():
    total_jobs = len(glob("__JOB_DIR__/job_*"))
    # Usage: python run_multicore.py [max_workers]  (default: one per core;
    # each job needs a few GB of memory, so lower it on a shared machine)
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()

    print(f"[INFO] Submitting {total_jobs} jobs with {max_workers} workers...\\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_script, i): i for i in range(0, total_jobs)}

        for future in as_completed(futures):
            i, returncode = future.result()
            if returncode == 0:
                print(f"[OK] Job {i} completed successfully.")
            else:
                print(f"[ERROR] Job {i} failed with return code {returncode}.\\n")

    print("\\n[INFO] All jobs have finished.")


if __name__ == "__main__":
    main()
"""


def main():
    an_dict = get_analysis_dict()
    era = an_dict["year"]
    runner_default = f"{get_fw_path()}/src/spritz/runners/runner_default.py"
    runner = an_dict.get("runner", runner_default)
    runner = os.path.split(runner)[-1]

    txt = f"""
    #!/bin/bash

    job_id=$1

    cd {get_batch_cfg()["BATCH_SYSTEM"]}/job_${{job_id}}

    mkdir -p tmp
    cd tmp
    cp ../chunks_job.pkl .
    cp ../../../config.py .
    cp ../../{runner} .
    cp ../../cfg.json .

    time python {runner} .
    rc=$?
    if [ $rc -eq 0 ]; then
        cp results.pkl ../chunks_job.pkl
    fi
    cd ..
    # tmp/results.pkl is a full copy of chunks_job.pkl; keep tmp only to debug failures
    if [ $rc -eq 0 ]; then
        rm -rf tmp
    fi
    cd ../../
    if [ $rc -eq 0 ]; then
        echo "Done ${{job_id}}"
    else
        echo "Failed ${{job_id}} (exit code $rc)"
    fi
    exit $rc
    """
    with open("run_local.sh", "w") as file:
        file.write(txt)
    proc = subprocess.Popen("chmod +x run_local.sh", shell=True)
    proc.wait()

    with open("run_multicore.py", "w") as file:
        file.write(MULTICORE_TEMPLATE.replace("__JOB_DIR__", get_batch_cfg()["BATCH_SYSTEM"]))
