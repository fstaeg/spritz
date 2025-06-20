import subprocess
import os
from spritz.framework.framework import get_analysis_dict, get_fw_path, get_batch_cfg


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

    mkdir tmp
    cd tmp
    cp ../chunks_job.pkl .
    cp ../../../config.py .
    cp ../../{runner} .
    cp ../../cfg.json .

    time python {runner} .
    cp results.pkl ../chunks_job.pkl
    cd ../../../
    echo "Done ${{job_id}}"
    """
    with open("run_local.sh", "w") as file:
        file.write(txt)
    proc = subprocess.Popen("chmod +x run_local.sh", shell=True)
    proc.wait()
