
    #!/bin/bash

    job_id=$1

    cd condor/job_${job_id}

    mkdir -p tmp
    cd tmp
    cp ../chunks_job.pkl .
    cp ../../../config.py .
    cp ../../runner_3DY_eft.py .
    cp ../../cfg.json .

    time python runner_3DY_eft.py .
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
        echo "Done ${job_id}"
    else
        echo "Failed ${job_id} (exit code $rc)"
    fi
    exit $rc
    