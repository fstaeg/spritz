
    #!/bin/bash

    ERA=Full2016v9noHIPM
    job_id=$1

    cd condor/job_${job_id}

    # configs/.../condor/job_0/tmp

    mkdir tmp
    cd tmp
    cp ../chunks_job.pkl .
    cp ../../run.sh .
    cp /gwpool/users/gpizzati/spritz/src/spritz/runners/script_worker_dumper.py .
    cp /gwpool/users/gpizzati/spritz/data/${ERA}/cfg.json .

    ./run.sh 2> err 1> out
    cp results.pkl ../chunks_job.pkl
    mv err ../err.txt
    mv out ../out.txt
    echo "Run locally" >> ../err.txt
    echo "Done ${job_id}"
    