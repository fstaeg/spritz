import concurrent.futures
import glob
import os
import sys
import traceback as tb

from spritz.framework.framework import read_chunks, get_batch_cfg


def bad_lines_fun(line):
    if line.strip() == "":
        return False

    if line.startswith("real"):
        return False
    if line.startswith("user"):
        return False
    if line.startswith("sys"):
        return False
    if line == "Run locally":
        return False
    if line.startswith("did not find anything for LHEPart "):
        return False
    if (
        "could not instantiate session cipher using cipher public info from server"
        in line
    ):
        return False
    return True


def check_job(job_id, batch_system):
    file = f"{batch_system}/{job_id}/chunks_job.pkl"

    if not os.path.exists(f"{batch_system}/{job_id}/err.txt"):
        return job_id, -1, ""

    chunks_total = 0
    chunks_err = 0
    erred_data = 0
    try:
        chunks = read_chunks(file)
        assert isinstance(chunks, list)
        for i in range(len(chunks)):
            chunks_total += 1
            if chunks[i]["result"] == {} and chunks[i]["error"] != "":
                chunks_err += 1
                if chunks[i]["is_data"]:
                    erred_data += 1
                break
        if chunks_total > 0 and chunks_err == 0:
            pass
        else:
            print("skipping job, should be retried")
            return job_id, 1 + erred_data, "Error found in chunks:" + chunks[i]["error"]
    except Exception as e:
        return job_id, True, "".join(tb.format_exception(None, e, e.__traceback__))

    if os.path.exists(f"{batch_system}/{job_id}/err.txt"):
        with open(f"{batch_system}/{job_id}/err.txt") as file:
            lines = file.read().split("\n")
            bad_lines = list(filter(bad_lines_fun, lines))
            error = "\n".join(bad_lines)
            if len(bad_lines) > 0:
                # print("\033[91m", job_id, "\033[0m")
                # print("\n".join(bad_lines))
                return job_id, 2, error
    return job_id, False, ""


def main():
    batch_cfg = get_batch_cfg()
    batch_system = batch_cfg["BATCH_SYSTEM"]
    files = glob.glob(f"{batch_system}/job_*/chunks_job.pkl")

    jobs = list(map(lambda k: k.split("/")[-2], files))

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        tasks = []
        for job_id in jobs:
            tasks.append(pool.submit(check_job, job_id, batch_system))
        concurrent.futures.wait(tasks)
        failed = []
        running = []
        total = 0
        for task in tasks:
            res = task.result()
            if res[1] > 0:
                failed.append(res[0])
            if res[1] == 2:
                print("Real error!", res[0])
            if res[1] == -1:
                running.append(res[0])
            total += 1

        if len(running)>0:
            print("\nStill running jobs")
            print(' '.join([j.replace('job_','') for j in sorted(running)]))
        if len(failed)>0:
            print("\nFailed jobs")
            print(' '.join([j.replace('job_','') for j in sorted(failed)]))
        print("\nFailed", len(failed))
        print("Total", total)
        print("Still running", len(running), "\n")


if __name__ == "__main__":
    main()
