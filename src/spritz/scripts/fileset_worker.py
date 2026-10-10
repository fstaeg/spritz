import sys
import traceback as tb

import uproot
from spritz.framework.framework import read_chunks, write_chunks

# Condor worker for the per-file event-counting step of spritz-fileset
# --condor/--merge. Mirrors the exact chunk convention analysis runners use
# (data/result/error per chunk, skip if already processed, write results.pkl)
# so it can be submitted via the existing batch.submit() machinery unchanged.

if __name__ == "__main__":
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks))

    for i in range(len(new_chunks)):
        new_chunk = new_chunks[i]

        if new_chunk["result"] != {}:
            print("Skip chunk", new_chunk["data"], "was already processed")
            continue

        try:
            f = uproot.open(new_chunk["data"]["file_path"])
            nevents = f["Events"].num_entries
            new_chunks[i]["result"] = {"nevents": nevents}
            new_chunks[i]["error"] = ""
        except Exception as e:
            print("\n\nError for chunk", new_chunk, file=sys.stderr)
            nice_exception = "".join(tb.format_exception(None, e, e.__traceback__))
            print(nice_exception, file=sys.stderr)
            new_chunks[i]["result"] = {"nevents": 0}
            new_chunks[i]["error"] = nice_exception

        print(f"Done {i+1}/{len(new_chunks)}")

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
