import sys
import traceback as tb

from spritz.framework.framework import add_dict_iterable, read_chunks, write_chunks
from spritz.scripts.merge import read_inputs

# Condor worker for `spritz-merge --condor`. Each chunk's "inputs" is a list
# of *absolute paths* to existing job_N/chunks_job.pkl files on the shared
# filesystem (not transferred via HTCondor -- read directly, same as the
# main runner reads NanoAOD over XRootD by absolute path). Writes its single
# merged result into the SAME "result": {"real_results": ...} shape a real
# analysis chunk uses, so the exact same read_inputs()/create_tree() used
# for local merging can also read these back transparently later.

if __name__ == "__main__":
    chunks_readable = False
    new_chunks = read_chunks("chunks_job.pkl", readable=chunks_readable)
    print("N chunks to process", len(new_chunks), flush=True)

    for i in range(len(new_chunks)):
        new_chunk = new_chunks[i]

        if new_chunk["result"] != {}:
            print("Skip chunk, was already processed", flush=True)
            continue

        inputs = new_chunk["data"]["inputs"]
        print(f"Merging {len(inputs)} inputs", flush=True)

        try:
            merged = add_dict_iterable(read_inputs(inputs))
            new_chunks[i]["result"] = {"real_results": merged}
            new_chunks[i]["error"] = ""
        except Exception as e:
            print("\n\nError merging chunk", new_chunk, file=sys.stderr)
            nice_exception = "".join(tb.format_exception(None, e, e.__traceback__))
            print(nice_exception, file=sys.stderr)
            new_chunks[i]["result"] = {}
            new_chunks[i]["error"] = nice_exception

        print(f"Done {i + 1}/{len(new_chunks)}", flush=True)

    write_chunks(new_chunks, "results.pkl", readable=chunks_readable)
