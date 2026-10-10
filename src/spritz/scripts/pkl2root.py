import argparse
import pickle

import uproot
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert the histos.pkl written by spritz-postproc into a ROOT "
        "file, for the scripts that read histos.root (spritz-plot-eft, "
        "spritz-cov-matrix, ...). Same layout spritz-postproc used to write: "
        "'region/variable/histo_X' keys become directories."
    )
    parser.add_argument("input", nargs="?", default="histos.pkl", help="default: histos.pkl")
    parser.add_argument("output", nargs="?", default="histos.root", help="default: histos.root (overwritten)")
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Reading {args.input}")
    with open(args.input, "rb") as file:
        histos = pickle.load(file)

    print(f"Writing {len(histos)} histograms to {args.output}")
    with uproot.recreate(args.output) as fout:
        for key in tqdm(list(histos), desc="writing"):
            # drop each histogram once written, to not hold two copies
            fout[key] = histos.pop(key)


if __name__ == "__main__":
    main()
