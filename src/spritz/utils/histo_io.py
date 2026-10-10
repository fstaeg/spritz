"""
Read and write the post-processed histograms either as histos.pkl (dict
'region/variable/histo_X' -> hist.Hist, written by spritz-postproc) or as
histos.root (same keys as directories, written by spritz-pkl2root). The
format is chosen from the file extension.
"""
import pickle

import uproot
from tqdm import tqdm


class RootHistos:
    """read-only, dict-like view of a ROOT file: histograms are read and
    converted to hist.Hist only when accessed"""

    def __init__(self, path):
        self.file = uproot.open(path)
        self._keys = [
            key for key, classname in self.file.classnames(cycle=False).items()
            if classname.startswith("TH")
        ]
        self._key_set = set(self._keys)

    def __getitem__(self, key):
        if key not in self._key_set:
            raise KeyError(key)
        return self.file[key].to_hist()

    def __contains__(self, key):
        return key in self._key_set

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return list(self._keys)


def is_root(path):
    return str(path).endswith(".root")


def open_histos(path):
    """dict-like 'region/variable/histo_X' -> hist.Hist. A .root file is read
    lazily, a .pkl one is loaded at once"""
    if is_root(path):
        return RootHistos(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_histos(path):
    """same as open_histos, but always a plain dict with every histogram"""
    histos = open_histos(path)
    if isinstance(histos, RootHistos):
        histos = {key: histos[key] for key in tqdm(histos.keys(), desc=f"reading {path}")}
    return histos


def save_histos(histos, path):
    if is_root(path):
        with uproot.recreate(path) as fout:
            for key in tqdm(list(histos), desc=f"writing {path}"):
                fout[key] = histos[key]
    else:
        with open(path, "wb") as f:
            pickle.dump(histos, f)
