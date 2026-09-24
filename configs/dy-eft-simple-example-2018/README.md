# DY EFT simple example (2018)

Minimal 2-operator (cql32, cqlm2) SMEFT reweighting example on the 8 mll-binned
2018 `DYMuMu_NLO_EFT_SMEFTatNLO_*_startingOne` samples, producing a datacard
with 6 processes: `sm`, `w1_cql32`, `wm1_cql32`, `w1_cqlm2`, `wm1_cqlm2`,
`w11_cql32_cqlm2`.

cql32 and cqlm2 were chosen specifically because both mll-bin groups' reweight
cards actually probe them (see below) -- cqlm1 does NOT appear anywhere in
this config, because one of the two groups' reweight cards never included it
at all, and there is no honest way to build a "cqlm1" template for those 3
datasets.

Every step below is a standard `spritz-*` command -- there are no bespoke
post-processing scripts. The EFT reweighting is built directly into
histograms by the runner, using the framework's native per-dataset
`subsamples` mechanism.

## How the EFT reweighting works

`config.py` sets `runner = runner_3DY_eft_reweight.py`, a small variant of the
standard `runner_3DY.py` where a `subsamples` entry can be either:
- a plain mask-expression string (existing behavior, e.g. jet-flavour splits), or
- a `(mask_expr, weight_expr)` tuple, where `weight_expr` replaces the nominal
  event weight for that subsample only.

Each of the 8 datasets in `config.py` lists 6 subsamples (`sm`, `w1_cql32`,
`wm1_cql32`, `w1_cqlm2`, `wm1_cqlm2`, `w11_cql32_cqlm2`), each selecting *all*
events (`ak.ones_like(events.weight, dtype=bool)`) but weighted by a different
`events.LHEReweightingWeight[:, idx]` point -- see the big comment block at
the top of `config.py` for the verified per-group indices (the two groups of
mll bins were generated with different reweight cards, so cql32 and cqlm2
each sit at a different index in each, even though both operators are
genuinely present in both cards).

`spritz.scripts.chunks.create_chunks()` copies every `datasets[dataset]` key
straight into that dataset's chunk kwargs, so this per-dataset `subsamples`
dict reaches the runner as-is -- no config-wide override machinery needed.

## Summing across the 8 mll bins into one shape per EFT point

This is standard `post_process.py` behavior, not anything special: `config.py`'s
`samples` dict groups all 8 `{dataset}_{subsample}` combinations sharing the
same subsample name into one entry (e.g. `samples["sm"]["samples"]` lists all
8 `..._sm"` dataset names). `post_process.py` already knows how to sum a
`"samples": [...]` list, each weighted by its own xsec/sumw/lumi -- and
`post_process.py`'s existing `if "subsamples" in datasets[dataset]:` branch
already builds the right xsec for every `{dataset}_{subsample}` key (each one
gets the *raw* dataset's xsec, unaffected by which EFT point it's weighted
by).

## Correlated MC-stat uncertainties (the covariance matrix)

All 6 templates come from the *same* underlying MC events, just reweighted
differently -- their statistical fluctuations are strongly correlated, not
independent. `autoMCStats` in the datacard treats each template's stat error
as independent, which is wrong here. To propagate this correctly you need the
full per-bin covariance matrix between all 6 templates, not just each one's
own variance.

`build_group_subsamples()` in `config.py` adds 21 extra subsamples per
dataset (one per unordered pair of the 6 EFT points, including the diagonal):
`cov_sm_sm`, `cov_sm_w1_cql32`, ..., `cov_w11_cql32_cqlm2_w11_cql32_cqlm2`.
Each computes `Sum(events.weight**2 * rwgt_i * rwgt_j)` per bin -- the runner
already multiplies a subsample's `weight_expr` by `events.weight` once, so
the config only needs to supply `events.weight * rwgt_i * rwgt_j`.

These are registered in `samples` like any other process, but with two flags:
- `is_variance: True` -- `post_process.py` now supports this flag (small,
  backward-compatible addition to `renorm()`/`single_post_process()`): a
  covariance term is quadratic in the per-event weight, so it needs
  `scale**2` normalization, not the usual linear `scale`. This has to happen
  *before* summing across the 8 differently-normalized mll bins, which is
  exactly what `post_process.py`'s per-sample xsec/sumw lookup already gives
  us for free.
- `exclude_from_datacard: True` -- `make_cards.py` now skips any sample with
  this flag (small, backward-compatible addition): the term still gets
  written into `histos.root` (so it survives the standard pipeline), but
  never becomes a process row in the datacard, per your requirement.
- `covariance_of: (name_i, name_j)` -- records which two templates this term
  is the covariance between. This is what lets the matrix-building tool below
  be generic: it discovers the whole matrix structure purely from these
  `samples` dict flags, and ignores anything else the config might contain
  (other backgrounds, unrelated processes, ...).

The diagonal (`cov_i_i`) is redundant with template `i`'s own histogram
variance (`hist.storage.Weight()` already accumulates `Sum(weight_i**2)`), but
is kept anyway for a uniform 21-term treatment and as a free consistency
check (see `--check` below).

After `spritz-postproc`, assemble the matrix with the package-level
`spritz-cov-matrix` tool (`spritz.scripts.build_covariance_matrix`) -- it
takes the config directory and a histos.root path explicitly, so it works
from anywhere, not just this config's own directory:

```bash
spritz-cov-matrix . histos.root -o covariance.root --check
```

`--check` prints `max|cov_i_i - histo_i.variances()|` for each template --
should be ~0, confirming the covariance terms are internally consistent with
the nominal templates. Region/variable default to this config's
`cards_regions`/`cards_variables` (override with `--region`/`--variable`).
The output is a 3D `hist.Hist` (x=mll, y/z=template name) at
`inc_mm/mll/covariance_matrix` in `covariance.root`, giving the full 6x6
correlated MC-stat covariance per mll bin.

## Pipeline

```bash
spritz-fileset
spritz-chunks
spritz-batch
# ... wait for condor jobs to finish (condor_q) ...
spritz-merge
spritz-postproc   # builds histos.root: 6 templates + 21 covariance terms
spritz-cards      # builds datacards/inc_mm/mll/: only the 6 templates
spritz-cov-matrix . histos.root -o covariance.root --check
```

`spritz-cards` (`make_cards.py`) was previously hardcoded to a different
analysis's region/variable names; it now reads optional `cards_regions`/
`cards_variables` config keys (falling back to the old hardcoded defaults if a
config doesn't set them), which is what lets this config drive it directly.

## Notes / things you may want to revisit

- Only "mll" is histogrammed (60 bins, 50-200 GeV) -- add more entries to
  `variables` in `config.py` for differential distributions; no other file
  needs to change to match, unlike the tree-based approach.
- `samples` marks `sm` as the sole background and the other 5 points as
  `is_signal: True` -- a reasonable default for building the EFT morphing,
  but reconsider if your fit expects something else.
- No lnN/shape systematics are defined, only `autoMCStats` (`nuisances["stat"]`).
- cqlm1 is deliberately absent from this config: one of the two mll-bin
  groups' reweight cards never probed it, so there's no honest way to build
  that template for those 3 datasets. cql32 was picked instead because both
  groups' cards actually include it.
- `scripts/dump_reweight.py` / `create_reweight_variables.py` are kept as
  utilities for (re-)deriving a card's index mapping from an actual
  `reweight_card.dat`, in case you need to redo this for a different operator
  pair or production.
