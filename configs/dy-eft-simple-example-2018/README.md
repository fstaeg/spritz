# DY EFT simple example (2018)

Minimal 2-operator (cql32, cpl2) SMEFT reweighting example on the 8 mll-binned
2018 `DYMuMu_NLO_EFT_SMEFTatNLO_*_startingOne` samples, producing a datacard
with 6 processes: `sm`, `w1_cql32`, `wm1_cql32`, `w1_cpl2`, `wm1_cpl2`,
`w11_cql32_cpl2`.

cql32 and cpl2 were chosen because both mll-bin groups' reweight cards
actually probe them (see below). For the same setup scaled up to every
operator the NLO cards probe in all 8 mll slices (16 operators, 153
templates), see [`../dy-eft-full-smeftatnlo-2018`](../dy-eft-full-smeftatnlo-2018).

Every spritz step below is a standard `spritz-*` command -- there are no
bespoke post-processing scripts. The EFT reweighting is built directly into
histograms by the runner.

## How the EFT reweighting works

`config.py` sets `runner = runner_3DY_eft_full_morphing_megahisto.py` and hands
it, per dataset, a structured `eft_reweighting` dict:

```python
{
    "weight_branch": "LHEReweightingWeight",
    "points": {"sm": 0, "wm1_cql32": 7, ...},        # point name -> column
    "covariance_pairs": [("sm", "sm"), ("sm", "w1_cql32"), ...],
    "n_weights": 406,                                 # expected branch width
}
```

Every point shares the same event selection (all events) and differs only in
which column of the 2D `LHEReweightingWeight` matrix multiplies the nominal
weight, so the runner computes all point and covariance weights as a few
vectorized numpy operations and fills them into a handful of batch
histograms -- instead of one `eval()` + `fill()` per name, which is what the
old `subsamples`-based `runner_3DY_eft_reweight.py` did. The output is
bit-for-bit the same as the old runner's (checked on a Group A and a Group B
chunk). `subsamples` is still the right mechanism for genuine event
selections (e.g. splitting a sample into Z->ee and Z->mumu); it is just not
used here.

The two groups of mll bins were generated with different reweight cards, so
each dataset carries its own point -> column mapping (`chunks.py` copies every
`datasets[dataset]` key straight into that dataset's chunk kwargs):

| group | datasets | card width | sm | cql32_m1 / cql32 | cpl2_m1 / cpl2 | cql32_cpl2 |
|---|---|---|---|---|---|---|
| A | mll200_400, 400_600, 600_800, 800_1000, 1500_inf | 406 | 0 | 7 / 8 | 27 / 28 | 139 |
| B | mll50_120, 120_200, 1000_1500 | 153 | 0 | 3 / 4 | 13 / 14 | 52 |

`n_weights` makes the runner fail (as a chunk error, listed by
`spritz-checkerrors`) any file whose branch has a different width than its
card: the column indices only mean something for that layout. This is not
hypothetical -- a survey of all 27,501 input files
(`../dy-eft-full-smeftatnlo-2018/scripts/survey_files.py`) found 92 (0.34%)
whose branch has fewer columns than their card, anywhere from 0 to 396.

`eft_operator_indices.py` has the point -> column tables for both cards,
restricted to the 16 operators both probe (same names in both groups, only the
columns differ). Both cards share one layout: `sm`, then
(`wm1_<op>`, `w1_<op>`) for each operator in card order, then one
`w11_<opi>_<opj>` per operator pair in `itertools.combinations` order.
`scripts/dump_reweight.py` / `create_reweight_variables.py` re-derive a
card's mapping from an actual `reweight_card.dat`.

## Summing across the 8 mll bins into one shape per EFT point

This is standard `post_process.py` behavior: `config.py`'s `samples` dict
groups all 8 `{dataset}_{point}` combinations sharing the same point name into
one entry (e.g. `samples["sm"]["samples"]` lists all 8 `..._sm` names).
`post_process.py` sums that list, each weighted by its own xsec/sumw/lumi. The
runner stores a dataset's EFT points as one combined entry;
`post_process.py` unpacks it back into per-`{dataset}_{point}` histograms
(`expand_eft_combined`) and gives each the raw dataset's xsec, unaffected by
which point it is.

## Correlated MC-stat uncertainties (the covariance matrix)

All 6 templates come from the *same* underlying MC events, just reweighted
differently -- their statistical fluctuations are strongly correlated, not
independent. `autoMCStats` in the datacard treats each template's stat error
as independent, which is wrong here. To propagate this correctly you need the
full per-bin covariance matrix between all 6 templates, not just each one's
own variance.

`config.py`'s `covariance_pairs` lists one pair per unordered pair of the 6
points (including the diagonal), 21 in total; for each the runner computes
`Sum(events.weight**2 * rwgt_i * rwgt_j)` per bin -- the per-bin MC-stat
covariance between templates i and j (nominal variation only).

These are registered in `samples` like any other process, but with:
- `is_variance: True` -- `post_process.py` normalizes a covariance term by
  `scale**2`, not the usual linear `scale` (it is quadratic in the per-event
  weight). This happens *before* summing across the 8 differently-normalized
  mll bins, which is exactly what the per-sample xsec/sumw lookup gives.
- `exclude_from_datacard: True` -- `make_cards.py` writes the term into
  `histos.root` but never turns it into a datacard process row.
- `covariance_of: (name_i, name_j)` -- which two templates this term is the
  covariance between; this lets `spritz-cov-matrix` discover the matrix
  structure purely from these `samples` flags.

The diagonal (`cov_i_i`) is redundant with template `i`'s own histogram
variance (`hist.storage.Weight()` already accumulates `Sum(weight_i**2)`) but
is kept as a free consistency check (see `--check` below).

After `spritz-postproc`, assemble the matrix with:

```bash
spritz-cov-matrix . histos.root -o covariance.root --check
```

`--check` prints `max|cov_i_i - histo_i.variances()|` for each template --
this should be ~0 relative to the values (float32 rounding aside). Region and
variable default to this config's `cards_regions`/`cards_variables` (override
with `--region`/`--variable`). The output is a 3D `hist.Hist` (x=mll, y/z=
template name) at `inc_mm/mll/covariance_matrix` in `covariance.root`, the
full 6x6 correlated MC-stat covariance per mll bin. `spritz-cards` copies it
into each datacard's `shapes.root` (config key `covariance_file`) and adds the
`autoMCCorr` line that points the CombinedLimit fork at it.

## Pipeline

```bash
spritz-fileset
spritz-chunks       # re-run whenever config.py changes: chunk kwargs embed the config
spritz-batch
# ... wait for condor jobs to finish (condor_q) ...
spritz-merge
spritz-postproc     # builds histos.root: 6 templates + 21 covariance terms
spritz-cov-matrix . histos.root -o covariance.root --check
spritz-cards        # builds datacards/inc_mm/mll/: only the 6 templates
```

For the fit itself (workspace, scans, plots) run `spritz-setup-eft-morphing`
once in this directory to check out CMSSW + the CombinedLimit fork +
AnalyticAnomalousCoupling + the scan tools, then follow `run_pipeline.sh`.
`spritz-cards` reads the optional `cards_regions`/`cards_variables` config keys
(falling back to the old hardcoded defaults if a config doesn't set them).

## Notes / things you may want to revisit

- Only "mll" is histogrammed (variable binning, 50-3000 GeV) -- add more
  entries to `variables` in `config.py` for differential distributions; no
  other file needs to change.
- `samples` marks `sm` as the sole background and the other 5 points as
  `is_signal: True` -- a reasonable default for building the EFT morphing,
  but reconsider if your fit expects something else.
- No lnN/shape systematics beyond a flat 2% lumi lnN are defined, only
  `autoMCStats` (`nuisances["stat"]`).
- Only operators present in both reweight cards can be used across all 8
  datasets (Group B's card probes 16 of Group A's 27); cqlm1, for instance, has
  no template for the three Group B datasets, so there is no honest way to
  build one.
