import awkward as ak
import numpy as np
import lhapdf

# DY MiNNLO NanoAODv9 samples only ship NNPDF31_nnlo_hessian_pdfas replicas in
# LHEPdfWeight, so swapping the nominal PDF set (e.g. to CT18Z) requires
# evaluating both PDF sets ourselves from the Generator-level parton kinematics.
_CVMFS_LHAPDF_PATH = "/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current"
_paths_set = False
_pdf_cache = {}

# In the thin sliver between two grids' differing heavy-flavor mass
# thresholds, one PDF can be a genuine (tiny but real) rising value while the
# other is still a denormalized near-zero -- individually finite and
# positive, so the earlier checks don't catch it, yet the ratio diverges
# continuously as the denominator shrinks towards its threshold. A single
# such event can carry a ratio of ~1e300+, which overflows the weighted sum
# over millions of events into inf. No swap of a heavy-flavor-inclusive PDF
# is physically a factor-of-100+ effect, so anything beyond that is capped
# back to the nominal (ratio 1.0) rather than left to blow up the sum.
_MAX_RATIO = 100.0


def _get_pdf(pdfset, member=0):
    global _paths_set
    if not _paths_set:
        lhapdf.setVerbosity(0)
        if _CVMFS_LHAPDF_PATH not in lhapdf.paths():
            lhapdf.pathsAppend(_CVMFS_LHAPDF_PATH)
        _paths_set = True
    key = (pdfset, member)
    if key not in _pdf_cache:
        _pdf_cache[key] = lhapdf.mkPDF(pdfset, member)
    return _pdf_cache[key]


def pdf_reweight(events, target_pdfset, source_pdfset="NNPDF31_nnlo_hessian_pdfas"):
    """Per-event ratio [xf(id1)*xf(id2)](target_pdfset) / [xf(id1)*xf(id2)](source_pdfset),
    evaluated at the Generator-level parton kinematics. Multiplying this into
    the nominal event weight swaps the sample's nominal PDF set.

    Generator_scalePDF is Q in GeV (despite its NanoAOD doc string reading
    "Q2 scale for PDF" -- verified against LHEPdfWeight replica ratios).
    Generator_xpdf1/2 are not used: they are unfilled (identically 0) in the
    DY MiNNLO samples, so both PDF sets are evaluated here instead of taking
    a ratio against the stored value.
    """
    target = _get_pdf(target_pdfset, 0)
    source = _get_pdf(source_pdfset, 0)

    id1 = np.asarray(events.Generator.id1)
    id2 = np.asarray(events.Generator.id2)
    x1 = np.asarray(events.Generator.x1)
    x2 = np.asarray(events.Generator.x2)
    q2 = np.asarray(events.Generator.scalePDF, dtype=np.float64) ** 2

    n = len(id1)
    ratio = np.empty(n)
    for i in range(n):
        i1, i2, xx1, xx2, qq2 = int(id1[i]), int(id2[i]), x1[i], x2[i], q2[i]
        f1 = source.xfxQ2(i1, xx1, qq2)
        f2 = source.xfxQ2(i2, xx2, qq2)
        g1 = target.xfxQ2(i1, xx1, qq2)
        g2 = target.xfxQ2(i2, xx2, qq2)
        # Near/below a heavy-flavor mass threshold the PDF is ~0 and can dip
        # slightly negative from spline interpolation in either grid; at
        # x==0 LHAPDF returns inf. Both must be checked individually first,
        # since e.g. two simultaneously-negative values multiply to a
        # spuriously positive, finite (but meaningless) product that would
        # otherwise slip past a check on the ratio alone.
        individually_ok = (
            np.isfinite(f1) and f1 > 0.0
            and np.isfinite(f2) and f2 > 0.0
            and np.isfinite(g1) and g1 > 0.0
            and np.isfinite(g2) and g2 > 0.0
        )
        if not individually_ok:
            ratio[i] = 1.0
            continue
        # Two individually-tiny-but-positive values can still multiply to a
        # denormal that underflows to exactly 0.0 (ZeroDivisionError), so
        # the division itself is guarded too. These are all a negligible
        # fraction of events with a physically tiny cross-section
        # contribution regardless -- leave the nominal weight untouched.
        try:
            r = (g1 * g2) / (f1 * f2)
        except ZeroDivisionError:
            r = np.nan
        valid_range = np.isfinite(r) and (1.0 / _MAX_RATIO) <= r <= _MAX_RATIO
        ratio[i] = r if valid_range else 1.0

    return ak.Array(ratio)
