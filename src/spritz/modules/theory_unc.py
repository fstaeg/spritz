import awkward as ak

# max factor between a PDF / alpha_s member weight and the nominal, per event
# (see the PDF section below). Not applied to QCD scale and PS weights: their
# negative ratios are genuine (capping them moves the inclusive muR shift by ~2%)
PDF_WEIGHT_CAP = 5.0


def theory_unc(events, variations):
    # QCD Scale 
    if "LHEScaleWeight" in ak.fields(events) and len(events.LHEScaleWeight) > 0:
        nVariations = len(events.LHEScaleWeight[0])

        # LHEScaleWeight index of each (muR, muF) point we use: muR and muF
        # varied alone (QCDScale_muR_*, QCDScale_muF_*) and together
        # (QCDScale_muRmuF_*). The anti-correlated (0.5, 2) and (2, 0.5) are
        # never read (marked "x" below).
        #   8 weights (no nominal): [0] muR=0.5 muF=0.5, [1] 0.5 1, [2] 0.5 2 x, [3] 1 0.5,
        #                           [4] 1 2, [5] 2 0.5 x, [6] 2 1, [7] 2 2
        #   9 weights: [0] muR=0.5 muF=0.5, [1] 0.5 1, [2] 0.5 2 x, [3] 1 0.5, [4] 1 1 (nominal),
        #              [5] 1 2, [6] 2 0.5 x, [7] 2 1, [8] 2 2
        #   18 weights: the 9-weight layout at the even indices (4 and 12 x)
        scale_indices = {
            8: {"muRmuF_down": 0, "muR_down": 1, "muF_down": 3, "muF_up": 4, "muR_up": 6, "muRmuF_up": 7},
            9: {"muRmuF_down": 0, "muR_down": 1, "muF_down": 3, "muF_up": 5, "muR_up": 7, "muRmuF_up": 8},
            18: {"muRmuF_down": 0, "muR_down": 2, "muF_down": 6, "muF_up": 10, "muR_up": 14, "muRmuF_up": 16},
        }

        if nVariations in scale_indices:
            for name, index in scale_indices[nVariations].items():
                events[f"weight_QCDScale_{name}"] = events.weight * events.LHEScaleWeight[:, index]
                variations.register_variation(columns=["weight"], variation_name=f"QCDScale_{name}")

    # PDF and alphaS variations
    if "LHEPdfWeight" in ak.fields(events) and len(events.LHEPdfWeight) > 0:
        nVariations = len(events.LHEPdfWeight[0])

        # LHEPdfWeight is the per-event ratio to the nominal PDF. Near x -> 1 the
        # nominal PDF vanishes and the ratios of single events explode (MiNNLO DY
        # above 1 TeV: from -1237 to +1204, negative for 3% of the events above
        # 2 TeV), while the LO PDF ratio of the same events stays within ~[0.24, 1.3].
        # Cap each ratio to a factor PDF_WEIGHT_CAP from the nominal: it changes the
        # PDF / alpha_s shifts at low mass by < 0.03% and removes the outliers that
        # dominate the high-mass bins. Bins where positive and negative weights
        # cancel (n_eff << N) stay unstable; no per-event cap can fix that.
        pdf_ratio = ak.where(events.LHEPdfWeight > PDF_WEIGHT_CAP, PDF_WEIGHT_CAP, events.LHEPdfWeight)
        pdf_ratio = ak.where(pdf_ratio < 1 / PDF_WEIGHT_CAP, 1 / PDF_WEIGHT_CAP, pdf_ratio)

        for i in range(nVariations):
            events[f"weight_PDFWeight_{i}"] = events.weight * pdf_ratio[:, i]
            variations.register_variation(columns=["weight"], variation_name=f"PDFWeight_{i}")

    # PS Weights
    if "PSWeight" in ak.fields(events) and len(events.PSWeight) > 0:
        nVariations = len(events.PSWeight[0])

        if nVariations == 4:
            events["weight_PSWeight_ISR_up"] = events.weight * events.PSWeight[:, 0]
            events["weight_PSWeight_FSR_up"] = events.weight * events.PSWeight[:, 1]
            events["weight_PSWeight_ISR_down"] = events.weight * events.PSWeight[:, 2]
            events["weight_PSWeight_FSR_down"] = events.weight * events.PSWeight[:, 3]

            variations.register_variation(columns=["weight"], variation_name="PSWeight_ISR_up")
            variations.register_variation(columns=["weight"], variation_name="PSWeight_FSR_up")
            variations.register_variation(columns=["weight"], variation_name="PSWeight_ISR_down")
            variations.register_variation(columns=["weight"], variation_name="PSWeight_FSR_down")
    
    return events, variations
