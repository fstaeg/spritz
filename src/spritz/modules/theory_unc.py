import awkward as ak

# max factor between a PDF / alpha_s member weight and the nominal, per event
# (see the PDF section below). Not applied to QCD scale and PS weights: their
# negative ratios are genuine (capping them moves the inclusive muR shift by ~2%)
PDF_WEIGHT_CAP = 5.0


def theory_unc(events, variations):
    # QCD Scale 
    if "HOScaleWeight" in ak.fields(events):
        # scale weights from higher order prediction (HO_reweight)
        scale_keys = ["muRmuF_down", "muR_down", "muF_down", "muRes_down", "muRes_up", "muF_up", "muR_up", "muRmuF_up"]
        for index, name in enumerate(scale_keys):
            events[f"weight_QCDScale_{name}"] = events.weight * events.HOScaleWeight[:, index]
            variations.register_variation(columns=["weight"], variation_name=f"QCDScale_{name}")
    
    elif "LHEScaleWeight" in ak.fields(events) and len(events.LHEScaleWeight) > 0:
        nVariations = len(events.LHEScaleWeight[0])
        #   8 weights: [0] muR=0.5 muF=0.5, [1] 0.5 1, [3] 1 0.5, [4] 1 2, [6] 2 1, [7] 2 2
        #   9 weights: [0] muR=0.5 muF=0.5, [1] 0.5 1, [3] 1 0.5, [5] 1 2, [7] 2 1, [8] 2 2
        #   18 weights: [0] muR=0.5 muF=0.5, [2] 0.5 1, [6] 1 0.5, [10] 1 2, [14] 2 1, [16] 2 2
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
