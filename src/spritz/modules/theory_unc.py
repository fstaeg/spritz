import awkward as ak

def theory_unc(events, variations):
    # QCD Scale 
    if "LHEScaleWeight" in ak.fields(events) and len(events.LHEScaleWeight) > 0:
        nVariations = len(events.LHEScaleWeight[0])

        if nVariations in [8, 9, 18]:
            if nVariations == 8:
                events["weight_QCDScale_muR_down"] = events.weight * events.LHEScaleWeight[:, 1]
                events["weight_QCDScale_muF_down"] = events.weight * events.LHEScaleWeight[:, 3]
                events["weight_QCDScale_muF_up"] = events.weight * events.LHEScaleWeight[:, 4]
                events["weight_QCDScale_muR_up"] = events.weight * events.LHEScaleWeight[:, 6]
            
            elif nVariations == 9:
                events["weight_QCDScale_muR_down"] = events.weight * events.LHEScaleWeight[:, 1]
                events["weight_QCDScale_muF_down"] = events.weight * events.LHEScaleWeight[:, 3]
                events["weight_QCDScale_muF_up"] = events.weight * events.LHEScaleWeight[:, 5]
                events["weight_QCDScale_muR_up"] = events.weight * events.LHEScaleWeight[:, 7]
            
            elif nVariations == 18:
                events["weight_QCDScale_muR_down"] = events.weight * events.LHEScaleWeight[:, 2]
                events["weight_QCDScale_muF_down"] = events.weight * events.LHEScaleWeight[:, 6]
                events["weight_QCDScale_muF_up"] = events.weight * events.LHEScaleWeight[:, 10]
                events["weight_QCDScale_muR_up"] = events.weight * events.LHEScaleWeight[:, 14]

            variations.register_variation(columns=["weight"], variation_name="QCDScale_muR_down")
            variations.register_variation(columns=["weight"], variation_name="QCDScale_muF_down")
            variations.register_variation(columns=["weight"], variation_name="QCDScale_muF_up")
            variations.register_variation(columns=["weight"], variation_name="QCDScale_muR_up")

    # PDF and alphaS variations
    if "LHEPdfWeight" in ak.fields(events) and len(events.LHEPdfWeight) > 0:
        nVariations = len(events.LHEPdfWeight[0])

        for i in range(nVariations):
            events[f"weight_PDFWeight_{i}"] = events.weight * events.LHEPdfWeight[:, i]
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
