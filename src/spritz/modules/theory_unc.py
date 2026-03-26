import awkward as ak

def theory_unc(events, variations):
    # QCD Scale
    if 'LHEScaleWeight' in ak.fields(events):
        nVariations = len(events.LHEScaleWeight[0])
        for i in range(nVariations):
            events[f"weight_QCDScale_{i}"] = events.weight * events.LHEScaleWeight[:, i]
            variations.register_variation(
                columns=["weight"], variation_name=f"QCDScale_{i}"
            )

    # Pdf Weights
    if 'LHEPdfWeight' in ak.fields(events):
        nVariations = len(events.LHEPdfWeight[0])
        for i in range(nVariations):
            events[f"weight_PDFWeight_{i}"] = events.weight * events.LHEPdfWeight[:, i]
            variations.register_variation(
                columns=["weight"], variation_name=f"PDFWeight_{i}"
            )

    # PS Weights
    if 'PSWeight' in ak.fields(events) and len(events.PSWeight)>0:
        if len(events.PSWeight[0])==4:
            for i in range(4):
                events[f"weight_PSWeight_{i}"] = events.weight * events.PSWeight[:, i]
                variations.register_variation(
                    columns=["weight"], variation_name=f"PSWeight_{i}"
                )
    
    return events, variations
