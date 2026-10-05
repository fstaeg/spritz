import os
import sys
from textwrap import dedent
from copy import deepcopy

import pickle
import numpy as np
import uproot

from spritz.framework.framework import get_analysis_dict
from spritz.utils.plotting_utils import add_to_samples


def get_datacard_header(bin_name, data_integral):
    return dedent(f"""\n
    ## Shape input card
    imax 1 number of channels
    jmax * number of background
    kmax * number of nuisance parameters
    ----------------------------------------------------------------------------------------------------
    bin         {bin_name}
    observation {data_integral}
    shapes  *           * shapes.root     histo_$PROCESS histo_$PROCESS_$SYSTEMATIC
    shapes  data_obs           * shapes.root     histo_Data
    """)


def make_datacard(
    input_file,
    region,
    variable,
    nuisances,
    samples,
    fakes_dict,
    covariance_file=None,
):
    output_path = f"datacards/{region}/{variable}"

    os.makedirs(output_path, exist_ok=True)
    output_file = uproot.recreate(f"{output_path}/shapes.root")

    covariance_written = False
    if covariance_file is not None:
        cov_key = f"{region}/{variable}/covariance_matrix"
        if cov_key in covariance_file:
            output_file["covariance_matrix"] = covariance_file[cov_key].to_hist()
            covariance_written = True
        else:
            print(f"Warning: {cov_key} not found in covariance file, skipping")

    sig_idx = 0
    bkg_idx = 1

    bin_name = f"{region}_{variable}"
    rows = [
        ["bin"],
        ["process"],
        ["process"],
        ["rate"],
    ]
    systs = {}

    h_data = 0
    enable_stat = False
    extra_lines = []
    if covariance_written:
        # points CMSHistErrorPropagator at MC-stat covariance matrix built by spritz-cov-matrix
        extra_lines.append(f"{bin_name} autoMCCorr shapes.root covariance_matrix")

    if fakes_dict is not None:
        if region in [r["target"] for r in fakes_dict.get("regions", [])]:
            for nuis in fakes_dict.get("nuisances", []):
                if "samples" in nuisances[nuis]:
                    nuisances[nuis]["samples"] = add_to_samples(nuisances[nuis]["samples"], "Fakes")

            samples = {"Fakes": {}} | samples

    for sample_name in samples:
        if samples[sample_name].get("exclude_from_datacard", False):
            continue
        
        name = samples[sample_name].get("name", sample_name)
        is_signal = samples[sample_name].get("is_signal", False)
        is_data = samples[sample_name].get("is_data", False)
        is_smeft = samples[sample_name].get("is_smeft", False)
        noStat = samples[sample_name].get("noStat", False)

        if is_smeft and not covariance_written:
            noStat = True

        final_name = f"{region}/{variable}/histo_{sample_name}"
        h = input_file[final_name].copy()

        if is_signal:
            idx = sig_idx
            sig_idx -= 1
        else:
            idx = bkg_idx
            bkg_idx += 1

        if is_data:
            h_data = h.copy()

        if is_data and name != "Data":
            raise Exception("Cannot use is_data with a name != 'Data'")

        if noStat:
            histo_view = h.view(True)
            histo_view.variance = np.zeros_like(histo_view.variance)

        output_file[f"histo_{name}"] = h
        if is_data:
            continue

        rows[0].append(bin_name)
        rows[1].append(name)
        rows[2].append(str(idx))
        rows[3].append(str(np.sum(h.values(True))))

        for systematic in nuisances:
            if nuisances[systematic]["type"] == "auto":
                enable_stat = True
                continue

            if nuisances[systematic]["type"] == "rateParam":
                if (
                    "samples" in nuisances[systematic]
                    and sample_name not in nuisances[systematic]["samples"]
                ):
                    continue
                if (
                    "cuts" in nuisances[systematic]
                    and region not in nuisances[systematic]["cuts"]
                ):
                    print(region, nuisances[systematic]["cuts"])
                    continue
                extra_lines.append(
                    f'{nuisances[systematic]["name"]} rateParam '
                    f'{bin_name} {sample_name} '  # FIXME should use region?
                    f'{nuisances[systematic]["samples"][sample_name]}'
                )
                continue

            if sample_name in nuisances[systematic]["samples"]:
                nuis_name = nuisances[systematic]["name"]
                if nuisances[systematic]["type"] == "lnN":
                    syst = nuisances[systematic]["samples"][sample_name]
                else:
                    syst = "1.0"
                    for tag in ["Up", "Down"]:
                        _final_name = final_name + f"_{nuis_name}{tag}"
                        _h = input_file[_final_name].copy()
                        output_file[f"histo_{name}_{nuis_name}{tag}"] = _h
            else:
                syst = "-"

            if systematic not in systs:
                systs[systematic] = [nuisances[systematic]["type"], syst]
            else:
                systs[systematic].append(syst)

    if isinstance(h_data, int):
        h_data = h.copy()
        histo_view = h_data.view(True)
        histo_view.value = np.zeros_like(histo_view.value)
        histo_view.variance = np.zeros_like(histo_view.variance)
        output_file["histo_Data"] = h_data

    datacard = get_datacard_header(bin_name, np.sum(h_data.values(True)))
    for row in rows:
        datacard += "\t".join(row) + "\n"
    datacard += "-" * 100 + "\n"
    for syst in systs:
        datacard += nuisances[syst]["name"] + "\t" + "\t".join(systs[syst]) + "\n"
    if enable_stat:
        extra_lines.append(f"{bin_name} autoMCStats 10 0 1")
    for line in extra_lines:
        datacard += line + "\n"

    with open(f"{output_path}/datacard.txt", "w") as file:
        file.write(datacard)


def main():
    analysis_dict = get_analysis_dict()
    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]
    fakes_dict = analysis_dict.get("fakes_dict")

    with open("histos.pkl", "rb") as f:
        fin = pickle.load(f)

    default_good_regions = [
        f"{region}_{cat}"
        for region in ["sr_inc", "dypu_cr", "top_cr"]
        for cat in ["ee", "mm"]
    ]
    default_good_variables = ["detajj_fits", "dnn_ptll", "MET_fits"]

    # override via cards_regions/cards_variables in config
    good_regions = analysis_dict.get("cards_regions", default_good_regions)
    good_variables = analysis_dict.get("cards_variables", default_good_variables)

    # covariance-matrix file (built by spritz-cov-matrix)
    covariance_file = None
    covariance_file_path = analysis_dict.get("covariance_file", None)
    if covariance_file_path is not None:
        if os.path.exists(covariance_file_path):
            covariance_file = uproot.open(covariance_file_path)
        else:
            print(
                f"Warning: covariance_file '{covariance_file_path}' not found, "
                "skipping covariance linking (run spritz-cov-matrix first)"
            )

    for region in good_regions:
        for variable in good_variables:
            if variable not in variables or "axis" not in variables[variable]:
                continue
            make_datacard(
                fin,
                region,
                variable,
                nuisances,
                samples,
                fakes_dict,
                covariance_file=covariance_file,
            )


if __name__ == "__main__":
    main()
