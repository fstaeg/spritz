# Merged config, written by spritz-merge-configs on 2026-10-13 09:44
# Input configs, in merge order (later ones update earlier ones):
#   smeftatnlo: /gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-split-test/smeftatnlo/config.py -> sources/smeftatnlo.py
#   bkg_data: /gwpool/users/gboldrini/spritz-fabian/configs/dy-eft-split-test/bkg_data/config.py -> sources/bkg_data.py
# Every merged name is built explicitly below: edit freely.
#
# Merge notes (what a later config replaced):
#   other names taken from the last config defining them, the configs differ: ['all_samples', 'mc_samples']
#   njobs: taken from bkg_data, the configs differ
import os
from spritz.scripts.merge_configs import (
    load_config, update_dicts, merge_entries, merge_fakes, merge_renorm,
)

_here = os.path.dirname(os.path.abspath(__file__))
_cfg = {
    'smeftatnlo': load_config(os.path.join(_here, 'sources/smeftatnlo.py')),
    'bkg_data': load_config(os.path.join(_here, 'sources/bkg_data.py')),
}

# taken from one config (the last that defines it)
year = _cfg['bkg_data'].year
year_label = _cfg['bkg_data'].year_label
lumi = _cfg['bkg_data'].lumi
lumi_unc = _cfg['bkg_data'].lumi_unc
runner = _cfg['bkg_data'].runner
njobs = _cfg['bkg_data'].njobs
eft_chunks_per_job = _cfg['smeftatnlo'].eft_chunks_per_job
special_analysis_cfg = _cfg['bkg_data'].special_analysis_cfg
regions = _cfg['bkg_data'].regions
variables = _cfg['bkg_data'].variables
cards_regions = _cfg['bkg_data'].cards_regions
cards_variables = _cfg['bkg_data'].cards_variables
covariance_file = _cfg['bkg_data'].covariance_file
split_uncertainties = _cfg['bkg_data'].split_uncertainties

# merged over all configs
datasets = update_dicts(_cfg['smeftatnlo'].datasets, _cfg['bkg_data'].datasets)
samples = update_dicts(_cfg['smeftatnlo'].samples, _cfg['bkg_data'].samples)
colors = update_dicts(_cfg['smeftatnlo'].colors, _cfg['bkg_data'].colors)
nuisances = merge_entries(_cfg['smeftatnlo'].nuisances, _cfg['bkg_data'].nuisances)
corrections = merge_entries(
    _cfg['smeftatnlo'].corrections,
    _cfg['bkg_data'].corrections,
)
fakes_dict = merge_fakes(_cfg['bkg_data'].fakes_dict)
renorm_samples = merge_renorm(_cfg['smeftatnlo'].renorm_samples)

# every other name of the configs, from the last that defines it:
# change or drop what is not needed
combinations_with_replacement = _cfg['smeftatnlo'].combinations_with_replacement
cmap_petroff = _cfg['bkg_data'].cmap_petroff
get_fw_path = _cfg['bkg_data'].get_fw_path
get_rw_idx = _cfg['smeftatnlo'].get_rw_idx
get_eft_points = _cfg['smeftatnlo'].get_eft_points
eft_operators = _cfg['smeftatnlo'].eft_operators
get_group = _cfg['smeftatnlo'].get_group
fw_path = _cfg['bkg_data'].fw_path
file = _cfg['bkg_data'].file
lumis = _cfg['bkg_data'].lumis
model = _cfg['smeftatnlo'].model
file_key = _cfg['smeftatnlo'].file_key
eft_bins = _cfg['smeftatnlo'].eft_bins
operators = _cfg['smeftatnlo'].operators
eft_points = _cfg['smeftatnlo'].eft_points
cov_pairs = _cfg['smeftatnlo'].cov_pairs
point_samples = _cfg['smeftatnlo'].point_samples
all_samples = _cfg['bkg_data'].all_samples
mc_samples = _cfg['bkg_data'].mc_samples
st_datasets_without_pdf = _cfg['bkg_data'].st_datasets_without_pdf
preselections = _cfg['bkg_data'].preselections
cos_theta_star = _cfg['bkg_data'].cos_theta_star
cmap_pastel = _cfg['bkg_data'].cmap_pastel
nnlo_n3ll_qcd = _cfg['bkg_data'].nnlo_n3ll_qcd
nlo_ew = _cfg['bkg_data'].nlo_ew
dy_bins = _cfg['bkg_data'].dy_bins
ggll_bins = _cfg['bkg_data'].ggll_bins
dataset = _cfg['bkg_data'].dataset
samples_data = _cfg['bkg_data'].samples_data
era = _cfg['bkg_data'].era

# your changes
