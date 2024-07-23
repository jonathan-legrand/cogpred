# %%

from pathlib import Path
import os
import nibabel as nib
import pandas as pd
from  joblib import dump
import matplotlib.pyplot as plt

from nilearn.signal import clean
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import MultiNiftiMasker
from nilearn.decomposition import DictLearning
from nilearn.plotting import plot_prob_atlas
from nilearn.regions import RegionExtractor

from cogpred.utils.naming import make_run_name
from cogpred.loading import make_training_data
from cogpred.utils.configuration import get_config

config = get_config()
conn_dir = config["connectivity_matrices"]
ATLAS = "msdl" # We don't really care about the atlas
k = 3
_, metadata = make_training_data(conn_dir, ATLAS, k)
func_filenames = metadata["file_path"].sample(n=100, random_state=1234).to_list()
# %%

clean_kwargs = {
    "clean__strategy": ["high_pass", "wm_csf", "motion"]
}

N_COMPONENTS = 10

from nilearn.image import binarize_img, resample_to_img
mask_img = nib.load(Path(config["data_dir"]) / "default_association-test_z_FDR_0.01.nii.gz")
mask_img = resample_to_img(mask_img, func_filenames[0])
mask_img = binarize_img(mask_img, 0.1, two_sided=False)
# TODO Resample mask image to the images' resolution
# %%

mask = MultiNiftiMasker(
    mask_img,
    smoothing_fwhm=6.0,
    mask_strategy="epi",
    standardize="zscore_sample",
    verbose=1,
    memory_level=2,
    **clean_kwargs
)

dict_learning = DictLearning(
    n_components=N_COMPONENTS,
    mask=mask,
    memory="nilearn_cache",
    memory_level=2,
    random_state=1234,
    verbose=1,
    n_jobs=10
)

dict_learning.fit(func_filenames)

components_img_ = dict_learning.components_img_

# %%
extractor = RegionExtractor(
	maps_img=components_img_,
	extractor='local_regions',
    min_region_size=900, # in mm^3
)
extractor.fit()
regions_extracted_img = extractor.regions_img_
regions_index = pd.Series(extractor.index_)
n_regions_extracted = regions_extracted_img.shape[-1]

plot_prob_atlas(
    regions_extracted_img, view_type="filled_contours"
)

#%%
run_path = Path(config["output_dir"]) / "parcellations" / make_run_name(
    ncomponents=N_COMPONENTS,
    nregions=n_regions_extracted,
    gsr=False
)
os.makedirs(run_path, exist_ok=True)

nib.save(regions_extracted_img, run_path / "parcellation.nii.gz")
regions_index.to_csv(run_path / "networks.csv")

plot_prob_atlas(
    regions_extracted_img, view_type="filled_contours"
)
plt.savefig(run_path / "parcellation.png")
print(f"Run saved in {run_path}")

# %%