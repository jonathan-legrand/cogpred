
from pathlib import Path
import os
import nibabel as nib
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
func_filenames = metadata.groupby("CEN_ANOM")["file_path"].sample(n=1, random_state=1234).to_list()

clean_kwargs = {
    "clean__strategy": ["high_pass", "wm_csf", "motion", "global_signal"]
}

N_COMPONENTS = 80

mask = MultiNiftiMasker(
    smoothing_fwhm=6.0,
    mask_strategy="whole-brain-template",
    standardize="zscore_sample",
    n_jobs=10,
    verbose=1,
    memory_level=2,
    **clean_kwargs
)

dict_learning = DictLearning(
    #n_components=80,
    n_components=N_COMPONENTS, # Debug config
    mask=mask,
    memory="nilearn_cache",
    memory_level=2,
    random_state=1234,
    verbose=1,
)

dict_learning.fit(func_filenames)

components_img_ = dict_learning.components_img_

extractor = RegionExtractor(
	maps_img=components_img_,
	extractor='local_regions'
)
extractor.fit()
regions_extracted_img = extractor.regions_img_
regions_index = extractor.index_
n_regions_extracted = regions_extracted_img.shape[-1]


run_path = Path(config["output_dir"]) / "parcellations" / make_run_name(
    ncomponents=N_COMPONENTS,
    nregions=n_regions_extracted,
)
os.makedirs(run_path, exist_ok=True)

nib.save(regions_extracted_img, run_path / "parcellation.nii.gz")

plot_prob_atlas(
    regions_extracted_img, view_type="filled_contours"
)
plt.savefig(run_path / "parcellation.png")
