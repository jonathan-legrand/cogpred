"""
Grid search script for CNN on BOLD time series
"""

import os
from datetime import datetime
from pathlib import Path

import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import seaborn as sns

from neuroginius.atlas import Atlas
from skorch.callbacks import EpochScoring, EarlyStopping
from skorch.dataset import ValidSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GroupShuffleSplit, cross_val_score

from cogpred.utils.configuration import get_config
from cogpred.utils.naming import make_run_path
from cogpred.loading import TSFetcher, make_training_data
from cogpred.plotting import plot_ts
from cogpred.features import make_features, generate_single_sub
from cogpred.supervised import macro_f1
from cogpred.models import (
    WindowNetClassifier,
    BOLDCNN, 
    default_channel_func,
    fast_increase,
    slow_increase,
    initial_bump,
    count_parameters
)

# Define script constants
WIN_SIZE = 48
BATCH_SIZE = 512
k=3
N_ITER = 10
ATLAS = "schaefer200"

torch.manual_seed(1234)
np.random.seed(1234)

config = get_config()
conn_dir = config["connectivity_matrices"]

# Load data
labels = pd.read_csv(f"data/cluster_{k}_labels.csv", index_col=0)

run_path = make_run_path(
    config["output_dir"],
    k=k,
    feat="series",
    experimental=True,
    atlas=ATLAS,
    winsize=WIN_SIZE,
    batchsize=BATCH_SIZE,
    niter=N_ITER,
    stamp=str(datetime.now())[:-10].replace(" ", "-")
)
os.makedirs(run_path, exist_ok=True)

tspath = Path("/georges/memento/BIDS/derivatives/schaeffer200_merged_phenotypes")
atlas = Atlas.from_name("schaefer200")
fetcher = TSFetcher(tspath)

dmn_indexer = np.where(np.array(atlas.macro_labels) == "Default", True, False)

_, metadata = make_training_data(conn_dir, atlas.name, k)
rest_dataset = fetcher.rest_dataset

metadata = pd.merge(
    rest_dataset,
    metadata,
    how="inner",
    on=["NUM_ID", "ses"],
    validate="many_to_one",
    suffixes=[None, "_"]
)

print("Creating features with sliding wiwdows", end="...")
features = make_features(fetcher, metadata, dmn_indexer)

X, y, centre = [], [], []

for idx, X_i in enumerate(features):
    
    y_i = int(metadata.loc[idx, "cluster_label"])
    centre_i = metadata.loc[idx, "CEN_ANOM"]
    
    # Augment underepresented cases with smaller stride
    if y_i in {1, 2}:
        win_kwargs = dict(stride=2)
    else:
        win_kwargs = dict(stride=4)

    win_kwargs["window_size"] = WIN_SIZE
        
    windows_i, targets_i = generate_single_sub(X_i, y_i, **win_kwargs)
    
    X += windows_i
    y += targets_i
    centre += [centre_i] * len(targets_i)


X = torch.tensor(np.stack(X, axis=0)).transpose(1, 2)
y = torch.tensor(y)


print("Done")
print("Define and start grid search...")


f1_cb = EpochScoring(macro_f1, lower_is_better=False, name="macro_f1")
early_stopping = EarlyStopping(
    monitor="macro_f1",
    lower_is_better=False,
    patience=5,
    load_best=True
)

counts = y.unique(return_counts=True)[1]

net = WindowNetClassifier(
    BOLDCNN,
    module__n_channels=sum(dmn_indexer),# We want 1 channel per ROI
    module__window_size=WIN_SIZE,
    max_epochs=25, # That seems too much
    criterion=nn.CrossEntropyLoss,
    criterion__weight=1/counts,
    optimizer=torch.optim.AdamW,
    iterator_train__shuffle=True,
    callbacks=[f1_cb, early_stopping],
    device="cuda",
    warm_start=False,
    batch_size=BATCH_SIZE, # We can make it even bigger
    train_split=ValidSplit(cv=8)
)

# TODO Allow single n_filters
grid_params = dict(
    module__num_conv_blocks=[1, 2, 3, 4],
    module__num_fc_blocks=[1, 2, 3],
    #module__conv_k=[3, 5, 7],
    #module__channel_func=(
    #    default_channel_func,
    #    initial_bump,
    #    slow_increase,
    #    fast_increase
    #),
    optimizer__lr=np.geomspace(1e-5, 0.1, num=5),
    optimizer__weight_decay=np.geomspace(1e-5, 0.1, num=5)
)
# We can't have deep networks with higher pool_k
# We could try having another dict of shallow confs

gkf = GroupShuffleSplit(n_splits=5, random_state=1999)
cv = gkf.split(X, y, groups=centre)

search = RandomizedSearchCV(
    net,
    param_distributions=grid_params,
    n_iter=N_ITER,
    scoring=macro_f1,
    cv=cv,
    random_state=1999,
    verbose=1,
    n_jobs=8,
    error_score=np.nan, # There will be errors due to invalid architectures
    refit=True,
)

search.fit(X, y)

cv_results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")

# Export results
search.best_estimator_.save_params(f_params=run_path / "params.pkl")
metadata.to_csv(run_path / "metadata.csv")
 # We need best hyperparameters to re-instantiate the model
cv_results.to_csv(run_path / "cv_results.csv")

print(f"Results exported to {run_path}")


