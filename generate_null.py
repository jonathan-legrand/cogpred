import sys
from pathlib import Path
import argparse
import warnings
from cogpred.loading import make_training_data
from cogpred.utils.configuration import get_config

import pandas as pd
import numpy as np
import joblib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

import random
from cogpred.supervised import run_cv_perms
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
from cogpred.transformers import MatrixMasker
from neuroginius.atlas import Atlas
from dask.distributed import Client, progress
from cogpred.utils.naming import make_run_path
                       

config = get_config()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate empirical null distribution with permutations")

    parser.add_argument(
        "--n_permutations",
        help="Number of point in the null distribution",
        type=int,
        default=50
    )
    
    parser.add_argument(
        "--n_jobs",
        help="Number of parallel processes for fitting on permuted data",
        type=int,
        default=10
    )
    
    parser.add_argument(
        "--seed",
        help="Custom seed for scan selection when there are multiple scans per subject",
        type=int,
        default=config["seed"]
    )
    parser.add_argument(
        "--atlas",
        help="Name of fc atlas",
        type=str,
        default="schaefer200"
    )
    return parser


def generate_null_dask(
    matrices:np.array,
    metadata:pd.DataFrame,
    client:Client,
    N:int=100,
    seed:int=1234,
    atlas_name:str="schaefer200"
    ):
    
    random.seed(seed)
    
    idx_range = list(range(len(matrices)))
    permutation_scheme = [
        random.sample(idx_range, k=len(idx_range)) for _ in range(N)
    ]

    atlas = Atlas.from_name(atlas_name, soft=False)
    REFNET = np.unique(atlas.macro_labels)
    INTER = REFNET

    net = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        max_iter=1000,
        random_state=1999,
    )
    clf = Pipeline(
        [
        ("matrixmasker", MatrixMasker(REFNET, INTER, atlas=atlas)),
        ("scaler", preprocessing.StandardScaler()),
        ("classifier", net)
        ],
        verbose=False
    )

    def single_call(permutation):
        p_matrices = matrices[permutation]
        p_metadata = metadata.iloc[permutation, :].reset_index(drop=True)

        outer_cv = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=2024)
        test_scores, hmat = run_cv_perms(clone(clf), p_matrices, p_metadata, outer_cv)
        return test_scores, hmat

    
    print(client)

    futures = client.map(single_call, permutation_scheme)
    progress(futures, notebook=False)
    permuted_res = [future.result() for future in futures]

    return permuted_res, permutation_scheme


def generate_and_export(
    n_permutations,
    n_jobs,
    seed,
    atlas
    ):
    conn_dir = config["connectivity_matrices"]
    matrices, metadata = make_training_data(conn_dir, atlas, 3, test_centre=None)

    with SLURMCluster(
        cores=1,
        memory="10GB",
        walltime="00:10:00",
        log_directory="/tmp/dask"
    ) as cluster:
        cluster.scale(n_jobs)
        client = Client(cluster)
        print(client.dashboard_link)
        permuted_res, permutation_scheme = generate_null_dask(
            matrices, metadata, client, N=n_permutations, seed=seed, atlas_name=atlas
        )
    print(permuted_res)
    print(permutation_scheme)

    run_path = make_run_path(
        config["output_dir"],
        k=3,
        feat="fc",
        atlas=atlas,
        net="all",
    )
    joblib.dump(
        run_path / permuted_res[0], "scores.joblib"
    )
    joblib.dump(
        run_path / permuted_res[1], "weights.joblib"
    )
    joblib.dump(
        run_path / permutation_scheme, "permutation_scheme.joblib"
    )
    print(f"Permutations exported in {run_path}")
    

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    generate_and_export(*vars(args).values())