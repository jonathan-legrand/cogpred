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
from sklearn.model_selection import GroupKFold
from cogpred.transformers import MatrixMasker
from neuroginius.atlas import Atlas
from dask.distributed import Client, progress
from cogpred.utils.naming import make_run_path
                       

config = get_config()

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate empirical null distribution with permutations")

    parser.add_argument(
        "refnet",
        help="Yeo's net to use as a reference",
        type=str,
        default="Default"
    )

    parser.add_argument(
        "inter",
        help="Yeo's net to use as an interaction",
        type=str,
        default="Default"
    )
    
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
    refnet,
    inter,
    matrices:np.array,
    metadata:pd.DataFrame,
    client:Client,
    N:int=100,
    seed:int=1234,
    atlas_name:str="schaefer200"
    ):
    
    random.seed(seed)
    
    idx_range = list(range(len(matrices)))
    print("Generating permutation scheme...", end="")
    permutation_scheme = [
        random.sample(idx_range, k=len(idx_range)) for _ in range(N)
    ]
    print("Done")

    atlas = Atlas.from_name(atlas_name, soft=False)

    warnings.warn("Change classififier to match actual model")
    net = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        max_iter=3000,
        random_state=2024,
    )

    clf = Pipeline(
        [
        ("matrixmasker", MatrixMasker(refnet, inter, atlas=atlas)),
        ("scaler", preprocessing.StandardScaler()),
        ("classifier", net)
        ],
        verbose=False
    )


    def single_call(permutation):
        p_metadata = metadata.iloc[permutation]
        outer_cv = GroupKFold(n_splits=8)
        test_scores, hmat = run_cv_perms(clone(clf), matrices, p_metadata, outer_cv)
        return test_scores, hmat

    
    print(client)

    futures = client.map(single_call, permutation_scheme, batch_size=100)
    #progress(futures, notebook=False)

    permuted_res = []
    for future in futures:
        permuted_res.append(future.result())

    return permuted_res, permutation_scheme


def generate_and_export(
    refnet,
    inter,
    n_permutations,
    n_jobs,
    seed,
    atlas
    ):
    conn_dir = config["connectivity_matrices"]
    matrices, metadata = make_training_data(conn_dir, atlas, 3, test_centre=None)
    metadata = metadata.loc[:, ["cluster_label", "CEN_ANOM"]]

    if refnet == "all" and inter == "all":
        refnet = np.unique(atlas.macro_labels)
        inter = refnet

    # TODO Do we need that much memory per worker?
    with SLURMCluster(
        cores=1,
        memory="10GB",
        walltime="10:00:00",
        log_directory="/tmp/dask"
    ) as cluster:
        cluster.scale(n_jobs)
        client = Client(cluster)
        print(client.dashboard_link)
        permuted_res, permutation_scheme = generate_null_dask(
            refnet, inter, matrices, metadata, client, N=n_permutations, seed=seed, atlas_name=atlas
        )

    run_path = make_run_path(
        config["output_dir"],
        k=3,
        feat="fc",
        atlas=atlas,
        net=refnet,
        inter=inter
    )

    if len(run_path.name) > 55:
        print("too long")
        run_path = make_run_path(
        config["output_dir"],
        k=3,
        feat="fc",
        atlas=atlas,
        net="all",
    )
    
    joblib.dump(
        permuted_res, run_path / f"{n_permutations}_permutations_res.joblib"
    )
    
    print(f"Permutations exported in {run_path}")
    

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    generate_and_export(*vars(args).values())