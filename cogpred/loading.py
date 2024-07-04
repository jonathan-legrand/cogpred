import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Lazy copy paste
def make_test_data(conn_dir, atlas, k, test_centre="RXE"):
    matrices = joblib.load(f"{conn_dir}/atlas-{atlas}_prediction/all_subs.joblib")
    metadata = pd.read_csv(f"{conn_dir}/atlas-{atlas}_prediction/balanced_all_subs.csv", index_col=0)
    labels = pd.read_csv(f"data/cluster_{k}_labels.csv", index_col=0)

    baseline_msk = (metadata.ses == "M000")
    if test_centre is not None:
        baseline_msk *= (metadata.CEN_ANOM == test_centre)
    
    metadata = metadata[baseline_msk]
    matrices = matrices[baseline_msk]

    metadata = metadata.merge(
        right=labels,
        how="left", # Preserves order of the left key
        on="NUM_ID",
        validate="many_to_one"
    )

    no_psych_mask = metadata.cluster_label.isna()
    print(
        f"Dropping {no_psych_mask.sum()} subjects because of lacking MMMSE"
    )

    metadata = metadata[np.logical_not(no_psych_mask)]
    matrices = matrices[np.logical_not(no_psych_mask)]

    return matrices, metadata

def make_training_data(conn_dir, atlas, k, test_centre="RXE"):
    matrices = joblib.load(f"{conn_dir}/atlas-{atlas}_prediction/all_subs.joblib")
    metadata = pd.read_csv(f"{conn_dir}/atlas-{atlas}_prediction/balanced_all_subs.csv", index_col=0)
    labels = pd.read_csv(f"data/cluster_{k}_labels.csv", index_col=0)

    baseline_msk = (metadata.ses == "M000")
    if test_centre is not None:
        baseline_msk *= (metadata.CEN_ANOM != test_centre)
    
    metadata = metadata[baseline_msk]
    matrices = matrices[baseline_msk]

    metadata = metadata.merge(
        right=labels,
        how="left", # Preserves order of the left key
        on="NUM_ID",
        validate="many_to_one"
    )

    no_psych_mask = metadata.cluster_label.isna()
    print(
        f"Dropping {no_psych_mask.sum()} subjects because of lacking MMMSE"
    )

    metadata = metadata[np.logical_not(no_psych_mask)]
    matrices = matrices[np.logical_not(no_psych_mask)]

    return matrices, metadata

def load_psychometry(fpath="P:/jlegrand/Documents/follow.csv"):
    df = pd.read_csv(
        fpath,
        encoding="unicode_escape",
        na_values="."
    )
    converters = df[df["CDRSCR"] > 0.5].NUM_ID.unique()
    df["converter"] = df.NUM_ID.isin(converters)
    df["RISCTOTRI"] = df["RISCTOTIM"] - df["RISCTOTRL"]
    return df

def load_uniq(fpath="P:/jlegrand/Documents/uniq_modif.csv"):
    df = pd.read_csv(
        fpath,
        encoding="unicode_escape",
    )
    return df

def load_phenotypes(fpath="P:/jlegrand/Documents/phenotypes.csv"):
    phenotypes =  pd.read_csv(
        fpath, index_col=0
    )
    phenotypes["NUM_ID"] = phenotypes["sub"].apply(lambda x: "SUBJ" + str(x).zfill(4))
    return phenotypes


class TSFetcher:
    """
    Simple dataset fetcher with no torch dependency
    """

    def __init__(self, input_dir:Path) -> None:
        self.input_dir = input_dir
        self.rest_dataset = pd.read_csv(
                    input_dir / "phenotypes.csv",
                    index_col=0,
                    parse_dates=True,
                    date_format="%Y/%m/%d",
            )

        with open(input_dir / "metadata.json", "r") as stream:
            config = json.load(stream)
        for k, w in config.items():
            setattr(self, k, w)

    def __getitem__(self, idx):
        row = self.rest_dataset.iloc[idx, :]
        return self.get_single_row(row)

    def get_single_row(self, row):
        ts = joblib.load(self.input_dir / f"time_series/{row.file_basename}")
        return ts

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        try:
            item = self.__getitem__(self.counter)
        except IndexError:
            raise StopIteration()
        except FileNotFoundError:
            print("Caching uncomplete, stop iterations")
            raise StopIteration()
        self.counter += 1
        return item