import numpy as np
from joblib import Memory
from sklearn.base import TransformerMixin, BaseEstimator, OneToOneFeatureMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from neuroginius.atlas import Atlas
from neuroginius.synthetic_data.generation import generate_topology, generate_topology_net_interaction
from neuroginius.plotting import plot_matrix
from nilearn.connectome.connectivity_matrices import sym_matrix_to_vec

mem = Memory("/tmp/masker_cache", verbose=0)

@mem.cache
def _transform(matrices, vec_idx):
        X = []
        for mat in matrices:
            vec = sym_matrix_to_vec(mat, discard_diagonal=True)
            X.append(vec[vec_idx])

        X = np.stack(X)
        return X

@mem.cache
def _make_topology(refnet, interaction, macro_labels):

    n_regions = len(macro_labels)
    topology_ = np.zeros((n_regions, n_regions))

    for net_a in refnet:
        topology_ += generate_topology(net_a, macro_labels)
        for net_b in interaction:
            topology_ += generate_topology_net_interaction(
                (net_a, net_b), macro_labels
            )

    topology_ = np.where(topology_ != 0, 1, 0)
    vectop = sym_matrix_to_vec(topology_, discard_diagonal=True)
    vec_idx_ = np.nonzero(vectop)[0]

    return topology_, vec_idx_
    

from typing import Iterable

# TODO Inverse transform to show full matrix
class MatrixMasker(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, refnet:Iterable, interaction:Iterable, atlas:Atlas=None):
        if atlas is None:
            atlas = Atlas.from_name("schaefer200")
        self.atlas = atlas
        self.refnet = refnet
        self.interaction = interaction

    def fit(self, matrices, y=None):

        self.n_regions_ = matrices.shape[1]

        self.topology_, self.vec_idx_ = _make_topology(
            self.refnet, self.interaction, self.atlas.macro_labels
        )

        return self

    def transform(self, matrices):
        return _transform(matrices, self.vec_idx_)

    def plot(self, **kwargs):
        check_is_fitted(self)
        axes = plot_matrix(self.topology_, self.atlas, **kwargs)
        axes.set_title(f"MatrixMasker, {self.refnet}-{self.interaction}")
        return axes
        