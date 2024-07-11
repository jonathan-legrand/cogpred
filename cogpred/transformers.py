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

# TODO Accept list of interactions?
# TODO Inverse transform to show full matrix
class MatrixMasker(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, refnet:str, interaction:str, atlas:Atlas=None):
        if atlas is None:
            atlas = Atlas.from_name("schaefer200")
        self.atlas = atlas
        self.refnet = refnet
        self.interaction = interaction

    def fit(self, matrices, y=None):
        topology_ = generate_topology(self.refnet, self.atlas.macro_labels)
        topology_ += generate_topology_net_interaction(
            (self.refnet, self.interaction), self.atlas.macro_labels
        )
        self.topology_ = np.where(topology_ != 0, 1, 0)
        vectop = sym_matrix_to_vec(self.topology_, discard_diagonal=True)
        self.vec_idx_ = np.nonzero(vectop)[0]

        return self

    def transform(self, matrices):
        return _transform(matrices, self.vec_idx_)

    def plot(self, **kwargs):
        check_is_fitted(self)
        axes = plot_matrix(self.topology_, self.atlas, bounds=(0, 1), **kwargs)
        axes.set_title(f"MatrixMasker, {self.refnet}-{self.interaction}")
        return axes
        