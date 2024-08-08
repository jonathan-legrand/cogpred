from scipy import stats
import numpy as np
from nilearn.connectome import vec_to_sym_matrix
from cogpred.matrices import compute_mat_size

from sklearn.metrics import make_scorer, f1_score, confusion_matrix

def run_cv_perms(estimator, matrices, metadata, cv):
    y = metadata.cluster_label.values.astype(int)
    scores = []
    maps = []

    for train_idx, test_idx in cv.split(matrices, y, groups=metadata.CEN_ANOM.values):
        X_train, y_train = matrices[train_idx], y[train_idx]
        X_test, y_test = matrices[test_idx], y[test_idx]
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)

        scores.append(
            f1_score(y_test, y_pred, average="macro")
        )

        reg = estimator.named_steps["classifier"]
        
        # This should be moved outisde the loop
        masker = estimator.named_steps["matrixmasker"]

        # Compute Haufe's transform to make coefs interpretable
        X = masker.transform(matrices)
        sigma_X = np.cov(X.T)
        W = reg.coef_.T
        patterns = sigma_X @ W

        maps.append(patterns)
    
    weights = np.stack(maps, axis=0)
    return scores, weights

# TODO Joblib that, I suppose there should be some very similar code in cross validate
# TODO Allow passing index to shuffle
def run_cv(estimator, matrices, metadata, cv):
    y = metadata.cluster_label.values.astype(int)
    n_classes = len(np.unique(y))
    scores = []
    maps = []
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for train_idx, test_idx in cv.split(matrices, y, groups=metadata.CEN_ANOM.values):
        X_train, y_train = matrices[train_idx], y[train_idx]
        X_test, y_test = matrices[test_idx], y[test_idx]
        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        cm += confusion_matrix(y_test, y_pred)

        scores.append(
            f1_score(y_test, y_pred, average="macro")
        )

        reg = estimator.named_steps["classifier"]
        
        # This should be moved outisde the loop
        masker = estimator.named_steps["matrixmasker"]
        l = len(masker.vec_idx_)
        n_regions = compute_mat_size(l)

        # Compute Haufe's transform to make coefs interpretable
        X = masker.transform(matrices)
        sigma_X = np.cov(X.T)
        W = reg.coef_.T
        patterns = sigma_X @ W

        maps.append(patterns)
    
    hmat = np.stack(maps, axis=0)
    hmat = vec_to_sym_matrix(
        hmat.transpose((0, 2, 1)), diagonal=np.zeros((cv.n_splits, n_classes, n_regions))
    )
    return scores, cm, hmat

macro_f1 = make_scorer(
    f1_score, average="macro"
)

def compute_CI(scores, alpha=0.05):
    s = scores.std()
    n = len(scores)
    mean = scores.mean()
    c = stats.t.ppf(1 - alpha / 2, df=n-1)

    length = (c * s) / np.sqrt(n)
    return mean, mean - length / 2, mean + length / 2