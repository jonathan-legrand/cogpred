from scipy import stats
import numpy as np

from sklearn.metrics import make_scorer, f1_score

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