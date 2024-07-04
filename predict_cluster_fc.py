# %%
import joblib
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.plotting import plot_matrix
from nilearn.connectome.connectivity_matrices import sym_matrix_to_vec


# %%
from neuroginius.synthetic_data.generation import generate_topology, generate_topology_net_interaction
from neuroginius.atlas import Atlas


atlas = Atlas.from_name("schaefer200")

topology = generate_topology("Default", atlas.macro_labels)
networks = np.unique(atlas.macro_labels)

for network in networks:
    new = generate_topology_net_interaction(("Default", network), atlas.macro_labels)
    topology += new

topology = np.where(topology != 0, 1, 0)
plt.imshow(topology)
plt.show()

# %%
from cogpred.utils.configuration import get_config

config = get_config()
conn_dir = config["connectivity_matrices"]

# %%
ATLAS = "schaeffer200"
k = 3
matrices = joblib.load(f"{conn_dir}/atlas-{ATLAS}_prediction/all_subs.joblib")
metadata = pd.read_csv(f"{conn_dir}/atlas-{ATLAS}_prediction/balanced_all_subs.csv", index_col=0)
labels = pd.read_csv(f"data/cluster_{k}_labels.csv", index_col=0)


baseline_msk = (metadata.ses == "M000")
metadata = metadata[baseline_msk]
matrices = matrices[baseline_msk]

metadata = metadata.merge(
    right=labels,
    how="left", # Preserves order of the left key
    on="NUM_ID",
    validate="many_to_one"
)
# For semi supervised settings
#metadata.replace(
#    {"cluster_label": math.nan},
#    -1,
#    inplace=True
#)
no_psych_mask = metadata.cluster_label.isna()
print(
    f"Dropping {no_psych_mask.sum()} subjects because of lacking MMMSE"
)

metadata = metadata[np.logical_not(no_psych_mask)]
matrices = matrices[np.logical_not(no_psych_mask)]

# %%
# Set non default to 0, transform to vec, and extract non zero coefficients
matrices *= topology

X = []
for mat in matrices:
    vec = sym_matrix_to_vec(mat, discard_diagonal=True)
    vec_idx = np.flatnonzero(vec) # Should only be computed once
    X.append(vec[vec_idx])

X = np.stack(X)
y = metadata.cluster_label
assert len(X) == len(y)

# %%
from neuroginius.plotting import plot_matrix
i = 2
row = metadata.iloc[i]
fig, ax = plt.subplots(2, 1, figsize=(6, 8))

plot_matrix(matrices[i], atlas, axes=ax[0])

ax[1].scatter(np.arange(X.shape[1]), X[i], s=5)
ax[1].set_xlabel("Region inside mask")
ax[1].set_ylabel("Connectivity value")

fig.suptitle(f"{row.file_basename}")
fig.tight_layout()
plt.show()

# %%
from scipy import stats

#lambda_param = 1 / 1e-4
lambda_param = 1000
alpha_distribution = stats.expon(scale=1/lambda_param)
l1_ratio = stats.uniform(0, 1)
#l1_ratio = stats.beta(1.5, 3) # TODO Try higher l1_ratio with high n_comps
#l1_ratio = stats.expon(scale=1/7)
power_t = stats.norm(loc=0.5, scale=0.1)

x = np.linspace(-0.1, 1, 1000)
da = alpha_distribution.pdf(x)
dl = l1_ratio.pdf(x)
dt = power_t.pdf(x)


#plt.plot(x, da, label="alpha")
plt.plot(x, dl, label="l1 ratio")
plt.plot(x, dt, label="power_t")
plt.title("Continuous params prior distributions")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

n_features = X.shape[1]
n_comps = list(range(5, 75))
dc = np.ones((len(n_comps))) * 1 / len(n_comps)
plt.plot(n_comps, dc, label="n_components", marker="x")
plt.title("Discrete params prior distributions")

plt.legend()
plt.show()

# %%
param_dist = {
    "classifier__loss": ["hinge", "log_loss", "modified_huber"],
    "classifier__alpha": alpha_distribution,
    "classifier__l1_ratio": l1_ratio,
    "classifier__power_t": power_t,
    "reduction__n_components": n_comps
}

# %%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.model_selection import (
    cross_validate, cross_val_predict, KFold, RandomizedSearchCV, StratifiedKFold
)

from sklearn.decomposition import PCA
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import f1_score
from functools import partial

from sklearn.model_selection import cross_val_predict, KFold, RandomizedSearchCV, cross_val_score
from cogpred.supervised import macro_f1

sgd = SGDClassifier(
    penalty="elasticnet",
    class_weight="balanced", 
    random_state=1999
)

clf = Pipeline(
    [
    ("scaler", preprocessing.StandardScaler()),
    ("reduction", PCA()),
    ("classifier", sgd)
    ],
    verbose=False
)

inner_cv = StratifiedKFold(n_splits=8, shuffle=True, random_state=1999)
outer_cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=1999)


search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=100,
    scoring=macro_f1,
    cv=inner_cv,
    random_state=1999,
    verbose=1,
    n_jobs=8,
    error_score="raise"
)


# %%
search.fit(X, y)
cv_results = pd.DataFrame(search.cv_results_).sort_values(by="mean_test_score", ascending=False)
cv_results

# %%
pca = search.best_estimator_.named_steps["reduction"]
vis = pd.DataFrame(pca.transform(X))
vis["label"] = y
sns.scatterplot(vis, x=0, y=1, hue="label", palette="husl", s=14)
plt.show()
sns.lineplot(
    cv_results,
    x="param_reduction__n_components",
    y="mean_test_score"
)
plt.show()

# %%
cv_results.param_classifier__l1_ratio = cv_results.param_classifier__l1_ratio.astype(float)

# %%
sns.regplot(cv_results, x="param_classifier__l1_ratio", y="mean_test_score")

# %%
from matplotlib import cm

x = cv_results.param_classifier__l1_ratio.values
y = cv_results.param_reduction__n_components.values
z = cv_results.mean_test_score.values
yy, xx = np.meshgrid(y, x)


fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_trisurf(x, y, z, cmap=cm.inferno, linewidth=0.2)
plt.xlabel("l1 ratio")
plt.ylabel("#components")
ax.set_zlabel("Classifier score")
ax.set_title("Randomized grid search")


# %%
test_scores = cross_val_score(
    search, X, y, cv=outer_cv, n_jobs=8, scoring=macro_f1, verbose=1
) # TODO Get hyperparameters

# %%
test_scores

# %%
import joblib
joblib.dump(test_scores, f"output/{ATLAS}_DMN_{k}_test_scores.joblib")

# %%
cv = StratifiedKFold(n_splits=3)
y_pred = cross_val_predict(search, X, y, n_jobs=-1, cv=cv)

# %%
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y, y_pred)
acc = accuracy_score(y, y_pred)
plt.title(f"{acc:.2f}")
plt.show()

# %%
from matplotlib.backends.backend_pdf import PdfPages
t2 = matrices[y == 2]
t0 = matrices[y == 0]
with PdfPages("output/stable.pdf") as pdf:
    for mat in t0[1::50]:
        plot_matrix(mat, vmin=-1, vmax=1)
        pdf.savefig()
        plt.close()

with PdfPages("output/declining.pdf") as pdf:
    for mat in t2[1::5]:
        plot_matrix(mat, vmin=-1, vmax=1)
        pdf.savefig()
        plt.close()



