import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools as it

def show_clustering(tsdf, time_series, subjects, labels):
    cmap = sns.color_palette("tab10")
    smap = {True: "solid", False: "dotted"}
    k = len(np.unique(labels))
    _, classes_count = np.unique(labels, return_counts=True)

    fig, axes = plt.subplots(k, 1, figsize=(8, 4 * k))
    counts = np.zeros((k,))

    for index, ts in enumerate(time_series):
        subject = subjects[index]
        label = labels[index]
        is_converter = tsdf.loc[subject, "converter"].values[0]

        counts[label] += int(is_converter)

        axes[label].plot(
            ts,
            color=cmap[label],
            alpha=0.5,
            linestyle=smap[is_converter]
        )


    for clust_idx in range(k):
        axes[clust_idx].set_ylim(0, 31)
        n = classes_count[clust_idx]
        converter_ratio = counts[clust_idx] / n
        axes[clust_idx].set_title(
            f"N = {n}, {converter_ratio * 100:.2f}% of converters"
        )

    fig.tight_layout()
    return fig


def plot_freqs(y):
    cat, c = np.unique(y, return_counts=True)
    fig, ax = plt.subplots()
    freqs = c / len(y)
    ax.bar(cat, freqs, color=mpl.colormaps["tab10"].colors)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(cat)))
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Frequency")
    return fig, ax