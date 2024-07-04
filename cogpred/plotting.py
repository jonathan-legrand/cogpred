import matplotlib.pyplot as plt

def plot_ts(ts):
    """
    ts: bold time series of shape (n_TR, n_regions)
    """
    from scipy.stats import zscore
    fig, ax = plt.subplots(figsize=(8, 4))
    ts = zscore(ts)
    ax.plot(ts)
    ax.set_xlabel("TR")
    ax.set_ylabel("Signal intensity")
    return fig

def plot_predictions(p):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p.T, label=["Stable", "Slow decline", "Steep decline"])
    ax.set_xlabel("TR")
    ax.set_ylabel("Model output")
    ax.set_ylim(0, 1)
    fig.legend()
    return fig