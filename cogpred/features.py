import joblib
from nilearn.interfaces.fmriprep import load_confounds
from nilearn import signal

# 24 TR of 2.5s = 60s, which is a duration somewhat
# motivated by my exhaustive litterature review

cachedir = "/bigdata/jlegrand/joblib_cache"
memory = joblib.Memory(cachedir, verbose=0)

@memory.cache
def make_features(fetcher, metadata, indexer):
    features = []
    
    for i, row in metadata.iterrows():
        ts = fetcher.get_single_row(row)
        confounds, mask = load_confounds(row.file_path)
    
        ts = ts[:, indexer]
        ts = signal.clean(
            ts,
            confounds=confounds, 
            sample_mask=mask, 
            standardize="zscore_sample"
        )
    
        features.append(ts)

    return features

def generate_windows(segment, window_size=24, stride=6):
    for start in range(0, len(segment) - window_size, stride):
        yield segment[start:start+window_size]


def generate_single_sub(X, y, **win_kwargs):
    windows = []
    targets = []
    for window in generate_windows(X, **win_kwargs):
        windows.append(window)
        targets.append(y)
    return windows, targets