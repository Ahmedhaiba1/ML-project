"""


Runs K-Means, Agglomerative (Ward, on a sample) and GMM with simple default
settings (k = 3) and records unsupervised metrics. No labels (IsCanceled) are used.
The helper functions here are reused by the tuning step.
"""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from cluster_data import OUTPUT_DIR, RANDOM_STATE, prepare_features

SILHOUETTE_SAMPLE = 10_000   # silhouette is O(n^2): computed on a fixed sample
HIERARCHICAL_SAMPLE = 10_000  # agglomerative is run on a fixed random subset
BASELINE_K = 3


# ---------------------------------------------------------------- helpers
def sample_indices(n_rows: int, size: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """Fixed random subset of row positions (reproducible)."""
    rng = np.random.RandomState(seed)
    size = min(size, n_rows)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


def evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette (on a sample), Davies-Bouldin (full) and cluster sizes."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    if n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "cluster_sizes": sizes,
            "smallest_cluster_pct": np.nan,
        }

    idx = sample_indices(len(X), SILHOUETTE_SAMPLE)
    sil = silhouette_score(X[idx], labels[idx])
    db = davies_bouldin_score(X, labels)
    smallest_pct = 100 * min(sizes.values()) / len(labels)
    return {
        "n_clusters": n_clusters,
        "silhouette": round(float(sil), 4),
        "davies_bouldin": round(float(db), 4),
        "cluster_sizes": sizes,
        "smallest_cluster_pct": round(smallest_pct, 2),
    }


# ---------------------------------------------------------------- baselines
def run_kmeans(X: np.ndarray) -> dict:
    start = time.time()
    model = KMeans(n_clusters=BASELINE_K, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    result = evaluate_labels(X, labels)
    result.update(
        algorithm="K-Means",
        settings=f"k={BASELINE_K}, n_init=10",
        rows_used=len(X),
        train_time_s=round(time.time() - start, 2),
    )
    return result


def run_agglomerative(X: np.ndarray) -> dict:
    idx = sample_indices(len(X), HIERARCHICAL_SAMPLE)
    X_sub = X[idx]
    start = time.time()
    model = AgglomerativeClustering(n_clusters=BASELINE_K, linkage="ward")
    labels = model.fit_predict(X_sub)
    result = evaluate_labels(X_sub, labels)
    result.update(
        algorithm="Agglomerative",
        settings=f"k={BASELINE_K}, linkage=ward",
        rows_used=len(X_sub),
        train_time_s=round(time.time() - start, 2),
    )
    return result


def run_gmm(X: np.ndarray) -> dict:
    start = time.time()
    model = GaussianMixture(
        n_components=BASELINE_K, covariance_type="diag", random_state=RANDOM_STATE
    )
    labels = model.fit_predict(X)
    result = evaluate_labels(X, labels)
    result.update(
        algorithm="GMM",
        settings=f"components={BASELINE_K}, covariance=diag",
        rows_used=len(X),
        train_time_s=round(time.time() - start, 2),
        bic=round(float(model.bic(X)), 1),
    )
    return result


def run_baselines() -> pd.DataFrame:
    _, X, _, _ = prepare_features()
    rows = [run_kmeans(X), run_agglomerative(X), run_gmm(X)]

    table = pd.DataFrame(rows)[
        [
            "algorithm",
            "settings",
            "rows_used",
            "n_clusters",
            "silhouette",
            "davies_bouldin",
            "smallest_cluster_pct",
            "train_time_s",
            "cluster_sizes",
        ]
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    return table


if __name__ == "__main__":
    results = run_baselines()
    print("\nBASELINE CLUSTERING RESULTS (k = 3, untuned)")
    print("=" * 60)
    print(results.drop(columns=["cluster_sizes"]).to_string(index=False))
    print("\nCluster sizes:")
    for _, row in results.iterrows():
        print(f"  {row['algorithm']}: {row['cluster_sizes']}")
    print(f"\nSaved to: {OUTPUT_DIR / 'baseline_results.csv'}")