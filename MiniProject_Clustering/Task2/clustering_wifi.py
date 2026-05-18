from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

RANDOM_STATE = 42
MISSING_RSS = -100.0          # padding value for absent BSSIDs (dBm floor)
K_MIN, K_MAX = 2, 10          # sweep range for k
KMEANS_N_INIT = 20            # restarts to dodge local optima
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


@dataclass
class FingerprintDataset:
    """Encapsulates one prepared dataset."""
    name: str                       # e.g. "DataSetKMeans1"
    X: np.ndarray                   # (n_samples, n_features) RSS matrix
    fin_ids: np.ndarray             # finLabel for each row of X
    y_true: np.ndarray              # ground-truth RoomLabel for each row
    bssids: List[str]               # column ordering for X
    raw: pd.DataFrame = field(repr=False)


def load_dataset(csv_path: Path, missing: float = MISSING_RSS) -> FingerprintDataset:

    df = pd.read_csv(csv_path, encoding="latin-1")


    rss_matrix = (
        df.pivot_table(
            index="finLabel",
            columns="BSSIDLabel",
            values="RSSLabel",
            aggfunc="mean",
        )
        .sort_index()
    )

    fp_to_room = (
        df.drop_duplicates("finLabel")
        .set_index("finLabel")["RoomLabel"]
        .reindex(rss_matrix.index)
    )

    X = rss_matrix.fillna(missing).to_numpy(dtype=np.float64)
    return FingerprintDataset(
        name=csv_path.stem,
        X=X,
        fin_ids=rss_matrix.index.to_numpy(),
        y_true=fp_to_room.to_numpy(dtype=int),
        bssids=list(rss_matrix.columns),
        raw=df,
    )

# Davies–Bouldin Index
def davies_bouldin_assignment(X: np.ndarray, labels: np.ndarray) -> float:

    labels = np.asarray(labels)
    unique = np.unique(labels)
    k = unique.size
    if k < 2:
        return float("nan")

    avg = np.zeros(k)
    centroids = np.zeros((k, X.shape[1]))
    for idx, c in enumerate(unique):
        members = X[labels == c]
        centroids[idx] = members.mean(axis=0)
        if members.shape[0] >= 2:
            # pdist returns the condensed vector of pairwise distances.
            avg[idx] = pdist(members, metric="euclidean").mean()
        else:
            avg[idx] = 0.0  

    cen_dist = squareform(pdist(centroids, metric="euclidean"))

    db_per_cluster = np.zeros(k)
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j or cen_dist[i, j] == 0.0:
                continue
            ratios.append((avg[i] + avg[j]) / cen_dist[i, j])
        db_per_cluster[i] = max(ratios) if ratios else 0.0

    return float(db_per_cluster.mean())

# Sweep k and collect metrics
@dataclass
class SweepResult:
    k_values: List[int]
    dbi_assignment: List[float]
    dbi_sklearn: List[float]
    silhouette: List[float]
    inertia: List[float]
    ari: List[float]                  # vs ground-truth rooms
    nmi: List[float]                  # vs ground-truth rooms
    labels_per_k: Dict[int, np.ndarray]
    centroids_per_k: Dict[int, np.ndarray]

    def best_k(self) -> int:
        """k that minimises the assignment-defined DBI."""
        return self.k_values[int(np.argmin(self.dbi_assignment))]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "k": self.k_values,
                "DBI (assignment)": self.dbi_assignment,
                "DBI (sklearn)": self.dbi_sklearn,
                "Silhouette": self.silhouette,
                "Inertia": self.inertia,
                "ARI (vs rooms)": self.ari,
                "NMI (vs rooms)": self.nmi,
            }
        )


def sweep_kmeans(
    ds: FingerprintDataset,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    random_state: int = RANDOM_STATE,
) -> SweepResult:
    """Run KMeans for every k in [k_min, k_max] and score it three ways."""
    res = SweepResult(
        k_values=list(range(k_min, k_max + 1)),
        dbi_assignment=[],
        dbi_sklearn=[],
        silhouette=[],
        inertia=[],
        ari=[],
        nmi=[],
        labels_per_k={},
        centroids_per_k={},
    )

    for k in res.k_values:
        km = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            random_state=random_state,
        ).fit(ds.X)
        labels = km.labels_

        res.dbi_assignment.append(davies_bouldin_assignment(ds.X, labels))
        res.dbi_sklearn.append(davies_bouldin_score(ds.X, labels))
        res.silhouette.append(silhouette_score(ds.X, labels))
        res.inertia.append(float(km.inertia_))
        res.ari.append(float(adjusted_rand_score(ds.y_true, labels)))
        res.nmi.append(float(normalized_mutual_info_score(ds.y_true, labels)))
        res.labels_per_k[k] = labels
        res.centroids_per_k[k] = km.cluster_centers_

    return res

# Plotting helpers

def plot_metric_curves(
    results: Dict[str, SweepResult],
    out_path: Path,
) -> None:
    """DBI (custom + sklearn) and Silhouette vs k for every dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Row 0: DBI curves; row 1: silhouette.
    for col, (ds_name, res) in enumerate(results.items()):
        ax = axes[0, col]
        ax.plot(
            res.k_values,
            res.dbi_assignment,
            marker="o",
            linewidth=2.5,
            label="DBI (assignment formula)",
            color="#C0392B",
        )
        ax.plot(
            res.k_values,
            res.dbi_sklearn,
            marker="s",
            linewidth=2,
            linestyle="--",
            label="DBI (sklearn baseline)",
            color="#2980B9",
        )
        best = res.best_k()
        ax.axvline(best, color="green", alpha=0.4, linestyle=":")
        ax.scatter(
            [best],
            [min(res.dbi_assignment)],
            color="green",
            s=180,
            zorder=5,
            label=f"best k = {best}",
        )
        ax.set_title(f"{ds_name}\nDavies–Bouldin Index vs. k")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel("DBI  (lower is better)")
        ax.legend(fontsize=11)

        ax = axes[1, col]
        ax.plot(
            res.k_values,
            res.silhouette,
            marker="^",
            linewidth=2.5,
            color="#27AE60",
        )
        best_sil_k = res.k_values[int(np.argmax(res.silhouette))]
        ax.axvline(best_sil_k, color="green", alpha=0.4, linestyle=":")
        ax.set_title(f"{ds_name}\nSilhouette vs. k")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel("Silhouette  (higher is better)")
        ax.text(
            0.97,
            0.05,
            f"best k = {best_sil_k}",
            transform=ax.transAxes,
            ha="right",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="green", alpha=0.85),
        )

    fig.suptitle(
        "Internal-index sweep over k",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_mds(
    datasets: Dict[str, FingerprintDataset],
    sweeps: Dict[str, SweepResult],
    out_path: Path,
    random_state: int = RANDOM_STATE,
) -> None:
    """2-D MDS embedding coloured by (a) true room and (b) cluster at best k."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)

    for col, (name, ds) in enumerate(datasets.items()):

        mds = MDS(
            n_components=2,
            random_state=random_state,
            metric_mds=True, #metric=True for older versions of scikit
            normalized_stress="auto",
            n_init=1,
            init="classical_mds",
        )
        emb = mds.fit_transform(ds.X)

        # (a) coloured by true room
        ax = axes[0, col]
        rooms = np.unique(ds.y_true)
        palette = sns.color_palette("Set1", n_colors=len(rooms))
        for r, c in zip(rooms, palette):
            mask = ds.y_true == r
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                color=c,
                edgecolor="black",
                linewidth=0.4,
                s=70,
                label=f"Room {r}",
                alpha=0.85,
            )
        ax.set_title(f"{name}\nMDS coloured by true room")
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.legend(fontsize=11, loc="best")

        # (b) coloured by predicted cluster at best k
        best_k = sweeps[name].best_k()
        labels = sweeps[name].labels_per_k[best_k]
        ax = axes[1, col]
        palette = sns.color_palette("tab10", n_colors=best_k)
        for cl, c in zip(np.unique(labels), palette):
            mask = labels == cl
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                color=c,
                edgecolor="black",
                linewidth=0.4,
                s=70,
                label=f"Cluster {cl}",
                alpha=0.85,
            )
        ax.set_title(f"{name}\nMDS coloured by KMeans cluster (k={best_k})")
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.legend(fontsize=11, loc="best")

    fig.suptitle(
        "MDS projection — true labels vs. KMeans assignments",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    datasets: Dict[str, FingerprintDataset],
    sweeps: Dict[str, SweepResult],
    out_path: Path,
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax, (name, ds) in zip(axes, datasets.items()):
        best_k = sweeps[name].best_k()
        labels = sweeps[name].labels_per_k[best_k]
        ct = pd.crosstab(
            pd.Series(ds.y_true, name="True room"),
            pd.Series(labels, name="Predicted cluster"),
        )
        ari = adjusted_rand_score(ds.y_true, labels)
        nmi = normalized_mutual_info_score(ds.y_true, labels)
        sns.heatmap(
            ct,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            ax=ax,
            xticklabels=[f"C{c}" for c in ct.columns],
            yticklabels=[f"Room {r}" for r in ct.index],
        )
        ax.set_title(
            f"{name}\n"
            f"best k = {best_k}   |   ARI = {ari:.3f}   NMI = {nmi:.3f}"
        )
        ax.set_xlabel("Predicted cluster")
        ax.set_ylabel("True room")

    fig.suptitle(
        "Contingency table — KMeans assignment vs. ground-truth room",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_best_vs_true_k(
    datasets: Dict[str, FingerprintDataset],
    sweeps: Dict[str, SweepResult],
    out_path: Path,
    true_k: int = 4,
) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    for col, (name, ds) in enumerate(datasets.items()):
        for row, k_choice_label in enumerate(("best DBI k", f"true k = {true_k}")):
            k = sweeps[name].best_k() if row == 0 else true_k
            labels = sweeps[name].labels_per_k[k]
            ct = pd.crosstab(
                pd.Series(ds.y_true, name="True room"),
                pd.Series(labels, name="Predicted cluster"),
            )
            ari = adjusted_rand_score(ds.y_true, labels)
            nmi = normalized_mutual_info_score(ds.y_true, labels)
            sns.heatmap(
                ct,
                annot=True,
                fmt="d",
                cmap="Blues" if row == 0 else "Greens",
                cbar=True,
                ax=axes[row, col],
                xticklabels=[f"C{c}" for c in ct.columns],
                yticklabels=[f"Room {r}" for r in ct.index],
            )
            axes[row, col].set_title(
                f"{name}  —  {k_choice_label}  (k = {k})\n"
                f"ARI = {ari:.3f}     NMI = {nmi:.3f}"
            )
            axes[row, col].set_xlabel("Predicted cluster")
            axes[row, col].set_ylabel("True room")

    fig.suptitle(
        "Contingency tables: DBI-optimal k vs. ground-truth k",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_inertia(
    results: Dict[str, SweepResult],
    out_path: Path,
) -> None:
    """Classic elbow curve as supporting evidence."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    palette = {"DataSetKMeans1": "#E67E22", "DataSetKMeans2": "#16A085"}
    for name, res in results.items():
        ax.plot(
            res.k_values,
            res.inertia,
            marker="o",
            linewidth=2.5,
            label=name,
            color=palette.get(name, None),
        )
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("KMeans inertia (within-cluster SSE)")
    ax.set_title("Elbow plot")
    ax.legend()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

# console pretty printer

def print_summary(name: str, ds: FingerprintDataset, res: SweepResult) -> None:
    print(f"\n=== {name} ===")
    print(f"  fingerprints (samples): {ds.X.shape[0]}")
    print(f"  BSSIDs (features):      {ds.X.shape[1]}")
    print(f"  true rooms:             {len(np.unique(ds.y_true))}")
    print(f"  density (non-padded):   "
          f"{(ds.X != MISSING_RSS).mean()*100:.2f}%")
    print()
    print(res.to_dataframe().to_string(index=False, float_format="%.4f"))
    best = res.best_k()
    print(f"\n  >>> best k by assignment-DBI: {best}")
    print(f"      DBI(assignment) = {min(res.dbi_assignment):.4f}")
    idx = res.k_values.index(best)
    print(f"      DBI(sklearn)    = {res.dbi_sklearn[idx]:.4f}")
    print(f"      Silhouette      = {res.silhouette[idx]:.4f}")

# Orchestrator

def run(
    csv_paths: List[Path],
    out_dir: Path = OUT_DIR,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> Tuple[Dict[str, FingerprintDataset], Dict[str, SweepResult]]:
    datasets: Dict[str, FingerprintDataset] = {}
    sweeps: Dict[str, SweepResult] = {}

    for path in csv_paths:
        ds = load_dataset(path)
        sweep = sweep_kmeans(ds, k_min=k_min, k_max=k_max)
        datasets[ds.name] = ds
        sweeps[ds.name] = sweep
        print_summary(ds.name, ds, sweep)
        sweep.to_dataframe().to_csv(
            out_dir / f"metrics_{ds.name}.csv", index=False
        )

    plot_metric_curves(sweeps, out_dir / "fig1_metric_curves.png")
    plot_mds(datasets, sweeps, out_dir / "fig2_mds.png")
    plot_confusion_matrices(datasets, sweeps, out_dir / "fig3_confusion.png")
    plot_best_vs_true_k(datasets, sweeps, out_dir / "fig4_k_comparison.png")
    plot_inertia(sweeps, out_dir / "fig5_inertia.png")

    return datasets, sweeps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[
            str(DATA_DIR / "DataSetKMeans1.csv"),
            str(DATA_DIR / "DataSetKMeans2.csv"),
        ],
        help="Paths to CSV datasets."
    )

    parser.add_argument(
        "--k-min",
        type=int,
        default=K_MIN
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=K_MAX
    )

    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory for output figures and CSVs."
    )

    args = parser.parse_args()

    csv_paths = [Path(p).resolve() for p in args.csvs]
    out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== PATH CONFIGURATION ===")
    print(f"Script directory : {BASE_DIR}")
    print(f"Data directory   : {DATA_DIR}")
    print(f"Output directory : {out_dir}")
    print()

    print("CSV files:")
    for p in csv_paths:
        print(f"  - {p}")

    print()

    missing_files = [p for p in csv_paths if not p.exists()]

    if missing_files:
        print("ERROR: Missing dataset files:\n")

        for p in missing_files:
            print(f"  - {p}")

        raise FileNotFoundError(
            "\nOne or more CSV dataset files could not be found."
        )

    run(
        csv_paths=csv_paths,
        out_dir=out_dir,
        k_min=args.k_min,
        k_max=args.k_max,
    )
