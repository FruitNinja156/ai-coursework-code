from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent #for portable file handling

def _read_map_image(path: str | Path) -> np.ndarray:

    return np.asarray(Image.open(path).convert("RGB"))


EARTH_RADIUS_KM = 6371.0


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    lon1, lat1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    lon2, lat2 = np.radians(b[:, 0]), np.radians(b[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def pairwise_haversine_km(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:

    p = points[:, None, :]
    c = centroids[None, :, :]
    lon1, lat1 = np.radians(p[..., 0]), np.radians(p[..., 1])
    lon2, lat2 = np.radians(c[..., 0]), np.radians(c[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

@dataclass
class KMeansResult:
    centroids: np.ndarray            
    labels: np.ndarray               
    sq_dists: np.ndarray            
    sse: float                      


def _random_centroids(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return lo + (hi - lo) * rng.random((k, points.shape[1]))


def kmeans(
    points: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_haversine_km,
    max_iter: int = 100,
) -> KMeansResult:

    centroids = _random_centroids(points, k, rng)
    labels = np.full(len(points), -1, dtype=int)

    for _ in range(max_iter):
        d = distance(points, centroids)              # (m, k)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            members = points[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
            else:
                centroids[j] = points[rng.integers(len(points))]

    final_d = distance(points, centroids)
    sq = final_d[np.arange(len(points)), labels] ** 2
    return KMeansResult(centroids, labels, sq, float(sq.sum()))


def kmeans_best_of(
    points: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    n_init: int = 10,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_haversine_km,
) -> KMeansResult:
    """Run K-Means `n_init` times with different seeds and keep the best SSE."""
    best: Optional[KMeansResult] = None
    for _ in range(n_init):
        res = kmeans(points, k, rng=rng, distance=distance)
        if best is None or res.sse < best.sse:
            best = res
    assert best is not None
    return best

@dataclass
class BisectingResult:
    centroids: np.ndarray            # (k, 2)
    labels: np.ndarray               # (m,)
    sq_dists: np.ndarray             # (m,) squared distance to assigned centroid
    sse_history: List[float] = field(default_factory=list)  # SSE after each split

    @property
    def sse(self) -> float:
        return float(self.sq_dists.sum())


def bisecting_kmeans(
    points: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_haversine_km,
    n_init_inner: int = 5,
    verbose: bool = True,
) -> BisectingResult:

    if k < 1:
        raise ValueError("k must be >= 1")

    m = len(points)
    labels = np.zeros(m, dtype=int)

    centroid0 = points.mean(axis=0, keepdims=True)
    centroids: np.ndarray = centroid0.copy()
    sq_dists = distance(points, centroid0).ravel() ** 2

    sse_history: List[float] = [float(sq_dists.sum())]
    if verbose:
        print(f"[init]  1 cluster   SSE = {sse_history[-1]:10.4f}")

    while len(centroids) < k:
        best = {"sse_after": np.inf, "cid": None, "subres": None}

        for cid in range(len(centroids)):
            members = labels == cid
            if members.sum() < 2:
                # Can't split a singleton - skip.
                continue
            subset = points[members]
            sub = kmeans_best_of(subset, 2, rng=rng, n_init=n_init_inner, distance=distance)

            sse_split = sub.sse
            sse_other = sq_dists[~members].sum()
            sse_total = sse_split + sse_other
            if sse_total < best["sse_after"]:
                best.update({"sse_after": sse_total, "cid": cid, "subres": sub, "mask": members})

        if best["cid"] is None:                       
            break

        cid = best["cid"]
        sub: KMeansResult = best["subres"]
        mask: np.ndarray = best["mask"]
        new_cid = len(centroids)


        new_labels_for_subset = np.where(sub.labels == 0, cid, new_cid)
        labels[mask] = new_labels_for_subset


        sq_dists[mask] = sub.sq_dists


        centroids = np.vstack([centroids, sub.centroids[1:2]])
        centroids[cid] = sub.centroids[0]

        sse_history.append(float(sq_dists.sum()))
        if verbose:
            print(
                f"[split] {len(centroids):>2d} clusters  "
                f"split #{cid:>2d} -> #{cid:>2d},#{new_cid:>2d}   "
                f"SSE = {sse_history[-1]:10.4f}"
            )

    return BisectingResult(centroids=centroids, labels=labels,
                           sq_dists=sq_dists, sse_history=sse_history)


@dataclass
class Place:
    name: str
    address: str
    city: str
    lat: float
    lon: float


def load_places(path: str | Path) -> Tuple[List[Place], np.ndarray]:

    places: List[Place] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            name, address, city, lat, lon = parts[:5]
            places.append(Place(name, address, city, float(lat), float(lon)))

    coords = np.array([[p.lon, p.lat] for p in places], dtype=float)
    return places, coords


_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
]
_MARKERS = ["o", "s", "^", "D", "P", "v", "<", ">", "h", "X"]


def _cluster_summary_lines(result: BisectingResult, places: List[Place]) -> List[str]:
    """Build a short text summary of each cluster: size, SSE, mean radius."""
    lines = []
    for cid in range(len(result.centroids)):
        mask = result.labels == cid
        size = int(mask.sum())
        sse_c = float(result.sq_dists[mask].sum())
        # Mean great-circle radius (km) of cluster members.
        radius_km = float(np.sqrt(result.sq_dists[mask]).mean()) if size else 0.0
        cities = {places[i].city for i, m in enumerate(mask) if m}
        lines.append(
            f"  Cluster {cid}: {size:>3d} place(s), "
            f"SSE = {sse_c:8.4f}, mean dist = {radius_km:5.2f} km, "
            f"cities = {{{', '.join(sorted(cities))}}}"
        )
    return lines


def plot_clusters_on_map(
    result: BisectingResult,
    coords: np.ndarray,
    places: List[Place],
    map_image_path: str | Path,
    save_path: str | Path,
    *,
    title: Optional[str] = None,
    show_hulls: bool = True,
    show_legend: bool = True,
) -> None:

    img = _read_map_image(map_image_path)

    fig = plt.figure(figsize=(12, 10), dpi=120)
    rect = [0.05, 0.05, 0.90, 0.90]
    ax_map = fig.add_axes(rect, label="map", xticks=[], yticks=[])
    ax_map.imshow(img, aspect="auto")
    ax_map.set_xticks([]); ax_map.set_yticks([])

    ax = fig.add_axes(rect, label="scatter", frameon=False)
    ax.set_xticks([]); ax.set_yticks([])


    pad_x = 0.02 * (coords[:, 0].max() - coords[:, 0].min())
    pad_y = 0.02 * (coords[:, 1].max() - coords[:, 1].min())
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    k = len(result.centroids)
    for cid in range(k):
        mask = result.labels == cid
        if not mask.any():
            continue
        colour = _PALETTE[cid % len(_PALETTE)]
        marker = _MARKERS[cid % len(_MARKERS)]
        pts = coords[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=110, c=colour, marker=marker,
            edgecolors="black", linewidths=0.8,
            alpha=0.92, zorder=3,
            label=f"Cluster {cid} (n={mask.sum()})",
        )

        if show_hulls and mask.sum() >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly = MplPolygon(
                    hull_pts, closed=True,
                    facecolor=colour, edgecolor=colour,
                    alpha=0.10, linewidth=1.4, zorder=2,
                )
                ax.add_patch(poly)
            except Exception:
                pass

   
    cx, cy = result.centroids[:, 0], result.centroids[:, 1]
    ax.scatter(cx, cy, marker="+", s=400, c="black",
               linewidths=3.5, zorder=5, label="Centroid")
    ax.scatter(cx, cy, marker="+", s=400, c="white",
               linewidths=1.5, zorder=6)
    for cid, (x, y) in enumerate(zip(cx, cy)):
        ax.annotate(
            f"C{cid}", xy=(x, y), xytext=(8, 8), textcoords="offset points",
            fontsize=11, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="black", lw=0.8, alpha=0.85),
            zorder=7,
        )

    if title is None:
        title = (f"Bisecting K-Means on {len(places)} Portland-area places  "
                 f"(k = {k},  total SSE = {result.sse:.4f})")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)

    if show_legend:
        leg = ax.legend(
            loc="lower left", fontsize=9, framealpha=0.92,
            facecolor="white", edgecolor="black",
        )
        leg.set_zorder(8)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sse_elbow(
    ks: List[int], sses: List[float], save_path: str | Path
) -> None:
    """Plot total SSE vs k (the classic 'elbow' diagnostic)."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(ks, sses, "o-", color="#377EB8", linewidth=2.0, markersize=7)
    ax.set_xlabel("Number of clusters k", fontsize=11)
    ax.set_ylabel("Total SSE  (sum of squared great-circle distances, km²)",
                  fontsize=11)
    ax.set_title("SSE vs k for Bisecting K-Means", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.35)
    ax.set_xticks(ks)
    for k_, s_ in zip(ks, sses):
        ax.annotate(f"{s_:.2f}", (k_, s_), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_compare(
    coords: np.ndarray,
    places: List[Place],
    map_image_path: str | Path,
    save_path: str | Path,
    *,
    ks: Tuple[int, ...] = (3, 4, 5, 6),
    seed: int = 42,
) -> None:
    """Render k = 3..6 in a 2x2 grid for at-a-glance comparison."""
    img = _read_map_image(map_image_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=110)
    axes = axes.ravel()

    for ax, k in zip(axes, ks):
        rng = np.random.default_rng(seed)                # reproducible per panel
        res = bisecting_kmeans(coords, k, rng=rng, verbose=False)

        ax.imshow(img, extent=(
            coords[:, 0].min() - 0.01, coords[:, 0].max() + 0.01,
            coords[:, 1].min() - 0.01, coords[:, 1].max() + 0.01,
        ), aspect="auto")

        for cid in range(k):
            mask = res.labels == cid
            if not mask.any():
                continue
            colour = _PALETTE[cid % len(_PALETTE)]
            marker = _MARKERS[cid % len(_MARKERS)]
            pts = coords[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=55, c=colour, marker=marker,
                       edgecolors="black", linewidths=0.6, alpha=0.92, zorder=3)
        ax.scatter(res.centroids[:, 0], res.centroids[:, 1],
                   marker="+", s=200, c="black", linewidths=2.5, zorder=5)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"k = {k}   |   SSE = {res.sse:.4f}",
                     fontsize=12, fontweight="bold")

    fig.suptitle("Bisecting K-Means - cluster count comparison",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--places",
        default=BASE_DIR / "places.txt",
        help="Path to the tab-delimited places file."
    )

    parser.add_argument(
        "--map",
        default=BASE_DIR / "Portland.png",
        help="Path to the background map image."
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters (default: 5)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--outdir",
        default=BASE_DIR / "outputs",
        help="Where to write the output figures and report."
    )

    parser.add_argument(
        "--sweep",
        nargs=2,
        type=int,
        metavar=("KMIN", "KMAX"),
        default=[1, 10],
        help="Range of k values for the elbow plot (default: 1 10)."
    )

    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the 2x2 multi-k comparison grid."
    )

    args = parser.parse_args()

    args.places = Path(args.places)
    args.map = Path(args.map)
    args.outdir = Path(args.outdir)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Script directory : {BASE_DIR}")
    print(f"Places file      : {args.places}")
    print(f"Map image        : {args.map}")
    print(f"Output directory : {outdir}")
    print()

    #Validate required files
    if not args.places.exists():
        raise FileNotFoundError(f"Places file not found: {args.places}")

    if not args.map.exists():
        raise FileNotFoundError(f"Map image not found: {args.map}")

    #Load data
    places, coords = load_places(args.places)

    print(f"Loaded {len(places)} places.")
    print(f"  Longitude range: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
    print(f"  Latitude  range: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
    print()

    # Main clustering at the requested k
    print(f"Running bisecting K-means with k = {args.k}, seed = {args.seed}\n")

    rng = np.random.default_rng(args.seed)

    result = bisecting_kmeans(
        coords,
        args.k,
        rng=rng,
        verbose=True
    )

    print(f"\nFinal SSE: {result.sse:.6f}")

    print("Per-cluster summary:")

    summary_lines = _cluster_summary_lines(result, places)

    for line in summary_lines:
        print(line)

    # Main figure 
    main_png = outdir / f"clusters_k={args.k}.png"

    plot_clusters_on_map(
        result,
        coords,
        places,
        args.map,
        main_png
    )

    print(f"\nWrote {main_png}")

    # SSE-vs-k diagnostic 
    kmin, kmax = args.sweep

    sses: List[float] = []

    ks = list(range(kmin, kmax + 1))

    for k_ in ks:
        rng_k = np.random.default_rng(args.seed)

        res_k = bisecting_kmeans(
            coords,
            k_,
            rng=rng_k,
            verbose=False
        )

        sses.append(res_k.sse)

    elbow_png = outdir / "sse_vs_k.png"

    plot_sse_elbow(
        ks,
        sses,
        elbow_png
    )

    print(f"Wrote {elbow_png}  (SSE for k in {kmin}..{kmax})")

    # Multi-k comparison grid
    if not args.no_grid:
        grid_png = outdir / "compare_k_3_4_5_6.png"

        plot_grid_compare(
            coords,
            places,
            args.map,
            grid_png,
            ks=(3, 4, 5, 6),
            seed=args.seed
        )

        print(f"Wrote {grid_png}")

    report_path = outdir / "results_summary.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Bisecting K-Means - Run Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Input file       : {args.places}\n")
        f.write(f"Map image        : {args.map}\n")
        f.write(f"# of places      : {len(places)}\n")
        f.write(f"k                : {args.k}\n")
        f.write(f"Random seed      : {args.seed}\n")
        f.write(f"Distance metric  : haversine (great-circle, km)\n\n")

        f.write("SSE history (after each split):\n")

        for i, s in enumerate(result.sse_history):
            f.write(f"  After {i+1} cluster(s): SSE = {s:.6f}\n")

        f.write(f"\nFinal total SSE  : {result.sse:.6f}\n\n")

        f.write("Per-cluster breakdown:\n")

        for line in summary_lines:
            f.write(line + "\n")

        f.write("\nCentroids (longitude, latitude):\n")

        for cid, (x, y) in enumerate(result.centroids):
            f.write(f"  C{cid}:  lon = {x:.6f},  lat = {y:.6f}\n")

        f.write("\nElbow sweep (k -> SSE):\n")

        for k_, s_ in zip(ks, sses):
            f.write(f"  k = {k_:>2d}  SSE = {s_:.6f}\n")

        f.write("\nFull cluster membership:\n")

        for cid in range(args.k):
            f.write(f"\n[Cluster {cid}]\n")

            for i, p in enumerate(places):
                if result.labels[i] == cid:
                    f.write(
                        f"  - {p.name:<28s}  ({p.city}) "
                        f"[{p.lat:.5f}, {p.lon:.5f}]\n"
                    )

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()
