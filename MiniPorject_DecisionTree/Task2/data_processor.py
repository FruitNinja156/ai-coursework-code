from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

RSS_MISSING_SENTINEL: float = -100.0

REQUIRED_COLUMNS = ("BSSIDLabel", "RSSLabel", "RoomLabel", "finLabel")


@dataclass
class FingerprintDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    fingerprint_ids: np.ndarray


def load_raw(path: str, encoding: str = "latin-1") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path!r} missing required columns: {missing}")
    # Coerce dtypes
    df["RSSLabel"] = pd.to_numeric(df["RSSLabel"], errors="raise")
    df["RoomLabel"] = pd.to_numeric(df["RoomLabel"], errors="raise").astype(int)
    df["finLabel"] = pd.to_numeric(df["finLabel"], errors="raise").astype(int)
    df["BSSIDLabel"] = df["BSSIDLabel"].astype(str)
    return df


def collect_feature_names(*frames: pd.DataFrame) -> List[str]:
    seen = set()
    feats: List[str] = []
    for df in frames:
        for b in df["BSSIDLabel"].unique():
            if b not in seen:
                seen.add(b)
                feats.append(b)
    return sorted(feats)


def _validate_one_room_per_fingerprint(df: pd.DataFrame) -> None:
    bad = df.groupby("finLabel")["RoomLabel"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise ValueError(
            "Some fingerprints map to more than one RoomLabel: "
            f"{bad.index.tolist()[:5]}…"
        )


def pivot_continuous(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    fill_value: float = RSS_MISSING_SENTINEL,
) -> FingerprintDataset:
    _validate_one_room_per_fingerprint(df)
    wide = (
        df.pivot_table(
            index="finLabel",
            columns="BSSIDLabel",
            values="RSSLabel",
            aggfunc="mean",  
        )
        .reindex(columns=list(feature_names))
        .astype(np.float64)
    )

    # Map fingerprint to room 
    rooms = df.groupby("finLabel")["RoomLabel"].first().reindex(wide.index)

    X = wide.to_numpy()
    np.nan_to_num(X, copy=False, nan=fill_value)
    y = rooms.to_numpy(dtype=int)

    return FingerprintDataset(
        X=X,
        y=y,
        feature_names=list(feature_names),
        fingerprint_ids=wide.index.to_numpy(),
    )


def pivot_binary(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> FingerprintDataset:

    _validate_one_room_per_fingerprint(df)

    wide = (
        df.assign(_one=1.0)
        .pivot_table(
            index="finLabel",
            columns="BSSIDLabel",
            values="_one",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(columns=list(feature_names), fill_value=0.0)
        .astype(np.float64)
    )

    rooms = df.groupby("finLabel")["RoomLabel"].first().reindex(wide.index)
    return FingerprintDataset(
        X=wide.to_numpy(),
        y=rooms.to_numpy(dtype=int),
        feature_names=list(feature_names),
        fingerprint_ids=wide.index.to_numpy(),
    )


def summary(ds: FingerprintDataset, name: str = "dataset") -> str:
    classes, counts = np.unique(ds.y, return_counts=True)
    lines = [
        f"── {name} ──",
        f"  fingerprints : {ds.X.shape[0]}",
        f"  features     : {ds.X.shape[1]}",
        f"  classes      : {len(classes)}  ({dict(zip(classes.tolist(), counts.tolist()))})",
        f"  X dtype      : {ds.X.dtype}",
        f"  X range      : [{ds.X.min():.1f}, {ds.X.max():.1f}]",
        f"  density      : {(ds.X != RSS_MISSING_SENTINEL).mean():.3f}  "
        f"(fraction of cells with an actual RSS reading)",
    ]
    return "\n".join(lines)
