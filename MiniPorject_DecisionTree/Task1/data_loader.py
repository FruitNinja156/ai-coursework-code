from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


FEATURE_NAMES: List[str] = ["age", "prescription", "astigmatic", "tear_rate"]
TARGET_NAME: str = "lens_type"
ALL_COLUMNS: List[str] = FEATURE_NAMES + [TARGET_NAME]


def parse_lenses_file(path: str | Path) -> List[List[str]]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: List[List[str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != len(ALL_COLUMNS):
                raise ValueError(
                    f"Line {line_no}: expected {len(ALL_COLUMNS)} tab-separated "
                    f"fields, found {len(fields)}: {line!r}"
                )
            records.append([f.strip() for f in fields])
    return records


def load_as_dataframe(path: str | Path) -> pd.DataFrame:

    records = parse_lenses_file(path)
    df = pd.DataFrame(records, columns=ALL_COLUMNS)
    for col in df.columns:
        df[col] = df[col].astype("category")
    return df


def quick_check(df: pd.DataFrame) -> None:

    bar = "=" * 72
    print(bar)
    print(" DATASET QUICK-CHECK ".center(72, "="))
    print(bar)

    print(f"Shape           : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns         : {list(df.columns)}")
    print(f"Missing values  : {int(df.isna().sum().sum())}")
    print()

    print("--- Head (first 5 rows) ---")
    print(df.head().to_string(index=True))
    print()
    print("--- Tail (last 5 rows) ---")
    print(df.tail().to_string(index=True))
    print()

    print("--- Value counts per column ---")
    for col in df.columns:
        print(f"\n[{col}]")
        counts = df[col].value_counts()
        for value, count in counts.items():
            pct = 100.0 * count / len(df)
            bar_str = "█" * int(round(pct / 4))
            print(f"  {str(value):<12} {count:>3}  ({pct:5.1f}%) {bar_str}")

    print()
    print("--- Class balance ---")
    cls_counts = df[TARGET_NAME].value_counts()
    most_common_pct = 100.0 * cls_counts.iloc[0] / len(df)
    print(f"Majority class baseline accuracy ≈ {most_common_pct:.1f}%")
    print(bar)


def encode_for_sklearn(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]], Dict[int, str]]:

    feature_encoders: Dict[str, Dict[str, int]] = {}
    X_cols = []
    for col in FEATURE_NAMES:
        cats = sorted(df[col].cat.categories.tolist())
        mapping = {v: i for i, v in enumerate(cats)}
        feature_encoders[col] = mapping
        X_cols.append(df[col].map(mapping).to_numpy())
    X = np.column_stack(X_cols).astype(int)

    target_cats = sorted(df[TARGET_NAME].cat.categories.tolist())
    target_mapping = {v: i for i, v in enumerate(target_cats)}
    target_decoder = {i: v for v, i in target_mapping.items()}
    y = df[TARGET_NAME].map(target_mapping).to_numpy().astype(int)
    return X, y, feature_encoders, target_decoder


if __name__ == "__main__":
    df = load_as_dataframe("lenses.txt")
    quick_check(df)
