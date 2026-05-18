
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn import tree as sktree

from data_loader import FEATURE_NAMES, TARGET_NAME
from decision_tree import ID3DecisionTree

def save_custom_tree_graphviz(
    clf: ID3DecisionTree,
    out_path: str | Path,
    title: str = "Contact-Lens ID3 Decision Tree (from scratch)",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = clf.to_graphviz(title=title)
    rendered = g.render(
        filename=out_path.stem,
        directory=str(out_path.parent),
        format="png",
        cleanup=True,
    )
    rendered_path = Path(rendered)
    if rendered_path != out_path:
        rendered_path.replace(out_path)
    return out_path

def save_sklearn_tree(
    sk_clf,
    feature_encoders: Dict[str, Dict[str, int]],
    target_decoder: Dict[int, str],
    out_path: str | Path,
    title: str = "Contact-Lens Decision Tree (scikit-learn)",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_labels = []
    for f in FEATURE_NAMES:
        mapping = feature_encoders[f]
        readable = ", ".join(f"{k}={v}" for k, v in mapping.items())
        feature_labels.append(f"{f}\n[{readable}]")

    class_labels = [target_decoder[i] for i in sorted(target_decoder)]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    sktree.plot_tree(
        sk_clf,
        feature_names=feature_labels,
        class_names=class_labels,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        ax=ax,
        fontsize=9,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

def save_data_overview(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    palette = ["#5B8FF9", "#5AD8A6", "#F6BD16", "#E86452", "#6DC8EC", "#945FB9"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=120)
    fig.suptitle(
        "Contact-Lens Dataset — Distribution of Features and Target",
        fontsize=15,
        fontweight="bold",
    )
    cols = list(df.columns)
    for ax, col in zip(axes.flatten(), cols):
        counts = df[col].value_counts()
        bars = ax.bar(
            counts.index.astype(str),
            counts.values,
            color=palette[: len(counts)],
            edgecolor="#333333",
        )
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("count")
        ax.set_ylim(0, max(counts.values) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for b, v in zip(bars, counts.values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.3,
                str(int(v)),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.tick_params(axis="x", rotation=15)

    if len(cols) < axes.size:
        for extra in axes.flatten()[len(cols):]:
            extra.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

def save_confusion_matrix(
    cm: np.ndarray,
    class_labels: List[str],
    out_path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    """Save a heatmap of a confusion matrix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("blues", ["#FFFFFF", "#1F4E79"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label", fontweight="bold")
    ax.set_ylabel("True label", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=13)

    threshold = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "#222",
                fontsize=12,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

def save_feature_importance(
    importances: Dict[str, float],
    out_path: str | Path,
    title: str = "Feature importance",
) -> Path:
    """Save a horizontal bar chart of feature importances."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(importances.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    bars = ax.barh(labels, values, color="#5B8FF9", edgecolor="#1F4E79")
    ax.set_xlabel("importance")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
