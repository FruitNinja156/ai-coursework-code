from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: Sequence[int]) -> np.ndarray:
    """Compute a confusion matrix indexed by `labels`."""
    label_to_idx = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    M = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            M[label_to_idx[t], label_to_idx[p]] += 1
    return M


def per_class_metrics(M: np.ndarray, labels: Sequence[int]) -> dict:

    K = len(labels)
    out = {"per_class": {}, "macro": {}}
    precisions, recalls, f1s = [], [], []
    for i, l in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out["per_class"][int(l)] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(M[i, :].sum()),
        }
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    out["macro"]["precision"] = float(np.mean(precisions))
    out["macro"]["recall"]    = float(np.mean(recalls))
    out["macro"]["f1"]        = float(np.mean(f1s))
    out["accuracy"]           = float(np.trace(M) / max(M.sum(), 1))
    return out

def _setup_mpl():
  
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def plot_confusion_matrix(M: np.ndarray, labels: Sequence[int],
                          out_path: Path, title: str = "Confusion matrix") -> None:
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([f"Room {l}" for l in labels])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([f"Room {l}" for l in labels])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)

    vmax = M.max() if M.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > vmax / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: Sequence[str],
                            out_path: Path, top_n: int = 20,
                            title: str = "Top discriminative BSSIDs") -> None:

    plt = _setup_mpl()
    nonzero_mask = importances > 0
    nonzero_idx  = np.where(nonzero_mask)[0]
    if nonzero_idx.size == 0:
        fig, ax = plt.subplots(figsize=(6, 2.4))
        ax.text(0.5, 0.5, "No features were used (degenerate tree).",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
        return

    order = nonzero_idx[np.argsort(importances[nonzero_idx])[::-1]][:top_n]
    used_total = nonzero_idx.size

    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.42 * len(order) + 1)))
    ax.barh(range(len(order)), importances[order][::-1], color="#3a76b5")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=9, family="monospace")
    ax.set_xlabel("Normalised information-gain contribution")
    suffix = f" ({used_total} of {len(feature_names)} BSSIDs used)"
    ax.set_title(title + suffix)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tree(root, classes, feature_names, out_path: Path,
              max_depth: int = 4, title: str = "Decision tree (top levels)") -> None:
    plt = _setup_mpl()

    positions: dict[int, tuple[float, float]] = {}
    next_x = [0.0]   

    def count_leaves(node, depth=0):

        if node is None:
            return 0
        if node.is_leaf or depth == max_depth:
            return 1
        return count_leaves(node.left, depth + 1) + count_leaves(node.right, depth + 1)

    def assign_pos(node, depth=0):
        if node is None:
            return None
        node_id = id(node)
        if node.is_leaf or depth == max_depth:
            x = next_x[0]
            next_x[0] += 1.0
            positions[node_id] = (x, -depth)
            return x
        lx = assign_pos(node.left,  depth + 1)
        rx = assign_pos(node.right, depth + 1)
        x = (lx + rx) / 2.0
        positions[node_id] = (x, -depth)
        return x

    assign_pos(root)
    if not positions:
        return

    n_leaves = count_leaves(root)
    fig_w = max(8, n_leaves * 2.3)
    fig_h = max(4.5, max_depth * 1.3 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    #color palette!
    cmap = ["#3a76b5", "#e57b3b", "#5fa56b", "#9c6dab", "#d05050", "#5fb6c4"]

    def label_for(node, depth):
        if node.is_leaf or depth == max_depth:
            counts = {int(classes[i]): int(c) for i, c in enumerate(node.class_counts) if c > 0}
            pred = classes[int(np.argmax(node.class_counts))]
            tag = "leaf" if node.is_leaf else "(truncated)"
            return f"{tag}\nRoom {pred}\nn={node.n_samples}\n{counts}"
        fname = feature_names[node.feature_idx]
        return (f"{fname[-12:]}\n≤ {node.threshold:.1f}\n"
                f"H={node.impurity:.2f}  n={node.n_samples}")

    def draw(node, depth=0, parent_pos=None):
        if node is None:
            return
        x, y = positions[id(node)]
        if parent_pos is not None:
            ax.plot([parent_pos[0], x], [parent_pos[1] - 0.18, y + 0.18],
                    color="#888", lw=1.0, zorder=1)
        is_terminal = node.is_leaf or depth == max_depth
        majority = int(np.argmax(node.class_counts))
        face = cmap[majority % len(cmap)] if is_terminal else "#f7f7f7"
        edge = "#333" if is_terminal else "#555"
        bbox = dict(boxstyle="round,pad=0.45", facecolor=face,
                    edgecolor=edge, linewidth=1.1, alpha=0.95)
        text_color = "white" if is_terminal else "black"
        ax.text(x, y, label_for(node, depth), ha="center", va="center",
                fontsize=8, family="monospace", color=text_color,
                bbox=bbox, zorder=2)
        if not is_terminal:
            draw(node.left,  depth + 1, (x, y))
            draw(node.right, depth + 1, (x, y))

    draw(root)

    if not root.is_leaf:
        rx, ry = positions[id(root)]
        if root.left is not None:
            lx, ly = positions[id(root.left)]
            ax.text((rx + lx) / 2, (ry + ly) / 2, "True", fontsize=8,
                    color="#2a6b3a", ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
        if root.right is not None:
            xx, yy = positions[id(root.right)]
            ax.text((rx + xx) / 2, (ry + yy) / 2, "False", fontsize=8,
                    color="#aa2a2a", ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.6, max(p[0] for p in positions.values()) + 0.6)
    ax.set_ylim(-max_depth - 0.7, 0.7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(y_train: np.ndarray, y_test: np.ndarray,
                       out_path: Path) -> None:
    plt = _setup_mpl()
    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    train_counts = [int(np.sum(y_train == c)) for c in classes]
    test_counts  = [int(np.sum(y_test  == c)) for c in classes]
    x = np.arange(len(classes)); w = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, train_counts, w, label="Train", color="#3a76b5")
    ax.bar(x + w/2, test_counts,  w, label="Test",  color="#e57b3b")
    ax.set_xticks(x); ax.set_xticklabels([f"Room {c}" for c in classes])
    ax.set_ylabel("Number of fingerprints")
    ax.set_title("Class balance per split")
    ax.legend()
    for xi, v in zip(x - w/2, train_counts): ax.text(xi, v + 0.7, str(v), ha="center", fontsize=9)
    for xi, v in zip(x + w/2,  test_counts): ax.text(xi, v + 0.7, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_accuracy_comparison(results: dict, out_path: Path) -> None:
    plt = _setup_mpl()
    names  = list(results.keys())
    values = [results[n] for n in names]
    colors = ["#3a76b5", "#5fa56b", "#e57b3b", "#9c6dab"][: len(names)]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Accuracy on test set")
    ax.set_title("Accuracy by representation / implementation")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", fontsize=10)
    ax.axhline(0.25, ls="--", lw=1, color="grey", alpha=0.6)
    ax.text(len(names) - 0.4, 0.255, "random baseline (¼)", color="grey", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_depth_sweep(depths: Sequence[int], train_acc: Sequence[float],
                     test_acc: Sequence[float], out_path: Path) -> None:
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(depths, train_acc, "-o", label="Train accuracy", color="#3a76b5")
    ax.plot(depths, test_acc,  "-o", label="Test accuracy",  color="#e57b3b")
    ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy")
    ax.set_title("Train vs. test accuracy as max_depth grows")
    ax.set_ylim(0, 1.04); ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
