from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Node:
    is_leaf: bool = False
    prediction: Optional[int] = None
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    n_samples: int = 0
    impurity: float = 0.0
    class_counts: np.ndarray = field(default_factory=lambda: np.array([]))

    depth: int = 0


class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: Optional[int] = 42,
    ) -> None:
        if criterion != "entropy":
            raise ValueError(f"Unsupported criterion: {criterion!r}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        # Populated by `fit`
        self.root_: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_nodes_: int = 0
        self.n_leaves_: int = 0
        self.depth_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build the tree from training data ``(X, y)``."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_, dtype=np.float64)

        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y_idx, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        self.n_nodes_, self.n_leaves_, self.depth_ = self._tree_stats(self.root_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
       
        if self.root_ is None:
            raise RuntimeError("Tree has not been fitted yet — call `fit` first.")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, x in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = self.classes_[node.prediction]
        return out

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = X.shape[0]
        class_counts = np.bincount(y, minlength=len(self.classes_))
        impurity = self._entropy_from_counts(class_counts, n_samples)

        node = Node(
            n_samples=n_samples,
            impurity=impurity,
            class_counts=class_counts,
            depth=depth,
        )
        stop = (
            impurity == 0.0 
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return self._make_leaf(node)

        best_feat, best_thr, best_gain, best_left, best_right = self._best_split(
            X, y, impurity, n_samples
        )

        if best_feat is None or best_gain <= self.min_impurity_decrease:
            return self._make_leaf(node)
        node.feature_idx = best_feat
        node.threshold = best_thr
        self.feature_importances_[best_feat] += best_gain * n_samples

        node.left = self._build(X[best_left], y[best_left], depth + 1)
        node.right = self._build(X[best_right], y[best_right], depth + 1)
        return node

    def _make_leaf(self, node: Node) -> Node:
        
        node.is_leaf = True
        
        node.prediction = int(np.argmax(node.class_counts))
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parent_impurity: float,
        n_samples: int,
    ):
        n_classes = len(self.classes_)
        best_gain = -np.inf
        best_feat = best_thr = None
        best_left_mask = best_right_mask = None

        for feat in range(self.n_features_):
            x = X[:, feat]
            order = np.argsort(x, kind="mergesort")
            x_sorted = x[order]
            y_sorted = y[order]

            if x_sorted[0] == x_sorted[-1]:
                continue

            left_counts = np.zeros(n_classes, dtype=np.int64)
            total_counts = np.bincount(y_sorted, minlength=n_classes)

            for i in range(n_samples - 1):
                left_counts[y_sorted[i]] += 1

                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                n_left = i + 1
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                right_counts = total_counts - left_counts
                e_left = self._entropy_from_counts(left_counts, n_left)
                e_right = self._entropy_from_counts(right_counts, n_right)
                weighted = (n_left * e_left + n_right * e_right) / n_samples
                gain = parent_impurity - weighted

                if gain > best_gain:
                    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
                    best_gain = gain
                    best_feat = feat
                    best_thr = threshold
                    best_split_pos = i + 1
                    best_order = order

        if best_feat is None:
            return None, None, -np.inf, None, None

        left_idx = best_order[:best_split_pos]
        right_idx = best_order[best_split_pos:]
        left_mask = np.zeros(n_samples, dtype=bool)
        right_mask = np.zeros(n_samples, dtype=bool)
        left_mask[left_idx] = True
        right_mask[right_idx] = True
        return best_feat, best_thr, best_gain, left_mask, right_mask

    @staticmethod
    def _entropy_from_counts(counts: np.ndarray, total: int) -> float:
        """Shannon entropy from class counts. Empty / single-class node → 0."""
        if total <= 0:
            return 0.0
        nz = counts[counts > 0]
        if nz.size <= 1:
            return 0.0
        p = nz / total
        return float(-np.sum(p * np.log2(p)))

    def _tree_stats(self, node: Optional[Node]) -> tuple[int, int, int]:
        if node is None:
            return 0, 0, -1
        if node.is_leaf:
            return 1, 1, node.depth
        ln, ll, ld = self._tree_stats(node.left)
        rn, rl, rd = self._tree_stats(node.right)
        return 1 + ln + rn, ll + rl, max(ld, rd)

    def export_text(self, feature_names: Optional[Sequence[str]] = None,
                    max_depth: Optional[int] = None) -> str:
        lines: List[str] = []

        def _fmt_counts(counts: np.ndarray) -> str:
            cls = self.classes_
            return "{" + ", ".join(f"{int(cls[i])}:{int(c)}" for i, c in enumerate(counts) if c > 0) + "}"

        def _walk(node: Node, prefix: str = "", is_last: bool = True) -> None:
            if max_depth is not None and node.depth > max_depth:
                return
            connector = "└── " if is_last else "├── "
            if node.is_leaf:
                pred = self.classes_[node.prediction]
                lines.append(
                    f"{prefix}{connector}leaf: predict={pred}  "
                    f"n={node.n_samples}  H={node.impurity:.3f}  "
                    f"counts={_fmt_counts(node.class_counts)}"
                )
                return
            fname = (
                feature_names[node.feature_idx]
                if feature_names is not None
                else f"x[{node.feature_idx}]"
            )
            lines.append(
                f"{prefix}{connector}{fname} <= {node.threshold:.3f}  "
                f"n={node.n_samples}  H={node.impurity:.3f}  "
                f"counts={_fmt_counts(node.class_counts)}"
            )
            new_prefix = prefix + ("    " if is_last else "│   ")
            _walk(node.left, new_prefix, is_last=False)
            _walk(node.right, new_prefix, is_last=True)

        if self.root_ is not None:
            _walk(self.root_, "", True)
        return "\n".join(lines)
