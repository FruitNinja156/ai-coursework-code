from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log2
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import graphviz

def entropy(labels: Sequence[Any]) -> float:

    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * log2(c / n) for c in counts.values() if c > 0)


def information_gain(
    parent_labels: Sequence[Any],
    child_partitions: Iterable[Sequence[Any]],
) -> float:

    n = len(parent_labels)
    if n == 0:
        return 0.0
    parent_h = entropy(parent_labels)
    weighted_child_h = 0.0
    for child in child_partitions:
        if not child:
            continue
        weighted_child_h += (len(child) / n) * entropy(child)
    return parent_h - weighted_child_h

@dataclass
class TreeNode:

    feature: Optional[str] = None
    children: Dict[Any, "TreeNode"] = field(default_factory=dict)
    info_gain: float = 0.0                

    is_leaf: bool = False
    prediction: Optional[Any] = None   
    class_distribution: Dict[Any, int] = field(default_factory=dict)
    samples: int = 0                     

    def majority_class(self) -> Any:
        """Return the most common class at this node (for fall-back prediction)."""
        if self.class_distribution:
            return max(self.class_distribution.items(), key=lambda kv: kv[1])[0]
        return self.prediction

class ID3DecisionTree:

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 1,
        min_info_gain: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain

        self.root: Optional[TreeNode] = None
        self.feature_names_: List[str] = []
        self.classes_: List[Any] = []
        self.n_nodes_: int = 0
        self.n_leaves_: int = 0
        self.depth_: int = 0

    def fit(
        self,
        X: Sequence[Sequence[Any]],
        y: Sequence[Any],
        feature_names: Sequence[str],
    ) -> "ID3DecisionTree":

        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) == 0:
            raise ValueError("Cannot fit on empty data.")
        if len(X[0]) != len(feature_names):
            raise ValueError(
                "Number of feature_names does not match number of columns in X."
            )

        self.feature_names_ = list(feature_names)
        self.classes_ = sorted(set(y))

        rows: List[Dict[str, Any]] = [
            dict(zip(self.feature_names_, row)) for row in X
        ]
        labels: List[Any] = list(y)

        self.root = self._build(
            rows=rows,
            labels=labels,
            available_features=set(self.feature_names_),
            depth=0,
        )

        self.n_nodes_, self.n_leaves_, self.depth_ = self._tree_stats(self.root)
        return self

    def predict_one(self, sample: Union[Dict[str, Any], Sequence[Any]]) -> Any:
        """Predict the class for a single sample (dict or positional sequence)."""
        if self.root is None:
            raise RuntimeError("The tree has not been fitted yet.")
        if not isinstance(sample, dict):
            sample = dict(zip(self.feature_names_, sample))
        return self._traverse(self.root, sample)

    def predict(
        self, X: Sequence[Union[Dict[str, Any], Sequence[Any]]]
    ) -> List[Any]:
        """Predict the class for a batch of samples."""
        return [self.predict_one(s) for s in X]

    def score(
        self,
        X: Sequence[Union[Dict[str, Any], Sequence[Any]]],
        y: Sequence[Any],
    ) -> float:
        """Return classification accuracy on the supplied data."""
        preds = self.predict(X)
        return sum(int(p == t) for p, t in zip(preds, y)) / len(y)

    def _build(
        self,
        rows: List[Dict[str, Any]],
        labels: List[Any],
        available_features: set,
        depth: int,
    ) -> TreeNode:

        class_counts = dict(Counter(labels))
        node = TreeNode(class_distribution=class_counts, samples=len(labels))

        if len(class_counts) == 1:
            node.is_leaf = True
            node.prediction = next(iter(class_counts))
            return node

        if not available_features:
            node.is_leaf = True
            node.prediction = node.majority_class()
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = node.majority_class()
            return node

        if len(rows) < self.min_samples_split:
            node.is_leaf = True
            node.prediction = node.majority_class()
            return node

        best_feature, best_gain, best_partitions = self._best_split(
            rows, labels, available_features
        )

        if best_feature is None or best_gain < self.min_info_gain:
            node.is_leaf = True
            node.prediction = node.majority_class()
            return node

        node.feature = best_feature
        node.info_gain = best_gain

        new_available = available_features - {best_feature}
        for value, (sub_rows, sub_labels) in best_partitions.items():
            if not sub_rows:

                child = TreeNode(
                    is_leaf=True,
                    prediction=node.majority_class(),
                    class_distribution=class_counts,
                    samples=0,
                )
            else:
                child = self._build(sub_rows, sub_labels, new_available, depth + 1)
            node.children[value] = child

        return node

    @staticmethod
    def _partition(
        rows: List[Dict[str, Any]],
        labels: List[Any],
        feature: str,
    ) -> Dict[Any, Tuple[List[Dict[str, Any]], List[Any]]]:
        """Split (rows, labels) into groups keyed by the value of *feature*."""
        out: Dict[Any, Tuple[List[Dict[str, Any]], List[Any]]] = {}
        for row, lab in zip(rows, labels):
            v = row[feature]
            if v not in out:
                out[v] = ([], [])
            out[v][0].append(row)
            out[v][1].append(lab)
        return out

    def _best_split(
        self,
        rows: List[Dict[str, Any]],
        labels: List[Any],
        available_features: set,
    ) -> Tuple[Optional[str], float, Dict[Any, Tuple[List[Dict[str, Any]], List[Any]]]]:
        """Find the feature with the highest information gain."""
        best_feature: Optional[str] = None
        best_gain: float = -1.0
        best_partitions: Dict[Any, Tuple[List[Dict[str, Any]], List[Any]]] = {}

        for feature in sorted(available_features):
            partitions = self._partition(rows, labels, feature)
            child_label_lists = [sub_labels for _, sub_labels in partitions.values()]
            gain = information_gain(labels, child_label_lists)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_partitions = partitions
        return best_feature, max(best_gain, 0.0), best_partitions

    def _traverse(self, node: TreeNode, sample: Dict[str, Any]) -> Any:

        if node.is_leaf:
            return node.prediction
        value = sample.get(node.feature)
        if value not in node.children:
            return node.majority_class()
        return self._traverse(node.children[value], sample)

    @staticmethod
    def _tree_stats(root: TreeNode) -> Tuple[int, int, int]:
        """Return (n_nodes, n_leaves, depth)."""
        n_nodes = 0
        n_leaves = 0
        depth = 0

        stack: List[Tuple[TreeNode, int]] = [(root, 0)]
        while stack:
            node, d = stack.pop()
            n_nodes += 1
            depth = max(depth, d)
            if node.is_leaf:
                n_leaves += 1
            else:
                for child in node.children.values():
                    stack.append((child, d + 1))
        return n_nodes, n_leaves, depth

    def to_dict(self) -> Dict[str, Any]:

        if self.root is None:
            return {}
        return self._node_to_dict(self.root)

    def _node_to_dict(self, node: TreeNode) -> Dict[str, Any]:
        if node.is_leaf:
            return {
                "leaf": True,
                "prediction": node.prediction,
                "samples": node.samples,
                "distribution": node.class_distribution,
            }
        return {
            "leaf": False,
            "feature": node.feature,
            "info_gain": round(node.info_gain, 4),
            "samples": node.samples,
            "distribution": node.class_distribution,
            "children": {
                str(v): self._node_to_dict(c) for v, c in node.children.items()
            },
        }

    def render_text(self) -> str:
        """Return an ASCII-art rendering of the decision tree."""
        if self.root is None:
            return "<empty tree>"
        lines: List[str] = []
        self._render_text(self.root, prefix="", connector="", lines=lines)
        return "\n".join(lines)

    def _render_text(
        self,
        node: TreeNode,
        prefix: str,
        connector: str,
        lines: List[str],
    ) -> None:
        if node.is_leaf:
            dist = ", ".join(
                f"{k}:{v}" for k, v in sorted(node.class_distribution.items())
            )
            lines.append(
                f"{prefix}{connector} PREDICT '{node.prediction}'  "
                f"(samples={node.samples}, [{dist}])"
            )
            return

        lines.append(
            f"{prefix}{connector}split on '{node.feature}' "
            f"(IG={node.info_gain:.4f}, samples={node.samples})"
        )
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last = i == len(children) - 1
            branch = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{prefix}{('    ' if connector else '')}│")
            sub_connector = f"{branch}[{node.feature} = {value}] → "
            self._render_text(child, prefix=child_prefix, connector=sub_connector, lines=lines)

    def to_graphviz(
        self,
        title: str = "ID3 Decision Tree (from scratch)",
    ) -> "graphviz.Digraph":
        import graphviz  

        class_colors = {
            "no lenses": "#FFD6D6",   
            "soft":      "#D6F0FF",   
            "hard":      "#D6FFD6",  
        }
        default_leaf_color = "#EEEEEE"

        g = graphviz.Digraph(
            "ID3",
            graph_attr={
                "label": title,
                "labelloc": "t",
                "fontsize": "20",
                "fontname": "Helvetica",
                "rankdir": "TB",
                "bgcolor": "white",
                "splines": "polyline",
                "nodesep": "0.5",
                "ranksep": "0.7",
            },
            node_attr={"fontname": "Helvetica", "fontsize": "11"},
            edge_attr={"fontname": "Helvetica", "fontsize": "10"},
        )

        counter = [0]

        def add(node: TreeNode) -> str:
            nid = f"n{counter[0]}"
            counter[0] += 1
            if node.is_leaf:
                color = class_colors.get(str(node.prediction), default_leaf_color)
                dist = "\\n".join(
                    f"{k}: {v}"
                    for k, v in sorted(node.class_distribution.items())
                )
                label = (
                    f"PREDICT\\n"
                    f"« {node.prediction} »\\n"
                    f"———————\\n"
                    f"samples = {node.samples}\\n"
                    f"{dist}"
                )
                g.node(
                    nid,
                    label=label,
                    shape="box",
                    style="rounded,filled",
                    fillcolor=color,
                    color="#444444",
                )
            else:
                label = (
                    f"{node.feature} ?\\n"
                    f"———————\\n"
                    f"info gain = {node.info_gain:.3f}\\n"
                    f"samples = {node.samples}"
                )
                g.node(
                    nid,
                    label=label,
                    shape="ellipse",
                    style="filled",
                    fillcolor="#FFF7C2",
                    color="#444444",
                )
                for value, child in node.children.items():
                    cid = add(child)
                    g.edge(nid, cid, label=f" {value} ")
            return nid

        if self.root is not None:
            add(self.root)
        return g

if __name__ == "__main__":
    from data_loader import load_as_dataframe, FEATURE_NAMES, TARGET_NAME

    df = load_as_dataframe("lenses.txt")
    X = df[FEATURE_NAMES].astype(str).values.tolist()
    y = df[TARGET_NAME].astype(str).tolist()

    clf = ID3DecisionTree()
    clf.fit(X, y, FEATURE_NAMES)

    print(clf.render_text())
    print()
    print(f"Training accuracy: {clf.score(X, y):.4f}")
    print(f"Tree size: {clf.n_nodes_} nodes, {clf.n_leaves_} leaves, depth {clf.depth_}")
