from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from decision_tree import ID3DecisionTree, TreeNode

def confusion_matrix_for(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    classes: Sequence[Any],
) -> np.ndarray:
    """Build a confusion matrix laid out as ``classes × classes``."""
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm

def leave_one_out_cv(
    X: List[List[Any]],
    y: List[Any],
    feature_names: List[str],
    factory: Callable[[], Any],
    is_custom: bool,
) -> Dict[str, Any]:

    n = len(X)
    n_correct = 0
    per_fold: List[float] = []
    preds: List[Any] = []
    truths: List[Any] = []

    for i in range(n):
        X_train = X[:i] + X[i + 1:]
        y_train = y[:i] + y[i + 1:]
        X_test = X[i]
        y_test = y[i]

        clf = factory()
        if is_custom:
            clf.fit(X_train, y_train, feature_names)
            pred = clf.predict_one(X_test)
        else:
            clf.fit(np.array(X_train), np.array(y_train))
            pred = clf.predict(np.array([X_test]))[0]

        preds.append(pred)
        truths.append(y_test)
        ok = int(pred == y_test)
        n_correct += ok
        per_fold.append(float(ok))

    return {
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "n_total": n,
        "per_fold": per_fold,
        "predictions": preds,
        "truths": truths,
    }

def classification_report_dict(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    classes: Sequence[Any],
) -> Dict[Any, Dict[str, float]]:
    """Return per-class precision, recall and F1 plus a macro average."""
    cm = confusion_matrix_for(y_true, y_pred, classes)
    report: Dict[Any, Dict[str, float]] = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }

    macro = {
        "precision": float(np.mean([r["precision"] for r in report.values()])),
        "recall": float(np.mean([r["recall"] for r in report.values()])),
        "f1": float(np.mean([r["f1"] for r in report.values()])),
    }
    report["__macro_avg__"] = macro
    return report

def custom_feature_importance(clf: ID3DecisionTree) -> Dict[str, float]:

    if clf.root is None:
        return {f: 0.0 for f in clf.feature_names_}

    raw: Dict[str, float] = defaultdict(float)
    stack: List[TreeNode] = [clf.root]
    while stack:
        node = stack.pop()
        if not node.is_leaf:
            raw[node.feature] += node.samples * node.info_gain
            stack.extend(node.children.values())

    total = sum(raw.values())
    if total == 0:
        return {f: 0.0 for f in clf.feature_names_}

    return {f: raw.get(f, 0.0) / total for f in clf.feature_names_}
