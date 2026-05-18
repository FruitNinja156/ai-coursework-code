from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from data_loader import (
    FEATURE_NAMES,
    TARGET_NAME,
    encode_for_sklearn,
    load_as_dataframe,
    quick_check,
)
from decision_tree import ID3DecisionTree
from evaluation import (
    classification_report_dict,
    confusion_matrix_for,
    custom_feature_importance,
    leave_one_out_cv,
)
from predictor import DEMO_PATIENTS, predict_for_samples
from tree_visualizer import (
    save_confusion_matrix,
    save_custom_tree_graphviz,
    save_data_overview,
    save_feature_importance,
    save_sklearn_tree,
)


HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def banner(title: str, char: str = "═") -> None:
    line = char * 78
    print(f"\n{line}\n║ {title}\n{line}")

#  Pipeline

def main() -> None:
 
    banner("STEP 1 · DATA PARSING")
    df = load_as_dataframe(HERE / "lenses.txt")
    quick_check(df)

    save_data_overview(df, OUTPUTS / "data_overview.png")
    print(f"\nSaved dataset overview : {OUTPUTS / 'data_overview.png'}")

    banner("STEP 2 · TRAINING — FROM-SCRATCH ID3")

    X_str: List[List[str]] = df[FEATURE_NAMES].astype(str).values.tolist()
    y_str: List[str] = df[TARGET_NAME].astype(str).tolist()

    custom = ID3DecisionTree()
    custom.fit(X_str, y_str, FEATURE_NAMES)

    print()
    print(f" Tree size            : {custom.n_nodes_} nodes "
          f"({custom.n_leaves_} leaves), depth {custom.depth_}")
    print(f" Training accuracy    : {custom.score(X_str, y_str):.4f}")

    text_tree_path = OUTPUTS / "custom_tree.txt"
    text_tree_path.write_text(custom.render_text(), encoding="utf-8")
    print(f"Text rendering      : {text_tree_path}")

    custom_png = OUTPUTS / "custom_tree.png"
    save_custom_tree_graphviz(custom, custom_png)
    print(f"Graphviz PNG        : {custom_png}")

    banner("STEP 3 · TRAINING — SCIKIT-LEARN")

    X_enc, y_enc, feat_enc, tgt_dec = encode_for_sklearn(df)
    sk = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
    ).fit(X_enc, y_enc)

    print(f"Tree size            : {sk.tree_.node_count} nodes, depth {sk.get_depth()}")
    print(f"Training accuracy    : {sk.score(X_enc, y_enc):.4f}")

    sklearn_png = OUTPUTS / "sklearn_tree.png"
    save_sklearn_tree(sk, feat_enc, tgt_dec, sklearn_png)
    print(f"scikit-learn PNG    : {sklearn_png}")

    banner("STEP 4 · FEATURE IMPORTANCE")

    custom_importance = custom_feature_importance(custom)
    sklearn_importance: Dict[str, float] = dict(zip(FEATURE_NAMES, sk.feature_importances_))

    print("From-scratch ID3 (info-gain weighted):")
    for k in FEATURE_NAMES:
        print(f"  {k:<14} {custom_importance[k]:.4f}")
    print()
    print("scikit-learn (Gini-style impurity reduction):")
    for k in FEATURE_NAMES:
        print(f"  {k:<14} {sklearn_importance[k]:.4f}")

    save_feature_importance(
        custom_importance,
        OUTPUTS / "feature_importance_custom.png",
        title="Feature importance (from-scratch ID3)",
    )
    save_feature_importance(
        sklearn_importance,
        OUTPUTS / "feature_importance_sklearn.png",
        title="Feature importance (scikit-learn)",
    )
    print(f"Importance charts   : {OUTPUTS / 'feature_importance_custom.png'}")
    print(f"                      {OUTPUTS / 'feature_importance_sklearn.png'}")

    banner("STEP 5 · LEAVE-ONE-OUT CROSS VALIDATION")

    print("Running LOOCV on the from-scratch ID3 tree …")
    loo_custom = leave_one_out_cv(
        X=X_str,
        y=y_str,
        feature_names=FEATURE_NAMES,
        factory=lambda: ID3DecisionTree(),
        is_custom=True,
    )

    print("Running LOOCV on the scikit-learn tree …")
    loo_sk = leave_one_out_cv(
        X=X_enc.tolist(),
        y=y_enc.tolist(),
        feature_names=FEATURE_NAMES,
        factory=lambda: DecisionTreeClassifier(criterion="entropy", random_state=42),
        is_custom=False,
    )

    print(f"\nFrom-scratch ID3   LOOCV accuracy : "
          f"{loo_custom['accuracy']:.4f} "
          f"({loo_custom['n_correct']} / {loo_custom['n_total']})")
    print(f"scikit-learn       LOOCV accuracy : "
          f"{loo_sk['accuracy']:.4f} "
          f"({loo_sk['n_correct']} / {loo_sk['n_total']})")

    banner("STEP 6 · CONFUSION MATRIX & CLASSIFICATION REPORT")

    classes = sorted(set(y_str))
    train_preds = custom.predict(X_str)
    cm = confusion_matrix_for(y_str, train_preds, classes)
    save_confusion_matrix(
        cm, classes, OUTPUTS / "confusion_matrix_train.png",
        title="Confusion matrix — training set (from-scratch)",
    )
    print(f"Confusion matrix    : {OUTPUTS / 'confusion_matrix_train.png'}")

    cm_loo = confusion_matrix_for(loo_custom["truths"], loo_custom["predictions"], classes)
    save_confusion_matrix(
        cm_loo, classes, OUTPUTS / "confusion_matrix_loocv.png",
        title="Confusion matrix — Leave-One-Out (from-scratch)",
    )
    print(f"LOOCV confusion mtx : {OUTPUTS / 'confusion_matrix_loocv.png'}")

    print("\nLOOCV per-class metrics (from-scratch tree):")
    report = classification_report_dict(loo_custom["truths"], loo_custom["predictions"], classes)
    print(f"  {'class':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for c in classes:
        m = report[c]
        print(f"  {c:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10d}")
    macro = report["__macro_avg__"]
    print(f"  {'macro avg':<12} {macro['precision']:>10.3f} {macro['recall']:>10.3f} {macro['f1']:>10.3f}")

    banner("STEP 7 · DEMONSTRATION PREDICTIONS")
    print(predict_for_samples(custom, DEMO_PATIENTS))

    banner("DONE")
    print(f"All outputs saved to: {OUTPUTS.resolve()}")
    for f in sorted(OUTPUTS.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name:<35} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
