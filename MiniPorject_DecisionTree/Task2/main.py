from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_processor import (                       
    FingerprintDataset, collect_feature_names, load_raw,
    pivot_binary, pivot_continuous, summary,
)
from decision_tree import DecisionTreeClassifier   
from evaluator import (                            
    confusion_matrix, per_class_metrics,
    plot_accuracy_comparison, plot_class_balance,
    plot_confusion_matrix, plot_depth_sweep, plot_feature_importance,
    plot_tree,
)

DATA_DIR = ROOT
OUT_DIR = ROOT / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CSV = DATA_DIR / "TrainDT.csv"
TEST_CSV  = DATA_DIR / "TestDT.csv"

TREE_HYPERPARAMS = dict(
    criterion="entropy",
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    min_impurity_decrease=0.0,
    random_state=42,
)
def banner(text: str, ch: str = "═") -> str:
    bar = ch * (len(text) + 4)
    return f"\n{bar}\n║ {text} ║\n{bar}"


def evaluate(name: str, clf, train: FingerprintDataset, test: FingerprintDataset,
             classes: list[int]) -> dict:
    """Fit → predict → metrics. Returns a dict of everything the report wants."""
    print(f"\n→ Training [{name}] …")
    t0 = time.perf_counter()
    clf.fit(train.X, train.y)
    fit_secs = time.perf_counter() - t0

    y_pred_train = clf.predict(train.X)
    y_pred_test  = clf.predict(test.X)
    train_acc = float(np.mean(y_pred_train == train.y))
    test_acc  = float(np.mean(y_pred_test  == test.y))

    M = confusion_matrix(test.y, y_pred_test, classes)
    metrics = per_class_metrics(M, classes)

    print(f"  fit time   : {fit_secs:.3f}s")
    if hasattr(clf, "n_nodes_"):
        print(f"  tree size  : {clf.n_nodes_} nodes "
              f"({clf.n_leaves_} leaves, depth={clf.depth_})")
    print(f"  train acc  : {train_acc:.4f}")
    print(f"  test acc   : {test_acc:.4f}  ← {int(round(test_acc * len(test.y)))}/{len(test.y)}")

    return {
        "name": name,
        "fit_seconds": fit_secs,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": M.tolist(),
        "per_class": metrics["per_class"],
        "macro": metrics["macro"],
        "y_pred_test": y_pred_test,
        "n_nodes": getattr(clf, "n_nodes_", None),
        "n_leaves": getattr(clf, "n_leaves_", None),
        "depth": getattr(clf, "depth_", None),
    }


def run_depth_sweep(train: FingerprintDataset, test: FingerprintDataset,
                    depths=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, None)
                    ) -> tuple[list, list, list]:
    """Fit one tree per max_depth and record train/test accuracy."""
    print(banner("Depth sweep"))
    xs, train_accs, test_accs = [], [], []
    for d in depths:
        clf = DecisionTreeClassifier(**{**TREE_HYPERPARAMS, "max_depth": d})
        clf.fit(train.X, train.y)
        ta = float(np.mean(clf.predict(train.X) == train.y))
        te = float(np.mean(clf.predict(test.X)  == test.y))
        label = d if d is not None else "∞"
        # Use a sentinel large number for plotting unconstrained depth.
        xs.append(d if d is not None else max([x for x in depths if x is not None]) + 4)
        train_accs.append(ta); test_accs.append(te)
        print(f"  max_depth={str(label):>3}  →  train={ta:.4f}  test={te:.4f}  "
              f"(nodes={clf.n_nodes_})")
    return xs, train_accs, test_accs

def main() -> None:
    print(banner("WiFi-fingerprint Room Prediction — Decision Tree"))
    print("\n=== PATH CONFIGURATION ===")
    print(f"Script directory : {ROOT}")
    print(f"Data directory   : {DATA_DIR}")
    print(f"Train CSV        : {TRAIN_CSV}")
    print(f"Test CSV         : {TEST_CSV}")
    print(f"Output directory : {OUT_DIR}")

    print("\n=== FILES IN DIRECTORY ===")

    for f in DATA_DIR.iterdir():
        print(f"  -> {f.name}")

    missing = []

    if not TRAIN_CSV.exists():
        missing.append(TRAIN_CSV)

    if not TEST_CSV.exists():
        missing.append(TEST_CSV)

    if missing:
        print("\nERROR: Missing files:\n")

        for m in missing:
            print(f"  - {m}")

        raise FileNotFoundError(
            "\nRequired CSV dataset files could not be found."
        )
    print("\nLoading raw CSVs …")

    train_raw = load_raw(TRAIN_CSV)
    test_raw  = load_raw(TEST_CSV)

    print(f"  TrainDT.csv : {len(train_raw):>6} rows")
    print(f"  TestDT.csv  : {len(test_raw):>6} rows")

    feature_names = collect_feature_names(train_raw, test_raw)

    print(f"\nFeature space (BSSID union) : {len(feature_names)} unique BSSIDs")
    print(banner("Pivoting tall → wide"))

    train_cont = pivot_continuous(train_raw, feature_names)
    test_cont  = pivot_continuous(test_raw, feature_names)

    print(summary(train_cont, "Train (continuous, fill=-100)"))
    print(summary(test_cont,  "Test  (continuous, fill=-100)"))

    train_bin = pivot_binary(train_raw, feature_names)
    test_bin  = pivot_binary(test_raw,  feature_names)
    assert np.array_equal(train_cont.y, train_bin.y)
    assert np.array_equal(test_cont.y,  test_bin.y)

    classes = sorted(set(train_cont.y.tolist()))

    plot_class_balance(
        train_cont.y,
        test_cont.y,
        OUT_DIR / "class_balance.png"
    )
    print(banner("Experiment A — custom DT, continuous RSS"))

    clf_main = DecisionTreeClassifier(**TREE_HYPERPARAMS)

    res_main = evaluate(
        "custom-DT (continuous RSS)",
        clf_main,
        train_cont,
        test_cont,
        classes
    )

    print(banner("Experiment B — custom DT, binary received/not-received"))

    clf_bin = DecisionTreeClassifier(**TREE_HYPERPARAMS)

    res_bin = evaluate(
        "custom-DT (binary)",
        clf_bin,
        train_bin,
        test_bin,
        classes
    )

    print(banner("Experiment C — sklearn DT (sanity check)"))

    try:
        from sklearn.tree import DecisionTreeClassifier as SKDT

        sk = SKDT(
            criterion="entropy",
            max_depth=TREE_HYPERPARAMS["max_depth"],
            min_samples_split=TREE_HYPERPARAMS["min_samples_split"],
            min_samples_leaf=TREE_HYPERPARAMS["min_samples_leaf"],
            random_state=42
        )

        res_sk = evaluate(
            "sklearn-DT (continuous RSS)",
            sk,
            train_cont,
            test_cont,
            classes
        )

    except ImportError:
        print("  sklearn not available — skipping sanity check.")
        res_sk = None

    print(banner("Rendering plots"))

    plot_confusion_matrix(
        np.array(res_main["confusion_matrix"]),
        classes,
        OUT_DIR / "confusion_matrix.png",
        title=f"Confusion matrix — custom DT (acc={res_main['test_accuracy']:.4f})",
    )

    plot_feature_importance(
        clf_main.feature_importances_,
        feature_names,
        OUT_DIR / "feature_importance.png",
        top_n=20,
    )

    plot_tree(
        clf_main.root_,
        clf_main.classes_,
        feature_names,
        OUT_DIR / "tree_diagram.png",
        max_depth=4,
        title=f"Custom decision tree — test accuracy {res_main['test_accuracy']:.4f}",
    )

    acc_compare = {
        res_main["name"]: res_main["test_accuracy"],
        res_bin["name"]:  res_bin["test_accuracy"]
    }

    if res_sk is not None:
        acc_compare[res_sk["name"]] = res_sk["test_accuracy"]

    plot_accuracy_comparison(
        acc_compare,
        OUT_DIR / "accuracy_comparison.png"
    )

    xs, tr_acc, te_acc = run_depth_sweep(train_cont, test_cont)

    plot_depth_sweep(
        xs,
        tr_acc,
        te_acc,
        OUT_DIR / "depth_sweep.png"
    )

    text_tree = clf_main.export_text(
        feature_names=feature_names,
        max_depth=4
    )

    (OUT_DIR / "tree_top4_levels.txt").write_text(
        text_tree,
        encoding="utf-8"
    )

    print(f"\nTop 4 levels of the trained tree:\n{text_tree}")
    n_test = len(test_cont.y)

    n_correct = int(round(res_main["test_accuracy"] * n_test))

    print(banner("FINAL RESULT", ch="█"))

    print(f"\n  Custom decision tree (continuous RSS, missing = -100 dBm)")

    print(
        f"  Test accuracy : {res_main['test_accuracy']:.4f}  "
        f"({n_correct} / {n_test} correct)"
    )

    print(f"  Macro F1      : {res_main['macro']['f1']:.4f}")

    print(
        f"  Tree size     : {res_main['n_nodes']} nodes, "
        f"{res_main['n_leaves']} leaves, depth={res_main['depth']}"
    )

    summary_payload = {
        "task": "Decision tree — predict room from WiFi fingerprint",
        "n_train_fingerprints": int(train_cont.X.shape[0]),
        "n_test_fingerprints":  int(test_cont.X.shape[0]),
        "n_features":           int(len(feature_names)),
        "classes":              classes,
        "hyperparameters":      TREE_HYPERPARAMS,
        "headline_accuracy":    res_main["test_accuracy"],
        "headline_correct":     n_correct,
        "headline_total":       n_test,
        "experiments": {
            "custom_continuous": {
                k: v for k, v in res_main.items()
                if k != "y_pred_test"
            },
            "custom_binary": {
                k: v for k, v in res_bin.items()
                if k != "y_pred_test"
            },
            **({
                "sklearn_continuous": {
                    k: v for k, v in res_sk.items()
                    if k != "y_pred_test"
                }
            } if res_sk is not None else {}),
        },
    }

    (OUT_DIR / "results.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8"
    )

    print(f"\nFull results JSON  →  {OUT_DIR / 'results.json'}")
    print(f" Access the Plots  →  {OUT_DIR}")


if __name__ == "__main__":
    main()
