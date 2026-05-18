from __future__ import annotations

from typing import Dict, Iterable, List

from decision_tree import ID3DecisionTree

DEMO_PATIENTS: List[Dict[str, str]] = [
    {"age": "young",      "prescription": "myope", "astigmatic": "no",  "tear_rate": "normal"},
    {"age": "young",      "prescription": "hyper", "astigmatic": "yes", "tear_rate": "normal"},
    {"age": "pre",        "prescription": "myope", "astigmatic": "yes", "tear_rate": "normal"},
    {"age": "presbyopic", "prescription": "hyper", "astigmatic": "no",  "tear_rate": "normal"},
    {"age": "presbyopic", "prescription": "myope", "astigmatic": "yes", "tear_rate": "reduced"},
    {"age": "young",      "prescription": "hyper", "astigmatic": "no",  "tear_rate": "reduced"},
]


def predict_for_samples(
    clf: ID3DecisionTree,
    patients: Iterable[Dict[str, str]] = DEMO_PATIENTS,
) -> str:

    patients = list(patients)
    headers = ["#", "age", "prescription", "astigmatic", "tear_rate", "→ predicted lens"]

    rows: List[List[str]] = []
    for i, p in enumerate(patients, start=1):
        pred = clf.predict_one(p)
        rows.append([
            str(i),
            p["age"],
            p["prescription"],
            p["astigmatic"],
            p["tear_rate"],
            str(pred),
        ])

    cols = list(zip(*([headers] + rows)))
    widths = [max(len(c) for c in col) for col in cols]

    def fmt(row: List[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(row, widths))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt(headers), sep]
    out.extend(fmt(r) for r in rows)
    return "\n".join(out)
