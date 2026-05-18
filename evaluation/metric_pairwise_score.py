import json
import os
import argparse
from collections import defaultdict

from sklearn.metrics import mean_squared_error, cohen_kappa_score

from utils import (
    load_jsonl,
    map_score,
    build_gt_scores,
)

CRITERIA = [
    "correctness",
    "efficiency",
    "readability"
]


def build_anchor_map(anchor_data):
    anchor_map = defaultdict(dict)

    for row in anchor_data:
        key = (row["id"], row["criteria"])

        for score, sub_id in row["anchors"].items():
            anchor_map[key][int(score)] = sub_id

    return anchor_map


def collect_anchor_sub_ids(anchor_data):
    anchor_ids = set()

    for row in anchor_data:
        for _, sub_id in row["anchors"].items():
            anchor_ids.add(sub_id)

    return anchor_ids


def build_relations(pairwise_data, swapped_data):
    relations = defaultdict(float)

    # original
    for r in pairwise_data:
        a, b = r["sub_id_1"], r["sub_id_2"]
        p = map_score(r["prediction"])

        relations[(a, b)] += p
        relations[(b, a)] += (1 - p)

    # swapped
    for r in swapped_data:
        a, b = r["sub_id_1"], r["sub_id_2"]
        p = map_score(r["prediction"])

        relations[(a, b)] += (1 - p)
        relations[(b, a)] += p

    return relations


def compare(a, b, relations):
    return relations[(a, b)] - relations[(b, a)]


def apply_soft_monotonic(rel, scores):
    """
    Enforce soft monotonic constraint:
    rel[1] >= rel[2] >= ... >= rel[5]

    Instead of hard overwrite,
    average conflicting neighbors.
    """

    fixed = rel.copy()

    for i in range(len(scores) - 1):
        low = scores[i]
        high = scores[i + 1]

        # contradiction
        if fixed[low] < fixed[high]:

            avg = (fixed[low] + fixed[high]) / 2

            fixed[low] = avg
            fixed[high] = avg

    return fixed

def infer_score(sub_id, anchors, relations):
    scores = sorted(anchors.keys())

    rel = {
        s: compare(sub_id, anchors[s], relations)
        for s in scores
    }

    # apply soft monotonic smoothing
    rel = apply_soft_monotonic(rel, scores)

    # above highest anchor
    if rel[scores[-1]] >= 0:
        return scores[-1]

    # below lowest anchor
    if rel[scores[0]] < 0:
        return scores[0]

    # find interval
    for i in range(len(scores) - 1):
        low, high = scores[i], scores[i + 1]

        if rel[low] >= 0 and rel[high] <= 0:
            return (low + high) // 2

    return scores[0]

def build_predictions(pairwise_data, swapped_data, anchor_map):
    preds = defaultdict(dict)

    relations = build_relations(pairwise_data, swapped_data)

    groups = defaultdict(list)

    for r in pairwise_data:
        groups[(r["id"], r["criteria"])].append(r)

    for (base_id, c), rows in groups.items():
        anchors = anchor_map.get((base_id, c))

        if not anchors:
            continue

        subs = set()

        for r in rows:
            subs.add(r["sub_id_1"])
            subs.add(r["sub_id_2"])

        for sub_id in subs:
            preds[sub_id][c] = infer_score(
                sub_id,
                anchors,
                relations
            )

    return preds

def evaluate(preds, test_data, anchor_ids):
    results = {}

    for c in CRITERIA:
        y_true, y_pred = [], []

        for r in test_data:
            sub = r["sub_id"]

            # exclude anchors
            if sub in anchor_ids:
                continue

            if sub not in preds or c not in preds[sub]:
                continue

            y_true.append(r[f"{c}_score"])
            y_pred.append(int(preds[sub][c]))

        if not y_true:
            continue

        mse = mean_squared_error(y_true, y_pred)

        kappa = cohen_kappa_score(
            y_true,
            y_pred,
            weights="quadratic"
        )

        results[c] = {
            "mse": mse,
            "kappa": kappa,
            "n": len(y_true)
        }

    return results

def run_model(pairwise_path, swapped_path, anchor_path, test_data, mode):
    print("\n==============================")
    print(f"Model: {os.path.basename(pairwise_path)}")

    pairwise_data = load_jsonl(pairwise_path)
    swapped_data = load_jsonl(swapped_path)
    anchor_data = load_jsonl(anchor_path)

    anchor_map = build_anchor_map(anchor_data)
    anchor_ids = collect_anchor_sub_ids(anchor_data)

    preds = build_predictions(
        pairwise_data,
        swapped_data,
        anchor_map
    )

    if mode in ["raw", "both"]:
        res = evaluate(preds, test_data, anchor_ids)

        print("{:<15} {:<15} {:<15} {:<10}".format(
            "Criterion", "Kappa", "MSE", "N"
        ))
        print("-" * 60)

        for k, v in res.items():
            print(
                f"{k:<15} "
                f"{v['kappa']:<15.4f} "
                f"{v['mse']:<15.4f} "
                f"{v['n']:<10}"
            )

def main(pairwise_dir, anchor_path, test_path, mode):
    test_data = load_jsonl(test_path)

    for f in os.listdir(pairwise_dir):
        if not f.endswith("_pairwise.jsonl"):
            continue

        pairwise_path = os.path.join(pairwise_dir, f)

        swapped_path = pairwise_path.replace(
            "_pairwise.jsonl",
            "_pairwise_swapped.jsonl"
        )

        run_model(
            pairwise_path,
            swapped_path,
            anchor_path,
            test_data,
            mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise_dir", required=True)
    parser.add_argument(
        "--anchor_path",
        default="data/pairwise_test_with_anchors.jsonl"
    )
    parser.add_argument(
        "--test_path",
        default="data/original_test.jsonl"
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["raw", "conf", "both"]
    )
    args = parser.parse_args()

    main(
        args.pairwise_dir,
        args.anchor_path,
        args.test_path,
        args.mode
    )