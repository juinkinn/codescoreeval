import json
import os
import argparse
from collections import defaultdict

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from utils import (
    load_jsonl,
    map_score,
    build_gt_scores,
)


def build_anchor_map(anchor_data):
    anchor_map = defaultdict(dict)

    for row in anchor_data:
        key = (row["id"], row["criteria"])

        for score, sub_id in row["anchors"].items():
            anchor_map[key][int(score)] = sub_id

    return anchor_map

def build_relations(pairwise_data, swapped_data):
    """
    relations[(a,b)]:
        cumulative support that a > b
    """

    relations = defaultdict(float)

    # original
    for r in pairwise_data:
        a, b = r["sub_id_1"], r["sub_id_2"]
        p = map_score(r["prediction"])

        relations[(a, b)] += p
        relations[(b, a)] += (1 - p)

    # swapped (debias)
    for r in swapped_data:
        a, b = r["sub_id_1"], r["sub_id_2"]
        p = map_score(r["prediction"])

        relations[(a, b)] += (1 - p)
        relations[(b, a)] += p

    return relations


def compare(a, b, relations):
    return relations[(a, b)] - relations[(b, a)]

def infer_score(sub_id, anchors, relations):
    scores = sorted(anchors.keys())

    rel = {
        s: compare(sub_id, anchors[s], relations)
        for s in scores
    }

    if rel[scores[-1]] >= 0:
        return scores[-1]

    if rel[scores[0]] < 0:
        return scores[0]

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

def evaluate(preds, test_data):
    results = {}

    for c in ["correctness", "efficiency", "readability"]:
        y_true, y_pred = [], []

        for r in test_data:
            sub = r["sub_id"]

            if sub not in preds or c not in preds[sub]:
                continue

            y_true.append(r[f"{c}_score"])
            y_pred.append(int(preds[sub][c]))

        if not y_true:
            continue

        results[c] = {
            "spearman": spearmanr(y_true, y_pred)[0],
            "kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
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
    gt = build_gt_scores(test_data)

    preds = build_predictions(
        pairwise_data,
        swapped_data,
        anchor_map
    )

    if mode in ["raw", "both"]:
        res = evaluate(preds, test_data)

        for k, v in res.items():
            print(
                f"\n[{k}] "
                f"Spearman={v['spearman']:.4f} "
                f"Kappa={v['kappa']:.4f} "
                f"n={v['n']}"
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