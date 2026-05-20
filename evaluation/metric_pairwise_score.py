import os
import argparse
from collections import defaultdict

from sklearn.metrics import (
    mean_squared_error,
    cohen_kappa_score
)

from utils import (
    load_jsonl,
    map_score,
)

CRITERIA = [
    "correctness",
    "efficiency",
    "readability"
]


def build_anchor_map(anchor_data):
    anchor_map = defaultdict(dict)

    for row in anchor_data:
        key = (
            row["id"],
            row["criteria"]
        )

        for score, sub_id in row["anchors"].items():
            anchor_map[key][int(score)] = sub_id

    return anchor_map


def collect_anchor_sub_ids(anchor_data):
    anchor_ids = set()

    for row in anchor_data:
        for _, sub_id in row["anchors"].items():
            anchor_ids.add(sub_id)

    return anchor_ids


def build_relations(
    pairwise_data,
    swapped_data
):
    """
    relations[(a,b)]:
        support that a > b
    """

    relations = defaultdict(float)

    # original
    for r in pairwise_data:
        a = r["sub_id_1"]
        b = r["sub_id_2"]
        p = map_score(r["prediction"])

        relations[(a, b)] += p
        relations[(b, a)] += (1 - p)

    # swapped
    for r in swapped_data:
        a = r["sub_id_1"]
        b = r["sub_id_2"]
        p = map_score(r["prediction"])

        relations[(a, b)] += (1 - p)
        relations[(b, a)] += p

    return relations


def build_global_ranking(relations):
    """
    Build ranking score from graph.

    node_score:
        overall strength of sample
    """

    node_score = defaultdict(float)

    for (a, b), score in relations.items():
        reverse = relations[(b, a)]
        margin = score - reverse
        node_score[a] += margin

    return node_score

def calibrate_with_anchors(
    sub_id,
    anchors,
    node_score
):
    """
    Convert ranking score
    -> absolute score using anchors.
    """
    scores = sorted(anchors.keys())

    sample_rank = node_score[sub_id]
    anchor_ranks = {}
    for s in scores:
        anchor_id = anchors[s]

        anchor_ranks[s] = node_score[anchor_id]

    # above highest anchor
    if sample_rank >= anchor_ranks[scores[-1]]:
        return scores[-1]

    # below lowest anchor
    if sample_rank <= anchor_ranks[scores[0]]:
        return scores[0]

    # interval localization
    for i in range(len(scores) - 1):
        low = scores[i]
        high = scores[i + 1]

        low_rank = anchor_ranks[low]
        high_rank = anchor_ranks[high]

        if (
            low_rank <= sample_rank
            <= high_rank
        ):

            dist_low = abs(
                sample_rank - low_rank
            )

            dist_high = abs(
                sample_rank - high_rank
            )

            if dist_high < dist_low:
                return high

            return low

    return scores[0]

def build_predictions(
    pairwise_data,
    swapped_data,
    anchor_map
):

    preds = defaultdict(dict)

    relations = build_relations(
        pairwise_data,
        swapped_data
    )

    node_score = build_global_ranking(
        relations
    )

    groups = defaultdict(list)

    for r in pairwise_data:

        key = (
            r["id"],
            r["criteria"]
        )

        groups[key].append(r)

    for (base_id, c), rows in groups.items():
        anchors = anchor_map.get(
            (base_id, c)
        )

        if not anchors:
            continue

        subs = set()

        for r in rows:
            subs.add(r["sub_id_1"])
            subs.add(r["sub_id_2"])

        for sub_id in subs:
            preds[sub_id][c] = calibrate_with_anchors(
                sub_id,
                anchors,
                node_score
            )

    return preds

def evaluate(
    preds,
    test_data,
    anchor_ids
):
    results = {}

    for c in CRITERIA:

        y_true = []
        y_pred = []

        for r in test_data:
            sub = r["sub_id"]

            # exclude anchors
            if sub in anchor_ids:
                continue

            if sub not in preds:
                continue

            if c not in preds[sub]:
                continue

            y_true.append(
                r[f"{c}_score"]
            )

            y_pred.append(
                int(preds[sub][c])
            )

        if not y_true:
            continue

        mse = mean_squared_error(
            y_true,
            y_pred
        )

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

def run_model(
    pairwise_path,
    swapped_path,
    anchor_path,
    test_data,
    mode
):
    print("\n==============================")

    print(
        f"Model: "
        f"{os.path.basename(pairwise_path)}"
    )

    pairwise_data = load_jsonl(pairwise_path)
    swapped_data = load_jsonl(swapped_path)
    anchor_data = load_jsonl(anchor_path)

    anchor_map = build_anchor_map(anchor_data)
    anchor_ids = collect_anchor_sub_ids(
        anchor_data
    )

    preds = build_predictions(
        pairwise_data,
        swapped_data,
        anchor_map
    )

    if mode in ["raw", "both"]:
        res = evaluate(
            preds,
            test_data,
            anchor_ids
        )

        print(
            "{:<15} {:<15} {:<15} {:<10}".format(
                "Criterion",
                "Kappa",
                "MSE",
                "N"
            )
        )

        print("-" * 60)

        for k, v in res.items():
            print(
                f"{k:<15} "
                f"{v['kappa']:<15.4f} "
                f"{v['mse']:<15.4f} "
                f"{v['n']:<10}"
            )

def main(
    pairwise_dir,
    anchor_path,
    test_path,
    mode
):
    test_data = load_jsonl(test_path)

    for f in os.listdir(pairwise_dir):
        if not f.endswith("_pairwise.jsonl"):
            continue

        pairwise_path = os.path.join(
            pairwise_dir,
            f
        )

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
    parser.add_argument(
        "--pairwise_dir",
        required=True
    )
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