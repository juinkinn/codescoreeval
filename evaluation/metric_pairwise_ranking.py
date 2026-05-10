import os
import argparse
from collections import defaultdict
from utils import (
    load_jsonl,
    map_score,
    build_gt_scores,
    pairwise_consistency,
    pairwise_consistency_conf,
    evaluate_ranking,
)


def build_scores_raw(pairwise_data):
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for row in pairwise_data:
        base_id = row["id"]
        c = row["criteria"]
        s1 = row["sub_id_1"]
        s2 = row["sub_id_2"]

        p = map_score(row["prediction"])

        scores[base_id][c][s1] += p
        scores[base_id][c][s2] += (1 - p)

    return scores

def build_scores_conf(pairwise_data, swapped_data):
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    soft_bias_sum = 0
    total = 0

    for row1, row2 in zip(pairwise_data, swapped_data):
        base_id = row1["id"]
        c = row1["criteria"]
        s1 = row1["sub_id_1"]
        s2 = row1["sub_id_2"]

        p1 = map_score(row1["prediction"])
        p2 = map_score(row2["prediction"])

        # CONF aggregation
        conf = (p1 + (1 - p2)) / 2

        scores[base_id][c][s1] += conf
        scores[base_id][c][s2] += (1 - conf)

        # SOFT BIAS (continuous)
        soft_bias_sum += abs(p1 - (1 - p2))
        total += 1

    soft_bias = soft_bias_sum / total if total > 0 else 0

    return scores, soft_bias

def build_predictions(scores):
    preds = defaultdict(dict)
    for base_id in scores:
        for c in scores[base_id]:
            for sub_id, sc in scores[base_id][c].items():
                preds[sub_id][c] = sc
    return preds

def build_predictions(scores):
    preds = defaultdict(dict)
    for base_id in scores:
        for c in scores[base_id]:
            for sub_id, sc in scores[base_id][c].items():
                preds[sub_id][c] = sc
    return preds


def print_results(title, results):
    print(f"\n=== {title} ===")
    for c, res in results.items():
        print(f"\n[{c}]")
        print(f"Spearman: {res['spearman']:.4f}")
        print(f"Kendall: {res['kendall']:.4f}")
        print(f"Samples: {res['n']}")


def run_model(pairwise_path, swapped_path, test_data, mode):
    print(f"\n==============================")
    print(f"Model: {os.path.basename(pairwise_path)}")

    pairwise_data = load_jsonl(pairwise_path)
    gt_scores = build_gt_scores(test_data)

    # RAW
    cons_raw = pairwise_consistency(pairwise_data, gt_scores)

    if mode in ["raw", "both"]:
        scores_raw = build_scores_raw(pairwise_data)
        preds_raw = build_predictions(scores_raw)
        results_raw = evaluate_ranking(preds_raw, gt_scores)

        print_results("RAW (NO SWAP)", results_raw)

    print(f"\nConsistency RAW: {cons_raw:.4f}")
    print(f"Bias RAW: N/A (no swap)")

    # CONF
    if os.path.exists(swapped_path):
        swapped_data = load_jsonl(swapped_path)

        scores_conf, bias_conf = build_scores_conf(pairwise_data, swapped_data)
        preds_conf = build_predictions(scores_conf)
        results_conf = evaluate_ranking(preds_conf, gt_scores)

        cons_conf = pairwise_consistency_conf(pairwise_data, swapped_data, gt_scores)

        if mode in ["conf", "both"]:
            print_results("CONF (WITH SWAP)", results_conf)

        print(f"\nConsistency CONF: {cons_conf:.4f}")
        print(f"Bias CONF: {bias_conf:.4f}")

    else:
        print("Missing swapped file → skip CONF")



def main(pairwise_dir, test_path, mode):
    test_data = load_jsonl(test_path)

    for file in os.listdir(pairwise_dir):
        if not file.endswith("_pairwise.jsonl"):
            continue

        pairwise_path = os.path.join(pairwise_dir, file)
        swapped_path = pairwise_path.replace("_pairwise.jsonl", "_pairwise_swapped.jsonl")

        run_model(pairwise_path, swapped_path, test_data, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="data/original_test.jsonl")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["raw", "conf", "both"]
    )

    args = parser.parse_args()

    main(
        pairwise_dir=args.pairwise_dir,
        test_path=args.test_path,
        mode=args.mode
    )

