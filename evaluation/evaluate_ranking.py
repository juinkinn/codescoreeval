import json
import os
import argparse
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_scores_raw(pairwise_data):
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def map_score(pred):
        if pred == 1:
            return 1
        elif pred == 0:
            return 0
        else:
            return 0.5

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

    def map_score(pred):
        if pred == 1:
            return 1
        elif pred == 0:
            return 0
        else:
            return 0.5

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

def build_gt_scores(test_data):
    gt = defaultdict(dict)

    for row in test_data:
        sub_id = row.get("sub_id")
        c = row.get("criteria")
        if c and sub_id and 'rank' in row:
            gt[c][sub_id] = row["rank"]
        else:
            for c in ["correctness", "efficiency", "readability"]:
                gt[c][sub_id] = row[f"{c}_score"]

    return gt


def pairwise_consistency(pairwise_data, gt_scores):
    correct = 0
    total = 0

    def map_pred(pred):
        if pred == 1:
            return 1
        elif pred == 0:
            return -1
        else:
            return 0

    for row in pairwise_data:
        c = row["criteria"]
        s1 = row["sub_id_1"]
        s2 = row["sub_id_2"]

        if s1 not in gt_scores[c] or s2 not in gt_scores[c]:
            continue

        gt_diff = gt_scores[c][s1] - gt_scores[c][s2]

        if gt_diff > 0:
            gt_label = 1
        elif gt_diff < 0:
            gt_label = -1
        else:
            gt_label = 0

        pred = map_pred(row["prediction"])

        if pred == gt_label:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0

def pairwise_consistency_conf(pairwise_data, swapped_data, gt_scores):
    correct = 0
    total = 0

    def map_score(pred):
        if pred == 1:
            return 1
        elif pred == 0:
            return 0
        else:
            return 0.5

    for r1, r2 in zip(pairwise_data, swapped_data):
        c = r1["criteria"]
        s1 = r1["sub_id_1"]
        s2 = r1["sub_id_2"]

        if s1 not in gt_scores[c] or s2 not in gt_scores[c]:
            continue

        gt_diff = gt_scores[c][s1] - gt_scores[c][s2]
        if gt_diff > 0:
            gt_label = 1
        elif gt_diff < 0:
            gt_label = -1
        else:
            gt_label = 0

        p1 = map_score(r1["prediction"])
        p2 = map_score(r2["prediction"])
        conf = (p1 + (1 - p2)) / 2

        if conf > 0.5:
            pred = 1
        elif conf < 0.5:
            pred = -1
        else:
            pred = 0

        if pred == gt_label:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0


def evaluate(preds, gt_scores):
    results = {}

    for c in ["correctness", "efficiency", "readability"]:
        gt_list = []
        pred_list = []

        # Duyệt qua các bài code có trong tập Ground Truth
        for sub_id, true_score in gt_scores[c].items():
            # Nếu AI có dự đoán cho bài này
            if sub_id in preds and c in preds[sub_id]:
                gt_list.append(true_score)
                pred_list.append(preds[sub_id][c])

        if len(gt_list) == 0:
            continue

        spearman, _ = spearmanr(gt_list, pred_list)
        kendall, _ = kendalltau(gt_list, pred_list)

        results[c] = {
            "spearman": spearman,
            "kendall": kendall,
            "n": len(gt_list)
        }

    return results


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
        results_raw = evaluate(preds_raw, gt_scores)

        print_results("RAW (NO SWAP)", results_raw)

    print(f"\nConsistency RAW: {cons_raw:.4f}")
    print(f"Bias RAW: N/A (no swap)")

    # CONF
    if os.path.exists(swapped_path):
        swapped_data = load_jsonl(swapped_path)

        scores_conf, bias_conf = build_scores_conf(pairwise_data, swapped_data)
        preds_conf = build_predictions(scores_conf)
        results_conf = evaluate(preds_conf, gt_scores)

        cons_conf = pairwise_consistency_conf(pairwise_data, swapped_data, gt_scores)

        if mode in ["conf", "both"]:
            print_results("CONF (WITH SWAP)", results_conf)

        print(f"\nConsistency CONF: {cons_conf:.4f}")
        print(f"Bias CONF: {bias_conf:.4f}")

        # DELTA
        print("\n--- DELTA (CONF - RAW) ---")
        print(f"Δ Consistency: {cons_conf - cons_raw:.4f}")
        print(f"Δ Bias: {-bias_conf:.4f} (↓ is better)")

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

