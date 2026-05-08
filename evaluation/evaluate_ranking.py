import json
import os
import argparse
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_scores_raw(pairwise_data):
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for row in pairwise_data:
        base_id = row["id"]
        c = row["criteria"]
        s1 = row["sub_id_1"]
        s2 = row["sub_id_2"]

        # Support both 'prediction' (from LLM) and 'label' (from GT)
        pred_val = row.get("prediction", row.get("label", 0.5))

        if pred_val == 1:
            p = 1
        elif pred_val == 0:
            p = 0
        else:
            p = 0.5

        scores[base_id][c][s1] += p
        scores[base_id][c][s2] += (1 - p)

    return scores

def build_predictions(scores):
    preds = defaultdict(dict)

    for base_id in scores:
        for c in scores[base_id]:
            for sub_id, raw_score in scores[base_id][c].items():
                preds[sub_id][c] = raw_score

    return preds

def build_gt_scores(test_data):
    gt = defaultdict(dict)

    for row in test_data:
        sub_id = row["sub_id"]
        for c in ["correctness", "efficiency", "readability"]:
            if f"{c}_score" in row:
                gt[c][sub_id] = row[f"{c}_score"]

    return gt

def pairwise_consistency(pairwise_data, gt_scores):
    correct = 0
    total = 0

    for row in pairwise_data:
        c = row["criteria"]
        s1 = row["sub_id_1"]
        s2 = row["sub_id_2"]

        if s1 not in gt_scores[c] or s2 not in gt_scores[c]:
            continue

        # Ground Truth relation
        gt_diff = gt_scores[c][s1] - gt_scores[c][s2]
        if gt_diff > 0:
            gt_label = 1
        elif gt_diff < 0:
            gt_label = -1
        else:
            gt_label = 0

        # Prediction relation
        pred_val = row.get("prediction", row.get("label", 0.5))
        if pred_val == 1:
            pred = 1
        elif pred_val == 0:
            pred = -1
        else:
            pred = 0

        if pred == gt_label:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0

def evaluate(preds, test_data):
    results = {}

    for c in ["correctness", "efficiency", "readability"]:
        gt_list = []
        pred_list = []

        for row in test_data:
            sub_id = row["sub_id"]

            if sub_id not in preds or c not in preds[sub_id]:
                continue

            gt_list.append(row[f"{c}_score"])
            pred_list.append(preds[sub_id][c])

        if len(gt_list) == 0:
            continue

        # Calculate Ranking Metrics
        spearman, _ = spearmanr(gt_list, pred_list)
        kendall, _ = kendalltau(gt_list, pred_list)

        results[c] = {
            "spearman": spearman,
            "kendall": kendall,
            "n": len(gt_list)
        }

    return results

def print_results(title, results, consistency):
    print(f"\n=== {title} ===")
    print(f"Pairwise Consistency (Accuracy): {consistency:.4f}")
    
    for c, res in results.items():
        print(f"\n[{c.upper()}]")
        print(f"Spearman Correlation: {res['spearman']:.4f}")
        print(f"Kendall Tau:          {res['kendall']:.4f}")
        print(f"Number of samples:    {res['n']}")

def run_evaluation(pairwise_path, test_data):
    print(f"\n==============================")
    print(f"Evaluating file: {os.path.basename(pairwise_path)}")

    pairwise_data = load_jsonl(pairwise_path)
    gt_scores = build_gt_scores(test_data)

    # 1. Calculate Pairwise Consistency (Accuracy)
    cons = pairwise_consistency(pairwise_data, gt_scores)

    # 2. Build Raw Scores (Points) -> Direct Predictions
    scores_raw = build_scores_raw(pairwise_data)
    preds_raw = build_predictions(scores_raw)

    # 3. Evaluate using Ranking metrics
    results = evaluate(preds_raw, test_data)

    # 4. Print Results
    print_results("RANKING EVALUATION", results, cons)

def main(args):
    test_data = load_jsonl(args.test_path)

    # Support running on a single file OR an entire directory
    if os.path.isfile(args.input_path):
        run_evaluation(args.input_path, test_data)
    elif os.path.isdir(args.input_path):
        for file in os.listdir(args.input_path):
            if file.endswith(".jsonl"):
                file_path = os.path.join(args.input_path, file)
                run_evaluation(file_path, test_data)
    else:
        print(f"Error: Path {args.input_path} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pointwise/Pairwise using Pure Ranking (No mapping)")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the pairwise JSONL file or directory")
    parser.add_argument("--test_path", type=str, default="data/original_test.jsonl", help="Path to Ground Truth test data")

    args = parser.parse_args()
    main(args)
