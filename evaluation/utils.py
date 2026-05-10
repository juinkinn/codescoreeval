import json
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

def load_jsonl(path):
    """Load JSONL file and return list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def map_score(pred):
    """
    Map prediction value to score.
    1 -> 1, 0 -> 0, -1 -> 0.5 (tie)
    """
    if pred == 1:
        return 1
    elif pred == 0:
        return 0
    else:
        return 0.5

def map_pred(pred):
    """
    Map prediction value to label.
    1 -> 1, 0 -> -1, -1 -> 0 (tie)
    """
    if pred == 1:
        return 1
    elif pred == 0:
        return -1
    else:
        return 0

def build_gt_scores(test_data):
    """
    Build ground truth scores from test data.
    Supports two formats:
    1. With 'sub_id' and 'correctness_score', 'efficiency_score', 'readability_score'
    2. With 'sub_id', 'criteria', and 'rank'
    """
    gt = defaultdict(dict)

    for row in test_data:
        sub_id = row.get("sub_id")

        # Format 1: sub_id + criteria_score
        if sub_id and "correctness_score" in row:
            for c in ["correctness", "efficiency", "readability"]:
                gt[c][sub_id] = row[f"{c}_score"]
        # Format 2: sub_id + criteria + rank
        elif sub_id and row.get("criteria") and "rank" in row:
            c = row["criteria"]
            gt[c][sub_id] = row["rank"]

    return gt

def pairwise_consistency(pairwise_data, gt_scores):
    """
    Calculate consistency of pairwise predictions with ground truth.
    Returns: accuracy of predictions matching GT ordering.
    """
    correct = 0
    total = 0

    for row in pairwise_data:
        c = row["criteria"]
        s1 = row["sub_id_1"]
        s2 = row["sub_id_2"]

        if s1 not in gt_scores[c] or s2 not in gt_scores[c]:
            continue

        # GT label: 1 if s1 > s2, -1 if s1 < s2, 0 if equal
        gt_diff = gt_scores[c][s1] - gt_scores[c][s2]
        gt_label = (
            1 if gt_diff > 0
            else -1 if gt_diff < 0
            else 0
        )

        # Predicted label
        pred = map_pred(row["prediction"])

        if pred == gt_label:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0


def pairwise_consistency_conf(pairwise_data, swapped_data, gt_scores):
    """
    Calculate consistency with confidence (debias) method using swap.
    Returns: accuracy of debiased predictions matching GT ordering.
    """
    correct = 0
    total = 0

    for r1, r2 in zip(pairwise_data, swapped_data):
        c = r1["criteria"]
        s1 = r1["sub_id_1"]
        s2 = r1["sub_id_2"]

        if s1 not in gt_scores[c] or s2 not in gt_scores[c]:
            continue

        # GT label
        gt_diff = gt_scores[c][s1] - gt_scores[c][s2]
        gt_label = (
            1 if gt_diff > 0
            else -1 if gt_diff < 0
            else 0
        )

        # Confidence score combining original and swapped
        p1 = map_score(r1["prediction"])
        p2 = map_score(r2["prediction"])
        conf = (p1 + (1 - p2)) / 2

        # Predicted label from confidence
        pred = (
            1 if conf > 0.5
            else -1 if conf < 0.5
            else 0
        )

        if pred == gt_label:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0

def evaluate_ranking(preds, gt_scores):
    """
    Evaluate ranking predictions using Spearman and Kendall correlation.
    Returns: dict with metrics for each criterion.
    """
    results = {}

    for c in ["correctness", "efficiency", "readability"]:
        gt_list = []
        pred_list = []

        for sub_id, true_score in gt_scores[c].items():
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

def evaluate_pointwise(preds, test_data):
    """
    Evaluate pointwise predictions using Spearman correlation and Cohen's kappa.
    Returns: dict with metrics for each criterion.
    """
    results = {}

    for c in ["correctness", "efficiency", "readability"]:
        y_true, y_pred = [], []

        for row in test_data:
            sub = row["sub_id"]

            if sub not in preds or c not in preds[sub]:
                continue

            y_true.append(row[f"{c}_score"])
            y_pred.append(int(preds[sub][c]))

        if not y_true:
            continue

        results[c] = {
            "spearman": spearmanr(y_true, y_pred)[0],
            "kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "n": len(y_true)
        }

    return results

def soft_bias_from_swap(pairwise_data, swapped_data):
    """
    Calculate soft bias (continuous) from original and swapped predictions.
    Returns: average absolute difference between p1 and (1 - p2).
    """
    bias_sum = 0
    total = 0

    for r1, r2 in zip(pairwise_data, swapped_data):
        p1 = map_score(r1["prediction"])
        p2 = map_score(r2["prediction"])

        bias_sum += abs(p1 - (1 - p2))
        total += 1

    return bias_sum / total if total > 0 else 0


def positional_bias_from_swap(pairwise_data, swapped_data):
    """
    Calculate positional bias by comparing predictions with swapped input.
    If prediction remains same -> positional bias exists.
    
    Returns: dict with bias metrics
        - bias_score: proportion of biased cases
        - biased_cases: count of prediction that stayed same
        - total_non_tie: count of non-tie predictions
        - tie_cases: count of tie predictions skipped
        - first_position_kept: times first position was favored
        - second_position_kept: times second position was favored
    """
    biased = 0
    total = 0
    tie_cases = 0
    first_position_kept = 0
    second_position_kept = 0

    for r1, r2 in zip(pairwise_data, swapped_data):
        p1 = r1["prediction"]
        p2 = r2["prediction"]

        # Skip ties
        if p1 == -1 or p2 == -1:
            tie_cases += 1
            continue

        total += 1

        # If prediction remains same -> positional bias
        if p1 == p2:
            biased += 1

            if p1 == 1:
                first_position_kept += 1
            elif p1 == 0:
                second_position_kept += 1

    bias_score = biased / total if total > 0 else 0

    return {
        "bias_score": bias_score,
        "biased_cases": biased,
        "total_non_tie": total,
        "tie_cases": tie_cases,
        "first_position_kept": first_position_kept,
        "second_position_kept": second_position_kept
    }
