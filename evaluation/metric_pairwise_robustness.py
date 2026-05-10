import os
import argparse

from utils import (
    load_jsonl,
    build_gt_scores,
    pairwise_consistency,
    pairwise_consistency_conf,
    soft_bias_from_swap,
    positional_bias_from_swap,
)


def run_model(pairwise_path, swapped_path, test_data):
    """
    Evaluate a pairwise ranking model for bias and consistency.
    Compares predictions with ground truth ordering.
    
    Metrics:
    - Consistency RAW: accuracy of original predictions vs GT
    - Consistency CONF: accuracy of debiased predictions vs GT
    - Soft Bias: continuous bias measure from swapped predictions
    - Positional Bias: whether model favors first/second position
    """
    print(f"\n==============================")
    print(f"Model: {os.path.basename(pairwise_path)}")

    pairwise_data = load_jsonl(pairwise_path)
    gt_scores = build_gt_scores(test_data)

    # ===== RAW (No Debiasing) =====
    cons_raw = pairwise_consistency(pairwise_data, gt_scores)
    print(f"\n[RAW] Consistency: {cons_raw:.4f}")

    # ===== CONF (With Debiasing via Swap) =====
    if os.path.exists(swapped_path):
        swapped_data = load_jsonl(swapped_path)

        cons_conf = pairwise_consistency_conf(pairwise_data, swapped_data, gt_scores)
        soft_bias = soft_bias_from_swap(pairwise_data, swapped_data)
        pos_bias = positional_bias_from_swap(pairwise_data, swapped_data)

        print(f"[CONF] Consistency: {cons_conf:.4f}")

        # ===== SOFT BIAS (Continuous) =====
        print(f"\n[SOFT BIAS]")
        print(f"  Soft Bias Score: {soft_bias:.4f}")
        print(f"  (↓ is better - indicates less bias)")

        # ===== POSITIONAL BIAS (Discrete) =====
        print(f"\n[POSITIONAL BIAS]")
        print(f"  Bias Score: {pos_bias['bias_score']:.4f}")
        print(f"  Biased Cases: {pos_bias['biased_cases']}/{pos_bias['total_non_tie']}")
        print(f"  Tie Cases Skipped: {pos_bias['tie_cases']}")
        print(f"  First Position Favored: {pos_bias['first_position_kept']}")
        print(f"  Second Position Favored: {pos_bias['second_position_kept']}")
        print(f"  (bias_score ↓ is better - 0 = unbiased, 1 = fully biased)")

        # ===== IMPROVEMENT (CONF vs RAW) =====
        print(f"\n[IMPROVEMENT - CONF vs RAW]")
        print(f"  Δ Consistency: {cons_conf - cons_raw:+.4f}")
        print(f"  Δ Soft Bias: {-soft_bias:+.4f} (negative is improvement)")

    else:
        print("\n⚠ Missing swapped file → cannot measure debiasing improvement")
        print("  Skipping CONF metrics and bias analysis")


def main(pairwise_dir, test_path):
    """
    Evaluate all pairwise models in a directory.
    """
    test_data = load_jsonl(test_path)

    models_evaluated = 0

    for file in sorted(os.listdir(pairwise_dir)):
        if not file.endswith("_pairwise.jsonl"):
            continue

        pairwise_path = os.path.join(pairwise_dir, file)
        swapped_path = pairwise_path.replace("_pairwise.jsonl", "_pairwise_swapped.jsonl")

        run_model(pairwise_path, swapped_path, test_data)
        models_evaluated += 1

    if models_evaluated == 0:
        print(f"⚠ No pairwise models found in {pairwise_dir}")
    else:
        print(f"\n\n{'='*50}")
        print(f"Evaluated {models_evaluated} model(s)")
        print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise ranking models for BIAS and CONSISTENCY only"
    )

    parser.add_argument(
        "--pairwise_dir",
        type=str,
        required=True,
        help="Directory containing pairwise prediction files (*_pairwise.jsonl)"
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default="data/original_test.jsonl",
        help="Path to test data with ground truth scores"
    )

    args = parser.parse_args()

    main(
        pairwise_dir=args.pairwise_dir,
        test_path=args.test_path,
    )