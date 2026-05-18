import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import argparse
import os

parser = argparse.ArgumentParser(description='Evaluate metrics for human and model ratings.')
parser.add_argument('--input', '-i', type=str, required=True, help='Path to file or folder')
args = parser.parse_args()

criteria = ['correctness', 'efficiency', 'readability']

ANCHOR_PATH = "data/pairwise_test_with_anchors.jsonl"

anchor_ids = set()
anchor_df = pd.read_json(ANCHOR_PATH, lines=True)

for _, row in anchor_df.iterrows():
    for _, sub_id in row["anchors"].items():
        anchor_ids.add(sub_id)


def evaluate_file(filepath):
    df = pd.read_json(filepath, lines=True)

    eval_df = df[~df["sub_id"].isin(anchor_ids)]
    n = len(eval_df)

    print(f"\nFile: {os.path.basename(filepath)}")
    print(f"N = {n}")
    print("{:<15} {:<15} {:<15}".format("Criterion", "Kappa", "MSE"))
    print("-" * 50)

    for criterion in criteria:
        try:
            y_true = eval_df[f'gt_{criterion}_score']
            y_pred = eval_df[f'{criterion}_score']

            kappa = cohen_kappa_score(
                y_true,
                y_pred,
                weights='quadratic'
            )

            mse = mean_squared_error(
                y_true,
                y_pred
            )

            print(
                f"{criterion:<15} "
                f"{kappa:<15.4f} "
                f"{mse:<15.4f}"
            )

        except Exception as e:
            print(f"{criterion:<15} Error: {e}")


if os.path.isdir(args.input):
    for filename in os.listdir(args.input):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            evaluate_file(os.path.join(args.input, filename))
else:
    evaluate_file(args.input)