import os
import argparse
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
)

parser = argparse.ArgumentParser(
    description="Evaluate metrics for human and model ratings."
)
parser.add_argument(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Path to file or folder",
)
parser.add_argument(
    "--anchor_path",
    "-a",
    type=str,
    default="data/pairwise_test_with_anchors.jsonl",
    help="Path to anchor data file",
)
args = parser.parse_args()

criteria = ["correctness", "efficiency", "readability"]

ANCHOR_PATH = args.anchor_path

anchor_df = pd.read_json(ANCHOR_PATH, lines=True)

anchor_ids = set()
for _, row in anchor_df.iterrows():
    for _, sub_id in row["anchors"].items():
        anchor_ids.add(sub_id)


def evaluate_file(filepath):
    df = pd.read_json(filepath, lines=True)

    if "sub_id" not in df.columns:
        print(f"Skipping {filepath}: missing 'sub_id'")
        return

    eval_df = df[~df["sub_id"].isin(anchor_ids)]

    print(f"\nFile: {os.path.basename(filepath)}")
    print(f"N = {len(eval_df)}")
    print(
        "{:<15} {:>10} {:>10} {:>10} {:>10}".format(
            "Criterion", "QWK", "MAE", "MSE", "RMSE"
        )
    )
    print("-" * 60)

    for criterion in criteria:
        try:
            y_true = eval_df[f"gt_{criterion}_score"]
            y_pred = eval_df[f"{criterion}_score"]

            qwk = cohen_kappa_score(
                y_true,
                y_pred,
                weights="quadratic",
            )

            mae = mean_absolute_error(
                y_true,
                y_pred,
            )

            mse = mean_squared_error(
                y_true,
                y_pred,
            )

            rmse = mse ** 0.5

            print(
                "{:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    criterion,
                    qwk,
                    mae,
                    mse,
                    rmse,
                )
            )

        except Exception as e:
            print(f"{criterion:<15} Error: {e}")


if os.path.isdir(args.input):
    for filename in sorted(os.listdir(args.input)):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            evaluate_file(os.path.join(args.input, filename))
else:
    evaluate_file(args.input)