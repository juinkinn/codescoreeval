import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import argparse
import os

parser = argparse.ArgumentParser(description='Evaluate metrics for human and model ratings.')
parser.add_argument('--input', '-i', type=str, required=True, help='Path to file or folder')
args = parser.parse_args()

criteria = ['correctness', 'efficiency', 'readability']

def evaluate_file(filepath):
    df = pd.read_json(filepath, lines=True)
    results = []

    print(f"\nFile: {os.path.basename(filepath)}")
    print("{:<15} {:<25} {:<20}".format("Criterion", "Spearman", "Kappa"))
    print("-" * 60)

    for criterion in criteria:
        try:
            spearman_corr, _ = spearmanr(df[f'gt_{criterion}_score'], df[f'{criterion}_score'])
            kappa = cohen_kappa_score(
                df[f'gt_{criterion}_score'],
                df[f'{criterion}_score'],
                weights='quadratic'
            )
            results.append((criterion, spearman_corr, kappa))
            print(f"{criterion:<15} {spearman_corr:<25.4f} {kappa:<20.4f}")
        except Exception as e:
            print(f"{criterion:<15} Error: {e}")

    print("-" * 60)
    return results


if os.path.isdir(args.input):
    for filename in os.listdir(args.input):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            filepath = os.path.join(args.input, filename)
            evaluate_file(filepath)
else:
    evaluate_file(args.input)