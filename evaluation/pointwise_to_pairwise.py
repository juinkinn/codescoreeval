import json
import argparse
import itertools
from collections import defaultdict
from pathlib import Path


CRITERIA = ['efficiency', 'correctness', 'readability']

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def compare_scores(s1, s2):
    if s1 is None or s1 == "null" or s2 is None or s2 == "null":
        return None
    if s1 > s2:
        return 1
    if s1 < s2:
        return 0
    return 0.5


def build_pairwise_records(data, args):
    # Group submissions by problem ID
    grouped_data = defaultdict(list)
    for row in data:
        # If data doesn't have an 'id' field, extract it from 'sub_id'
        problem_id = row.get("id")
        if not problem_id and "sub_id" in row:
            problem_id = row["sub_id"].rsplit("_", 1)[0]

        grouped_data[problem_id].append(row)

    pairwise_records = []

    for problem_id, submissions in grouped_data.items():
        # Create all possible pairs within a single problem
        for sub1, sub2 in itertools.combinations(submissions, 2):
            for criteria in CRITERIA:

                record = {
                    "id": problem_id,
                    "criteria": criteria,
                    "sub_id_1": sub1["sub_id"],
                    "sub_id_2": sub2["sub_id"],
                }

                # Logic to calculate Ground Truth 'label'
                if args.mode in ["gt", "both"]:
                    gt_col = f"{args.gt_prefix}{criteria}{args.gt_suffix}"
                    if gt_col in sub1 and gt_col in sub2:
                        val = compare_scores(sub1[gt_col], sub2[gt_col])
                        if val is not None:
                            record["label"] = val

                # Logic to calculate LLM 'prediction'
                if args.mode in ["llm", "both"]:
                    pred_col = f"{args.pred_prefix}{criteria}{args.pred_suffix}"
                    if pred_col in sub1 and pred_col in sub2:
                        val = compare_scores(sub1[pred_col], sub2[pred_col])
                        if val is not None:
                            record["prediction"] = val

                # Only append the record if it successfully generated at least one target field
                if "label" in record or "prediction" in record:
                    pairwise_records.append(record)

    return pairwise_records


def output_dir_for_input(input_path, args):
    if args.output:
        return Path(args.output)

    path = Path(input_path)
    parts = list(path.parent.parts)
    if "pointwise" in parts:
        parts[parts.index("pointwise")] = "pairwise"
        return Path(*parts)

    return path.parent


def output_name_for_input(input_path):
    path = Path(input_path)
    stem = path.stem.replace(" ", "")

    if stem.endswith("_pointwise_raw"):
        stem = stem[:-len("_pointwise_raw")] + "_raw_pairwise_from_pointwise"
    elif stem.endswith("_pointwise"):
        stem = stem[:-len("_pointwise")] + "_pairwise_from_pointwise"
    else:
        stem = stem + "_pairwise_from_pointwise"

    return stem + path.suffix


def iter_input_files(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]

    return sorted(
        p for p in path.iterdir()
        if p.is_file()
        and p.suffix in [".json", ".jsonl"]
        and "pointwise" in p.stem
    )


def process_file(input_path, args):
    print(f"Reading data from: {input_path}")
    data = load_jsonl(input_path)
    pairwise_records = build_pairwise_records(data, args)

    output_path = output_dir_for_input(input_path, args) / output_name_for_input(input_path)
    save_jsonl(pairwise_records, output_path)
    print(f"Done! Saved {len(pairwise_records)} pairwise records to: {output_path}")


def main(args):
    input_files = iter_input_files(args.input)
    if not input_files:
        raise FileNotFoundError(f"No pointwise JSON/JSONL files found in: {args.input}")

    for input_path in input_files:
        process_file(input_path, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pointwise scores to Pairwise comparisons")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to a pointwise JSONL file or folder")
    parser.add_argument("--output", "-o", type=str, default="", help="Optional output folder; defaults by replacing /pointwise/ with /pairwise/")
    parser.add_argument(
        "--mode", 
        "-m",
        type=str, 
        required=True, 
        choices=["gt", "llm", "both"], 
        help="Choose 'gt' for labels only, 'llm' for predictions only, or 'both' for both columns."
    )
    
    # Customization for Prediction columns (e.g., correctness_score)
    parser.add_argument("--pred_prefix", type=str, default="", help="Prefix for prediction score column")
    parser.add_argument("--pred_suffix", type=str, default="_score", help="Suffix for prediction score column")
    
    # Customization for Ground Truth columns (e.g., gt_correctness_score)
    parser.add_argument("--gt_prefix", type=str, default="gt_", help="Prefix for ground truth score column")
    parser.add_argument("--gt_suffix", type=str, default="_score", help="Suffix for ground truth score column")
    
    args = parser.parse_args()
    main(args)
