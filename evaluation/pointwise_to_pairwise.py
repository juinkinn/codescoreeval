import json
import argparse
import itertools
from collections import defaultdict

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

def main(args):
    print(f"Reading data from: {args.input}")
    data = load_jsonl(args.input)
    
    # Group submissions by problem ID
    grouped_data = defaultdict(list)
    for row in data:
        # If data doesn't have an 'id' field, extract it from 'sub_id'
        problem_id = row.get("id")
        if not problem_id and "sub_id" in row:
            problem_id = row["sub_id"].rsplit("_", 1)[0]
            
        grouped_data[problem_id].append(row)
        
    criteria_list = ['efficiency', 'correctness', 'readability']
    pairwise_records = []

    for problem_id, submissions in grouped_data.items():
        # Create all possible pairs within a single problem
        for sub1, sub2 in itertools.combinations(submissions, 2):
            for criteria in criteria_list:
                
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
                        s1 = sub1[gt_col]
                        s2 = sub2[gt_col]
                        
                        if s1 > s2: val = 1
                        elif s1 < s2: val = 0
                        else: val = 0.5
                        
                        record["label"] = val

                # Logic to calculate LLM 'prediction'
                if args.mode in ["llm", "both"]:
                    pred_col = f"{args.pred_prefix}{criteria}{args.pred_suffix}"
                    if pred_col in sub1 and pred_col in sub2:
                        s1 = sub1[pred_col]
                        s2 = sub2[pred_col]
                        
                        if s1 > s2: val = 1
                        elif s1 < s2: val = 0
                        else: val = 0.5
                        
                        record["prediction"] = val
                
                # Only append the record if it successfully generated at least one target field
                if "label" in record or "prediction" in record:
                    pairwise_records.append(record)

    save_jsonl(pairwise_records, args.output)
    print(f"Done! Saved {len(pairwise_records)} pairwise records to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pointwise scores to Pairwise comparisons")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the original pointwise file (JSONL)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save the output pairwise file (JSONL)")
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
