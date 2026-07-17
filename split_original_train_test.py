import argparse
import json
import random
from pathlib import Path

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_id(row):
    return row["sub_id"].rsplit("_", 1)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final_dataset_gt.jsonl")
    parser.add_argument("--annotated", nargs="+", default=[
        "data/calibration_set/annotated_subset_1.jsonl",
        "data/calibration_set/annotated_subset_2.jsonl",
    ])
    parser.add_argument("--train-out", default="data/original_train.jsonl")
    parser.add_argument("--test-out", default="data/original_test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load final dataset
    final_rows = load_jsonl(args.input)

    # Derive id and store
    for row in final_rows:
        row["id"] = get_id(row)

    # Load annotated subsets
    ann_rows = []
    for path in args.annotated:
        ann_rows.extend(load_jsonl(path))

    ann_ids = set()
    ann_sub_ids = set()
    for row in ann_rows:
        pid = get_id(row)
        ann_ids.add(pid)
        ann_sub_ids.add(row["sub_id"])

    # Split by problem id
    all_ids = sorted({row["id"] for row in final_rows})
    rng.shuffle(all_ids)

    split_idx = round(len(all_ids) * args.test_ratio)
    split_idx = max(1, min(split_idx, len(all_ids) - 1))
    test_ids = set(all_ids[:split_idx])
    train_ids = set(all_ids[split_idx:])

    train_rows = []
    test_rows = []
    for row in final_rows:
        if row["id"] in test_ids:
            test_rows.append(row)
        else:
            train_rows.append(row)

    # Remove annotated samples
    # Train: remove by problem id
    train_filtered = [r for r in train_rows if r["id"] not in ann_ids]
    # Test: remove by sub_id
    test_filtered = [r for r in test_rows if r["sub_id"] not in ann_sub_ids]

    with open(args.train_out, "w") as f:
        for row in train_filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.test_out, "w") as f:
        for row in test_filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Final dataset: {len(final_rows)} rows, {len(all_ids)} unique ids")
    print(f"Annotated ids (train removal): {len(ann_ids)}")
    print(f"Annotated sub_ids (test removal): {len(ann_sub_ids)}")
    print(f"Train before removal: {len(train_rows)} rows, {len(train_ids)} ids")
    print(f"Test before removal: {len(test_rows)} rows, {len(test_ids)} ids")
    print(f"Train after removal: {len(train_filtered)} rows")
    print(f"Test after removal: {len(test_filtered)} rows")
    print(f"Removed from train: {len(train_rows) - len(train_filtered)} rows")
    print(f"Removed from test: {len(test_rows) - len(test_filtered)} rows")

if __name__ == "__main__":
    main()
