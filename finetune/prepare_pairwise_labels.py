import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from evaluation.utils import load_jsonl


CRITERIA = ("correctness", "efficiency", "readability")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_problem_id(row):
    if row.get("id"):
        return row["id"]
    sub_id = row.get("sub_id_1") or row.get("sub_id")
    if sub_id:
        return sub_id.rsplit("_", 1)[0]
    return None


def validate_labels(rows):
    required = {"id", "criteria", "sub_id_1", "sub_id_2", "label"}
    errors = []

    for idx, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            errors.append(f"line {idx}: missing fields {sorted(missing)}")
            continue

        if row["label"] not in {0, 0.5, 1}:
            errors.append(f"line {idx}: label must be 0, 0.5, or 1")

        if row["criteria"] not in CRITERIA:
            errors.append(f"line {idx}: criteria must be one of {CRITERIA}")

    if errors:
        preview = "\n".join(errors[:20])
        extra = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Invalid label file:\n{preview}{extra}")


def validate_join_sources(rows, submissions_path, metadata_path, allow_missing):
    submissions = load_jsonl(submissions_path)
    metadata = load_jsonl(metadata_path)

    submission_ids = {row["sub_id"] for row in submissions if "sub_id" in row}
    metadata_ids = {row["id"] for row in metadata if "id" in row}

    missing_submissions = set()
    missing_metadata = set()

    for row in rows:
        if row["sub_id_1"] not in submission_ids:
            missing_submissions.add(row["sub_id_1"])
        if row["sub_id_2"] not in submission_ids:
            missing_submissions.add(row["sub_id_2"])

        problem_id = get_problem_id(row)
        if problem_id and problem_id not in metadata_ids:
            missing_metadata.add(problem_id)

    if not missing_submissions and not missing_metadata:
        return

    messages = []
    if missing_submissions:
        missing_list = sorted(missing_submissions)
        messages.append(
            f"missing {len(missing_list)} sub_id values in {submissions_path}: "
            f"{missing_list[:10]}"
        )
    if missing_metadata:
        missing_list = sorted(missing_metadata)
        messages.append(
            f"missing {len(missing_list)} problem id values in {metadata_path}: "
            f"{missing_list[:10]}"
        )

    message = "\n".join(messages)
    if allow_missing:
        print(f"[WARN] {message}")
    else:
        raise ValueError(message)


def split_by_problem_id(rows, valid_ratio, seed):
    problem_ids = sorted({get_problem_id(row) for row in rows if get_problem_id(row)})
    if not problem_ids:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(problem_ids)

    if valid_ratio <= 0:
        valid_ids = set()
    elif len(problem_ids) == 1:
        valid_ids = set(problem_ids)
    else:
        valid_size = round(len(problem_ids) * valid_ratio)
        valid_size = max(1, min(valid_size, len(problem_ids) - 1))
        valid_ids = set(problem_ids[:valid_size])

    train_rows = []
    valid_rows = []
    for row in rows:
        if get_problem_id(row) in valid_ids:
            valid_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, valid_rows


def main():
    parser = argparse.ArgumentParser(
        description="Create pairwise train/valid label files from pairwise_train.jsonl."
    )
    parser.add_argument("--input", default="data/pairwise_train.jsonl")
    parser.add_argument("--output-dir", default="data/finetune")
    parser.add_argument("--train-name", default="pairwise_train_labels.jsonl")
    parser.add_argument("--valid-name", default="pairwise_valid_labels.jsonl")
    parser.add_argument("--submissions", default="data/submissions.jsonl")
    parser.add_argument("--metadata", default="data/metadata.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Warn instead of failing when labels cannot be joined with submissions/metadata.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_name
    valid_path = output_dir / args.valid_name

    rows = load_jsonl(input_path)
    validate_labels(rows)
    validate_join_sources(rows, args.submissions, args.metadata, args.allow_missing)

    train_rows, valid_rows = split_by_problem_id(rows, args.valid_ratio, args.seed)

    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    train_ids = {get_problem_id(row) for row in train_rows if get_problem_id(row)}
    valid_ids = {get_problem_id(row) for row in valid_rows if get_problem_id(row)}

    print(f"Loaded labels: {len(rows)} rows, {len(train_ids | valid_ids)} problem ids")
    print(f"Train labels: {len(train_rows)} rows, {len(train_ids)} problem ids -> {train_path}")
    print(f"Valid labels: {len(valid_rows)} rows, {len(valid_ids)} problem ids -> {valid_path}")


if __name__ == "__main__":
    main()
