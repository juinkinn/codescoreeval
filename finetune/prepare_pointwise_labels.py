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
    return row["sub_id"].rsplit("_", 1)[0]


def validate_labels(rows):
    required = {"sub_id", "correctness_score", "efficiency_score", "readability_score"}
    errors = []

    for idx, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            errors.append(f"line {idx}: missing fields {sorted(missing)}")
            continue

        for criterion in CRITERIA:
            field = f"{criterion}_score"
            if row[field] not in {1, 2, 3, 4, 5}:
                errors.append(f"line {idx}: {field} must be an integer from 1 to 5")

    if errors:
        preview = "\n".join(errors[:20])
        extra = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Invalid label file:\n{preview}{extra}")


def validate_join_sources(rows, submissions_path, metadata_path, allow_missing):
    submissions = load_jsonl(submissions_path)
    metadata = load_jsonl(metadata_path)

    submission_ids = {row["sub_id"] for row in submissions if "sub_id" in row}
    metadata_ids = {row["id"] for row in metadata if "id" in row}

    missing_submissions = sorted(
        {row["sub_id"] for row in rows if row["sub_id"] not in submission_ids}
    )
    missing_metadata = sorted(
        {get_problem_id(row) for row in rows if get_problem_id(row) not in metadata_ids}
    )

    if not missing_submissions and not missing_metadata:
        return

    messages = []
    if missing_submissions:
        messages.append(
            f"missing {len(missing_submissions)} sub_id values in {submissions_path}: "
            f"{missing_submissions[:10]}"
        )
    if missing_metadata:
        messages.append(
            f"missing {len(missing_metadata)} problem id values in {metadata_path}: "
            f"{missing_metadata[:10]}"
        )

    message = "\n".join(messages)
    if allow_missing:
        print(f"[WARN] {message}")
    else:
        raise ValueError(message)


def split_by_problem_id(rows, valid_ratio, seed):
    problem_ids = sorted({get_problem_id(row) for row in rows})
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
        description="Create pointwise train/valid label files from original_train.jsonl."
    )
    parser.add_argument("--input", default="data/original_train.jsonl")
    parser.add_argument("--output-dir", default="data/finetune")
    parser.add_argument("--train-name", default="pointwise_train_labels.jsonl")
    parser.add_argument("--valid-name", default="pointwise_valid_labels.jsonl")
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

    train_ids = {get_problem_id(row) for row in train_rows}
    valid_ids = {get_problem_id(row) for row in valid_rows}

    print(f"Loaded labels: {len(rows)} rows, {len(train_ids | valid_ids)} problem ids")
    print(f"Train labels: {len(train_rows)} rows, {len(train_ids)} problem ids -> {train_path}")
    print(f"Valid labels: {len(valid_rows)} rows, {len(valid_ids)} problem ids -> {valid_path}")


if __name__ == "__main__":
    main()
