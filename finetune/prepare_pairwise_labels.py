import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
            f"missing {len(missing_list)} sub_id values "
            f"in {submissions_path}: {missing_list[:10]}"
        )

    if missing_metadata:
        missing_list = sorted(missing_metadata)
        messages.append(
            f"missing {len(missing_list)} problem id values "
            f"in {metadata_path}: {missing_list[:10]}"
        )

    message = "\n".join(messages)
    if allow_missing:
        print(f"[WARN] {message}")
    else:
        raise ValueError(message)


def analyze_pair_code_length(pairwise_rows, submissions_path):
    submissions = load_jsonl(submissions_path)
    code_map = {row["sub_id"]: row["code"] for row in submissions if "sub_id" in row and "code" in row}

    lengths = []
    missing = 0

    for row in pairwise_rows:
        sub1, sub2 = row["sub_id_1"], row["sub_id_2"]
        code1, code2 = code_map.get(sub1), code_map.get(sub2)

        if code1 is None or code2 is None:
            missing += 1
            continue

        lengths.append(len(code1) + len(code2))

    if not lengths:
        print("No valid pairs found.")
        return

    arr = np.array(lengths)

    print("\n" + "=" * 80)
    print("PAIR CODE LENGTH ANALYSIS")
    print("=" * 80)
    print(f"Pairs analyzed : {len(arr):,}")
    print(f"Missing lookup : {missing:,}")
    print("\nCharacter statistics")
    print("-" * 80)
    print(f"Mean   : {arr.mean():,.1f}")
    print(f"Median : {np.median(arr):,.1f}")
    print(f"P90    : {np.percentile(arr, 90):,.1f}")
    print(f"P95    : {np.percentile(arr, 95):,.1f}")
    print(f"P99    : {np.percentile(arr, 99):,.1f}")
    print(f"Max    : {arr.max():,}")

    approx_tokens = arr / 4.0
    print("\nApprox token statistics (chars / 4)")
    print("-" * 80)
    print(f"Mean   : {approx_tokens.mean():,.1f}")
    print(f"Median : {np.median(approx_tokens):,.1f}")
    print(f"P90    : {np.percentile(approx_tokens, 90):,.1f}")
    print(f"P95    : {np.percentile(approx_tokens, 95):,.1f}")
    print(f"P99    : {np.percentile(approx_tokens, 99):,.1f}")
    print(f"Max    : {approx_tokens.max():,.1f}")

    print("\nHistogram (characters)")
    print("-" * 80)
    bins = [0, 2000, 4000, 8000, 12000, 16000, 20000, float("inf")]
    for left, right in zip(bins[:-1], bins[1:]):
        count = np.sum((arr >= left) & (arr < right))
        pct = count / len(arr) * 100
        right_str = "inf" if right == float("inf") else str(int(right))
        print(f"[{int(left):>5}, {right_str:>5}) : {count:>8,} ({pct:6.2f}%)")


def analyze_score_gap_distribution(pairwise_rows, original_train_path):
    original_rows = load_jsonl(original_train_path)
    score_map = {
        row["sub_id"]: {
            "correctness": row["correctness_score"],
            "efficiency": row["efficiency_score"],
            "readability": row["readability_score"],
        }
        for row in original_rows
    }

    gap_counter = Counter()
    label_counter = Counter()
    criterion_gap_counter = defaultdict(Counter)
    
    gap_criterion_counter = defaultdict(Counter) 
    
    all_gaps = []
    missing = 0

    for row in pairwise_rows:
        sub1, sub2, criterion = row["sub_id_1"], row["sub_id_2"], row["criteria"]

        if sub1 not in score_map or sub2 not in score_map:
            missing += 1
            continue

        score1 = score_map[sub1][criterion]
        score2 = score_map[sub2][criterion]
        gap = abs(score1 - score2)

        gap_counter[gap] += 1
        criterion_gap_counter[criterion][gap] += 1
        
        gap_criterion_counter[gap][criterion] += 1 
        
        label_counter[row["label"]] += 1
        all_gaps.append(gap)

    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    total_labels = sum(label_counter.values())
    for label, count in sorted(label_counter.items()):
        pct = count / total_labels * 100
        print(f"label={label:<4} {count:>10,} ({pct:6.2f}%)")

    print("\n" + "=" * 80)
    print("GLOBAL SCORE GAP DISTRIBUTION")
    print("=" * 80)
    total_gaps = sum(gap_counter.values())
    for gap in sorted(gap_counter):
        count = gap_counter[gap]
        pct = count / total_gaps * 100
        print(f"gap={gap:<3} {count:>10,} ({pct:6.2f}%)")

    print("\n" + "=" * 80)
    print("QUANTILES")
    print("=" * 80)
    all_gaps_np = np.array(all_gaps)
    for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        value = np.quantile(all_gaps_np, q)
        print(f"Q{int(q*100):>3}: {value:.2f}")

    print("\n" + "=" * 80)
    print("PER CRITERION (Gap distribution within each criteria)")
    print("=" * 80)
    for criterion in CRITERIA:
        print(f"\n{criterion.upper()}")
        print("-" * 60)
        total = sum(criterion_gap_counter[criterion].values())
        if total > 0:
            for gap in sorted(criterion_gap_counter[criterion]):
                count = criterion_gap_counter[criterion][gap]
                pct = count / total * 100
                print(f"gap={gap:<3} {count:>10,} ({pct:6.2f}%)")

    print("\n" + "=" * 80)
    print("PER GAP (Criteria distribution within each gap)")
    print("=" * 80)
    for gap in sorted(gap_criterion_counter):
        print(f"\nGAP = {gap}")
        print("-" * 60)
        total_in_gap = sum(gap_criterion_counter[gap].values())
        for crit in CRITERIA:
            count = gap_criterion_counter[gap][crit]
            pct = count / total_in_gap * 100 if total_in_gap > 0 else 0
            print(f"{crit:<15}: {count:>8,} ({pct:6.2f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Pairs analyzed : {len(all_gaps):,}")
    print(f"Missing lookup : {missing:,}")
    print(f"Unique gaps    : {len(gap_counter)}")
    if all_gaps:
        print(f"Min gap        : {min(all_gaps)}")
        print(f"Max gap        : {max(all_gaps)}")
        print(f"Mean gap       : {np.mean(all_gaps):.3f}")
        print(f"Median gap     : {np.median(all_gaps):.3f}")


def filter_and_resample(rows, submissions_path, original_train_path, seed):
    rng = random.Random(seed)
    
    print("\n[1/2] Filtering Context Length Outliers...")
    submissions = load_jsonl(submissions_path)
    code_map = {r["sub_id"]: r.get("code", "") for r in submissions if "sub_id" in r}

    valid_length_rows = []
    lengths = []
    for row in rows:
        sub1 = row["sub_id_1"]
        sub2 = row["sub_id_2"]
        if sub1 in code_map and sub2 in code_map:
            l = len(code_map[sub1]) + len(code_map[sub2])
            lengths.append(l)
            valid_length_rows.append(row)

    if not lengths:
        return rows

    p99_length = np.percentile(lengths, 99)
    print(f"-> P99 Context Length limit: {p99_length:,.0f} chars")

    filtered_by_length = []
    for row, length in zip(valid_length_rows, lengths):
        if length <= p99_length:
            filtered_by_length.append(row)

    print(f"-> Kept {len(filtered_by_length):,} / {len(valid_length_rows):,} samples (<= P99)")

    print("\n[2/2] Resampling by Global Gap...")
    original_train = load_jsonl(original_train_path)
    score_map = {
        r["sub_id"]: {
            "correctness": r.get("correctness_score", 0),
            "efficiency": r.get("efficiency_score", 0),
            "readability": r.get("readability_score", 0),
        }
        for r in original_train if "sub_id" in r
    }

    gap_groups = defaultdict(list)
    for row in filtered_by_length:
        sub1, sub2, crit = row["sub_id_1"], row["sub_id_2"], row["criteria"]
        if sub1 in score_map and sub2 in score_map:
            gap = round(abs(score_map[sub1][crit] - score_map[sub2][crit]), 4)
            gap_groups[gap].append(row)

    min_gap_count = min(len(g_rows) for g_rows in gap_groups.values())
    print(f"-> Target sample size per gap: {min_gap_count:,}")

    resampled_rows = []

    for gap, g_rows in sorted(gap_groups.items()):
        total_in_gap = len(g_rows)
        
        crit_groups = defaultdict(list)
        for r in g_rows:
            crit_groups[r["criteria"]].append(r)

        sampled_for_this_gap = 0
        print(f"\nGap {gap} (Original: {total_in_gap:,}) -> Target: {min_gap_count:,}")
        
        for crit, c_rows in crit_groups.items():
            crit_prop = len(c_rows) / total_in_gap
            target_crit_count = int(min_gap_count * crit_prop)
            
            if target_crit_count > 0:
                sampled = rng.sample(c_rows, target_crit_count)
                resampled_rows.extend(sampled)
                sampled_for_this_gap += len(sampled)
            
            print(f"   - {crit.capitalize()}: {len(c_rows):,} ({crit_prop:.1%}) -> Sampled: {target_crit_count:,}")

    print(f"\nFinal dataset size after resampling: {len(resampled_rows):,}")
    rng.shuffle(resampled_rows)
    return resampled_rows


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pairwise_train.jsonl")
    parser.add_argument("--original-train", default="data/original_train.jsonl")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--analyze-length", action="store_true", help="Analyze pair code length distribution.")
    parser.add_argument("--resample", action="store_true", help="Filter P99 context length and downsample by gap.")
    parser.add_argument("--output-dir", default="data/finetune")
    parser.add_argument("--train-name", default="pairwise_train_labels.jsonl")
    parser.add_argument("--valid-name", default="pairwise_valid_labels.jsonl")
    parser.add_argument("--submissions", default="data/submissions.jsonl")
    parser.add_argument("--metadata", default="data/metadata.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    validate_labels(rows)

    if args.analyze_only:
        analyze_score_gap_distribution(rows, args.original_train)
        return

    if args.analyze_length:
        analyze_pair_code_length(rows, args.submissions)
        return

    validate_join_sources(rows, args.submissions, args.metadata, args.allow_missing)

    if args.resample:
        rows = filter_and_resample(
            rows=rows,
            submissions_path=args.submissions,
            original_train_path=args.original_train,
            seed=args.seed
        )

    output_dir = Path(args.output_dir)
    train_path = output_dir / args.train_name
    valid_path = output_dir / args.valid_name

    train_rows, valid_rows = split_by_problem_id(rows, args.valid_ratio, args.seed)

    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    print(f"\nTrain labels: {len(train_rows):,} -> {train_path}")
    print(f"Valid labels: {len(valid_rows):,} -> {valid_path}")


if __name__ == "__main__":
    main()