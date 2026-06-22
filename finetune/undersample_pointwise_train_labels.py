import argparse
import json
import random
from collections import Counter
from pathlib import Path


CRITERIA = ("correctness", "efficiency", "readability")


def parse_scores(value):
    if not value.strip():
        return set()
    return {int(score.strip()) for score in value.split(",") if score.strip()}


def load_rows(path):
    rows = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                rows.append((line_number, json.loads(line)))
    return rows


def get_problem_id(row):
    return row.get("id") or row["sub_id"].rsplit("_", 1)[0]


def count_remaining(rows, removed, score_name, score):
    return sum(1 for idx, row in rows if idx not in removed and row[score_name] == score)


def remove_from_candidates(rows, removed, rng, predicate, limit, reason, removed_reasons):
    if limit <= 0:
        return 0

    candidates = [idx for idx, row in rows if idx not in removed and predicate(row)]
    rng.shuffle(candidates)
    chosen = candidates[:limit]

    for idx in chosen:
        removed.add(idx)
        removed_reasons[reason] += 1

    return len(chosen)


def undersample_rows(rows, args):
    rng = random.Random(args.seed)
    removed = set()
    removed_reasons = Counter()
    preferred_readability = parse_scores(args.preferred_readability_scores)
    protected_readability = parse_scores(args.protected_readability_scores)

    target_c5 = args.target_correctness_5
    if target_c5 is not None:
        need_c5_drop = max(0, count_remaining(rows, removed, "correctness_score", 5) - target_c5)
        correctness_priorities = [
            (
                "c5_e5_preferred_readability",
                lambda row: row["correctness_score"] == 5
                and row["efficiency_score"] == 5
                and row["readability_score"] in preferred_readability,
            ),
            (
                "c5_not_e5_preferred_readability",
                lambda row: row["correctness_score"] == 5
                and row["efficiency_score"] != 5
                and row["readability_score"] in preferred_readability,
            ),
            (
                "c5_non_protected_readability",
                lambda row: row["correctness_score"] == 5
                and row["readability_score"] not in protected_readability,
            ),
            (
                "c5_any",
                lambda row: row["correctness_score"] == 5,
            ),
        ]

        for reason, predicate in correctness_priorities:
            if need_c5_drop <= 0:
                break
            dropped = remove_from_candidates(
                rows,
                removed,
                rng,
                predicate,
                need_c5_drop,
                reason,
                removed_reasons,
            )
            need_c5_drop -= dropped

    target_e5 = args.target_efficiency_5
    if target_e5 is not None:
        need_e5_drop = max(0, count_remaining(rows, removed, "efficiency_score", 5) - target_e5)
        efficiency_priorities = [
            (
                "e5_not_c5_preferred_readability",
                lambda row: row["efficiency_score"] == 5
                and row["correctness_score"] != 5
                and row["readability_score"] in preferred_readability,
            ),
            (
                "e5_preferred_readability",
                lambda row: row["efficiency_score"] == 5
                and row["readability_score"] in preferred_readability,
            ),
            (
                "e5_non_protected_readability",
                lambda row: row["efficiency_score"] == 5
                and row["readability_score"] not in protected_readability,
            ),
            (
                "e5_any",
                lambda row: row["efficiency_score"] == 5,
            ),
        ]

        for reason, predicate in efficiency_priorities:
            if need_e5_drop <= 0:
                break
            dropped = remove_from_candidates(
                rows,
                removed,
                rng,
                predicate,
                need_e5_drop,
                reason,
                removed_reasons,
            )
            need_e5_drop -= dropped

    kept_rows = [(idx, row) for idx, row in rows if idx not in removed]
    removed_rows = [(idx, row) for idx, row in rows if idx in removed]
    return kept_rows, removed_rows, removed_reasons


def score_stats(rows):
    stats = {}
    for criterion in CRITERIA:
        counts = Counter(row[f"{criterion}_score"] for _, row in rows)
        total = sum(counts.values())
        mean = sum(score * count for score, count in counts.items()) / total if total else 0.0
        stats[criterion] = (counts, total, mean)

    all_counts = Counter()
    for _, row in rows:
        for criterion in CRITERIA:
            all_counts[row[f"{criterion}_score"]] += 1
    all_total = sum(all_counts.values())
    all_mean = sum(score * count for score, count in all_counts.items()) / all_total if all_total else 0.0
    stats["all"] = (all_counts, all_total, all_mean)
    return stats


def print_distribution(title, rows):
    stats = score_stats(rows)
    print(title)
    for criterion in (*CRITERIA, "all"):
        counts, total, mean = stats[criterion]
        print(f"{criterion}\tn={total}\tmean={mean:.4f}")
        for score in range(1, 6):
            count = counts[score]
            pct = count / total * 100 if total else 0.0
            print(f"{score}\t{count}\t{pct:.2f}%")
        print()


def write_rows(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for _, row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Row-level targeted undersampling for pointwise finetune labels."
    )
    parser.add_argument("--input", default="data/finetune/pointwise_train_labels.jsonl")
    parser.add_argument(
        "--output",
        default="data/finetune/pointwise_train_labels_row_balanced_c5_4000_e5_6000.jsonl",
    )
    parser.add_argument("--target-correctness-5", type=int, default=4000)
    parser.add_argument("--target-efficiency-5", type=int, default=6000)
    parser.add_argument("--preferred-readability-scores", default="2,3")
    parser.add_argument("--protected-readability-scores", default="1,5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input)
    kept_rows, removed_rows, removed_reasons = undersample_rows(rows, args)

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Original rows: {len(rows)}")
    print(f"Kept rows: {len(kept_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print(f"Original pointwise examples: {len(rows) * len(CRITERIA)}")
    print(f"Kept pointwise examples: {len(kept_rows) * len(CRITERIA)}")
    print(f"Removed pointwise examples: {len(removed_rows) * len(CRITERIA)}")

    original_problems = {get_problem_id(row) for _, row in rows}
    kept_problems = {get_problem_id(row) for _, row in kept_rows}
    print(f"Original problems: {len(original_problems)}")
    print(f"Kept problems: {len(kept_problems)}")
    print(f"Problems fully removed: {len(original_problems - kept_problems)}")

    print("Removed reasons:")
    for reason, count in removed_reasons.items():
        print(f"{reason}\t{count}")
    print()

    print_distribution("Original distribution", rows)
    print_distribution("Balanced distribution", kept_rows)
    print_distribution("Removed distribution", removed_rows)

    if args.dry_run:
        print("Dry run: did not write output file")
        return

    write_rows(args.output, kept_rows)
    print(f"Wrote {len(kept_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
