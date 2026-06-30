import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CRITERIA = ("correctness", "efficiency", "readability")


def load_rows(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_problem_id(row):
    return row.get("id") or row["sub_id"].rsplit("_", 1)[0]


def expand_rows(rows):
    examples = []
    for row in rows:
        for criterion in CRITERIA:
            examples.append(
                {
                    "sub_id": row["sub_id"],
                    "id": get_problem_id(row),
                    "criterion": criterion,
                    "score": row[f"{criterion}_score"],
                }
            )
    return examples


def parse_score_caps(value):
    caps = {}
    if not value.strip():
        return caps

    for item in value.split(","):
        score, cap = item.split(":", 1)
        caps[int(score.strip())] = int(cap.strip())
    return caps


def capped_undersample(examples, default_cap, score_caps, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)

    for example in examples:
        groups[(example["criterion"], example["score"])].append(example)

    kept = []
    removed_counts = Counter()
    for key in sorted(groups):
        group = groups[key]
        _, score = key
        cap = score_caps.get(score, default_cap)
        rng.shuffle(group)
        kept.extend(group[:cap])
        removed_counts[key] = max(0, len(group) - cap)

    rng.shuffle(kept)
    return kept, removed_counts


def score_stats(examples):
    stats = {}
    for criterion in CRITERIA:
        counts = Counter(
            example["score"] for example in examples if example["criterion"] == criterion
        )
        total = sum(counts.values())
        mean = sum(score * count for score, count in counts.items()) / total if total else 0.0
        stats[criterion] = (counts, total, mean)

    all_counts = Counter(example["score"] for example in examples)
    all_total = sum(all_counts.values())
    all_mean = sum(score * count for score, count in all_counts.items()) / all_total if all_total else 0.0
    stats["all"] = (all_counts, all_total, all_mean)
    return stats


def print_distribution(title, examples):
    stats = score_stats(examples)
    print(title)
    for criterion in (*CRITERIA, "all"):
        counts, total, mean = stats[criterion]
        print(f"{criterion}\tn={total}\tmean={mean:.4f}")
        for score in range(1, 6):
            count = counts[score]
            pct = count / total * 100 if total else 0.0
            print(f"{score}\t{count}\t{pct:.2f}%")
        print()


def write_jsonl(path, examples):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Expand pointwise labels and cap each criterion-score group."
    )
    parser.add_argument("--input", default="data/finetune/pointwise_train_labels.jsonl")
    parser.add_argument(
        "--output",
        default="data/finetune/pointwise_train_labels_expanded_balanced.jsonl",
    )
    parser.add_argument("--cap-per-criterion-score", type=int, default=1500)
    parser.add_argument(
        "--score-caps",
        default="2:1000",
        help="Optional score-specific caps, e.g. '2:1000,5:1200'. Overrides --cap-per-criterion-score for those scores.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.input)
    examples = expand_rows(rows)
    score_caps = parse_score_caps(args.score_caps)
    kept, removed_counts = capped_undersample(
        examples,
        default_cap=args.cap_per_criterion_score,
        score_caps=score_caps,
        seed=args.seed,
    )

    original_sub_ids = {row["sub_id"] for row in rows}
    kept_sub_ids = {example["sub_id"] for example in kept}
    original_problem_ids = {get_problem_id(row) for row in rows}
    kept_problem_ids = {example["id"] for example in kept}

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Cap per criterion-score: {args.cap_per_criterion_score}")
    print(f"Score-specific caps: {score_caps or 'none'}")
    print(f"Original rows: {len(rows)}")
    print(f"Original expanded examples: {len(examples)}")
    print(f"Kept expanded examples: {len(kept)}")
    print(f"Removed expanded examples: {len(examples) - len(kept)}")
    print(f"Original sub_ids: {len(original_sub_ids)}")
    print(f"Kept sub_ids: {len(kept_sub_ids)}")
    print(f"Sub_ids with no kept criterion: {len(original_sub_ids - kept_sub_ids)}")
    print(f"Original problems: {len(original_problem_ids)}")
    print(f"Kept problems: {len(kept_problem_ids)}")
    print(f"Problems with no kept example: {len(original_problem_ids - kept_problem_ids)}")
    print("Removed by criterion-score:")
    for criterion in CRITERIA:
        for score in range(1, 6):
            print(f"{criterion}\t{score}\t{removed_counts[(criterion, score)]}")
    print()

    print_distribution("Original expanded distribution", examples)
    print_distribution("Balanced expanded distribution", kept)

    if args.dry_run:
        print("Dry run: did not write output file")
        return

    write_jsonl(args.output, kept)
    print(f"Wrote {len(kept)} examples to {args.output}")


if __name__ == "__main__":
    main()
