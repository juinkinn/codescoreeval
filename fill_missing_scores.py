import os
import json
import random
import argparse
from pathlib import Path


SCORE_FIELDS = [
    "gt_correctness_score",
    "gt_efficiency_score",
    "gt_readability_score",
    "correctness_score",
    "efficiency_score",
    "readability_score"
]


def fill_missing(record):
    for k in SCORE_FIELDS:
        if k not in record or record[k] is None or record[k] == "null":
            record[k] = random.randint(1, 5)
    return record


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            obj = fill_missing(obj)

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".jsonl"):
            continue

        input_path = os.path.join(input_folder, file_name)

        new_name = file_name.replace("_raw", "")
        output_path = os.path.join(output_folder, new_name)

        process_file(input_path, output_path)

        print(f"Processed: {file_name} -> {new_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="./output/pointwise/processed")

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)