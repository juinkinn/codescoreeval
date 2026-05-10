import json
import random
from collections import defaultdict
import os

ORIGINAL_TEST_PATH = "../data/original_test.jsonl"
PAIRWISE_TEST_PATH = "../data/pairwise_test.jsonl"

OUTPUT_PATH = "../data/pairwise_test_with_anchors.jsonl"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

random.seed(42)


anchors_pool = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)

with open(ORIGINAL_TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        problem_id = row["id"]
        sub_id = row["sub_id"]

        criteria_scores = {
            "correctness": row["correctness_score"],
            "efficiency": row["efficiency_score"],
            "readability": row["readability_score"],
        }

        for criterion, score in criteria_scores.items():
            anchors_pool[problem_id][criterion][score].append(sub_id)

selected_anchors = defaultdict(lambda: defaultdict(dict))

for problem_id in anchors_pool:
    for criterion in anchors_pool[problem_id]:
        for score in anchors_pool[problem_id][criterion]:

            candidates = anchors_pool[problem_id][criterion][score]

            anchor_sub_id = random.choice(candidates)

            selected_anchors[problem_id][criterion][score] = anchor_sub_id


with open(PAIRWISE_TEST_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in fin:
        row = json.loads(line)

        problem_id = row["id"]
        criterion = row["criteria"]
        anchor_dict = selected_anchors[problem_id][criterion]

        row["anchors"] = anchor_dict

        fout.write(json.dumps(row) + "\n")

print(f"Saved to: {OUTPUT_PATH}")