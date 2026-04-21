import torch
import json
import os
from tqdm import tqdm
import argparse
import re

from loader import load_tokenizer, load_model


PAIRWISE_PROMPT = """\
You are a Senior Code Reviewer.

Compare TWO code submissions based ONLY on the given criterion.

Criterion: {criterion}

DO NOT consider any other aspects.

Problem Description:
{description}

Code A ({lang1}):
{code1}

Code B ({lang2}):
{code2}

Which code is better?

Final answer (A, B, or both only):
"""


# ===== truncate =====
def smart_truncate(code, max_chars=10000):
    if not code:
        return code
        
    if len(code) <= max_chars:
        return code

    half = max_chars // 2
    return code[:half] + "\n...\n" + code[-half:]


# ===== load submissions map =====
def load_submissions(path):
    sub_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            sub_map[j["sub_id"]] = j
    return sub_map


# ===== load metadata =====
def load_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            meta[j["id"]] = j.get("description", "")
    return meta


# ===== build prompt =====
def build_prompt(sample, sub_map, metadata_map):
    sub1 = sub_map[sample["sub_id_1"]]
    sub2 = sub_map[sample["sub_id_2"]]

    base_id = sample["sub_id_1"].rsplit("_", 1)[0]
    description = metadata_map.get(base_id, "")

    code1 = smart_truncate(sub1.get("code", ""))
    code2 = smart_truncate(sub2.get("code", ""))

    return PAIRWISE_PROMPT.format(
        criterion=sample["criteria"],
        description=description,
        code1=code1,
        code2=code2,
        lang1=sub1.get("lang", ""),
        lang2=sub2.get("lang", "")
    )


# ===== extract prediction =====
def extract_choice(text: str):
    if not text:
        return None

    text = text.strip().upper()

    # ưu tiên đầu
    if text.startswith("A"):
        return 1
    if text.startswith("B"):
        return 2
    if text.startswith("both"):
        return 0

    # regex fallback
    if re.search(r"\bboth\b", text):
        return 0
    if re.search(r"\bA\b", text):
        return 1
    if re.search(r"\bB\b", text):
        return 2

    return None


# ===== infer =====
def infer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=5,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    print(output)
    pred = extract_choice(output)
    print(pred)
    del inputs, output_ids
    torch.cuda.empty_cache()

    return pred


# ===== main =====
def main(model_name,
         pairwise_path="./data/pairwise_test.jsonl",
         submissions_path="./data/submissions.jsonl",
         metadata_path="./data/metadata.jsonl",
         use_bnb=False,
         limit=None):

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, use_bnb=use_bnb, device_map="cuda")

    sub_map = load_submissions(submissions_path)
    metadata_map = load_metadata(metadata_path)

    os.makedirs("output", exist_ok=True)
    output_file = os.path.join(
        "output", f"{model_name.replace('/', '_')}_pairwise.jsonl"
    )

    total = sum(1 for _ in open(pairwise_path, "r", encoding="utf-8"))

    with open(pairwise_path, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(tqdm(f_in, desc="Pairwise infer", total=total)):
            if limit and i >= limit:
                break

            sample = json.loads(line)

            prompt = build_prompt(sample, sub_map, metadata_map)
            pred = infer(model, tokenizer, prompt)

            out = {
                "id": sample["id"],
                "criteria": sample["criteria"],
                "sub_id_1": sample["sub_id_1"],
                "sub_id_2": sample["sub_id_2"],
                "label": sample["label"],   # 1 / 2 / 0
                "prediction": pred
            }

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"Inference done! Saved to {output_file}")


# ===== entry =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pairwise_path", type=str, default="./data/pairwise_test.jsonl")
    parser.add_argument("--submissions_path", type=str, default="./data/submissions.jsonl")
    parser.add_argument("--metadata_path", type=str, default="./data/metadata.jsonl")
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        pairwise_path=args.pairwise_path,
        submissions_path=args.submissions_path,
        metadata_path=args.metadata_path,
        use_bnb=args.use_bnb,
        limit=args.limit
    )