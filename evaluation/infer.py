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

Final answer (A or B only):
"""

def smart_truncate(code, max_chars=10000):
    if not code:
        return code
        
    if len(code) <= max_chars:
        return code

    half = max_chars // 2
    head = code[:half]
    tail = code[-half:]

    return head + "\n...\n" + tail


def build_prompt(sample, metadata_map):
    description = metadata_map.get(sample["id"], "")

    code1 = smart_truncate(sample["code1"])
    code2 = smart_truncate(sample["code2"])
    return PAIRWISE_PROMPT.format(
        criterion=sample["criteria"],
        description=description,
        code1=code1,
        code2=code2,
        lang1=sample.get("lang1", ""),
        lang2=sample.get("lang2", "")
    )


def load_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            meta[j["id"]] = j.get("description", "")
    return meta

def extract_choice(text: str):
    if not text:
        return None

    text = text.strip().upper()

    # Ưu tiên match đầu
    if text.startswith("A"):
        return 1
    if text.startswith("B"):
        return 2

    # Regex fallback
    match = re.search(r"\b(A|B)\b", text)
    if match:
        return 1 if match.group(1) == "A" else 2

    return None


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

    pred = extract_choice(output)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return pred


def main(model_name,
         pairwise_path="./data/pairwise_test.jsonl",
         metadata_path="./data/metadata.jsonl",
         use_bnb=False,
         limit=None):

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, use_bnb=use_bnb, device_map="cuda")

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

            prompt = build_prompt(sample, metadata_map)
            pred = infer(model, tokenizer, prompt)

            out = {
                "id": sample["id"],
                "criteria": sample["criteria"],
                "label": sample["label"],  
                "prediction": pred          
            }

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Inference done! Saved to {output_file}")


# ================= ENTRY =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pairwise_path", type=str, default="./data/pairwise_test.jsonl")
    parser.add_argument("--metadata_path", type=str, default="./data/metadata.jsonl")
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        pairwise_path=args.pairwise_path,
        metadata_path=args.metadata_path,
        use_bnb=args.use_bnb,
        limit=args.limit
    )