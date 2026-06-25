import torch
import json
import os
from tqdm import tqdm
import argparse
import re

from loader import load_tokenizer, load_model
from prompts import CORRECTNESS_PAIRWISE, EFFICIENCY_PAIRWISE, READABILITY_PAIRWISE


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
def build_prompts(sample, sub_map, metadata_map, swapped=False, chat_format=False):
    sub1 = sub_map[sample["sub_id_1"]]
    sub2 = sub_map[sample["sub_id_2"]]

    base_id = sample["sub_id_1"].rsplit("_", 1)[0]
    description = metadata_map.get(base_id, "")

    code1 = sub1.get("code", "")
    code2 = sub2.get("code", "")

    lang1 = sub1.get("lang", "")
    lang2 = sub2.get("lang", "")

    if swapped:
        code1, code2 = code2, code1
        lang1, lang2 = lang2, lang1

    criterion = sample["criteria"]

    if criterion == "correctness":
        prompt_template = CORRECTNESS_PAIRWISE
    elif criterion == "efficiency":
        prompt_template = EFFICIENCY_PAIRWISE
    elif criterion == "readability":
        prompt_template = READABILITY_PAIRWISE
    else:
        prompt_template = READABILITY_PAIRWISE

    prompt = prompt_template.format(
        description=description,
        code1=code1,
        code2=code2,
        lang1=lang1,
        lang2=lang2
    )

    if not chat_format:
        return prompt

    messages = []

    messages.append({
        "role": "user",
        "content": prompt
    })

    return messages

# ===== extract prediction =====
def extract_choice(text: str):
    if not text:
        return None

    text = text.strip().upper()

    if text.startswith("BOTH"):
        return 0.5
    if text.startswith("A"):
        return 1
    if text.startswith("B"):
        return 0
    if re.search(r"\bBOTH\b", text):
        return 0.5
    if re.search(r"\bA\b", text):
        return 1
    if re.search(r"\bB\b", text):
        return 0

    return None

def make_key(sample):
    return (
        sample["id"],
        sample["criteria"],
        sample["sub_id_1"],
        sample["sub_id_2"],
    )

# ===== load completed ids =====
def load_completed_ids(output_file):
    completed = set()

    if not os.path.exists(output_file):
        return completed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)

                key = (
                    j["id"],
                    j["criteria"],
                    j["sub_id_1"],
                    j["sub_id_2"],
                )

                completed.add(key)

            except:
                pass

    return completed

# ===== infer =====
def infer(model, tokenizer, prompt, max_retry=2):
    for _ in range(max_retry):
        if isinstance(prompt, list):
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=32,
                eos_token_id=tokenizer.eos_token_id
            )

        output = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        pred = extract_choice(output)

        if pred is not None:
            del inputs, output_ids
            torch.cuda.empty_cache()
            return pred

    del inputs, output_ids
    torch.cuda.empty_cache()
    return None


# ===== base dir of current file =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== HF -> local path mapping =====
MODEL_PATHS = {
    "google/gemma-3-4b-it":
        os.path.abspath(
            os.path.join(BASE_DIR,
            "../../../models/gemma-3-4b-it")
        ),

    "Qwen/Qwen2.5-3B-Instruct":
        os.path.abspath(
            os.path.join(BASE_DIR,
            "../../../models/qwen-3b")
        ),

    "Qwen/Qwen2.5-Coder-3B-Instruct":
        os.path.abspath(
            os.path.join(BASE_DIR,
            "../../../models/qwen-coder-3b")
        ),

    "Qwen/Qwen2.5-Coder-7B-Instruct":
        os.path.abspath(
            os.path.join(BASE_DIR,
            "../../../models/qwen-coder-7b")
        ),

    "stabilityai/stable-code-instruct-3b":
        os.path.abspath(
            os.path.join(BASE_DIR,
            "../../../models/stable-code-3b")
        ),
}

def resolve_model_path(model_name, MODEL_PATHS):
    local_path = MODEL_PATHS.get(model_name)

    if local_path is not None and os.path.isdir(local_path):
        print(f"[MODEL] Using LOCAL: {local_path}")
        return local_path

    print(f"[MODEL] Using HF: {model_name}")
    return model_name

# ===== main =====
def main(model_name,
         pairwise_path="data/pairwise_test.jsonl",
         submissions_path="data/submissions.jsonl",
         metadata_path="data/metadata.jsonl",
         use_bnb=False,
         swapped=False,
         limit=None,
         adapter_path=None):

    # convert HF name -> local path if exists
    resolved_model_path = resolve_model_path(model_name, MODEL_PATHS)

    print(f"Loading model from: {resolved_model_path}")

    tokenizer = load_tokenizer(resolved_model_path)

    model = load_model(
        resolved_model_path,
        use_bnb=use_bnb,
        device_map="cuda",
        adapter_path=adapter_path
    )

    sub_map = load_submissions(submissions_path)
    metadata_map = load_metadata(metadata_path)
    use_tuning = adapter_path is not None

    os.makedirs("output", exist_ok=True)
    suffix = "tuning" if adapter_path else "raw"
    suffix += "_swapped" if swapped else ""
    output_file = os.path.join(
        "output",
        f"{model_name.replace('/', '_')}_pairwise{suffix}.jsonl"
    )

    completed_ids = load_completed_ids(output_file)

    print(f"Found {len(completed_ids)} completed samples")

    total = sum(
        1 for _ in open(pairwise_path, "r", encoding="utf-8")
    )

    with open(pairwise_path, "r", encoding="utf-8") as f_in, \
         open(output_file, "a", encoding="utf-8") as f_out:

        for i, line in enumerate(
            tqdm(f_in, desc="Pairwise infer", total=total)
        ):

            if limit and i >= limit:
                break

            sample = json.loads(line)

            key = (
                sample["id"],
                sample["criteria"],
                sample["sub_id_1"],
                sample["sub_id_2"],
            )

            if key in completed_ids:
                continue

            prompt = build_prompts(sample, sub_map, metadata_map, swapped=swapped, chat_format=use_tuning) 
            pred = infer(model, tokenizer, prompt)

            out = {
                "id": sample["id"],
                "criteria": sample["criteria"],
                "sub_id_1": sample["sub_id_1"],
                "sub_id_2": sample["sub_id_2"],
                "label": sample["label"],
                "prediction": pred
            }

            f_out.write(
                json.dumps(out, ensure_ascii=False) + "\n"
            )

            f_out.flush()

    print(f"Inference done! Saved to {output_file}")


# ===== entry =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pairwise_path", type=str,
                        default="data/pairwise_test.jsonl")
    parser.add_argument("--submissions_path", type=str,
                        default="data/submissions.jsonl")
    parser.add_argument("--metadata_path", type=str,
                        default="data/metadata.jsonl")
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--swapped", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter_path", type=str, default=None)

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        pairwise_path=args.pairwise_path,
        submissions_path=args.submissions_path,
        metadata_path=args.metadata_path,
        use_bnb=args.use_bnb,
        swapped=args.swapped,
        limit=args.limit,
        adapter_path=args.adapter_path
    )