import torch
import json
import os
import re
from dataset import SubmissionDataset
from loader import load_tokenizer, load_model
from prompts import CORRECTNESS_PROMPT, EFFICIENCY_PROMPT, SYNTAX_PROMPT
from tqdm import tqdm
from collections import Counter

def build_prompts(submission):
    code = submission["code"]
    lang = submission.get("lang", "")
    desc = submission.get("description", "")

    return {
        "correctness": CORRECTNESS_PROMPT.format(code=code, lang=lang, description=desc),
        "efficiency": EFFICIENCY_PROMPT.format(code=code, lang=lang, description=desc),
        "readability": SYNTAX_PROMPT.format(code=code, lang=lang, description=desc),
    }


def extract_score(text: str):
    if not text or not isinstance(text, str):
        return None

    text = text.lower().strip()

    # ===== 1. direct number =====
    try:
        val = float(text)
        if 1 <= val <= 5:
            return int(round(val))
    except:
        pass

    # ===== 2. strong structured patterns =====
    patterns = [
        r"score\s*[:=]?\s*([1-5])",
        r"rating\s*[:=]?\s*([1-5])",
        r"label\s*[:=]?\s*([1-5])",
        r"answer\s*[:=]?\s*([1-5])",
        r"final\s*[:=]?\s*([1-5])",

        r"([1-5])\s*/\s*5",
        r"([1-5])\s*out of\s*5",
        r"([1-5])\s*stars?",

        r"i (?:would )?give (?:it )?([1-5])",
        r"i rate (?:it )?([1-5])",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))

    # ===== 3. sentence-style patterns =====
    sentence_patterns = [
        r"score is\s*([1-5])",
        r"should be\s*([1-5])",
        r"would be\s*([1-5])",
        r"rated\s*([1-5])",
        r"get[s]?\s*([1-5])",
    ]

    for p in sentence_patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))

    # ===== 4. fallback: most frequent digit =====
    numbers = re.findall(r"\b[1-5]\b", text)

    if not numbers:
        return None

    numbers = list(map(int, numbers))

    # majority vote if noisy
    return Counter(numbers).most_common(1)[0][0]


def infer_criteria(model, tokenizer, prompts):
    results = {}

    for criterion, prompt_text in prompts.items():
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
            )

        output = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        print(output)
        score = extract_score(output)

        results[f"{criterion}_score"] = score

        del output_ids
        torch.cuda.empty_cache()

    return results

def main(
    model_name,
    use_bnb=False,
    submissions_path="../../data/original_test.jsonl",
    metadata_path="../../data/metadata.jsonl",
    limit=None,
):
    dataset = SubmissionDataset(submissions_path, metadata_path, limit=limit)
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, use_bnb=use_bnb, device_map="cuda")

    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", f"{model_name.replace('/', '_')}_pointwise.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for sub in tqdm(dataset, desc="Infer submissions", unit="sub"):
            prompts = build_prompts(sub)
            scores = infer_criteria(model, tokenizer, prompts)
            out = {
                "sub_id": sub["sub_id"],
                "gt_correctness_score": sub["gt_correctness_score"],
                "gt_efficiency_score": sub["gt_efficiency_score"],
                "gt_readability_score": sub["gt_readability_score"],
                **scores,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Inference done! Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.model_name, args.use_bnb, limit=args.limit)