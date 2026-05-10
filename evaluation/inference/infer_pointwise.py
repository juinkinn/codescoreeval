import torch
import json
import os
from evaluation.inference.dataset import SubmissionDataset
from evaluation.inference.loader import load_tokenizer, load_model
from prompts import CORRECTNESS_PROMPT, EFFICIENCY_PROMPT, SYNTAX_PROMPT
from tqdm import tqdm
import re
from collections import Counter

def build_prompts(submission):
    code = submission['code']
    lang = submission.get('lang', '')
    desc = submission.get('description', '')

    return {
        "correctness": CORRECTNESS_PROMPT.format(code=code, lang=lang, description=desc),
        "efficiency": EFFICIENCY_PROMPT.format(code=code, lang=lang, description=desc),
        "readability": SYNTAX_PROMPT.format(code=code, lang=lang, description=desc)
    }


def extract_score(output: str):
    if not output or not isinstance(output, str):
        return None

    text = output.lower().strip()

    try:
        val = float(text)
        if 1 <= val <= 5:
            return int(val)
    except:
        pass

    patterns = [
        r"score\s*[:=]?\s*([1-5])",
        r"([1-5])\s*/\s*5",
        r"([1-5])\s*out of\s*5",
        r"rating\s*[:=]?\s*([1-5])",
        r"i give (?:it )?([1-5])",
    ]

    for p in patterns:
        match = re.search(p, text)
        if match:
            return int(match.group(1))

    sentence_patterns = [
        r"score (?:is|of)?\s*([1-5])",
        r"would be\s*([1-5])",
        r"should be\s*([1-5])",
        r"rated\s*([1-5])",
    ]

    for p in sentence_patterns:
        match = re.search(p, text)
        if match:
            return int(match.group(1))

    numbers = re.findall(r"\b[1-5]\b", text)

    if not numbers:
        return None

    numbers = [int(n) for n in numbers]

    if len(set(numbers)) > 1:
        return Counter(numbers).most_common(1)[0][0]

    return numbers[0]

def infer_criteria(model, tokenizer, prompts, model_name=None):
    results = {}

    # ===== build allowed tokens (1–5) =====
    allowed_token_ids = set()

    for i in range(1, 6):
        for variant in [str(i), f" {i}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                allowed_token_ids.add(ids[0])

    allowed_token_ids = list(allowed_token_ids)

    if len(allowed_token_ids) == 0:
        raise ValueError("No valid single-token digits found!")

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return allowed_token_ids

    # ===== infer =====
    for criterion, prompt_text in prompts.items():
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,                     
                max_new_tokens=1,                   
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                eos_token_id=tokenizer.eos_token_id
            )

        output = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        try:
            score = int(output)
        except:
            score = None

        results[f"{criterion}_score"] = score

        del output_ids
        torch.cuda.empty_cache()

    return results

def main(model_name, use_bnb=False, submissions_path="./data/original_test.jsonl", metadata_path="./data/metadata.jsonl", limit=None):
    dataset = SubmissionDataset(submissions_path, metadata_path, limit=limit)
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, use_bnb=use_bnb, device_map="cuda")

    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", f"{model_name.replace('/', '_')}_pointwise.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for sub in tqdm(dataset, desc="Infer submissions", unit="sub"):
            prompts = build_prompts(sub)
            scores = infer_criteria(model, tokenizer, prompts, model_name=model_name)
            out = {
                "sub_id": sub['sub_id'],
                "gt_correctness_score": sub['gt_correctness_score'],
                "gt_efficiency_score": sub['gt_efficiency_score'],
                "gt_readability_score": sub['gt_readability_score'],
                **scores
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Inference done! Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--use_bnb', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    main(args.model_name, args.use_bnb, limit=args.limit)