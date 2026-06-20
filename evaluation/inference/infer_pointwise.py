import torch
import json
import os
import re
from dataset import SubmissionDataset
from loader import load_tokenizer, load_model
from prompts import CORRECTNESS_PROMPT, EFFICIENCY_PROMPT, SYNTAX_PROMPT
from tuning_prompts import POINTWISE_PROMPTS
from tqdm import tqdm
from collections import Counter

def build_prompts_tuning(submission):
    code = submission["code"]
    lang = submission.get("lang", "")
    desc = submission.get("description", "")

    prompts = {}

    for criterion, (system_prompt, user_template) in POINTWISE_PROMPTS.items():

        user_prompt = user_template.format(
            code=code,
            lang=lang,
            description=desc
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompts[criterion] = messages

    return prompts

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


def infer_criteria(model, tokenizer, prompts, max_retry=3):
    results = {}

    for criterion, prompt in prompts.items():
        score = None
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
                    max_new_tokens=128,
                    eos_token_id=tokenizer.eos_token_id,
                )

            output = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            score = extract_score(output)

            if score is not None and 1 <= score <= 5:
                break

        results[f"{criterion}_score"] = score

        del output_ids
        torch.cuda.empty_cache()

    return results

def main(
    model_name,
    use_bnb=False,
    submissions_path="data/original_test.jsonl",
    metadata_path="data/metadata.jsonl",
    limit=None,
    adapter_path=None,
):
    dataset = SubmissionDataset(submissions_path, metadata_path, limit=limit)
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, adapter_path=adapter_path, use_bnb=use_bnb, device_map="cuda")

    os.makedirs("output", exist_ok=True)
    suffix = "tuning" if adapter_path else "raw"
    output_file = os.path.join("output", f"{model_name.replace('/', '_')}_{suffix}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for sub in tqdm(dataset, desc="Infer submissions", unit="sub"):
            use_tuning = adapter_path is not None
            prompts = build_prompts_tuning(sub) if use_tuning else build_prompts(sub)
            scores = infer_criteria(model, tokenizer, prompts)
            out = {
                "sub_id": sub["sub_id"],
                "gt_correctness_score": sub["gt_correctness_score"],
                "gt_efficiency_score": sub["gt_efficiency_score"],
                "gt_readability_score": sub["gt_readability_score"],
                **scores,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Inference done! Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter_path", type=str, default=None)
    args = parser.parse_args()
    main(args.model_name, args.use_bnb, limit=args.limit, adapter_path=args.adapter_path)