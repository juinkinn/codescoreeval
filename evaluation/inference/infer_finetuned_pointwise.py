import sys
import torch
import json
import os
import re
from pathlib import Path
from dataset import SubmissionDataset
from tqdm import tqdm
from collections import Counter
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from finetune.prompt import POINTWISE_PROMPTS

CRITERIA = ("correctness", "efficiency", "readability")


def _read_base_model_name(adapter_path: str):
    """Read base_model_name_or_path from adapter_config.json, which PEFT
    writes automatically when an adapter is saved."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"adapter_config.json at {adapter_path} has no base_model_name_or_path; "
            "pass --base-model-name explicitly."
        )
    return base_model_name


def load_finetuned(adapter_path: str, base_model_name: str = None, use_bnb: bool = False, device_map="cuda"):
    """Load base model + LoRA adapter (PEFT) directly. adapter_path must be a
    local directory containing adapter_config.json + adapter_model.safetensors,
    as saved by Trainer.save_model() with a PEFT-wrapped model."""

    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise ValueError(
            f"{adapter_path} does not look like a PEFT adapter dir "
            "(missing adapter_config.json)."
        )

    resolved_base = base_model_name or _read_base_model_name(adapter_path)

    # Tokenizer: prefer the one saved alongside the adapter (guaranteed
    # consistent with training); fall back to the base model's tokenizer.
    if os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json")):
        print(f"[Tokenizer] Loading from ADAPTER dir: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    else:
        print(f"[Tokenizer] No tokenizer in adapter dir, loading BASE: {resolved_base}")
        tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    if use_bnb:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"[Model] Loading BASE: {resolved_base}")
    base_model = AutoModelForCausalLM.from_pretrained(resolved_base, **kwargs)

    print(f"[Model] Attaching ADAPTER: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def build_messages(criterion, submission):
    code = submission["code"]
    lang = submission.get("lang", "")
    description = submission.get("description", "")

    system_prompt, user_prompt = POINTWISE_PROMPTS[criterion]
    user_prompt = user_prompt.format(
        code=code,
        lang=lang,
        description=description,
    ).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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


def infer_criteria(model, tokenizer, submission, max_retry=3, greedy=True):
    results = {}

    for criterion in CRITERIA:
        score = None

        messages = build_messages(criterion, submission)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        for attempt in range(max_retry):
            gen_kwargs = dict(
                max_new_tokens=8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

            if greedy and attempt == 0:
                gen_kwargs.update(do_sample=False, num_beams=1)
            else:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95,
                    repetition_penalty=1.15,
                )

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            output = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            score = extract_score(output)

            del output_ids
            torch.cuda.empty_cache()

            if score is not None and 1 <= score <= 5:
                break

        results[f"{criterion}_score"] = score

    return results


def main(
    model_name,
    use_bnb=False,
    base_model_name=None,
    submissions_path="../../data/original_test.jsonl",
    metadata_path="../../data/metadata.jsonl",
    limit=None,
):
    dataset = SubmissionDataset(submissions_path, metadata_path, limit=limit)
    model, tokenizer = load_finetuned(
        model_name,
        base_model_name=base_model_name,
        use_bnb=use_bnb,
        device_map="cuda",
    )

    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", f"{model_name.replace('/', '_')}_pointwise.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for sub in tqdm(dataset, desc="Infer submissions", unit="sub"):
            scores = infer_criteria(model, tokenizer, sub)
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
    parser.add_argument("--model_name", type=str, required=True, help="Path to the PEFT adapter dir (output_dir from training).")
    parser.add_argument("--base-model-name", type=str, default=None, help="Override the base model; defaults to adapter_config.json's base_model_name_or_path.")
    parser.add_argument("--use_bnb", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.model_name, args.use_bnb, base_model_name=args.base_model_name, limit=args.limit)
