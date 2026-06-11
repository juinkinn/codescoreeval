import argparse
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from evaluation.utils import load_jsonl
from finetune.prompt import POINTWISE_PROMPTS


CRITERIA = ("correctness", "efficiency", "readability")


def load_map(path, key):
    return {row[key]: row for row in load_jsonl(path)}


def get_problem_id(label):
    if label.get("id"):
        return label["id"]
    return label["sub_id"].rsplit("_", 1)[0]


def parse_target_modules(value):
    return [module.strip() for module in value.split(",") if module.strip()]


def prepare_output_dir(output_dir):
    if not output_dir or not output_dir.strip():
        raise ValueError("--output-dir must not be empty")

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_precision(precision):
    if precision == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested, but this GPU does not support bf16")
        return torch.bfloat16, True, False, "bf16"

    if precision == "fp16":
        return torch.float16, False, True, "fp16"

    if precision == "fp32":
        return torch.float32, False, False, "fp32"

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False, "bf16"

    if torch.cuda.is_available():
        return torch.float16, False, True, "fp16"

    return torch.float32, False, False, "fp32"


def build_messages(criterion, label, submission, metadata):
    lang = submission.get("lang", "")
    code = submission.get("code", "")
    description = metadata.get("description", "")
    score = str(label[f"{criterion}_score"])

    system_prompt, user_prompt = POINTWISE_PROMPTS[criterion]
    user_prompt = user_prompt.format(
        code=code,
        lang=lang,
        description=description,
    ).strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": score},
    ]


def tokenize_messages(tokenizer, messages, max_length, overlength_policy):
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, verbose=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, verbose=False)["input_ids"]

    if len(prompt_ids) >= len(full_ids):
        return None, "empty_response"

    if len(full_ids) > max_length:
        if overlength_policy == "skip":
            return None, "overlength"

        response_ids = full_ids[len(prompt_ids):]
        available_prompt_length = max_length - len(response_ids)
        if available_prompt_length <= 0:
            return None, "overlength"

        input_ids = prompt_ids[-available_prompt_length:] + response_ids
        labels = [-100] * available_prompt_length + response_ids
    else:
        input_ids = full_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }, None


def build_tokenized_examples(
    labels_path,
    submissions_path,
    metadata_path,
    tokenizer,
    max_length,
    overlength_policy,
    max_samples=None,
    seed=42,
    desc="Tokenizing examples",
):
    labels = load_jsonl(labels_path)
    submissions = load_map(submissions_path, "sub_id")
    metadata = load_map(metadata_path, "id")

    examples = []
    skipped_missing = 0
    skipped_overlength = 0
    skipped_empty_response = 0

    for label in tqdm(labels, desc=desc, unit="labels", dynamic_ncols=True):
        sub_id = label["sub_id"]
        problem_id = get_problem_id(label)
        submission = submissions.get(sub_id)
        meta = metadata.get(problem_id)

        if not submission or not meta:
            skipped_missing += 1
            continue

        for criterion in CRITERIA:
            messages = build_messages(criterion, label, submission, meta)
            example, reason = tokenize_messages(tokenizer, messages, max_length, overlength_policy)

            if reason == "overlength":
                skipped_overlength += 1
                continue
            if reason == "empty_response":
                skipped_empty_response += 1
                continue

            example.update({"sub_id": sub_id, "id": problem_id, "criterion": criterion})
            examples.append(example)

    if max_samples is not None and len(examples) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_samples]

    if skipped_missing:
        print(f"[WARN] Skipped {skipped_missing} labels because code or metadata was missing")
    if skipped_overlength:
        print(f"[WARN] Skipped {skipped_overlength} examples over max_length={max_length}")
    if skipped_empty_response:
        print(f"[WARN] Skipped {skipped_empty_response} examples with empty assistant response")

    return examples


def build_data_collator(tokenizer):
    def collate(features):
        labels = [feature["labels"] for feature in features]
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = tokenizer.pad(model_features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]

        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_length - len(label)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return collate


def load_model_and_tokenizer(args):
    torch_dtype, _, _, precision_name = resolve_precision(args.precision)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.gradient_checkpointing:
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer, precision_name


def make_training_arguments(args):
    _, bf16, fp16, _ = resolve_precision(args.precision)

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        optim=args.optim,
        seed=args.seed,
        report_to=args.report_to,
        remove_unused_columns=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pointwise code-scoring model with Transformers + PEFT.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--train-labels", default="data/finetune/pointwise_train_labels.jsonl")
    parser.add_argument("--valid-labels", default="data/finetune/pointwise_valid_labels.jsonl")
    parser.add_argument("--submissions", default="data/submissions.jsonl")
    parser.add_argument("--metadata", default="data/metadata.jsonl")
    parser.add_argument("--output-dir", default="output/finetune/qwen2_5_coder_3b_pointwise_transformers")

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--overlength-policy", choices=("skip", "truncate_prompt"), default="skip")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Training precision. auto uses bf16 when supported, otherwise fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    args = parser.parse_args()
    args.output_dir = prepare_output_dir(args.output_dir)

    model, tokenizer, precision_name = load_model_and_tokenizer(args)

    train_examples = build_tokenized_examples(
        args.train_labels,
        args.submissions,
        args.metadata,
        tokenizer,
        args.max_length,
        args.overlength_policy,
        max_samples=args.max_train_samples,
        seed=args.seed,
        desc="Tokenizing train labels",
    )
    valid_examples = build_tokenized_examples(
        args.valid_labels,
        args.submissions,
        args.metadata,
        tokenizer,
        args.max_length,
        args.overlength_policy,
        max_samples=args.max_valid_samples,
        seed=args.seed,
        desc="Tokenizing valid labels",
    )

    if not train_examples:
        raise ValueError("No training examples were built. Check paths or increase --max-length.")

    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples) if valid_examples else None

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset) if valid_dataset is not None else 0}")
    print(f"Output dir: {args.output_dir}")
    print(f"Precision: {precision_name}")
    print(f"Overlength policy: {args.overlength_policy}")

    trainer = Trainer(
        model=model,
        args=make_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=build_data_collator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
