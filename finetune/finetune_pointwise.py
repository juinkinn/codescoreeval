import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from evaluation.inference.prompts import CORRECTNESS_PROMPT, EFFICIENCY_PROMPT, SYNTAX_PROMPT
from evaluation.utils import load_jsonl
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments


CRITERIA = ("correctness", "efficiency", "readability")
PROMPTS = {
    "correctness": CORRECTNESS_PROMPT,
    "efficiency": EFFICIENCY_PROMPT,
    "readability": SYNTAX_PROMPT,
}


def load_map(path, key):
    return {row[key]: row for row in load_jsonl(path)}


def get_problem_id(label):
    if label.get("id"):
        return label["id"]
    return label["sub_id"].rsplit("_", 1)[0]


def build_messages(criterion, label, submission, metadata):
    lang = submission.get("lang", "")
    code = submission.get("code", "")
    description = metadata.get("description", "")
    score = str(label[f"{criterion}_score"])

    prompt = PROMPTS[criterion].format(
        code=code,
        lang=lang,
        description=description,
    ).strip()

    return [
        {
            "role": "system",
            "content": "You are a senior competitive programming code reviewer. Return only one integer from 1 to 5.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": score},
    ]


def build_examples(labels_path, submissions_path, metadata_path, tokenizer, max_samples=None, seed=42):
    labels = load_jsonl(labels_path)
    submissions = load_map(submissions_path, "sub_id")
    metadata = load_map(metadata_path, "id")

    examples = []
    skipped = 0

    for label in labels:
        sub_id = label["sub_id"]
        problem_id = get_problem_id(label)
        submission = submissions.get(sub_id)
        meta = metadata.get(problem_id)

        if not submission or not meta:
            skipped += 1
            continue

        for criterion in CRITERIA:
            messages = build_messages(criterion, label, submission, meta)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            examples.append(
                {
                    "text": text,
                    "sub_id": sub_id,
                    "id": problem_id,
                    "criterion": criterion,
                }
            )

    if max_samples is not None and len(examples) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_samples]

    if skipped:
        print(f"[WARN] Skipped {skipped} labels because code or metadata was missing")

    return examples


def parse_target_modules(value):
    return [module.strip() for module in value.split(",") if module.strip()]


def make_training_arguments(args):
    try:
        return SFTConfig(
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
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
            bf16=not args.fp16,
            fp16=args.fp16,
            optim=args.optim,
            seed=args.seed,
            report_to=args.report_to,
        )
    except ImportError:
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
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps if args.eval_steps > 0 else None,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            gradient_checkpointing=args.gradient_checkpointing,
            bf16=not args.fp16,
            fp16=args.fp16,
            optim=args.optim,
            seed=args.seed,
            report_to=args.report_to,
        )


def create_trainer(model, tokenizer, train_dataset, valid_dataset, training_args, args):

    common = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset if len(valid_dataset) else None,
        "args": training_args,
    }

    try:
        return SFTTrainer(
            **common,
            processing_class=tokenizer,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
        )
    except TypeError:
        return SFTTrainer(
            **common,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
        )


def maybe_train_on_responses_only(trainer, args):
    if args.disable_train_on_responses_only:
        return trainer

    try:

        return train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    except Exception as exc:
        print(f"[WARN] Could not enable response-only loss masking: {exc}")
        return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pointwise code-scoring model with Unsloth QLoRA.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--train-labels", default="data/finetune/pointwise_train_labels.jsonl")
    parser.add_argument("--valid-labels", default="data/finetune/pointwise_valid_labels.jsonl")
    parser.add_argument("--submissions", default="data/submissions.jsonl")
    parser.add_argument("--metadata", default="data/metadata.jsonl")
    parser.add_argument("--output-dir", default="output/finetune/qwen2_5_coder_3b_pointwise")

    parser.add_argument("--max-seq-length", type=int, default=4096)
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
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--disable-train-on-responses-only", action="store_true")
    parser.add_argument("--save-merged-16bit", action="store_true")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=parse_target_modules(args.target_modules),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=args.seed,
    )

    train_examples = build_examples(
        args.train_labels,
        args.submissions,
        args.metadata,
        tokenizer,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    valid_examples = build_examples(
        args.valid_labels,
        args.submissions,
        args.metadata,
        tokenizer,
        max_samples=args.max_valid_samples,
        seed=args.seed,
    )

    if not train_examples:
        raise ValueError("No training examples were built. Check label, submission, and metadata paths.")

    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples) if valid_examples else Dataset.from_list([])

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")
    print(f"Output dir: {args.output_dir}")

    training_args = make_training_arguments(args)
    trainer = create_trainer(model, tokenizer, train_dataset, valid_dataset, training_args, args)
    trainer = maybe_train_on_responses_only(trainer, args)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        model.save_pretrained_merged(
            str(Path(args.output_dir) / "merged_16bit"),
            tokenizer,
            save_method="merged_16bit",
        )


if __name__ == "__main__":
    main()
