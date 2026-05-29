import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def _is_valid_local(path: str):
    return isinstance(path, str) and os.path.isdir(path)


def load_tokenizer(model_name: str):
    # local path
    if _is_valid_local(model_name):
        print(f"[Tokenizer] Loading LOCAL: {model_name}")
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    # HF fallback
    print(f"[Tokenizer] Loading HF: {model_name}")
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def load_model(model_name: str, use_bnb: bool = False, device_map="cuda"):

    kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 4bit quant
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        kwargs["quantization_config"] = bnb_config

    # LOCAL path
    if _is_valid_local(model_name):
        print(f"[Model] Loading LOCAL: {model_name}")
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs
        )

    # HF fallback
    print(f"[Model] Loading HF: {model_name}")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs
    )