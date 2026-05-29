import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


def _is_local_path(path: str):
    return os.path.isdir(path)


def load_tokenizer(model_name: str):
    # local path or HF repo auto-handle
    if _is_local_path(model_name):
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
    else:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )


def load_model(model_name: str, use_bnb: bool = False, device_map="cuda"):

    kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
    )

    kwargs["torch_dtype"] = torch.bfloat16

    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        kwargs["quantization_config"] = bnb_config

    if _is_local_path(model_name):
        model_path = model_name
    else:
        model_path = model_name

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **kwargs
    )

    model.eval()
    return model