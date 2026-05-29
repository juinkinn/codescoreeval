from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def load_model(model_name: str, use_bnb: bool = False, device_map="cuda"):

    if use_bnb:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )