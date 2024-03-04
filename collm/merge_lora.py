import argparse
import copy
import logging
import os
import shutil

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import dequantize_4bit
from peft import PeftConfig, PeftModel
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Enable logging
logger = logging.getLogger(__name__)


def dequantize_model(model, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                quant_state[2] = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        # to save model, you have to unset this attribute
        model.is_loaded_in_4bit = False

        return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--qlora", action="store_true")  # qlora requires special treatment.
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    return parser.parse_args()


def load_lora_model_tokenizer(
    lora_model_name_or_path,
    base_model_name_or_path,
    use_qlora,
    tokenizer_padding_side,
    use_fast_tokenizer=True,
):
    peft_config = PeftConfig.from_pretrained(lora_model_name_or_path)
    base_model_name_or_path = (
        base_model_name_or_path if base_model_name_or_path else peft_config.base_model_name_or_path
    )
    logger.info("Loading the base model...")

    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",  # TODO: double check this -- it was {"":0}
        )
        base_model = dequantize_model(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
        )

    logger.info("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, lora_model_name_or_path)

    logger.info("Merging the lora modules...")
    merged_model = lora_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        lora_model_name_or_path, use_fast=use_fast_tokenizer, padding_side=tokenizer_padding_side
    )

    embedding_size = merged_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.warning(
            f"The vocabulary size of the tokenizer in the lora model folder contains {len(tokenizer)-embedding_size} more tokens than the base model."
        )
        logger.warning("Resizing the token embeddings of the merged model...")
        merged_model.resize_token_embeddings(len(tokenizer))

    print(merged_model.lm_head.weight.data)  # Sanity check
    return merged_model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    peft_config = PeftConfig.from_pretrained(args.lora_model_name_or_path)
    print("Loading the base model...")
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path if args.base_model_name_or_path else peft_config.base_model_name_or_path,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        base_model = dequantize_model(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path if args.base_model_name_or_path else peft_config.base_model_name_or_path,
        )
    print("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_name_or_path)
    print("Merging the lora modules...")
    merged_model = lora_model.merge_and_unload()
    output_dir = args.output_dir if args.output_dir else args.lora_model_name_or_path
    # Thanks to @TJKlein for the suggestion.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_model_name_or_path, use_fast=args.use_fast_tokenizer)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=args.use_fast_tokenizer)
    embedding_size = merged_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(
            f"The vocabulary size of the tokenizer in the lora model folder contains {len(tokenizer)-embedding_size} more tokens than the base model."
        )
        print("Resizing the token embeddings of the merged model...")
        merged_model.resize_token_embeddings(len(tokenizer))
    print(f"Saving to {output_dir}...")
    merged_model.save_pretrained(output_dir)
