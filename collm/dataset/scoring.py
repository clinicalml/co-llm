import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import datasets
import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    OPTConfig,
    OPTForCausalLM,
    SchedulerType,
    get_scheduler,
)

try:
    from transformers import CodeLlamaTokenizer, CodeLlamaTokenizerFast
except ImportError:
    print("You might install transformers >= 4.33.")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this.",
    )
    parser.add_argument(
        "--use_completion_format",
        action="store_true",
        help="If passed, will use the completion format to train the model.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the scoring on the first 5 samples for debugging and testing purposes.",
    )
    parser.add_argument(
        "--preprocessing_format",
        type=str,
        default="tulu_chat",
        help="The format of the dataset to use for the preprocessing. {tulu_chat, llama2_chat}",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json",
                "jsonl",
            ], "`train_file` should be a json/jsonl file."
    return args


def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += (
                "<|assistant|>\n" + message["content"].strip() + eos + "\n"
            )
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def create_prompt_with_llama2_chat_format(
    messages, bos="<s>", eos="</s>", add_bos=True
):
    """
    This function is adapted from the official llama2 chat completion script:
    https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
    """
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    formatted_text = ""
    # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
    # However, see here that removing the system prompt actually reduce the false refusal rates: https://github.com/facebookresearch/llama/blob/main/UPDATES.md?utm_source=twitter&utm_medium=organic_social&utm_campaign=llama2&utm_content=text#observed-issue
    if messages[0]["role"] == "system":
        assert (
            len(messages) >= 2 and messages[1]["role"] == "user"
        ), "LLaMa2 chat cannot start with a single system message."
        messages = [
            {
                "role": "user",
                "content": B_SYS
                + messages[0]["content"]
                + E_SYS
                + messages[1]["content"],
            }
        ] + messages[2:]
    for message in messages:
        if message["role"] == "user":
            formatted_text += bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
        elif message["role"] == "assistant":
            formatted_text += f" {(message['content'])} " + eos
        else:
            raise ValueError(
                "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    # The llama2 chat template by default has a bos token at the start of each user message.
    # The next line removes the bos token if add_bos is False.
    formatted_text = formatted_text[len(bos) :] if not add_bos else formatted_text
    return formatted_text


def encode_with_messages_llama2_chat_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    example_text = create_prompt_with_llama2_chat_format(
        messages, add_bos=False
    ).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    create_prompt_with_llama2_chat_format(
                        messages[:message_idx], add_bos=False
                    ),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = create_prompt_with_llama2_chat_format(
                    messages[: message_idx + 1], add_bos=False
                )
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example[
        "completion"
    ].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"],
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def score_dataset_items_with_model(data_dict, tokenizer, model):
    """
    This function will add a new column to the data_dict called "output_prob"
    which represents the probability of the next token in the sequence according the input model.
    """

    logging.info("Scoring inputs with regular model...")
    # data_dict = data_dict.map(score_examples, batched=True, batch_size=1) # This is slow since it has to hash the function

    reference_log_probs = []
    # reference_entropy = []
    reference_max_prob = []
    # reference_nll_loss = []

    with torch.no_grad():
        for i in tqdm(range(len(data_dict))):
            examples = data_dict[i : i + 1]
            input_ids = [
                torch.LongTensor(instance) for instance in examples["input_ids"]
            ]
            labels = [torch.LongTensor(instance) for instance in examples["input_ids"]]
            # A list of LongTensors

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(model.device)

            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            ).to(model.device)

            model_inputs = dict(
                input_ids=input_ids,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
            )

            # fmt: off
            model_outputs = model(**model_inputs)
            ref_shift_logits = model_outputs.logits[..., :-1, :].contiguous() # [batch_size, seq_len, vocab_size]
            ref_shift_log_probs = torch.log_softmax(ref_shift_logits, dim=-1)
            ref_log_probs_flat = ref_shift_log_probs.view(-1, ref_shift_log_probs.size(-1)) # [batch_size * seq_len, vocab_size]

            # Compute Entropy
            # ref_probs_flat = torch.exp(ref_log_probs_flat)
            # p_log_p = ref_probs_flat * ref_log_probs_flat
            # entropy = -p_log_p.mean(dim=-1)

            # Get max prob
            max_prob = ref_log_probs_flat.max(dim=-1)[0]

            shift_labels = labels[..., 1:].contiguous() # [batch_size, seq_len]

            # ref_nll_loss = torch.nn.functional.nll_loss(
            #     ref_log_probs_flat, 
            #     shift_labels.view(-1,), reduction='none')
            
            # When there's a padding token, it is originally set to -100 by default
            # Here we want to set it to 0 such that we can use it as an index in the gather function
            padding_mask = shift_labels == IGNORE_INDEX
            shift_labels[padding_mask] = 0 

            ref_log_prob_flat = torch.gather(ref_log_probs_flat, 1, shift_labels.reshape(-1,).unsqueeze(-1)) # [batch_size * seq_len, 1]
            
            ref_log_prob = ref_log_prob_flat.view_as(shift_labels).detach().to('cpu').float() # [batch_size, seq_len]
            # entropy = entropy.view_as(shift_labels).detach().to('cpu').float()
            max_prob = max_prob.view_as(shift_labels).detach().to('cpu').float()
            # ref_nll_loss = ref_nll_loss.view_as(shift_labels).detach().cpu().float()
            padding_mask = padding_mask.detach().to('cpu')

            # We want to remove the padded tokens from the output
            ref_log_prob = [row[~padding_mask[idx]].numpy().tolist() for idx, row in enumerate(ref_log_prob)]
            reference_log_probs.extend(ref_log_prob)

            # entropy = [row[~padding_mask[idx]].numpy().tolist() for idx, row in enumerate(entropy)]
            # reference_entropy.extend(entropy)

            max_prob = [row[~padding_mask[idx]].numpy().tolist() for idx, row in enumerate(max_prob)]
            reference_max_prob.extend(max_prob)

            # ref_nll_loss = [row[~padding_mask[idx]].numpy().tolist() for idx, row in enumerate(ref_nll_loss)]
            # reference_nll_loss.extend(ref_nll_loss)
            # fmt: on

    data_dict = data_dict.add_column("reference_log_probs", reference_log_probs)
    # data_dict = data_dict.add_column("reference_entropy", reference_entropy)
    data_dict = data_dict.add_column("reference_max_prob", reference_max_prob)
    # data_dict = data_dict.add_column("reference_nll_loss", reference_nll_loss)

    # Some checks
    for example in data_dict:
        assert len(example["reference_log_probs"]) == len(example["input_ids"]) - 1

    return data_dict


if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(logging.INFO)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = datasets.load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    # elif args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "Please provide either a tokenizer name specifically. We might need to train models with some different tokenizations;"
            "So we want to make sure explicitly specify the tokenizer and use it."
        )

    print(tokenizer)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if (
        isinstance(tokenizer, LlamaTokenizer)
        or isinstance(tokenizer, LlamaTokenizerFast)
        or isinstance(tokenizer, CodeLlamaTokenizer)
        or isinstance(tokenizer, CodeLlamaTokenizerFast)
    ):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
            }
        )
        assert (
            num_added_tokens == 1
        ), "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer):
        config = AutoConfig.from_pretrained(
            args.model_name_or_path
        )  # TODO: NOT TESTED so far.
        # We just don't want to load any models here, so we can use the config to check if we need to add special tokens.
        if isinstance(config, OPTConfig):
            num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})

    raw_datasets = raw_datasets["train"]

    if args.use_completion_format:
        if "messages" in raw_datasets.column_names:
            # Making sure the
            # assert all([len(example["messages"]) == 2 for example in raw_datasets])

            # We just take the first two messages as prompt and completion.
            raw_datasets = raw_datasets.map(
                lambda example: {
                    "prompt": example["messages"][0]["content"].strip() + "\n",
                    "completion": example["messages"][1]["content"].strip(),
                },
                batched=False,
                remove_columns=["messages"],
            )

    # Preprocessing the datasets.
    if (
        "prompt" in raw_datasets.column_names
        and "completion" in raw_datasets.column_names
    ):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        if args.preprocessing_format == "tulu_chat":
            encode_function = partial(
                encode_with_messages_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
            )
        elif args.preprocessing_format == "llama2_chat":
            encode_function = partial(
                encode_with_messages_llama2_chat_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
            )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names."
        )

    # Step 1: Reformatting the dataset
    logger.info("Reformatting the dataset")
    if args.debug:
        raw_datasets = raw_datasets.select(range(5))
    elif args.max_train_samples:
        max_train_samples = min(len(raw_datasets), args.max_train_samples)
        random_indices = list(range(len(raw_datasets)))
        random.seed(42)
        random.shuffle(random_indices)
        raw_datasets = raw_datasets.select(random_indices[:max_train_samples])

    data_dict = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[
            name
            for name in raw_datasets.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )
    data_dict.set_format(type="pt")
    data_dict = data_dict.filter(lambda example: (example["labels"] != -100).any())

    # Step 2: Load the models
    logger.info("Loading the model")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
        # use_flash_attention_2=True,
    )
    model.eval()

    print(model.lm_head.weight.data)  # Sanity check

    # Step 3: Run Scoring with the model
    data_dict = score_dataset_items_with_model(data_dict, tokenizer, model)

    # Finally, save to the target folder
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data_dict.save_to_disk(args.output_dir)
