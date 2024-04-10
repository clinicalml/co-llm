import asyncio
import functools
import json
import logging
import math
import multiprocessing
import os
import random
import re
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import datasets
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import shortuuid
import torch
import transformers
from rich.console import Console
from rich.text import Text
from tqdm import tqdm
from vllm.outputs import CompletionOutput, RequestOutput

DEFAULT_COLORS = [
    "#f3f0f0",
    "#f3e7e3",
    "#f4ddd7",
    "#f5d3ca",
    "#f5cabe",
    "#f6c0b1",
    "#f7b7a5",
    "#f8ad98",
    "#f8a38b",
    "#f99a7f",
]

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def visualize_deferral_generation(cur_generated_tokens, is_generated_token_deferred, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(cur_generated_tokens)

    current_sub_tokens = []
    current_sub_deferral = []
    out_string = ""
    prev_is_special = False
    for token, is_deferred in zip(tokens, is_generated_token_deferred):
        # make sure that special tokens are not decoded using sentencepiece model
        if token in tokenizer.all_special_tokens:
            continue
        else:
            current_sub_tokens.append(token)
            current_sub_deferral.append(is_deferred)

    proto = tokenizer.sp_model.decode(current_sub_tokens, out_type="immutable_proto")
    assert len(proto.pieces) == len(current_sub_deferral)

    console = Console()
    styled_text = Text()

    for piece, is_deferred in zip(proto.pieces, current_sub_deferral):
        word = proto.text[piece.begin : piece.end]
        if is_deferred:
            styled_text.append(Text().from_markup(f"[bold on {DEFAULT_COLORS[-1]}]{word}[/]"))
        else:
            styled_text.append(Text().from_markup(f"[on {DEFAULT_COLORS[0]}]{word}[/]"))

    console.print(styled_text)


def get_response(response: Union[requests.Response, Dict]) -> List[RequestOutput]:
    if isinstance(response, requests.Response):
        data = json.loads(response.content)
    else:
        data = response
    return dict_to_request_output(data)


def request_output_to_dict(request_output):
    def output_to_dict(output) -> dict:
        return {
            "index": output.index,
            "text": output.text,
            "token_ids": output.token_ids,
            "cumulative_logprob": output.cumulative_logprob,
            "logprobs": output.logprobs if output.logprobs else None,
            "finish_reason": output.finish_reason,
        }

    return {
        "request_id": request_output.request_id,
        "prompt": request_output.prompt,
        "prompt_token_ids": request_output.prompt_token_ids,
        "prompt_logprobs": request_output.prompt_logprobs,  # Assuming PromptLogprobs has a to_dict method
        "outputs": [output_to_dict(output) for output in request_output.outputs],
        "finished": request_output.finished,
    }


def dict_to_request_output(data: Dict):
    outputs_data = data["outputs"]
    outputs = [
        CompletionOutput(
            index=output["index"],
            text=output["text"],
            token_ids=output["token_ids"],
            cumulative_logprob=output["cumulative_logprob"],
            logprobs=(
                [{int(key): val for key, val in logprob.items()} for logprob in output.get("logprobs")]
                if output.get("logprobs")
                else None
            ),
            finish_reason=output.get("finish_reason"),
        )
        for output in outputs_data
    ]

    obj = RequestOutput(
        request_id=data["request_id"],
        prompt=data["prompt"],
        prompt_token_ids=data["prompt_token_ids"],
        prompt_logprobs=data["prompt_logprobs"],
        outputs=outputs,
        finished=data["finished"],
    )
    if "logits" in data:
        obj.logits = data["logits"]

    return obj


def is_contain_stop_sequence(generated: List[int], stop_sequences: List[List[int]], tokens_per_call: int):
    for stop_index in reversed(range(0, tokens_per_call)):
        for stop_seq in stop_sequences:
            start_index = max(len(generated) - stop_index - len(stop_seq), 0)
            end_index = len(generated) - stop_index
            if generated[start_index:end_index] == stop_seq:
                return True
    return False


def find_stop_sequence(generated: List[int], stop_sequences: List[List[int]], tokens_per_call: int):
    for stop_index in reversed(range(0, tokens_per_call)):
        for stop_seq in stop_sequences:
            start_index = max(len(generated) - stop_index - len(stop_seq), 0)
            end_index = len(generated) - stop_index
            if generated[start_index:end_index] == stop_seq:
                return (
                    True,
                    generated[len(generated) - tokens_per_call : len(generated) - stop_index],
                )
    return False, None


def is_contain_stop_pattern(pattern: str, generated: List[int], tokenizer: transformers.PreTrainedTokenizer):
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    # print(bool(re.search(pattern, generated_text)), generated_text)
    return bool(re.search(pattern, generated_text))


DEFAULT_ARGS = {
    "n": 1,
    "top_p": 1,
    "top_k": -1,
    "temperature": 0.0,
}


def get_next_prediction(
    prompt_token_ids,
    max_tokens,
    url: Union[List, str],
    vocab_size,
    target_vocab_size=None,
    return_logprobs_tensor=True,
    **kwargs,
):
    if isinstance(url, list):
        url = random.choice(url)

    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt_token_ids": prompt_token_ids,
        "max_tokens": max_tokens,
    }
    for key in ["n", "top_p", "top_k", "temperature"]:
        pload[key] = kwargs.pop(key, DEFAULT_ARGS[key])

    for key, value in kwargs.items():
        pload[key] = value

    pload["logprobs"] = vocab_size

    response = requests.post(url, headers=headers, json=pload, stream=False)
    reference_output = get_response(response)

    if not return_logprobs_tensor:
        return reference_output

    all_logprobs = []
    for logprob in reference_output.outputs[0].logprobs:
        log_prob_tensor = torch.zeros(vocab_size)
        for key, value in logprob.items():
            log_prob_tensor[key] = value

        all_logprobs.append(log_prob_tensor)
    all_logprobs = torch.vstack(all_logprobs)
    if target_vocab_size:
        all_logprobs = all_logprobs[:, :target_vocab_size]
    return reference_output, all_logprobs


def get_next_prediction_batch(forward_url: Union[List, str], prompt_token_id_list, max_tokens, url, **kwargs):
    if isinstance(forward_url, list):
        forward_url = random.choice(forward_url)

    headers = {"User-Agent": "Test Client"}
    # We need to be a bit prescriptive here, since we need to match exactly on the
    # javascript backend
    pload = {
        "forward_url": forward_url,
        "prompt_token_id_list": prompt_token_id_list,
        "n": 1,
        "top_p": 1,
        "top_k": -1,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stop": None,
        "logprobs": None,
    }

    for key in ["n", "top_p", "top_k", "temperature", "stop", "logprobs"]:
        pload[key] = kwargs.pop(key, pload[key])

    response = requests.post(url, headers=headers, json=pload, stream=False)
    data = response.json()
    reference_outputs = [get_response(example) for example in data]
    return reference_outputs


MODEL_INPUT_COL_NAME = "model_input"


def deferral_search(
    generated_token_ids,
    first_deferral_position,
    prompt_token_ids,
    beam_size,
    search_length,
    get_next_prediction_ref,
    get_next_prediction_base,
    get_next_prediction_batch_base,
    use_batch_scoring=False,
):
    # is_deferral = torch.sigmoid(log_probs_base[:, -1]) > def_threshold

    # deferred_positions = torch.where(is_deferral.squeeze())[0]
    # first_deferral_position = deferred_positions[0]

    output_ref, log_probs_ref = get_next_prediction_ref(
        prompt_token_ids + generated_token_ids[:first_deferral_position], 1
    )

    reference_top_tokens = log_probs_ref[0].topk(beam_size).indices.tolist()
    if generated_token_ids[-1] not in reference_top_tokens:
        reference_top_tokens.append(generated_token_ids[-1])

    if not use_batch_scoring:
        base_model_scoring = [
            get_next_prediction_base(
                prompt_token_ids + generated_token_ids[:first_deferral_position] + [token],
                search_length,
            )[0]
            for token in reference_top_tokens
        ]
    else:
        base_model_scoring = get_next_prediction_batch_base(
            [
                prompt_token_ids + generated_token_ids[:first_deferral_position] + [token]
                for token in reference_top_tokens
            ],
            search_length,
        )

    top_token_idx = np.argmax([item.outputs[0].cumulative_logprob for item in base_model_scoring])
    top_token = reference_top_tokens[top_token_idx]

    return top_token


def find_optimal_deferral_threshold(
    save_path: Path,
    ds: datasets.Dataset,
    n_samples: int,
    tokenizer: transformers.PreTrainedTokenizer,
    get_next_prediction_base,
    get_next_prediction_batch_ref,
) -> float:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    is_token_diff = []
    def_logits = []
    generated_examples = []

    ds_indices = list(range(n_samples))
    random.seed(42)
    random.shuffle(ds_indices)

    for example in ds.select(ds_indices):
        prompt_tokens = tokenizer(example["model_input"])["input_ids"]
        all_output_base, logprobs_base = get_next_prediction_base(prompt_tokens, 512)

        generated_tokens = all_output_base.outputs[0].token_ids
        all_segmented_sequences = [prompt_tokens + generated_tokens[:i] for i in range(0, len(generated_tokens))]

        cur_example = {
            "prompt": example["model_input"],
            "base_generated_text": all_output_base.outputs[0].text,
            "base_generated_tokens": all_output_base.outputs[0].token_ids,
            "deferral_probs": torch.sigmoid(logprobs_base[:, -1]).tolist(),
            "ref_model_tokens": [],
        }
        batch_size = 8
        ref_model_results = []
        for i in tqdm(range(0, len(all_segmented_sequences), batch_size)):
            ref_model_results.extend(
                get_next_prediction_batch_ref(
                    prompt_token_id_list=all_segmented_sequences[i : i + batch_size],
                    max_tokens=1,
                )
            )

        for idx in range(len(ref_model_results)):
            ref_model_token = ref_model_results[idx].outputs[0].token_ids[0]
            base_model_token = generated_tokens[idx]
            deferral_prob = logprobs_base[idx, -1]

            cur_example["ref_model_tokens"].append(ref_model_token)
            is_token_diff.append(ref_model_token != base_model_token)
            def_logits.append(deferral_prob)

        generated_examples.append(cur_example)

    predictions = torch.sigmoid(torch.stack(def_logits))
    true_labels = is_token_diff

    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    J = tpr - fpr  # Calculate Youden's J index
    idx = np.argmax(J)  # Find the optimal threshold index
    optimal_threshold = thresholds[idx]

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    # Set limits to prevent overlap with edges
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Threshold labeling
    for i in range(0, len(thresholds), 15):  # Adjust the step to your preference
        plt.scatter(fpr[i], tpr[i], s=10, color="black")
        plt.text(
            fpr[i],
            tpr[i],
            s=round(thresholds[i], 2),
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    # Highlight the optimal threshold on the plot
    plt.scatter(
        fpr[idx],
        tpr[idx],
        s=50,
        c="red",
        marker="x",
        label="Optimal threshold (%.2f)" % optimal_threshold,
        zorder=3,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Picking the best deferral threshold")
    plt.legend(loc="lower right")

    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path / f"deferral_threshold-{n_samples}.png")

    with open(save_path / f"generated_examples-{n_samples}.json", "w") as f:
        json.dump(generated_examples, f)

    return optimal_threshold


def calibrate_deferral_probability(
    n_samples: int,
    save_path: Path,
):
    deferral_threshold_search_path = save_path / f"generated_examples-{n_samples}.json"
    assert deferral_threshold_search_path.exists(), f"Could not find {deferral_threshold_search_path}"

    ds = datasets.load_dataset("json", data_files=str(deferral_threshold_search_path))["train"]

    all_probs = sum(ds["deferral_probs"], [])
    all_probs.sort()  # Sort the list of probabilities

    all_thresholds = []
    for target_portion in np.linspace(0, 1.0, 11):
        threshold = None

        for p in all_probs:
            # Calculate the proportion of predictions that are 1 at the current threshold p
            proportion = sum(value >= p for value in all_probs) / len(all_probs)
            if proportion <= target_portion:
                threshold = p
                break

        if threshold is None:
            threshold = 1.0  # If no threshold found, default to 1 so all probabilities are below threshold

        all_thresholds.append((target_portion, threshold))

    plt.figure(figsize=(10, 6), dpi=250)
    sns.histplot(all_probs, bins=20, kde=True)

    colors = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(all_thresholds))

    for cid, (target_portion, threshold) in enumerate(all_thresholds):
        plt.axvline(
            x=threshold,
            color=colors[cid],
            ls="--",
            label=f"%={target_portion:.1f}, th={threshold:.4f}",
        )
        plt.text(
            x=threshold,
            y=100,
            s=f"portion={target_portion:.1f}, th={threshold:.4f}",
            rotation=90,
        )

    plt.legend()
    plt.savefig(save_path / f"picking_thresholds-{n_samples}.png")

    return all_thresholds


def _exponential_warmup(step, warmup_steps, start_value, end_value):
    if step < warmup_steps:
        value = start_value * (end_value / start_value) ** (step / warmup_steps)
    else:
        value = end_value
    return value


def _cosine_warmup(step, warmup_steps, start_value, end_value):
    if step < warmup_steps:
        value = end_value + (start_value - end_value) / 2 * (math.cos(math.pi * step / warmup_steps) + 1)
    else:
        value = end_value
    return value


def _linear_warmup(step, warmup_steps, start_value, end_value):
    if step < warmup_steps:
        value = start_value + (end_value - start_value) * step / warmup_steps
    else:
        value = end_value
    return value


def create_deferral_threshold_schedule(
    total_step: int,
    warmup_schedule: str,
    warmup_steps: int,
    start_value: float,
    end_value: float,
):
    if warmup_schedule is None:
        warmup_schedule = "constant"

    if warmup_schedule == "exponential":
        warmup_func = _exponential_warmup
    elif warmup_schedule == "cosine":
        warmup_func = _cosine_warmup
    elif warmup_schedule == "linear":
        warmup_func = _linear_warmup
    elif warmup_schedule == "constant":
        warmup_func = lambda step, warmup_steps, start_value, end_value: end_value
    else:
        raise ValueError(f"Unknown warmup schedule: {warmup_schedule}")

    return torch.Tensor([warmup_func(step, warmup_steps, start_value, end_value) for step in range(total_step)])


def generate_response(
    prompt: Tuple[int, str],
    tokenizer_base: transformers.PreTrainedTokenizer,
    tokenizer_ref: transformers.PreTrainedTokenizer,
    max_tokens: int,
    def_threshold: float,
    tokens_per_call: int,
    deferral_strategy: str,
    get_next_prediction_base: Callable,
    get_next_prediction_ref: Callable,
    stop_sequences=None,
    verbose=False,
    debug=False,
    threshold_warmup_schedule: Optional[str] = None,
    threshold_warmup_steps: int = 15,
):
    if stop_sequences is not None:
        if isinstance(stop_sequences[0], str):
            stop_sequences = [tokenizer_base.encode(" " + seq, add_special_tokens=False)[1:] for seq in stop_sequences]

    try:
        idx, prompt = prompt
        prompt_token_ids = tokenizer_base(prompt)["input_ids"]
        vocab_size_ref = len(tokenizer_ref)
        eos_token_id = tokenizer_base.eos_token_id

        if stop_sequences is None:
            stop_sequences = []

        deferral_thresholds = create_deferral_threshold_schedule(
            total_step=max_tokens,
            warmup_schedule=threshold_warmup_schedule,
            warmup_steps=threshold_warmup_steps,
            start_value=1.0,  # not deferring at the beginning
            end_value=def_threshold,
        )
        # print(deferral_thresholds)

        total_generated = 0  # Counter for the total generated tokens
        generated = []
        deferral_for_tokens = []
        step_count = 0
        eos_reached = False

        while not eos_reached and total_generated < max_tokens:
            output_base, log_probs_base = get_next_prediction_base(prompt_token_ids, tokens_per_call)

            # generated_token_ids = output_base.outputs[0].token_ids
            generated_token_ids = log_probs_base[:, :vocab_size_ref].argmax(dim=-1).tolist()
            # Make sure not generating tokens outside of the base vocab

            # Firstly we handle the deferral logic here.
            # We assume the deferral position is the last token in the sequence
            deferral_probs = torch.sigmoid(log_probs_base[:, -1])

            # We find the deferral thresholds for the current steps
            cur_deferral_thresholds = deferral_thresholds[total_generated : total_generated + len(deferral_probs)]
            # print(total_generated, cur_deferral_thresholds)
            is_deferral = deferral_probs > cur_deferral_thresholds
            # if False:
            if torch.any(is_deferral):
                deferred_positions = torch.where(is_deferral.squeeze())[0]
                first_deferral_position = deferred_positions[0]
                # deferral_token = deferral_search(
                #     generated_token_ids,
                #     first_deferral_position,
                #     prompt_token_ids,
                #     beam_size=1,
                #     search_length=10,
                #     use_batch_scoring=True,
                # )
                output_ref, log_probs_ref = get_next_prediction_ref(
                    prompt_token_ids + generated_token_ids[:first_deferral_position], 1
                )

                if debug:
                    top_5_tokens = log_probs_ref[0].topk(5)
                    for _token_idx in range(len(top_5_tokens.indices)):
                        print(
                            tokenizer_base.decode(top_5_tokens.indices[_token_idx]),
                            top_5_tokens.values[_token_idx],
                        )

                if deferral_strategy == "defer":
                    deferral_token = output_ref.outputs[0].token_ids[0]

                elif deferral_strategy == "compose":
                    def_prob = deferral_probs[first_deferral_position]
                    base_probs = log_probs_base[first_deferral_position, :vocab_size_ref].exp()
                    ref_probs = log_probs_ref[0, :vocab_size_ref].exp()
                    combined_probs = (1 - def_prob) * base_probs + def_prob * ref_probs
                    deferral_token = combined_probs.argmax().item()
                    if debug:
                        top_5_tokens = combined_probs.topk(5)
                        for _token_idx in range(len(top_5_tokens.indices)):
                            print(
                                tokenizer_base.decode(top_5_tokens.indices[_token_idx]),
                                top_5_tokens.values[_token_idx],
                            )

                generated_token_ids = generated_token_ids[:first_deferral_position] + [deferral_token]

            # The total generated tokens here is used for checking the max_tokens condition
            total_generated += len(generated_token_ids)

            # Check for potential early stopping
            stop_seq_reached, final_sequences = find_stop_sequence(
                generated + generated_token_ids, stop_sequences, tokens_per_call
            )
            if stop_seq_reached:
                assert len(final_sequences) > 0
                generated.extend(final_sequences)
                deferral_for_tokens.append(deferral_probs[: len(final_sequences)])
                eos_reached = True
            elif eos_token_id in generated_token_ids or total_generated >= max_tokens:
                eos_index = (
                    generated_token_ids.index(eos_token_id)
                    if eos_token_id in generated_token_ids
                    else len(generated_token_ids)
                )
                generated.extend(generated_token_ids[:eos_index])
                deferral_for_tokens.append(deferral_probs[:eos_index])
                eos_reached = True
            else:
                generated.extend(generated_token_ids)
                deferral_for_tokens.append(deferral_probs[: len(generated_token_ids)])
                prompt_token_ids = prompt_token_ids + generated_token_ids

            step_count += 1
            if debug:
                print(f"step {step_count}")
                # TODO: fix this
                visualize_deferral_generation(
                    generated,
                    torch.cat(deferral_for_tokens) > cur_deferral_thresholds,
                    tokenizer_base,
                )

            if eos_reached:
                break  # Break from while not eos_reached
        deferral_for_tokens = torch.cat(deferral_for_tokens)
        deferral_threshold_for_tokens = deferral_thresholds[: len(deferral_for_tokens)]
        if verbose:
            print(f"Generation for {idx}, Deferral Threshold: {def_threshold}")
            visualize_deferral_generation(
                generated,
                deferral_for_tokens > deferral_threshold_for_tokens,
                tokenizer_base,
            )

        return {
            "idx": idx,
            "generated_tokens": generated,
            "generated_text": tokenizer_base.decode(generated, skip_special_tokens=True),
            "deferral_prob": deferral_for_tokens.tolist(),
            "def_threshold": def_threshold,
            "actual_deferral_threshold": deferral_threshold_for_tokens.tolist(),
        }
    except Exception as e:
        print(f"Error for {idx}")
        error_msg = traceback.format_exc()
        print(error_msg)
        with open("error.txt", "a") as f:
            f.write(f"{idx}\n")
            f.write(f"{error_msg}\n")
            f.write(f"=========\n")

        return {
            "idx": idx,
            "generated_tokens": None,
            "generated_text": None,
            "deferral_prob": None,
            "def_threshold": def_threshold,
        }


def pick_deferral_threshold(
    dataset,
    save_path,
    base_model_port,
    ref_model_port,
    batch_gen_port=8003,
    n_samples_for_search=None,
    num_proc=1,
    max_tokens=512,
    tokenizer_name_ref=None,
    tokenizer_name_base=None,
    tokens_per_call=10,
    deferral_strategy="defer",
    threshold_warmup_schedule: Optional[str] = None,
    threshold_warmup_steps: int = 15,
    debug=False,
    skip_generation=False,
):
    logger.info(f"Searching for optimal deferral threshold for {dataset}")

    if debug:
        logger.setLevel(logging.DEBUG)

    check_url(base_model_port)
    check_url(ref_model_port)

    save_path = Path(save_path + "_deferral_search")
    save_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving to {save_path}")

    ds = datasets.load_from_disk(dataset)
    logger.debug(f"Loaded dataset {dataset}")

    tokenizer_base = transformers.AutoTokenizer.from_pretrained(tokenizer_name_base, use_fast=False)
    tokenizer_ref = transformers.AutoTokenizer.from_pretrained(tokenizer_name_ref, use_fast=False)

    base_model_url = f"http://localhost:{base_model_port}/generate"
    get_next_prediction_base = functools.partial(
        get_next_prediction, url=base_model_url, vocab_size=len(tokenizer_base)
    )

    ref_model_url = f"http://localhost:{ref_model_port}/generate"
    get_next_prediction_ref = functools.partial(get_next_prediction, url=ref_model_url, vocab_size=len(tokenizer_ref))

    batch_gen_url = f"http://localhost:{batch_gen_port}/batch_generate"
    get_next_prediction_batch_ref = functools.partial(
        get_next_prediction_batch, forward_url=ref_model_url, url=batch_gen_url
    )

    if n_samples_for_search is None:
        n_samples_for_search = len(ds)

    if not (save_path / "deferral_threshold.csv").exists():
        logger.debug("Searching for optimal deferral threshold")
        optimal_deferral_threshold = find_optimal_deferral_threshold(
            save_path,
            ds,
            n_samples_for_search,
            tokenizer_base,
            get_next_prediction_base,
            get_next_prediction_batch_ref,
        )
        optimal_deferral_threshold = math.floor(optimal_deferral_threshold * 100) / 100

        all_thresholds = calibrate_deferral_probability(n_samples_for_search, save_path)
        all_thresholds = {f"{key:.2f}": value for key, value in all_thresholds}
        all_thresholds["optimal"] = optimal_deferral_threshold
        pd.DataFrame(all_thresholds.items(), columns=["portion", "deferral_threshold"]).to_csv(
            save_path / "deferral_threshold.csv", index=False
        )
    else:
        logger.debug("Loading deferral thresholds")
        all_thresholds = calibrate_deferral_probability(n_samples_for_search, save_path)
        all_thresholds = {f"{key:.2f}": value for key, value in all_thresholds}
        optimal_deferral_threshold = pd.read_csv(save_path / "deferral_threshold.csv").iloc[-1, -1]
        print(optimal_deferral_threshold)
        all_thresholds["optimal"] = optimal_deferral_threshold

    if skip_generation:
        return

    for idx, (portion, deferral_threshold) in enumerate(all_thresholds.items()):
        if portion == "1.00" or portion == "optimal":
            continue

        # if portion == "0.10":
        #     continue

        # if portion == "0.00":
        #     continue

        # if idx % 3 != 0 and portion != "0.10":
        #     continue

        generation_path = save_path / f"defer-{portion}" / "generations.json"
        if generation_path.exists():
            logger.info(f"Skipping {generation_path}")
            continue

        generate_deferral(
            dataset,
            save_path / f"defer-{portion}",
            base_model_port,
            ref_model_port,
            batch_gen_port,
            num_proc=num_proc,
            search_deferral_threshold=False,
            max_tokens=max_tokens,
            tokenizer_name_ref=tokenizer_name_ref,
            tokenizer_name_base=tokenizer_name_base,
            tokens_per_call=tokens_per_call if deferral_threshold < 1 else max_tokens,
            deferral_threshold=deferral_threshold,
            deferral_strategy=deferral_strategy,
            threshold_warmup_schedule=threshold_warmup_schedule,
            threshold_warmup_steps=threshold_warmup_steps,
            debug=False,
        )


def generate_deferral(
    dataset,
    save_path,
    base_model_port,
    ref_model_port,
    batch_gen_port=8003,
    num_proc=1,
    search_deferral_threshold=False,
    max_tokens=512,
    tokenizer_name_ref=None,
    tokenizer_name_base=None,
    tokens_per_call=10,
    deferral_threshold=None,
    deferral_strategy="defer",
    threshold_warmup_schedule: Optional[str] = None,
    threshold_warmup_steps: int = 15,
    debug=False,
):
    logger.info(f"Run deferral generation for {dataset}")

    if debug:
        logger.setLevel(logging.DEBUG)

    check_url(base_model_port)
    check_url(ref_model_port)

    save_path = Path(save_path)

    if deferral_threshold is None:
        if save_path / "_deferral_search" / "eval_with_deferral_threshold.csv":
            deferral_threshold = pd.read_csv(save_path / "_deferral_search" / "eval_with_deferral_threshold.csv").iloc[
                0
            ]["deferral_threshold"]
            logger.warn(f"Using deferral threshold {deferral_threshold}")
        elif save_path / "_deferral_search" / "deferral_threshold.csv":
            deferral_threshold = pd.read_csv(save_path / "_deferral_search" / "deferral_threshold.csv").iloc[
                -1, -1
            ]  # TODO
            logger.warn(f"Using deferral threshold {deferral_threshold}")
        else:
            deferral_threshold = 0.5
            logger.warn(f"Using deferral threshold {deferral_threshold}")
        save_path = save_path / f"deferral-{deferral_threshold}"

    save_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving to {save_path}")

    ds = datasets.load_from_disk(dataset)
    logger.debug(f"Loaded dataset {dataset}")

    tokenizer_ref = transformers.AutoTokenizer.from_pretrained(tokenizer_name_ref, use_fast=False)
    tokenizer_base = transformers.AutoTokenizer.from_pretrained(tokenizer_name_base, use_fast=False)

    base_model_url = f"http://localhost:{base_model_port}/generate"
    get_next_prediction_base = functools.partial(
        get_next_prediction, url=base_model_url, vocab_size=len(tokenizer_base)
    )

    ref_model_url = f"http://localhost:{ref_model_port}/generate"
    get_next_prediction_ref = functools.partial(get_next_prediction, url=ref_model_url, vocab_size=len(tokenizer_ref))

    batch_gen_url = f"http://localhost:{batch_gen_port}/batch_generate"
    get_next_prediction_batch_ref = functools.partial(
        get_next_prediction_batch, forward_url=ref_model_url, url=batch_gen_url
    )

    if search_deferral_threshold:
        logger.debug("Searching for optimal deferral threshold")
        optimal_deferral_threshold = find_optimal_deferral_threshold(
            save_path,
            ds,
            10,
            tokenizer_base,
            get_next_prediction_base,
            get_next_prediction_batch_ref,
        )
        optimal_deferral_threshold = math.floor(optimal_deferral_threshold * 100) / 100
        logger.debug(f"Found optimal deferral threshold: {optimal_deferral_threshold}")
    else:
        optimal_deferral_threshold = deferral_threshold
        logger.debug(f"Using deferral threshold {optimal_deferral_threshold}")

    batch_size = 1
    prompts = [(i, ds[i : i + batch_size][MODEL_INPUT_COL_NAME][0]) for i in range(0, len(ds), batch_size)]

    partial_generate_response = functools.partial(
        generate_response,
        get_next_prediction_base=get_next_prediction_base,
        get_next_prediction_ref=get_next_prediction_ref,
        tokenizer_base=tokenizer_base,
        tokenizer_ref=tokenizer_ref,
        max_tokens=max_tokens,
        def_threshold=optimal_deferral_threshold,
        tokens_per_call=tokens_per_call,
        deferral_strategy=deferral_strategy,
        verbose=True,
        stop_sequences=None,
        threshold_warmup_schedule=threshold_warmup_schedule,
        threshold_warmup_steps=threshold_warmup_steps,
    )

    if debug:
        prompts = prompts[:3]
        ds = ds.select(range(3))

    logger.debug(f"Generating responses for {len(prompts)} prompts")
    if num_proc == 1:
        all_responses = []
        for prompt in tqdm(prompts, desc="generate_response"):
            all_responses.append(partial_generate_response(prompt))
    else:
        with multiprocessing.Pool(num_proc) as p:
            all_responses = list(
                tqdm(
                    p.imap(partial_generate_response, prompts),
                    desc="generate_response",
                    total=len(prompts),
                )
            )

    with open(save_path / "generations.json", "w") as f:
        json.dump(all_responses, f)

    pred = datasets.Dataset.from_list(all_responses)
    ds.add_column("generated_tokens", pred["generated_tokens"]).save_to_disk(f"{save_path}/dataset")


# We have to move it outside to make it picklable
def _generate_response_with_per_instance_threshold(
    model_input: Tuple[int, float, str],
    tokenizer_base,
    tokenizer_ref,
    max_tokens,
    tokens_per_call,
    deferral_strategy,
    threshold_warmup_schedule,
    threshold_warmup_steps,
):
    index, optimal_deferral_threshold, prompt = model_input
    return generate_response(
        prompt=(index, prompt),
        tokenizer_base=tokenizer_base,
        tokenizer_ref=tokenizer_ref,
        max_tokens=max_tokens,
        def_threshold=optimal_deferral_threshold,
        tokens_per_call=tokens_per_call,
        deferral_strategy=deferral_strategy,
        verbose=True,
        stop_sequences=None,
        threshold_warmup_schedule=threshold_warmup_schedule,
        threshold_warmup_steps=threshold_warmup_steps,
    )


def generate_response_deferral_adaptative_threshold(
    dataset,
    save_path,
    threshold_file,
    num_proc=1,
    max_tokens=512,
    tokenizer_name_ref=None,
    tokenizer_name_base=None,
    tokens_per_call=10,
    deferral_strategy="defer",
    threshold_warmup_schedule: Optional[str] = None,
    threshold_warmup_steps: int = 15,
    debug=False,
):
    logger.info(f"Run deferral generation for {dataset}")

    if debug:
        logger.setLevel(logging.DEBUG)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving to {save_path}")

    ds = datasets.load_from_disk(dataset)
    logger.debug(f"Loaded dataset {dataset}")

    tokenizer_ref = transformers.AutoTokenizer.from_pretrained(tokenizer_name_ref, use_fast=False)
    tokenizer_base = transformers.AutoTokenizer.from_pretrained(tokenizer_name_base, use_fast=False)

    threshold_per_instance = pd.read_csv(threshold_file, index_col=0)["threshold"]

    batch_size = 1
    composite_prompts = [
        (i, threshold_per_instance[i], ds[i : i + batch_size][MODEL_INPUT_COL_NAME][0])
        for i in range(0, len(ds), batch_size)
    ]

    generate_response_with_per_instance_threshold_partial = functools.partial(
        _generate_response_with_per_instance_threshold,
        tokenizer_base=tokenizer_base,
        tokenizer_ref=tokenizer_ref,
        max_tokens=max_tokens,
        tokens_per_call=tokens_per_call,
        deferral_strategy=deferral_strategy,
        threshold_warmup_schedule=threshold_warmup_schedule,
        threshold_warmup_steps=threshold_warmup_steps,
    )

    if debug:
        composite_prompts = composite_prompts[:3]
        ds = ds.select(range(3))

    logger.debug(f"Generating responses for {len(composite_prompts)} prompts")
    if num_proc == 1:
        all_responses = []
        for prompt in tqdm(composite_prompts, desc="generate_response"):
            all_responses.append(generate_response_with_per_instance_threshold_partial(prompt))
    else:
        with multiprocessing.Pool(num_proc) as p:
            all_responses = list(
                tqdm(
                    p.imap(
                        generate_response_with_per_instance_threshold_partial,
                        composite_prompts,
                    ),
                    desc="generate_response",
                    total=len(composite_prompts),
                )
            )

    with open(save_path / "generations.json", "w") as f:
        json.dump(all_responses, f)

    pred = datasets.Dataset.from_list(all_responses)
    ds.add_column("generated_tokens", pred["generated_tokens"]).save_to_disk(f"{save_path}/dataset")


def generate_multi_turn(
    dataset,
    save_path,
    num_proc=1,
    max_tokens=1024,
    tokenizer_name_ref=None,
    tokenizer_name_base=None,
    tokens_per_call=10,
    deferral_threshold=0.25,
    deferral_strategy="defer",
    no_deferral=False,
    model_port=None,
    debug=False,
):
    logger.info(f"Run deferral generation for {dataset}")

    if debug:
        logger.setLevel(logging.DEBUG)
    if no_deferral:
        assert model_port is not None

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving to {save_path}")

    # Right now we only do this for MT-bench, so it's hardcoded
    ds = datasets.load_dataset("json", data_files=dataset)["train"]
    logger.debug(f"Loaded dataset {dataset}")
    if debug:
        ds = ds.select(range(3))

    if tokenizer_name_ref:
        tokenizer_ref = transformers.AutoTokenizer.from_pretrained(tokenizer_name_ref, use_fast=False)
    if tokenizer_name_base:
        tokenizer_base = transformers.AutoTokenizer.from_pretrained(tokenizer_name_base, use_fast=False)

    flatten_questions = []
    all_responses = []

    for question in tqdm(ds, "Generate responses"):
        current_message = []
        generated_responses = []
        for turn_id in range(len(question["turns"])):
            current_message.append({"role": "user", "content": question["turns"][turn_id].strip()})
            prompt = create_prompt_with_tulu_chat_format(current_message)

            if no_deferral:
                model_url = f"http://localhost:{model_port}/generate"
                get_next_prediction_api = functools.partial(
                    get_next_prediction, url=model_url, vocab_size=len(tokenizer_base)
                )
                response = generate_response_default(
                    prompt=(f"{question['question_id']}-{turn_id}", prompt),
                    tokenizer_base=tokenizer_base,
                    get_next_prediction_api=get_next_prediction_api,
                    max_tokens=max_tokens,
                    stop_sequences=None,
                    verbose=True,
                )
            else:
                response = generate_response(
                    prompt=(f"{question['question_id']}-{turn_id}", prompt),
                    tokenizer_base=tokenizer_base,
                    tokenizer_ref=tokenizer_ref,
                    max_tokens=max_tokens,
                    def_threshold=deferral_threshold,
                    tokens_per_call=tokens_per_call,
                    deferral_strategy=deferral_strategy,
                    verbose=True,
                    stop_sequences=None,
                )
            all_responses.append(response)

            generated_responses.append(response["generated_text"].strip())
            current_message.append({"role": "assistant", "content": response["generated_text"].strip()})

            flatten_questions.append(
                {
                    "question_id": question["question_id"],
                    "turn_id": turn_id,
                    "category": question["category"],
                    "turn_text": question["turns"][turn_id].strip(),
                    "generated_tokens": response["generated_tokens"],
                }
            )
        # save mt-generations.json
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": "deferral-model",
            "choices": [{"index": 0, "turns": generated_responses}],
            "tstamp": time.time(),
        }
        with open(save_path / "mt-generations.json", "a") as f:
            f.write(json.dumps(ans_json) + "\n")

    with open(save_path / "generations.json", "w") as f:
        json.dump(all_responses, f)

    datasets.Dataset.from_list(flatten_questions).save_to_disk(f"{save_path}/dataset")


def generate_response_default(
    prompt,
    tokenizer_base,
    get_next_prediction_api,
    max_tokens: int,
    stop_sequences=None,
    verbose=False,
):
    try:
        idx, prompt = prompt

        prompt_token_ids = tokenizer_base(prompt)["input_ids"]
        output_base = get_next_prediction_api(
            prompt_token_ids,
            max_tokens=max_tokens,
            return_logprobs_tensor=False,
            stop=stop_sequences,
        )
        if verbose:
            print(f"Generation for {idx}")
            print(output_base.outputs[0].text)
        return {
            "idx": idx,
            "generated_tokens": output_base.outputs[0].token_ids,
            "generated_text": output_base.outputs[0].text,
        }
    except Exception as e:
        print(f"Error for {idx}")
        error_msg = traceback.format_exc()
        print(error_msg)
        with open("error.txt", "a") as f:
            f.write(f"{idx}\n")
            f.write(f"{error_msg}\n")
            f.write(f"=========\n")

        return {
            "idx": idx,
            "generated_tokens": None,
            "generated_text": None,
        }


def generate_default(
    dataset,
    save_path,
    model_port,
    num_proc=1,
    max_tokens=512,
    stop_sequences: Optional[List[str]] = None,
    tokenizer_name_base=None,
    debug=False,
):
    check_url(model_port)

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving to {save_path}")

    ds = datasets.load_from_disk(dataset)
    logger.debug(f"Loaded dataset {dataset}")

    tokenizer_base = transformers.AutoTokenizer.from_pretrained(tokenizer_name_base, use_fast=False)

    batch_size = 1
    prompts = [(i, ds[i : i + batch_size][MODEL_INPUT_COL_NAME][0]) for i in range(0, len(ds), batch_size)]

    model_url = f"http://localhost:{model_port}/generate"
    get_next_prediction_api = functools.partial(get_next_prediction, url=model_url, vocab_size=len(tokenizer_base))

    partial_generate_response_default = functools.partial(
        generate_response_default,
        tokenizer_base=tokenizer_base,
        get_next_prediction_api=get_next_prediction_api,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        verbose=True,
    )

    if debug:
        prompts = prompts[:3]

    logger.debug(f"Generating responses for {len(prompts)} prompts")
    if num_proc == 1:
        all_responses = []
        for prompt in tqdm(prompts, desc="generate_response"):
            all_responses.append(partial_generate_response_default(prompt))
    else:
        with multiprocessing.Pool(num_proc) as p:
            all_responses = list(
                tqdm(
                    p.imap(partial_generate_response_default, prompts),
                    desc="generate_response",
                    total=len(prompts),
                )
            )

    with open(save_path / "generations.json", "w") as f:
        json.dump(all_responses, f)


def generate_composite_response(
    prompt: Tuple[int, str],
    tokenizer_base: transformers.PreTrainedTokenizer,
    tokenizer_ref: transformers.PreTrainedTokenizer,
    get_next_prediction_base: Callable,
    get_next_prediction_ref: Callable,
    max_tokens: int,
    deferral_strategy: str,
    get_next_prediction_extra: Optional[Callable] = None,
    stop_sequences=None,
    verbose=False,
    debug=False,
    **kwargs,
):
    tokens_per_call = 1

    if stop_sequences is not None:
        if isinstance(stop_sequences[0], str):
            stop_sequences = [tokenizer_base.encode(" " + seq, add_special_tokens=False)[1:] for seq in stop_sequences]

    assert deferral_strategy in ["random", "greedy", "cd", "cd_v2", "proxy"]

    try:
        idx, prompt = prompt
        prompt_token_ids = tokenizer_base(prompt)["input_ids"]
        vocab_size_ref = len(tokenizer_ref)
        eos_token_id = tokenizer_base.eos_token_id

        if stop_sequences is None:
            stop_sequences = []

        total_generated = 0  # Counter for the total generated tokens
        generated = []
        deferral_for_tokens = []
        step_count = 0
        eos_reached = False

        while not eos_reached and total_generated < max_tokens:
            output_base, log_probs_base = get_next_prediction_base(prompt_token_ids, tokens_per_call)

            # generated_token_ids = output_base.outputs[0].token_ids
            generated_token_ids = log_probs_base[:, :vocab_size_ref].argmax(dim=-1).tolist()
            # Make sure not generating tokens outside of the base vocab

            # Firstly we handle the deferral logic here.
            if deferral_strategy == "random":
                is_deferral = deferral_probs = torch.bernoulli(torch.full((len(generated_token_ids),), 0.5)).bool()
            elif deferral_strategy in ["greedy", "cd", "cd_v2", "proxy"]:
                is_deferral = deferral_probs = torch.tensor([True] * len(generated_token_ids))

            assert len(is_deferral) == tokens_per_call

            # if False:
            if torch.any(is_deferral):
                deferred_positions = torch.where(is_deferral.squeeze())[0]
                first_deferral_position = deferred_positions[0]

                assert (
                    first_deferral_position == 0
                )  # Apparently it should be the 0-th token since tokens_per_call is 1

                output_ref, log_probs_ref = get_next_prediction_ref(
                    prompt_token_ids + generated_token_ids[:first_deferral_position], 1
                )

                if deferral_strategy == "random":
                    deferral_token = output_ref.outputs[0].token_ids[0]
                    if debug:
                        top_5_tokens = log_probs_ref[0].topk(5)
                        for _token_idx in range(len(top_5_tokens.indices)):
                            print(
                                tokenizer_base.decode(top_5_tokens.indices[_token_idx]),
                                top_5_tokens.values[_token_idx],
                            )

                elif deferral_strategy == "greedy":
                    base_token_prob = log_probs_base[:, :vocab_size_ref].softmax(dim=-1)
                    ref_token_prob = log_probs_ref[:, :vocab_size_ref].softmax(dim=-1)

                    assert base_token_prob.shape[0] == tokens_per_call

                    if debug:
                        print(
                            f"base_prob_top_5_ids_and_values: {base_token_prob[0].topk(5).indices} {base_token_prob[0].topk(5).values}"
                        )
                        print(
                            f"ref_prob_top_5_ids_and_values: {ref_token_prob[0].topk(5).indices} {ref_token_prob[0].topk(5).values}"
                        )

                    base_prob_max = base_token_prob[0].max(dim=-1).values
                    ref_prob_max = ref_token_prob[0].max(dim=-1).values

                    if base_prob_max > ref_prob_max:
                        deferral_token = generated_token_ids[first_deferral_position]
                        deferral_probs[0] = False
                    else:
                        deferral_token = output_ref.outputs[0].token_ids[0]

                    if debug:
                        print(f"deferral_token_id: {deferral_token}")
                        print(f"deferral_token: {tokenizer_base.decode(deferral_token)}")

                elif deferral_strategy == "cd":
                    # Conversion between the terminologies
                    # Our Name        ===> Contrastive Decoding Name
                    # log_probs_base  ===> Amateur Model
                    # log_probs_ref   ===> Expert Model

                    # cut_off = alpha*ref_token_prob.max(dim=-1, keepdim=True).values
                    diffs = log_probs_ref[:, :vocab_size_ref] - log_probs_base[:, :vocab_size_ref]
                    # cd_logits = diffs.masked_fill(ref_token_prob < cut_off, -float('inf'))

                    # Since we do not do sampling, we do not need to worry about the cutoff
                    deferral_token = diffs.argmax(dim=-1)[0].item()

                elif deferral_strategy == "cd_v2":
                    # Conversion between the terminologies
                    # Our Name        ===> Contrastive Decoding Name
                    # log_probs_base  ===> Amateur Model
                    # log_probs_ref   ===> Expert Model
                    #
                    # Also in this case, the log_probs_ref and log_probs_base are
                    # unnormalized logits from LMs (we guarantee from the caller)

                    alpha = kwargs.get("alpha", 0.1)  # 0.1 is the default value used in the paper
                    cutoff = torch.log(torch.tensor(alpha)) + log_probs_ref.max(dim=-1, keepdim=True).values

                    beta = kwargs.get("beta", 0.5)  # 0.5 is the default value used in the paper
                    diffs = (1 + beta) * log_probs_ref[:, :vocab_size_ref] - beta * log_probs_base[:, :vocab_size_ref]

                    diffs = diffs.masked_fill(diffs < cutoff, -float("inf"))
                    deferral_token = diffs.argmax(dim=-1)[0].item()

                    if debug:
                        print(cutoff)
                        print(
                            f"base_prob_top_5_ids_and_values: {log_probs_base[0].topk(5).indices} {log_probs_base[0].topk(5).values}"
                        )
                        print(
                            f"ref_prob_top_5_ids_and_values: {log_probs_ref[0].topk(5).indices} {log_probs_ref[0].topk(5).values}"
                        )
                        print(f"cd_top_5_ids_and_values: {diffs[0].topk(5).indices} {diffs[0].topk(5).values}")
                        print(deferral_token)

                elif deferral_strategy == "proxy":

                    # Conversion between the terminologies
                    # Our Name        ===> Proxy Tuning Name
                    # log_probs_base  ===> a untuned small model M-
                    # log_probs_extra ===> a finetuned small model M+
                    # log_probs_ref   ===> a large pretrained model M
                    #
                    # Also in this case, the log_probs_ref and log_probs_base are
                    # unnormalized logits from LMs (we guarantee from the caller)

                    output_extra, log_probs_extra = get_next_prediction_extra(
                        prompt_token_ids + generated_token_ids[:first_deferral_position],
                        1,
                    )

                    combined = (
                        log_probs_ref[:, :vocab_size_ref]
                        + log_probs_extra[:, :vocab_size_ref]
                        - log_probs_base[:, :vocab_size_ref]
                    )
                    # Since we do not do sampling for now, we do not need softmax

                    deferral_token = combined.argmax(dim=-1)[0].item()

                    if debug:
                        print(
                            f"base_prob_top_5_ids_and_values: {log_probs_base[0].topk(5).indices} {log_probs_base[0].topk(5).values}"
                        )
                        print(
                            f"ref_prob_top_5_ids_and_values: {log_probs_ref[0].topk(5).indices} {log_probs_ref[0].topk(5).values}"
                        )
                        print(
                            f"extra_prob_top_5_ids_and_values: {log_probs_extra[0].topk(5).indices} {log_probs_extra[0].topk(5).values}"
                        )
                        print(
                            f"proxy_top_5_ids_and_values: {combined[0].topk(5).indices} {combined[0].topk(5).values}"
                        )
                        print(deferral_token)

                generated_token_ids = generated_token_ids[:first_deferral_position] + [deferral_token]

            # The total generated tokens here is used for checking the max_tokens condition
            total_generated += len(generated_token_ids)

            # Check for potential early stopping
            stop_seq_reached, final_sequences = find_stop_sequence(
                generated + generated_token_ids, stop_sequences, tokens_per_call
            )
            if stop_seq_reached:
                assert len(final_sequences) > 0
                generated.extend(final_sequences)
                deferral_for_tokens.append(deferral_probs[: len(final_sequences)])
                eos_reached = True
            elif eos_token_id in generated_token_ids or total_generated >= max_tokens:
                eos_index = (
                    generated_token_ids.index(eos_token_id)
                    if eos_token_id in generated_token_ids
                    else len(generated_token_ids)
                )
                generated.extend(generated_token_ids[:eos_index])
                deferral_for_tokens.append(deferral_probs[:eos_index])
                eos_reached = True
            else:
                generated.extend(generated_token_ids)
                deferral_for_tokens.append(deferral_probs[: len(generated_token_ids)])
                prompt_token_ids = prompt_token_ids + generated_token_ids

            if stop_pattern := kwargs.get("stop_pattern", None):
                if is_contain_stop_pattern(stop_pattern, generated, tokenizer_base):
                    eos_reached = True

            step_count += 1
            if debug:
                print(f"step {step_count}")
                # TODO: fix this
                visualize_deferral_generation(generated, torch.cat(deferral_for_tokens), tokenizer_base)

            if eos_reached:
                break  # Break from while not eos_reached
        deferral_for_tokens = torch.cat(deferral_for_tokens)
        if verbose:
            print(f"Generation for {idx}")
            visualize_deferral_generation(
                generated,
                deferral_for_tokens,
                tokenizer_base,
            )

        return {
            "idx": idx,
            "generated_tokens": generated,
            "generated_text": tokenizer_base.decode(generated, skip_special_tokens=True),
            "deferral_prob": deferral_for_tokens.tolist(),
        }
    except Exception as e:
        print(f"Error for {idx}")
        error_msg = traceback.format_exc()
        print(error_msg)
        with open("error.txt", "a") as f:
            f.write(f"{idx}\n")
            f.write(f"{error_msg}\n")
            f.write(f"=========\n")

        return {
            "idx": idx,
            "generated_tokens": None,
            "generated_text": None,
            "deferral_prob": None,
        }


def generate_ablation(
    dataset,
    save_path,
    base_model_port,
    ref_model_port,
    tokenizer_name_base,
    tokenizer_name_ref,
    deferral_strategy,
    extra_model_port=None,  # Only used for proxy tuning
    tokenizer_name_extra=None,  # Only used for proxy tuning
    num_proc=1,
    max_tokens=512,
    stop_sequences: Optional[List[str]] = None,
    debug=False,
    **kwargs,
):
    """Quick Reference for running baselines:

    In Contrastive Decoding
    # Conversion between the terminologies
    # Our Name        ===> Contrastive Decoding Name
    # log_probs_base  ===> Amateur Model
    # log_probs_ref   ===> Expert Model
    # e.g.,
    # base: Llama 7B  (8103)
    # ref:  Llama 70B (8102)

    In Proxy Tuning
    # Conversion between the terminologies
    # Our Name ===> Proxy Tuning Name
    # log_probs_base  ===> a untuned small model M-
    # log_probs_extra ===> a finetuned small model M+
    # log_probs_ref   ===> a large pretrained model M
    # e.g.,
    # base:  Llama 7B             (8103)
    # extra: Llama 7B (finetuned) (8104)
    # ref:   Llama 70B            (8102)
    """

    check_url(base_model_port)
    check_url(ref_model_port)
    if extra_model_port:
        check_url(extra_model_port)

    assert deferral_strategy in ["random", "greedy", "cd", "cd_v2", "proxy"]

    logger.info(f"Setting stop sequences to {stop_sequences}")
    logger.info("Extra Args")
    for key, val in kwargs.items():
        logger.info(f"[{key}]: {val}")

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving to {save_path}")

    ds = datasets.load_from_disk(dataset)
    logger.debug(f"Loaded dataset {dataset}")

    tokenizer_base = transformers.AutoTokenizer.from_pretrained(tokenizer_name_base, use_fast=False)
    tokenizer_ref = transformers.AutoTokenizer.from_pretrained(tokenizer_name_ref, use_fast=False)

    batch_size = 1
    prompts = [(i, ds[i : i + batch_size][MODEL_INPUT_COL_NAME][0]) for i in range(0, len(ds), batch_size)]

    base_model_url = f"http://localhost:{base_model_port}/generate"
    get_next_prediction_base = functools.partial(
        get_next_prediction, url=base_model_url, vocab_size=len(tokenizer_base)
    )

    ref_model_url = f"http://localhost:{ref_model_port}/generate"
    get_next_prediction_ref = functools.partial(get_next_prediction, url=ref_model_url, vocab_size=len(tokenizer_ref))

    if deferral_strategy == "proxy":
        assert extra_model_port is not None
        assert tokenizer_name_extra is not None

        extra_model_url = f"http://localhost:{extra_model_port}/generate"
        tokenizer_extra = transformers.AutoTokenizer.from_pretrained(tokenizer_name_extra, use_fast=False)
        get_next_prediction_extra = functools.partial(
            get_next_prediction, url=extra_model_url, vocab_size=len(tokenizer_extra)
        )
    else:
        get_next_prediction_extra = None

    if deferral_strategy in ["cd_v2", "proxy"]:
        # We need to make sure the used backend is not returning log_probs but un-normalized logits
        for idx, _test_prompt in prompts[:15]:
            test_prompt_token_ids = tokenizer_base(_test_prompt)["input_ids"]
            print("idx", idx)
            print("test_prompt_token_ids", test_prompt_token_ids[-5:])

            _, logits = get_next_prediction_base(test_prompt_token_ids, max_tokens=1)
            print("base_logits", logits)
            logits_sum = logits.exp().sum(dim=-1)
            print(logits_sum)
            if torch.isclose(logits_sum, torch.tensor([1.0])).all():
                warnings.warn(
                    "The base backend is not returning un-normalized logits. This will cause issues with CDv2 and Proxy Tuning."
                )

            _, logits = get_next_prediction_ref(test_prompt_token_ids, max_tokens=1)
            print("ref_logits", logits)
            logits_sum = logits.exp().sum(dim=-1)
            print(logits_sum)
            if torch.isclose(logits_sum, torch.tensor([1.0])).all():
                warnings.warn(
                    "The ref backend is not returning un-normalized logits. This will cause issues with CDv2 and Proxy Tuning."
                )

            if deferral_strategy == "proxy":
                _, logits = get_next_prediction_extra(test_prompt_token_ids, max_tokens=1)
                print("extra_logits", logits)
                logits_sum = logits.exp().sum(dim=-1)
                print(logits_sum)
                if torch.isclose(logits_sum, torch.tensor([1.0])).all():
                    warnings.warn(
                        "The extra backend is not returning un-normalized logits. This will cause issues with CDv2 and Proxy Tuning."
                    )

    partial_generate_composite_response = functools.partial(
        generate_composite_response,
        tokenizer_base=tokenizer_base,
        tokenizer_ref=tokenizer_ref,
        get_next_prediction_base=get_next_prediction_base,
        get_next_prediction_ref=get_next_prediction_ref,
        get_next_prediction_extra=get_next_prediction_extra,
        deferral_strategy=deferral_strategy,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
        verbose=True,
        debug=debug,
        **kwargs,
    )

    if debug:
        prompts = prompts[:3]

    if os.path.exists(save_path / "generations.json"):
        logger.debug(f"Loading existing generations from {save_path}")
        with open(save_path / "generations.json", "r") as f:
            existing_responses = json.load(f)

        empty_ids = [ele["idx"] for ele in existing_responses if ele["generated_tokens"] is None]
        prompts = [ele for ele in prompts if ele[0] in empty_ids]

    logger.debug(f"Generating responses for {len(prompts)} prompts")
    if num_proc == 1:
        all_responses = []
        for prompt in tqdm(prompts, desc="generate_response"):
            all_responses.append(partial_generate_composite_response(prompt))
    else:
        with multiprocessing.Pool(num_proc) as p:
            all_responses = list(
                tqdm(
                    p.imap(partial_generate_composite_response, prompts),
                    desc="generate_response",
                    total=len(prompts),
                )
            )

    if os.path.exists(save_path / "generations.json"):
        logger.debug(f"Merge new responses with existing ones")
        all_responses = all_responses + [ele for ele in existing_responses if ele["generated_tokens"] is not None]
        all_responses = sorted(all_responses, key=lambda x: x["idx"])

    with open(save_path / "generations.json", "w") as f:
        json.dump(all_responses, f)


def check_url(
    port_number: int,
    interval: int = 20,
):
    url = f"http://localhost:{port_number}/health"
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Service is up!")
                return
        except requests.exceptions.RequestException as e:
            print("Checking again...")
            time.sleep(interval)


if __name__ == "__main__":
    cli = {
        "deferral_generate_adaptative_th": generate_response_deferral_adaptative_threshold,
        "deferral_threshold_search": pick_deferral_threshold,
        "generate_deferral": generate_deferral,
        "generate_multi_turn": generate_multi_turn,
        "generate_default": generate_default,
        "generate_ablation": generate_ablation,
        "check_url": check_url,
    }
    fire.Fire(cli)
