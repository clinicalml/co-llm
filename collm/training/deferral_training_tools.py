import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import datasets
import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer

from collm.safe_save_trainer import SafeSaveTrainer

IGNORE_INDEX = -100
DEFAULT_DEFERRAL_TOKEN = "<|defer|>"


class DeferralTrainerV1(SafeSaveTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if inputs.get("reference_probs") is not None:
            # For compatibility with older versions of the code
            ref_probs = inputs.pop("reference_probs")
            # [batch_size, seq_len-1] <- the same shape as shift_labels
            ref_log_probs = torch.log(ref_probs)
        elif inputs.get("reference_log_probs") is not None:
            ref_log_probs = inputs.pop("reference_log_probs")
            # [batch_size, seq_len-1] <- the same shape as shift_labels

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        original_loss = outputs.get("loss")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # fmt: off
        # below this line, we use seq_len to refer to "seq_len - 1" (since we've changed the length before) 
        ref_log_probs_flat = ref_log_probs.view(-1, 1)              # [batch_size * seq_len, 1]
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # [batch_size * seq_len, vocab_size+1]
        labels_flat = shift_labels.view(-1, )                       # [batch_size * seq_len]

        # we need to remove padded tokens before running gathering 
        no_padding_mask = labels_flat != -100                       # [batch_size * seq_len]
        labels_flat = labels_flat[no_padding_mask]
        logits_flat = logits_flat[no_padding_mask]
        ref_log_probs_flat = ref_log_probs_flat[no_padding_mask]

        def_logits = logits_flat[..., -1:]                         # [batch_size * seq_len, 1]
        def_prob = torch.nn.functional.logsigmoid(def_logits)
        not_def_prob = torch.nn.functional.logsigmoid(-def_logits)

        target_logits_flat = torch.gather(
            torch.nn.functional.log_softmax(logits_flat[..., :-1], dim=-1), 
            dim=-1, index=labels_flat.unsqueeze(-1))
                                                                   # [batch_size * seq_len, 1]

        combined_logits = torch.cat(
            [not_def_prob + target_logits_flat, def_prob + ref_log_probs_flat], 
        dim=-1)
        
        marginal_likelihood_loss = -torch.logsumexp(combined_logits, dim=-1).mean()
        # fmt: on

        self.log(
            {
                "marginal_likelihood_loss": marginal_likelihood_loss.detach().cpu().item(),
                "original_loss": original_loss.detach().cpu().item() if original_loss is not None else None,
            }
        )

        return (marginal_likelihood_loss, outputs) if return_outputs else marginal_likelihood_loss


class DeferralTrainerV2(SafeSaveTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # In this version of trainer, the reference_log_probs
        # is actually a binary variable -- 1 means when the reference
        # model has a higher probability than the base model.

        assert inputs.get("reference_log_probs") is not None
        ref_log_probs = inputs.pop("reference_log_probs")
        # [batch_size, seq_len-1] <- the same shape as shift_labels

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        original_loss = outputs.get("loss")
        assert original_loss is not None

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # fmt: off
        # below this line, we use seq_len to refer to "seq_len - 1" (since we've changed the length before) 
        ref_log_probs_flat = ref_log_probs.view(-1, 1)              # [batch_size * seq_len, 1]
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # [batch_size * seq_len, vocab_size+1]
        labels_flat = shift_labels.view(-1, )                       # [batch_size * seq_len]

        # we need to remove padded tokens before running gathering 
        no_padding_mask = labels_flat != -100                       # [batch_size * seq_len]
        logits_flat = logits_flat[no_padding_mask]
        ref_log_probs_flat = ref_log_probs_flat[no_padding_mask]

        def_logits = logits_flat[..., -1:]                         # [batch_size * seq_len, 1]

        loss_func = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.ones([1], dtype=def_logits.dtype, device=def_logits.device)*5
        )
        z_loss = loss_func(def_logits, ref_log_probs_flat.type(def_logits.dtype))

        combined_loss = 2/3 * original_loss + 1/3 * z_loss

        # def_prob = torch.nn.functional.logsigmoid(def_logits)
        # not_def_prob = torch.nn.functional.logsigmoid(-def_logits)

        # target_logits_flat = torch.gather(
        #     torch.nn.functional.log_softmax(logits_flat[..., :-1], dim=-1),
        #     dim=-1, index=labels_flat.unsqueeze(-1))
        #                                                            # [batch_size * seq_len, 1]

        # combined_logits = torch.cat(
        #     [not_def_prob + target_logits_flat, def_prob + ref_log_probs_flat],
        # dim=-1)

        # marginal_likelihood_loss = -torch.logsumexp(combined_logits, dim=-1).mean()
        # fmt: on

        self.log(
            {
                "combined_loss": combined_loss.detach().cpu().item(),
                "z_loss": z_loss.detach().cpu().item(),
                "original_loss": original_loss.detach().cpu().item() if original_loss is not None else None,
            }
        )

        return (combined_loss, outputs) if return_outputs else combined_loss


ALL_DEFERRAL_TRAINERS = {
    "v1": DeferralTrainerV1,  # The current marginal likelihood loss (decoupling def prob from other token prediction)
    "v2": DeferralTrainerV2,  # The binary classification loss, should not be used with any initialization
}


class DefaultTrainer(SafeSaveTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if inputs.get("reference_probs") is not None:
            # For compatibility with older versions of the code
            ref_probs = inputs.pop("reference_probs")
            # [batch_size, seq_len-1] <- the same shape as shift_labels
            ref_log_probs = torch.log(ref_probs)
        elif inputs.get("reference_log_probs") is not None:
            ref_log_probs = inputs.pop("reference_log_probs")
            # [batch_size, seq_len-1] <- the same shape as shift_labels

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        original_loss = outputs.get("loss")

        return (original_loss, outputs) if return_outputs else original_loss


def search_deferral_initialization_v1(
    model,
    data_module,
    batch_size=32,
    max_steps=None,
    deferral_weight=None,
):
    if torch.cuda.is_available():
        model = model.cuda()

    if hasattr(model, "lm_head"):
        output_embeddings = model.lm_head.weight.data
    elif hasattr(model, "embed_out"):
        output_embeddings = model.embed_out.weight.data

    gperp = output_embeddings.mean(dim=0, keepdim=True).detach()
    gperp.requires_grad = True
    print("output embeddings")
    print(output_embeddings)
    print(gperp)

    optimizer = torch.optim.AdamW([gperp], lr=2e-5, weight_decay=0)
    if deferral_weight is None:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.BCELoss(torch.tensor([1, deferral_weight]).float().to(model.device))

    N = len(data_module["train_dataset"])
    data_collator = data_module["data_collator"]

    if max_steps is None:
        max_steps = N // batch_size

    model_dtype = None

    for epoch in range(1):
        for step, index in enumerate(tqdm(range(0, N, batch_size))):
            cur_examples = [data_module["train_dataset"][i] for i in range(index, min(index + batch_size, N))]

            model_input = data_collator(cur_examples)
            input_ids = model_input["input_ids"].to(model.device)
            attention_mask = model_input["attention_mask"].to(model.device)
            ref_log_probs = model_input["reference_log_probs"].to(model.device)  # we store as binary labels

            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden_state = output.hidden_states[-1]

            deferral_token_logits = last_hidden_state @ gperp.T  # [batch_size, seq_len, 1]
            deferral_token_logits = deferral_token_logits[:, :-1]  # We take till the second last token
            B, L, _ = deferral_token_logits.size()

            deferral_token_probs_flat = torch.sigmoid(deferral_token_logits).reshape(B * L)
            ref_log_probs_flat = ref_log_probs.reshape(B * L).float()
            ignore_mask = ref_log_probs_flat != IGNORE_INDEX

            # BCELoss does not support setting ignore_index to -100
            # So we need to manually mask them

            if deferral_weight is not None:
                # Also we have to manually set the weights
                loss_weights = torch.ones_like(ref_log_probs_flat[ignore_mask])  # .type(model_dtype)
                loss_weights[ref_log_probs_flat[ignore_mask] == 1] = deferral_weight
                criterion = torch.nn.BCELoss(loss_weights)

            if model_dtype is None:
                model_dtype = deferral_token_probs_flat.dtype

            loss = criterion(deferral_token_probs_flat[ignore_mask], ref_log_probs_flat[ignore_mask].type(model_dtype))
            print(loss.item())  # , gperp[0, :10], gperp.norm())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step > max_steps:
                break

    gperp = gperp.to("cpu")
    return gperp


ALL_INITIALIZATION_SEARCH = {
    "v1": search_deferral_initialization_v1,  # Regular v3 objective
}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # See the explanation of the code here in the __getitem__ of the dataset class
        input_ids, labels_reference_log_probs = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        labels, reference_log_probs = tuple(zip(*labels_reference_log_probs))

        # We need to have left padding during training
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [ele.flip(dims=[0]) for ele in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).flip(dims=[1])
        labels = torch.nn.utils.rnn.pad_sequence(
            [ele.flip(dims=[0]) for ele in labels], batch_first=True, padding_value=IGNORE_INDEX
        ).flip(dims=[1])
        reference_log_probs = torch.nn.utils.rnn.pad_sequence(
            [ele.flip(dims=[0]) for ele in reference_log_probs], batch_first=True, padding_value=IGNORE_INDEX
        ).flip(dims=[1])

        return dict(
            input_ids=input_ids,
            labels=labels,
            reference_log_probs=reference_log_probs,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str):
        super(SupervisedDataset, self).__init__()

        data_dict = datasets.load_from_disk(data_path)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.reference_log_probs = data_dict["reference_log_probs"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: WARNING: this is an ugly hack to pass the reference probs to the data collator
        # Because huaggingface trainer will filter out the arguments that are not in the signature of
        # the model, so we need to "fake" the labels to be the reference probs
        # While it works, we should find a better way to do this in the future
        # Also see the data collator below
        return dict(
            input_ids=self.input_ids[i],
            labels=(self.labels[i], self.reference_log_probs[i]),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_path=data_args.dataset_name)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    embedding_size = model.get_input_embeddings().weight.shape[0]
    num_new_tokens = len(tokenizer) - embedding_size
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )  # try to use max value for training
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )  # try to use max value for training

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def smart_tokenizer_and_embedding_resize_for_def_token(
    tokenizer: transformers.PreTrainedTokenizer,
    model: Optional[transformers.PreTrainedModel] = None,
    gperp: Optional[torch.Tensor] = None,
):
    """Different from smart_tokenizer_and_embedding_resize, which uses the mean
    vector values for the new tokens, this function will use a different type of initialization for the new deferral tokens based on the paper.
    """

    num_new_tokens = 1  # Only one deferral token
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True
    )  # try to use max value for training
    input_embeddings[-num_new_tokens:] = input_embeddings_avg

    if gperp is not None:
        output_embeddings[-num_new_tokens:, :] = gperp
    else:
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].logsumexp(
        #     dim=0, keepdim=True
        # )  # try to use max value for training
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
