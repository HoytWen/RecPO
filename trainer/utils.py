import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback


# @dataclass
# class RecPODataCollatorWithPadding:
#     r"""
#     DPO DataCollator class that pads the inputs to the maximum length of the batch.
#     Args:
#         tokenizer (`PreTrainedTokenizerBase`):
#             The tokenizer used for encoding the data.
#         padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
#             padding_strategy to pass to the tokenizer.
#         max_length (`Optional[int]`, `optional`, defaults to `None`):
#             The maximum length of the sequence to be processed.
#         max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
#             The maximum length of the prompt to be processed.
#         label_pad_token_id (`int`, defaults to -100):
#             The label used for masking.
#         padding_value (`int`, defaults to 0):
#             The value used for padding.
#         truncation_mode: (`str`, defaults to "keep_end"):
#             The truncation mode to use when truncating the prompt + chosen/rejected responses.
#     """
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str] = True
#     max_length: Optional[int] = None
#     max_prompt_length: Optional[int] = None
#     label_pad_token_id: int = -100
#     padding_value: int = 0
#     truncation_mode: str = "keep_end"
#     use_score: bool = False
#
#     def tokenize_batch_element(
#             self,
#             prompt: str,
#             chosen: str,
#             rejected: Dict[str, str],
#             chosen_score=None,
#             rejected_score: Dict=None,
#     ) -> Dict:
#         """Tokenize a single batch element.
#
#         At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
#             in case the prompt + chosen or prompt + rejected responses is/are too long. First
#             we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
#
#         We also create the labels for the chosen/rejected responses, which are of length equal to
#             the sum of the length of the prompt and the chosen/rejected response, with
#             label_pad_token_id  for the prompt tokens.
#         """
#         chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
#         prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
#         rejected_tokens = {}
#         for key in rejected:
#             rejected_tokens[key] = self.tokenizer(rejected[key], add_special_tokens=False)
#
#         assert self.tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
#         assert (
#                 self.tokenizer.eos_token_id not in chosen_tokens["input_ids"]
#         ), f"Chosen response contains EOS token: {chosen}"
#         assert (
#             all([self.tokenizer.eos_token_id not in rejected_tokens[key]["input_ids"] for key in rejected_tokens])
#         ), f"Rejected response contains EOS token: {rejected}"
#
#         chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
#         chosen_tokens["attention_mask"].append(1)
#         for key in rejected_tokens:
#             rejected_tokens[key]["input_ids"].append(self.tokenizer.eos_token_id)
#             rejected_tokens[key]["attention_mask"].append(1)
#
#         max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
#         longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)
#
#         # if combined sequence is too long, truncate the prompt
#         if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
#             if self.truncation_mode == "keep_start":
#                 prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
#             elif self.truncation_mode == "keep_end":
#                 prompt_tokens = {k: v[-self.max_prompt_length:] for k, v in prompt_tokens.items()}
#             else:
#                 raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
#
#         # if that's still too long, truncate the response
#         if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
#             chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
#             rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}
#
#         # Create labels
#         chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
#         rejected_sequence_tokens = {}
#         # rejected_tokens: Dict[str, Dict]
#         for key in rejected_tokens:
#             rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in
#                                              rejected_tokens[key]}
#         chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
#         chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
#             prompt_tokens["input_ids"]
#         )
#         for key in rejected_sequence_tokens:
#             rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
#             rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = (
#                     [self.label_pad_token_id] * len(prompt_tokens["input_ids"]))
#
#         batch = {}
#
#         batch["prompt"] = prompt
#         batch["chosen"] = prompt + chosen
#         for key in rejected:
#             batch[key] = prompt + rejected[key]
#         batch["chosen_response_only"] = chosen
#         for key in rejected:
#             batch[f"{key}_response_only"] = rejected[key]
#
#         for k, toks in {
#             "chosen": chosen_sequence_tokens,
#             # "rejected": rejected_sequence_tokens,
#             "prompt": prompt_tokens,
#         }.items():
#             for type_key, tokens in toks.items():
#                 if type_key == "token_type_ids":
#                     continue
#                 batch[f"{k}_{type_key}"] = tokens
#         # rejected_sequence_tokens: Dict[str, Dict]
#         for k, toks in rejected_sequence_tokens.items():
#             for type_key, tokens in toks.items():
#                 if type_key == "token_type_ids":
#                     continue
#                 batch[f"{k}_{type_key}"] = tokens
#
#         if self.use_score:
#             batch["chosen_score"] = chosen_score
#             for key in rejected_score:
#                 batch[key] = rejected_score[key]
#
#         return batch
#
#     def collate(self, batch):
#         # first, pad everything to the same length
#         padded_batch = {}
#         for k in batch[0].keys():
#             if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
#                 # adapted from https://stackoverflow.com/questions/73256206
#                 if "prompt" in k:
#                     to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
#                 else:
#                     to_pad = [torch.LongTensor(ex[k]) for ex in batch]
#                 if k.endswith("_input_ids"):
#                     padding_value = self.tokenizer.pad_token_id
#                 elif k.endswith("_labels"):
#                     padding_value = self.label_pad_token_id
#                 elif k.endswith("_attention_mask"):
#                     padding_value = self.padding_value
#                 else:
#                     raise ValueError(f"Unexpected key in batch '{k}'")
#
#                 padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
#                 # for the prompt, flip back so padding is on left side
#                 if "prompt" in k:
#                     padded_batch[k] = padded_batch[k].flip(dims=[1])
#             elif k.endswith("score"):
#                 padded_batch[k] = [torch.FloatTensor([ex[k]]) for ex in batch]
#                 # padded_batch[k] = torch.FloatTensor(temp)
#             else:
#                 padded_batch[k] = [ex[k] for ex in batch]
#
#         return padded_batch
#
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         tokenized_batch = []
#
#         for feature in features:
#             prompt = feature["prompt"]
#             chosen = feature["chosen"]
#
#             rejected = {}
#             for key in feature:
#                 if key.startswith("rejected") and not key.endswith("score"):
#                     rejected[key] = feature[key]
#
#             if self.use_score:
#                 chosen_score = feature["chosen_score"]
#                 rejected_score = {}
#                 for key in feature:
#                     if key.startswith("rejected") and key.endswith("score"):
#                         rejected_score[key] = feature[key]
#                 batch_element = self.tokenize_batch_element(prompt, chosen, rejected, chosen_score, rejected_score)
#             else:
#                 batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
#
#             tokenized_batch.append(batch_element)
#
#         # return collated batch
#         return self.collate(tokenized_batch)


@dataclass
class RecPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            elif k.endswith("score"):
                padded_batch[k] = [torch.FloatTensor([ex[k]]) for ex in features]
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output