
import inspect
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib


import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (DataCollator,
                          PreTrainedModel,
                          AutoModelForCausalLM,
                          PreTrainedTokenizerBase,
                          Trainer,
                          is_wandb_available)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import is_peft_available
from accelerate import PartialState

from .utils import RecPODataCollatorWithPadding, pad_to_length
from .recpo_config import RecPOConfig

if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel


class RecCPOTrainer(Trainer):
    r"""
    Initialize RecPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: Optional[RecPOConfig] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
                    None,
                    None,
            ),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional[Dict] = None,
    ):

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the CPOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the CPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the CPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # self._peft_has_been_casted_to_bf16 = False
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)


        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id


        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if args.max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            else:
                max_length = args.max_length
            if args.max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128
            else:
                max_prompt_length = args.max_prompt_length

            if args.max_completion_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using an encoder decoder architecture, you should set `max_completion_length` in the CPOConfig's init"
                    " it will default to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_completion_length = 128
            else:
                max_completion_length = args.max_completion_length

            if data_collator is None:
                data_collator = RecPODataCollatorWithPadding(
                    pad_token_id=tokenizer.pad_token_id,
                    label_pad_token_id=args.label_pad_token_id,
                    is_encoder_decoder=self.is_encoder_decoder,
                )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_recpo_data_collator = True
        else:
            self.use_recpo_data_collator = False


        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = args.max_completion_length
        self.tokenizer = tokenizer
        self.use_score = args.use_score
        self.ratio = args.ratio

        if args.loss_type in ["hinge", "ipo"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        if args.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in CPOTrainer. Please use KTOTrainer.")

        self.beta = args.beta
        self.simpo_gamma = args.simpo_gamma
        self.margin_lambda = args.margin_lambda
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.cpo_alpha = args.cpo_alpha
        self.ln = args.ln
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            None,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def tokenize_row(self, feature) -> Dict:

        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = {}
        for key in feature:
            if key.startswith("rejected") and not key.endswith("score"):
                rejected[key] = feature[key]

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = {}
        for key in rejected:
            rejected_tokens[key] = self.tokenizer(rejected[key], add_special_tokens=False)

        assert self.tokenizer.bos_token_id not in prompt_tokens["input_ids"], f"Prompt contains BOS token: {prompt}"
        assert self.tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
        assert (
                self.tokenizer.eos_token_id not in chosen_tokens["input_ids"]
        ), f"Chosen response contains EOS token: {chosen}"
        assert (
            all([self.tokenizer.eos_token_id not in rejected_tokens[key]["input_ids"] for key in rejected_tokens])
        ), f"Rejected response contains EOS token: {rejected}"

        prompt_tokens["input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["input_ids"]
        prompt_tokens["attention_mask"] = [1] + prompt_tokens["attention_mask"]
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)
        for key in rejected_tokens:
            rejected_tokens[key]["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens[key]["attention_mask"].append(1)

        max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
        longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length:] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()}

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {}
        # rejected_tokens: Dict[str, Dict]
        for key in rejected_tokens:
            rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in
                                             rejected_tokens[key]}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        for key in rejected_sequence_tokens:
            rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
            rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = (
                    [self.label_pad_token_id] * len(prompt_tokens["input_ids"]))

        batch = {}

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        for key in rejected:
            batch[key] = prompt + rejected[key]
        batch["chosen_response_only"] = chosen
        for key in rejected:
            batch[f"{key}_response_only"] = rejected[key]

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            # "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        # rejected_sequence_tokens: Dict[str, Dict]
        for k, toks in rejected_sequence_tokens.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        if self.use_score:
            chosen_score = feature["chosen_score"]
            rejected_score = {}
            for key in feature:
                if key.startswith("rejected") and key.endswith("score"):
                    rejected_score[key] = feature[key]

            batch["chosen_score"] = chosen_score
            for key in rejected_score:
                batch[key] = rejected_score[key]

        return batch

    @staticmethod
    def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]],
                            is_encoder_decoder: bool = False,
                            label_pad_token_id: int = -100,
                            padding_value: int = 0,
                            device: Optional[torch.device] = None,
                            ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        # 把 chosen 和 rejected response 拼接起来
        rejected_max_len = max(
            [batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = label_pad_token_id if "labels" in k else padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = label_pad_token_id if "labels" in k else padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):]
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        return concatenated_batch

    def recpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: Dict[str, torch.FloatTensor],
            chosen_score:  Optional[List[torch.FloatTensor]] = None,
            rejected_score: Optional[Dict[str, List[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        # chosen_logratios = policy_chosen_logps - reference_chosen_logps
        # rejected_logratios = {}
        # for key in policy_rejected_logps:
        #     rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key]

        logits_dict = {}
        for key in policy_rejected_logps:
            logits_dict[key] = (policy_rejected_logps[key] - policy_chosen_logps).to(self.accelerator.device)

        if self.use_score:
            if chosen_score is not None and rejected_score is not None:
                for key in logits_dict:
                    if self.ratio:
                        margin = (torch.cat(chosen_score, dim=0) / torch.cat(rejected_score[f"{key}_score"],
                                                                                 dim=0)).to(
                            self.accelerator.device)
                        margin = margin / self.beta
                        logits_dict[key] = logits_dict[key] + self.margin_lambda * margin
                    else:
                        margin = (torch.cat(chosen_score, dim=0) - torch.cat(rejected_score[f"{key}_score"],
                                                                                 dim=0)).to(
                            self.accelerator.device) / self.beta
                        margin = torch.log(1 + margin) / self.beta
                        logits_dict[key] = logits_dict[key] + self.margin_lambda * margin
            else:
                raise ValueError("The score value is not assigned")

        if self.loss_type == "simpo":
            gamma_beta_ratio = self.simpo_gamma / self.beta
            logits = sum(torch.exp(self.beta * (logits_dict[key] + gamma_beta_ratio)) for key in logits_dict)
            logits = -torch.log(logits)
            # logits = logits - gamma_beta_ratio
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-1 * logits) * self.label_smoothing
            )
        elif self.loss_type == "sigmoid":
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            logits = sum(torch.exp(self.beta * logits_dict[key]) for key in logits_dict)
            # assert logits.shape == policy_chosen_logps.shape
            logits = -torch.log(logits)
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-1 * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            logits = sum(torch.exp(self.beta * logits_dict[key]) for key in logits_dict)
            logits = -torch.log(logits)
            losses = torch.relu(1 - logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            logits = sum(torch.exp(self.beta * logits_dict[key]) for key in logits_dict)
            logits = -torch.log(logits)
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'simpo']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = {}
        for key in policy_rejected_logps:
            rejected_rewards[key] = self.beta * (policy_rejected_logps[key].to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def _get_batch_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            average_log_prob: bool = False,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(batch,
                                                      is_encoder_decoder=self.is_encoder_decoder,
                                                      label_pad_token_id=self.label_pad_token_id,
                                                      padding_value=self.padding_value,
                                                      device=self.accelerator.device,
                                                      )
        # print(concatenated_batch["concatenated_input_ids"].shape)
        len_chosen = batch["chosen_labels"].shape[0]

        output = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
        )
        all_logits = output.logits.to(torch.float32)

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0).to(self.accelerator.device)
        else:
            nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.ln,
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder
        )

        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step * cnt: step * (cnt + 1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step * cnt: step * (cnt + 1)]


        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def get_batch_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = self.concatenated_forward(model, batch)

        if self.use_score:
            chosen_score = batch["chosen_score"]
            rejected_score = {}
            for key in batch:
                if key.startswith("rejected") and key.endswith("score"):
                    rejected_score[key] = batch[key]
            losses, chosen_rewards, rejected_rewards = self.recpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                chosen_score,
                rejected_score,
            )
        else:
            losses, chosen_rewards, rejected_rewards = self.recpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
            )

        loss = losses.mean() + self.cpo_alpha * policy_nll_loss
        # reward_accuracies 记录 chosen 比所有 rejected 的收益都大的比例是多少
        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        for key in policy_rejected_logits:
            metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        return loss, metrics

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if not self.use_recpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        # compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_recpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "logits_test/chosen": metrics["logits_test/chosen"],
            # "logits_test/rejected": metrics["logits_test/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)




