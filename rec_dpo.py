import os
os.environ["HF_HOME"] = "/mnt/ssd3/chunhui/research"
import random

import torch
from typing import Any, Dict, Literal, Optional

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig

import bitsandbytes as bnb
from accelerate import Accelerator
import fire

from Prompt import Prompt, BeerPrompt
from trainer.rec_dpo_trainer import RecDPOTrainer
from trainer.recpo_config import RecPOConfig


random.seed(1958)

def train(
        # train
        output_dir="output/",
        logging_dir="log/",
        model_name="Qwen/Qwen2.5-7B",
        prompt_path="./prompt/book_rating.txt",
        train_dataset: str = "beeradvocate_10000",
        resume_from_checkpoint: str = "output/amazon-books/Base-qwen-7B-SFT-gpu8/final_checkpoint/",  # either training checkpoint or final adapter
        # wandb config
        report_to: str = "none",
        wandb_project: str = "RecPO",
        wandb_name: str = "CPO",  # the name of the wandb run
        # training hyperparameters.
        beta: float = 1.,
        simpo_gamma: float = 0.,
        margin_lambda: float = 2,
        sft_weight: float = 0.,
        loss_type: Literal["sigmoid", "hinge", "simpo", "ipo", "cpo"] = "sigmoid",
        ln: bool = False,
        neg_num: int = 3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 3,
        learning_rate: float = 1e-5,
        prompt_cutoff_len: int = 924,
        cutoff_len: int = 1024,
        eval_step=1,
        use_score: bool = True,
        ratio: bool = True,
        negative_selection: str = "both",
):

    ds_name, training_size = train_dataset.split('_')
    data_files = {
        f"train": f"data/{ds_name}/{ds_name}-size10000-cans20-train.json",
        "validation": f"data/{ds_name}/{ds_name}-cans20-val.json",
    }
    os.environ['WANDB_PROJECT'] = "-".join([wandb_project, ds_name])

    def convert_dict_to_prompt(d: dict):
        t = BeerPrompt(prompt_path) if train_dataset.startswith("beeradvocate") else Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.historyRatingList = d["historyRatingList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def process_data(examples, selection_mode=negative_selection):
        dic = {"prompt": [], "chosen": [], "chosen_score": []}
        for i in range(1, neg_num + 1):
            dic[f"rejected{i}"] = []
            dic[f"rejected{i}_score"] = []

        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {}
            data_point["trueSelection"] = examples["trueSelection"][i]
            data_point["itemList"] = examples["itemList"][i]
            data_point["historyList"] = examples["historyList"][i]
            data_point["historyRatingList"] = examples["historyRatingList"][i]
            data_point["itemScoreList"] = examples["itemScoreList"][i]
            data_point["selectionScore"] = examples["selectionScore"][i]

            data_point["ratingNegative"] = examples["ratingNegative"][i]
            data_point["randomNegative"] = examples["randomNegative"][i]

            t = convert_dict_to_prompt(data_point)
            prompt = str(t)
            prompt = prompt.replace("\\n", "\n")

            chosen = data_point["trueSelection"]
            chosen_score = data_point["selectionScore"]

            if selection_mode == "both":
                # negative_items = data_point["ratingNegative"] + data_point["randomNegative"]
                negative_items = [x for x in data_point["itemList"] if x != chosen]
            elif selection_mode == "random":
                negative_items = data_point["randomNegative"]
            else:
                if data_point["ratingNegative"]:
                    negative_items = data_point["ratingNegative"]
                else:
                    negative_items = [x for x in data_point["itemList"] if x != chosen]

            if "sort" in selection_mode:
                sample_negs = negative_items[-neg_num:]
            else:
                negative_indices = random.sample(range(len(negative_items)), neg_num)
                sample_negs = [negative_items[x] for x in negative_indices]

            if data_point["itemScoreList"]:
                sample_neg_scores = []
                for neg in sample_negs:
                    negative_index = data_point["itemList"].index(neg)
                    sample_neg_scores.append(data_point["itemScoreList"][negative_index])
            else:
                sample_neg_scores = [0.] * neg_num

            dic["prompt"].append(prompt)
            dic["chosen"].append(chosen)
            dic["chosen_score"].append(chosen_score)
            for j in range(neg_num):
                rejected = sample_negs[j]
                rejected_score = sample_neg_scores[j]
                dic[f"rejected{j + 1}"].append(rejected)
                dic[f"rejected{j + 1}_score"].append(rejected_score)
        return dic


    data = load_dataset("json", data_files=data_files)
    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, num_proc=8, batched=True,
                                   load_from_cache_file=False).shuffle(seed=42)

    if train_data.num_rows > int(training_size):
        train_data = train_data.select(range(training_size))
    print(train_data)

    # random 2000 samples for validation
    val_data = data["validation"].map(process_data, remove_columns=columns, num_proc=8, batched=True,
                                      load_from_cache_file=False).shuffle(seed=42)
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))
    print(val_data)

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    policy_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config)
    policy_model.config.use_cache = False
    policy_model = prepare_model_for_kbit_training(policy_model)

    if resume_from_checkpoint:
        policy_model = PeftModel.from_pretrained(policy_model, resume_from_checkpoint, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        policy_model = get_peft_model(policy_model, peft_config)
    policy_model.print_trainable_parameters()

    reference_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                           device_map=device_map,
                                                           quantization_config=bnb_config)
    if resume_from_checkpoint:
        reference_model = PeftModel.from_pretrained(reference_model, resume_from_checkpoint)

    reference_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    output_dir = os.path.join(output_dir, ds_name, wandb_name)
    training_args = RecPOConfig(
        beta=beta,
        margin_lambda=margin_lambda,
        simpo_gamma=simpo_gamma,
        sft_weight=sft_weight,
        ln=ln,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        loss_type=loss_type,
        truncation_mode="keep_end",
        save_strategy="epoch",
        # save_steps=eval_step,
        eval_strategy="no",
        # eval_steps=eval_step,
        logging_steps=1,
        output_dir=output_dir,
        run_name=wandb_name,
        report_to=report_to,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        ddp_find_unused_parameters=False,
        max_prompt_length=prompt_cutoff_len,
        max_length=cutoff_len,
        use_score=use_score,
        ratio=ratio,
    )

    rec_po_trainer = RecDPOTrainer(
        policy_model,
        reference_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )
    rec_po_trainer.train()

    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    rec_po_trainer.save_model(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)


if __name__ == "__main__":
    fire.Fire(train)

