import os
import torch
import re
import random

from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
# from trainer.rec_po_trainer import RecPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import find_all_linear_names, print_trainable_parameters

import bitsandbytes as bnb
from accelerate import Accelerator
import fire

from Prompt import Prompt

random.seed(1958)
def train(
        # train
        output_dir="output/",
        logging_dir="log/",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        prompt_path="./prompt/movie_rating2.txt",
        dataset="",
        resume_from_checkpoint: str = "output/sft_checkpoint/",  # either training checkpoint or final adapter
        # wandb config
        wandb_project: str = "Rec-PO",
        wandb_name: str = "PO",  # the name of the wandb run
        # training hyperparameters
        beta: float = 0.1,
        neg_num: int = 1,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 1,
        learning_rate: float = 1e-5,
        cutoff_len: int = 512,
        eval_step=1,
):

    data_files = {
        "train": "/home/ericwen/Rec-PO/data/movielens-1m/movielens-size10000-cans20-train.json",
        "validation": "/home/ericwen/Rec-PO/data/movielens-1m/movielens-cans20-val.json",
    }

    def convert_dict_to_prompt(d: dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.historyRatingList = d["historyRatingList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def process_train_data(examples):
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

            t = convert_dict_to_prompt(data_point)
            prompt = str(t)

            chosen = data_point["trueSelection"]
            chosen_score = data_point["selectionScore"]

            selection_index = data_point["itemList"].index(data_point["trueSelection"])
            negative_items = [data_point["itemList"][x] for x in range(len(data_point["itemList"])) if
                              x != selection_index]
            negative_scores = [data_point["itemScoreList"][x] for x in range(len(data_point["itemScoreList"])) if
                               x != selection_index]

            indices = random.sample(range(len(negative_items)), neg_num)
            sample_negs = [negative_items[x] for x in indices]
            sample_neg_scores = [negative_scores[x] for x in indices]

            dic["prompt"].append(prompt)
            dic["chosen"].append(chosen)
            dic["chosen_score"].append(chosen_score)
            for j in range(neg_num):
                rejected = sample_negs[j]
                rejected_score = sample_neg_scores[j]
                dic[f"rejected{j+1}"].append(rejected)
                dic[f"rejected{j+1}_score"].append(rejected_score)
        return dic

    def process_eval_data(examples):
        dic = {"prompt": [], "chosen": []}
        for i in range(1, neg_num + 1):
            dic[f"rejected{i}"] = []

        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {}
            data_point["trueSelection"] = examples["trueSelection"][i]
            data_point["itemList"] = examples["itemList"][i]
            data_point["historyList"] = examples["historyList"][i]
            data_point["historyRatingList"] = examples["historyRatingList"][i]

            t = convert_dict_to_prompt(data_point)
            prompt = str(t)

            chosen = data_point["trueSelection"]

            selection_index = data_point["itemList"].index(data_point["trueSelection"])
            negative_items = [data_point["itemList"][x] for x in range(len(data_point["itemList"])) if
                              x != selection_index]

            indices = random.sample(range(len(negative_items)), neg_num)
            sample_negs = [negative_items[x] for x in indices]

            dic["prompt"].append(prompt)
            dic["chosen"].append(chosen)
            for j in range(neg_num):
                rejected = sample_negs[j]
                dic[f"rejected{j + 1}"].append(rejected)
        return dic

    def process_data(examples):
        dic = {"prompt":[], "chosen":[], "rejected":[]}
        columns = list(examples.keys())
        for i in range(len(examples[columns[0]])):
            data_point = {}
            data_point["trueSelection"] = examples["trueSelection"][i]
            data_point["itemList"] = examples["itemList"][i]
            data_point["historyList"] = examples["historyList"][i]
            data_point["historyRatingList"] = examples["historyRatingList"][i]
            t = convert_dict_to_prompt(data_point)
            prompt = str(t)
            chosen = data_point["trueSelection"]
            negative_items = [item for item in data_point["itemList"] if item != data_point["trueSelection"]]
            sample_negs = random.sample(negative_items, neg_num)
            for rejected in sample_negs:
                dic['prompt'].append(prompt)
                dic['chosen'].append(chosen)
                dic['rejected'].append(rejected)

        return dic

    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, num_proc=8, batched=True).shuffle(seed=42)
    print(train_data)

    # random 2000 samples for validation
    val_data = data["validation"].map(process_data, remove_columns=columns, num_proc=8, batched=True).shuffle(seed=42)

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

    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    # )
    # peft_model = get_peft_model(base_model, peft_config)

    peft_policy_model = PeftModel.from_pretrained(policy_model, resume_from_checkpoint, is_trainable=True)
    peft_policy_model.print_trainable_parameters()


    ref_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map=device_map,
                                                     quantization_config=bnb_config)
    peft_ref_model = PeftModel.from_pretrained(ref_model, resume_from_checkpoint)
    peft_ref_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training

    training_args = DPOConfig(
        beta=beta,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_step,
        save_total_limit=100,
        evaluation_strategy="steps",
        eval_steps=eval_step,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to="wandb",
        run_name=wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        peft_policy_model,
        peft_ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )

    dpo_trainer.train()

    output_dir = os.path.join(output_dir, "final_checkpoint")
    # dpo_trainer.model.save_pretrained(output_dir)
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)

