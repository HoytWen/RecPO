import os
import random
from typing import Literal

import torch
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from Prompt import Prompt

import bitsandbytes as bnb
from accelerate import Accelerator
import fire

def train(
        # train
        output_dir="output/",
        logging_dir="log/",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        prompt_path="./prompt/movie_rating.txt",
        train_dataset: str = "50000",
        resume_from_checkpoint: str = "output/SFT-gpu4/",  # either training checkpoint or final adapter
        # wandb config
        wandb_project: str = "RecPo",
        wandb_name: str = "DPO",  # the name of the wandb run
        # training hyperparameters
        loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid",
        beta: float = 0.1,
        rpo_alpha: float = 0,
        neg_num: int = 1,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: int = 5,
        learning_rate: float = 1e-5,
        prompt_cutoff_len: int = 924,
        cutoff_len: int = 1024,
        eval_step=0.1,
        report_to: str = "none"
):

    ds_name, training_size = train_dataset.split('_')
    data_files = {
        f"train": f"./data/{ds_name}/{ds_name}-size10000-cans20-train.json",
        "validation": f"./data/{ds_name}/{ds_name}-cans20-val.json",
    }
    os.environ['WANDB_PROJECT'] = "-".join([wandb_project, ds_name])

    def convert_dict_to_prompt(d: dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.historyRatingList = d["historyRatingList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

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
            prompt = prompt.replace("\\n", "\n")
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
    training_args = DPOConfig(
        beta=beta,
        rpo_alpha=rpo_alpha if rpo_alpha > 0 else None,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        loss_type=loss_type,
        bf16=True,
        save_strategy="no",
        # save_steps=eval_step,
        eval_strategy="steps",
        eval_steps=eval_step,
        logging_steps=1,
        output_dir=output_dir,
        report_to=report_to,
        run_name=wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        ddp_find_unused_parameters=False,
        max_prompt_length=prompt_cutoff_len,
        max_length=cutoff_len,
    )

    dpo_trainer = DPOTrainer(
        policy_model,
        reference_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()

    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.save_model(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)


if __name__ == "__main__":
    fire.Fire(train)
